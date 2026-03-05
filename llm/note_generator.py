"""LLM-based study note generator.

Calls OpenAI chat completion API with serialized Document blocks
and returns a structured learning note dict.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai
from dotenv import load_dotenv

from models.document import Document, ProcessingStatus
from prompts.note_generation import PROMPT, PROMPT_VERSION
from utils.logging import log_api_call

load_dotenv()

LOGGER = logging.getLogger(__name__)

_MAX_BLOCKS = 40
_MAX_CONTENT_LEN = 800       # per block in normal mode
_MAX_CONTENT_LEN_LARGE = 400 # per block when doc has many blocks
_LARGE_DOC_THRESHOLD = 30    # blocks: above this, use large-doc strategy

_LLM_COST_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

_openai_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    """Return a module-level cached OpenAI client (lazy init)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    info = _LLM_COST_PER_1M.get(model, {})
    return (
        input_tokens * info.get("input", 0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


def _sample_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> list:
    """Return a representative block sample from the document.

    For large documents, evenly samples across the full block list so the
    LLM sees content from beginning, middle, and end rather than just the
    first N blocks.
    """
    blocks = doc.blocks
    if len(blocks) <= max_blocks:
        return blocks

    # Evenly spaced indices across the full document
    step = len(blocks) / max_blocks
    indices = {int(i * step) for i in range(max_blocks)}
    return [blocks[i] for i in sorted(indices)]


def _serialize_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> str:
    """Serialize document blocks into '[{type}] {content}' lines.

    For large documents (> _LARGE_DOC_THRESHOLD blocks), samples evenly
    across the document and applies a shorter per-block content limit so
    the LLM receives representative coverage rather than just the beginning.
    """
    is_large = len(doc.blocks) > _LARGE_DOC_THRESHOLD
    content_limit = _MAX_CONTENT_LEN_LARGE if is_large else _MAX_CONTENT_LEN
    sampled = _sample_blocks(doc, max_blocks)

    lines = []
    for block in sampled:
        content = block.content[:content_limit]
        lines.append(f"[{block.type.value}] {content}")
    return "\n".join(lines)


def _strip_markdown_fence(text: str) -> str:
    """Remove a single outer markdown code fence if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _make_fallback(doc: Document, raw_response: str, error_msg: str) -> dict[str, Any]:
    """Build a fallback dict when JSON parsing fails or API errors out."""
    return {
        "title": doc.source,
        "summary": "",
        "note_markdown": raw_response,
        "key_concepts": [],
        "difficulty_level": "unknown",
        "estimated_read_time_min": 0,
        "schema_version": PROMPT_VERSION,
        "confidence": 0.0,
        "errors": [error_msg],
    }


def generate_note(doc: Document, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Generate a structured study note dict from a Document.

    Calls OpenAI chat completion API with document blocks serialized as
    '[{type}] {content}' lines (system: PROMPT, user: serialized blocks).

    On success:
    - Parses the JSON response and returns it as a dict.
    - Sets doc.status = ProcessingStatus.NOTE_GENERATED.

    On JSON parse failure:
    - Returns fallback dict with raw_response in note_markdown field.
    - Adds 'note_generation_failed' tag to doc.metadata.tags.

    On API failure:
    - Logs the error and returns fallback dict with empty note_markdown.
    - Adds 'note_generation_failed' tag to doc.metadata.tags.
    - Never raises exceptions.

    Args:
        doc: Source document with populated blocks.
        model: OpenAI model identifier (default: gpt-4o-mini).

    Returns:
        dict with keys: title, summary, note_markdown, key_concepts,
        difficulty_level, estimated_read_time_min, schema_version,
        confidence, errors.
    """
    user_content = _serialize_blocks(doc)
    t0 = time.perf_counter()

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content or ""
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        cost_usd = _compute_cost(model, input_tokens, output_tokens)

        try:
            result = json.loads(_strip_markdown_fence(raw))
            if not isinstance(result, dict):
                raise ValueError("LLM response JSON must be an object")
        except (json.JSONDecodeError, ValueError) as parse_exc:
            LOGGER.warning("Note generation JSON parse failed: %s", parse_exc)
            log_api_call(
                model=model,
                stage="note_generation",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                success=False,
                error=f"JSON parse failed: {parse_exc}",
            )
            if "note_generation_failed" not in doc.metadata.tags:
                doc.metadata.tags.append("note_generation_failed")
            return _make_fallback(doc, raw, "JSON parse failed")

        log_api_call(
            model=model,
            stage="note_generation",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=True,
            error=None,
        )
        doc.status = ProcessingStatus.NOTE_GENERATED
        return result

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        error_msg = str(exc)
        LOGGER.error("Note generation API call failed: %s", exc)
        log_api_call(
            model=model,
            stage="note_generation",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=False,
            error=error_msg,
        )
        if "note_generation_failed" not in doc.metadata.tags:
            doc.metadata.tags.append("note_generation_failed")
        return _make_fallback(doc, "", error_msg)


__all__ = ["generate_note"]
