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

_MAX_BLOCKS = 50
_MAX_CONTENT_LEN = 2000

_LLM_COST_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    info = _LLM_COST_PER_1M.get(model, {})
    return (
        input_tokens * info.get("input", 0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


def _serialize_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> str:
    """Serialize document blocks into '[{type}] {content}' lines.

    Truncates each block's content to _MAX_CONTENT_LEN chars to avoid
    token overflow. Limits total blocks to max_blocks.
    """
    lines = []
    for block in doc.blocks[:max_blocks]:
        content = block.content[:_MAX_CONTENT_LEN]
        lines.append(f"[{block.type.value}] {content}")
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
        client = openai.OpenAI()
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
            result = json.loads(raw)
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
