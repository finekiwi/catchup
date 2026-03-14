"""LLM-based study note generator — multi-provider.

Supports OpenAI, Anthropic, and Google Gemini for note generation.
Calls chat completion API with serialized Document blocks and returns
a structured learning note dict.

Supported providers:
- OpenAI   : gpt-4o-mini, gpt-4o
- Anthropic: claude-haiku-4-5-20251001, claude-sonnet-4-6
- Google   : gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from models.document import BlockType, Document, ProcessingStatus
from prompts.note_generation import PROMPT, PROMPT_VERSION
from utils.logging import log_api_call
from llm.schemas import NoteGenerationOutput

load_dotenv()

LOGGER = logging.getLogger(__name__)

_MAX_BLOCKS = 200
_MAX_CONTENT_LEN = 1200      # per block in normal mode
_MAX_CONTENT_LEN_LARGE = 600 # per block when doc has many blocks
_LARGE_DOC_THRESHOLD = 40    # blocks: above this, use large-doc strategy
_MAX_CODE_LINES = 15         # code blocks: only first N lines passed to LLM

# ---------------------------------------------------------------------------
# Model registry: model_id → {provider, input_cost_per_1m, output_cost_per_1m}
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini":               {"provider": "openai",     "input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"provider": "openai",     "input": 2.50,  "output": 10.00},
    "gpt-4.1-mini":              {"provider": "openai",     "input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":              {"provider": "openai",     "input": 0.10,  "output": 0.40},
    "gpt-5-nano":                {"provider": "openai",     "input": 0.20,  "output": 0.80},
    "claude-haiku-4-5-20251001": {"provider": "anthropic",  "input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"provider": "anthropic",  "input": 3.00,  "output": 15.00},
    "gemini-3-flash-preview":         {"provider": "google",     "input": 0.10,  "output": 0.40},
    "gemini-3.1-pro-preview":         {"provider": "google",     "input": 1.25,  "output": 10.00},
    "gemini-3.1-flash-lite-preview":  {"provider": "google",     "input": 0.04,  "output": 0.15},
}

SUPPORTED_LLM_MODELS: list[str] = list(_MODEL_REGISTRY.keys())


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    info = _MODEL_REGISTRY.get(model, {})
    return (
        input_tokens * info.get("input", 0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


# ---------------------------------------------------------------------------
# Provider-level call functions — each returns (raw_text, input_tokens, output_tokens)
# ---------------------------------------------------------------------------

def _call_openai(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call OpenAI chat completion API."""
    import openai  # lazy import

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content or ""
    return raw, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_anthropic(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call Anthropic Claude chat API."""
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text if resp.content else ""
    return raw, resp.usage.input_tokens, resp.usage.output_tokens


def _call_google(model: str, system: str, user: str) -> tuple[str, int, int]:
    """Call Google Gemini chat API."""
    import google.generativeai as genai  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = genai.GenerativeModel(model, system_instruction=system)
    resp = gemini.generate_content(user)
    raw = resp.text or ""
    meta = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(meta, "prompt_token_count", 0) or 0
    output_tokens = getattr(meta, "candidates_token_count", 0) or 0
    return raw, input_tokens, output_tokens


_PROVIDER_DISPATCH: dict[str, Any] = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "google":    _call_google,
}


_MIN_FIGURE_CONTENT_LEN = 20  # figure blocks shorter than this carry no useful text


def _sample_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> list:
    """Return a representative block sample from the document.

    Figure blocks with very short content (< _MIN_FIGURE_CONTENT_LEN chars) are
    excluded before sampling — they have no extractable text and only waste tokens.
    Figure blocks with VLM-generated descriptions (longer content) are kept.

    For large documents, evenly samples across the filtered block list so the
    LLM sees content from beginning, middle, and end rather than just the
    first N blocks.
    """
    blocks = [
        b for b in doc.blocks
        if not (b.type == BlockType.FIGURE and len(b.content.strip()) < _MIN_FIGURE_CONTENT_LEN)
    ]
    if len(blocks) <= max_blocks:
        return blocks

    # Evenly spaced indices across the full document
    step = len(blocks) / max_blocks
    indices = {int(i * step) for i in range(max_blocks)}
    return [blocks[i] for i in sorted(indices)]


def _truncate_code(content: str, max_lines: int = _MAX_CODE_LINES) -> str:
    """Return first max_lines of a code block with an omission note if truncated."""
    code_lines = content.splitlines()
    if len(code_lines) <= max_lines:
        return content
    kept = "\n".join(code_lines[:max_lines])
    omitted = len(code_lines) - max_lines
    return f"{kept}\n# ... ({omitted} lines omitted)"


def _serialize_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> str:
    """Serialize document blocks into '[{type}] {content}' lines.

    For large documents (> _LARGE_DOC_THRESHOLD blocks), samples evenly
    across the document and applies a shorter per-block content limit so
    the LLM receives representative coverage rather than just the beginning.
    CODE blocks are truncated to _MAX_CODE_LINES to prevent the LLM from
    copying raw code into note_markdown.
    """
    is_large = len(doc.blocks) > _LARGE_DOC_THRESHOLD
    content_limit = _MAX_CONTENT_LEN_LARGE if is_large else _MAX_CONTENT_LEN
    sampled = _sample_blocks(doc, max_blocks)

    lines = []
    for block in sampled:
        if block.type == BlockType.CODE:
            content = _truncate_code(block.content)
            lines.append(f"[code] {content}")
        else:
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

    Routes to the appropriate provider (OpenAI / Anthropic / Google) based on
    the model identifier. Document blocks are serialized as '[{type}] {content}'
    lines and sent as user content with PROMPT as the system message.

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
        model: Model identifier. Must be one of SUPPORTED_LLM_MODELS.

    Returns:
        dict with keys: title, summary, note_markdown, key_concepts,
        difficulty_level, estimated_read_time_min, schema_version,
        confidence, errors.

    Raises:
        ValueError: If model is not in SUPPORTED_LLM_MODELS.
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model!r}. Choose from: {SUPPORTED_LLM_MODELS}")

    provider = _MODEL_REGISTRY[model]["provider"]
    call_fn = _PROVIDER_DISPATCH[provider]
    user_content = _serialize_blocks(doc)
    t0 = time.perf_counter()

    try:
        raw, input_tokens, output_tokens = call_fn(model, PROMPT, user_content)
        latency_ms = (time.perf_counter() - t0) * 1000
        cost_usd = _compute_cost(model, input_tokens, output_tokens)

        try:
            result = json.loads(_strip_markdown_fence(raw))
            if not isinstance(result, dict):
                raise ValueError("LLM response JSON must be an object")
            try:
                result = NoteGenerationOutput.model_validate(result).model_dump()
            except Exception as val_exc:
                LOGGER.warning("NoteGenerationOutput schema validation warning: %s", val_exc)
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
            return _make_fallback(doc, raw, "노트 구조 분석이 불완전합니다. 원문 형식으로 표시됩니다.")

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


__all__ = ["generate_note", "SUPPORTED_LLM_MODELS"]
