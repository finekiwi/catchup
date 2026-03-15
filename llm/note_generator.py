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
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv

from models.document import Block, BlockType, Document, ProcessingStatus
from prompts.note_generation import PROMPT, PROMPT_VERSION
from utils.logging import log_api_call
from llm.schemas import NoteGenerationOutput
from llm.block_filter import is_noise_block

load_dotenv()

LOGGER = logging.getLogger(__name__)

_MAX_BLOCKS = 200
_MAX_CONTENT_LEN = 1200  # per block in normal mode
_MAX_CONTENT_LEN_LARGE = 600  # per block when doc has many blocks
_LARGE_DOC_THRESHOLD = 40  # blocks: above this, use large-doc strategy
_MAX_CODE_LINES = 15  # code blocks: only first N lines passed to LLM

# ---------------------------------------------------------------------------
# Model registry: model_id → {provider, input_cost_per_1m, output_cost_per_1m}
# Mirrors utils/models.MODEL_REGISTRY — keep in sync when adding or removing models.
# NOTE: This duplicate exists because note_generator uses force_json / per-provider
#       timeout extensions that are not yet supported by utils.models.call_llm.
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60},
    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"provider": "openai", "input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"provider": "openai", "input": 0.10, "output": 0.40},
    "gpt-5-nano": {"provider": "openai", "input": 0.20, "output": 0.80},
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "input": 0.80,
        "output": 4.00,
    },
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    "gemini-3-flash-preview": {"provider": "google", "input": 0.10, "output": 0.40},
    "gemini-3.1-pro-preview": {"provider": "google", "input": 1.25, "output": 10.00},
    "gemini-3.1-flash-lite-preview": {
        "provider": "google",
        "input": 0.04,
        "output": 0.15,
    },
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


_API_TIMEOUT = 90  # seconds — hard HTTP timeout for all provider calls


def _call_openai(model: str, system: str, user: str, force_json: bool = True) -> tuple[str, int, int]:
    """Call OpenAI chat completion API.

    Args:
        force_json: When True, enforces ``response_format=json_object``. Set False
            for calls that expect plain text / markdown output (e.g. section notes).
    """
    import openai  # lazy import

    client = openai.OpenAI(timeout=_API_TIMEOUT)
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=8192,
    )
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    return raw, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_anthropic(model: str, system: str, user: str, force_json: bool = True) -> tuple[str, int, int]:
    """Call Anthropic Claude chat API."""
    import anthropic  # lazy import

    client = anthropic.Anthropic(timeout=_API_TIMEOUT)
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text if resp.content else ""
    return raw, resp.usage.input_tokens, resp.usage.output_tokens


def _call_google(model: str, system: str, user: str, force_json: bool = True) -> tuple[str, int, int]:
    """Call Google Gemini chat API."""
    import google.generativeai as genai  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini = genai.GenerativeModel(model, system_instruction=system)
    resp = gemini.generate_content(
        user,
        request_options={"timeout": _API_TIMEOUT},
    )
    raw = resp.text or ""
    meta = getattr(resp, "usage_metadata", None)
    input_tokens = getattr(meta, "prompt_token_count", 0) or 0
    output_tokens = getattr(meta, "candidates_token_count", 0) or 0
    return raw, input_tokens, output_tokens


_PROVIDER_DISPATCH: dict[str, Any] = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
}


# _is_noise_block moved to llm.block_filter; kept as alias for internal use
_is_noise_block = is_noise_block


def _sample_blocks(doc: Document, max_blocks: int = _MAX_BLOCKS) -> list:
    """Return a representative block sample from the document.

    Noise blocks are excluded before sampling: empty figure blocks, page numbers,
    TOC dot-leader lines, and short header/footer text. This ensures the same
    max_blocks budget is spent on substantive content blocks.

    For large documents, evenly samples across the filtered block list so the
    LLM sees content from beginning, middle, and end rather than just the
    first N blocks.
    """
    blocks = [b for b in doc.blocks if not _is_noise_block(b)]
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
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {SUPPORTED_LLM_MODELS}"
        )

    provider = _MODEL_REGISTRY[model]["provider"]
    call_fn = _PROVIDER_DISPATCH[provider]
    user_content = _serialize_blocks(doc)

    # Guard: if noise filtering removed every block, fail fast rather than letting
    # the LLM hallucinate a note from an empty context.
    if not user_content.strip():
        error_msg = "All document blocks were filtered as noise — no content to generate a note from."
        LOGGER.warning(
            "generate_note aborted for document id=%s: %s", doc.id, error_msg
        )
        if "note_generation_failed" not in doc.metadata.tags:
            doc.metadata.tags.append("note_generation_failed")
        return _make_fallback(doc, "", error_msg)

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
                LOGGER.warning(
                    "NoteGenerationOutput schema validation warning: %s", val_exc
                )
        except (json.JSONDecodeError, ValueError) as parse_exc:
            LOGGER.warning(
                "Note generation JSON parse failed (attempt 1): %s — retrying",
                parse_exc,
            )
            # Log attempt-1 failure so every API call is accounted for in the audit log
            log_api_call(
                model=model,
                stage="note_generation",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                success=False,
                error=f"JSON parse failed (attempt 1): {parse_exc}",
            )
            # Retry once with an explicit JSON-only nudge in the user message
            retry_user = (
                user_content
                + "\n\nIMPORTANT: Respond with valid JSON only. No markdown fences, no extra text."
            )
            t1 = time.perf_counter()
            try:
                raw, input_tokens, output_tokens = call_fn(model, PROMPT, retry_user)
                retry_latency_ms = (time.perf_counter() - t1) * 1000
                retry_cost_usd = _compute_cost(model, input_tokens, output_tokens)
                result = json.loads(_strip_markdown_fence(raw))
                if not isinstance(result, dict):
                    raise ValueError("LLM retry response JSON must be an object")
                try:
                    result = NoteGenerationOutput.model_validate(result).model_dump()
                except Exception as val_exc:
                    LOGGER.warning(
                        "NoteGenerationOutput schema validation warning (retry): %s",
                        val_exc,
                    )
                # Log successful retry with its own latency/tokens — not end-to-end
                log_api_call(
                    model=model,
                    stage="note_generation_retry",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=retry_latency_ms,
                    cost_usd=retry_cost_usd,
                    success=True,
                    error=None,
                )
            except (json.JSONDecodeError, ValueError) as retry_exc:
                retry_latency_ms = (time.perf_counter() - t1) * 1000
                retry_cost_usd = _compute_cost(model, input_tokens, output_tokens)
                LOGGER.warning(
                    "Note generation JSON parse failed (retry): %s", retry_exc
                )
                log_api_call(
                    model=model,
                    stage="note_generation_retry",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=retry_latency_ms,
                    cost_usd=retry_cost_usd,
                    success=False,
                    error=f"JSON parse failed after retry: {retry_exc}",
                )
                if "note_generation_failed" not in doc.metadata.tags:
                    doc.metadata.tags.append("note_generation_failed")
                return _make_fallback(
                    doc, raw, "노트 구조 분석이 불완전합니다. 원문 형식으로 표시됩니다."
                )
            # Retry succeeded — return early without logging via the normal success path below
            doc.status = ProcessingStatus.NOTE_GENERATED
            return result

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


# ---------------------------------------------------------------------------
# Section-based note generation (CU-14)
# ---------------------------------------------------------------------------


_SECTION_TEXT_LIMIT = 500   # text blocks: opening sentences carry the key idea
_SECTION_CODE_LIMIT = 800   # code blocks: need more context to be meaningful

def _serialize_section_blocks(
    blocks: list[Block],
    max_blocks: int = 40,
) -> str:
    """Serialize a section's blocks into '[{type}] {content}' lines.

    Applies per-section sampling if the section has more than max_blocks.
    Per-type content limits:
    - TEXT: 500 chars (leading sentences hold the key idea)
    - CODE: 800 chars (code needs more context to stay meaningful)
    - TABLE/FIGURE: passed through as-is (already compact)
    """
    sampled = blocks
    if len(blocks) > max_blocks:
        step = len(blocks) / max_blocks
        indices = {int(i * step) for i in range(max_blocks)}
        sampled = [blocks[i] for i in sorted(indices)]

    def _sanitize(text: str) -> str:
        """Strip characters that break JSON serialization (null bytes, surrogates, non-printable)."""
        # Keep printable chars + newline + tab; drop null bytes, surrogates, other control chars
        return "".join(c for c in text if c.isprintable() or c in "\n\t")

    lines = []
    for block in sampled:
        if block.type == BlockType.CODE:
            content = _sanitize(_truncate_code(block.content))[:_SECTION_CODE_LIMIT]
            lines.append(f"[code] {content}")
        elif block.type == BlockType.TEXT:
            lines.append(f"[text] {_sanitize(block.content)[:_SECTION_TEXT_LIMIT]}")
        else:
            # TABLE / FIGURE: also truncate to avoid oversized payloads
            lines.append(f"[{block.type.value}] {_sanitize(block.content)[:_SECTION_TEXT_LIMIT]}")
    return "\n".join(lines)


def _generate_section_note(
    section: Any,
    doc_title: str,
    section_idx: int,
    total_sections: int,
    model: str,
    max_blocks: int,
) -> str:
    """Generate markdown body for a single section.

    Returns raw markdown text on success, or a warning placeholder on failure.
    """
    from prompts.note_generation_section import SECTION_PROMPT

    provider = _MODEL_REGISTRY[model]["provider"]
    call_fn = _PROVIDER_DISPATCH[provider]

    system = SECTION_PROMPT.format(
        doc_title=doc_title,
        section_heading=section.heading,
        section_idx=section_idx,
        total_sections=total_sections,
    )
    user_content = _serialize_section_blocks(section.blocks, max_blocks=max_blocks)

    if not user_content.strip():
        return "> ⚠️ 이 섹션은 내용이 부족하여 생략되었습니다."

    t0 = time.perf_counter()
    try:
        raw, input_tokens, output_tokens = call_fn(model, system, user_content, force_json=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        cost_usd = _compute_cost(model, input_tokens, output_tokens)

        log_api_call(
            model=model,
            stage="note_generation_section",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=True,
            error=None,
        )
        return raw.strip()

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        LOGGER.warning(
            "Section note generation failed for '%s': %s", section.heading, exc
        )
        log_api_call(
            model=model,
            stage="note_generation_section",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return "> ⚠️ 이 섹션은 생성에 실패했습니다."


def _assemble_sections(
    sections: list[Any],
    section_markdowns: list[str],
    doc: Document,
    model: str,
) -> dict[str, Any]:
    """Assemble per-section markdowns and extract metadata via LLM.

    Concatenates section markdowns with ``## {heading}`` headers, then
    calls the ASSEMBLY_PROMPT to generate metadata JSON.

    Returns the final note dict matching NoteGenerationOutput schema.
    """
    from prompts.note_generation_section import (
        ASSEMBLY_PROMPT,
        PROMPT_VERSION as SECTION_PROMPT_VERSION,
    )

    def _normalize_heading(s: str) -> str:
        """Strip markdown/punctuation markers and whitespace for fuzzy heading comparison."""
        return re.sub(r"[#:*_\s]", "", s).lower()

    def _strip_leading_heading(md: str, heading: str) -> str:
        """Remove a leading heading line the LLM may have added despite instructions.

        Handles:
        - ``## Heading`` (markdown heading, any level) where the heading text
          matches the section heading (normalized comparison). Legitimate
          sub-headings like ``### Example`` or ``### 핵심 포인트`` are preserved.
        - Plain-text restatements: normalized first line *starts with* the
          normalized heading AND the remainder is ≤ 3 chars (punctuation only).
          This prevents stripping real body sentences that happen to open with
          a word also present in the heading (e.g. "Section B 내용입니다.").
        """
        lines = md.lstrip("\n").splitlines()
        if not lines:
            return md
        first = lines[0].lstrip()
        norm_h = _normalize_heading(heading)
        if first.startswith("#"):
            # Only strip if the markdown heading text matches the section heading
            # Strip leading '#' characters and surrounding whitespace to get heading text
            heading_text = first.lstrip("#").strip()
            norm_f = _normalize_heading(heading_text)
            is_md_heading = bool(norm_h) and norm_f.startswith(norm_h) and len(norm_f) - len(norm_h) <= 3
        else:
            norm_f = _normalize_heading(first)
            # Restatement: first line starts with heading AND has ≤ 3 extra chars
            # (allows trailing period, colon, or minor decoration but not body content)
            is_md_heading = False
        is_plain_restatement = (
            not first.startswith("#")
            and bool(norm_h)
            and norm_f.startswith(norm_h)
            and len(norm_f) - len(norm_h) <= 3
        )
        if is_md_heading or is_plain_restatement:
            md = "\n".join(lines[1:]).lstrip("\n")
        return md

    parts: list[str] = []
    for section, md in zip(sections, section_markdowns):
        parts.append(f"## {section.heading}\n\n{_strip_leading_heading(md, section.heading)}")
    note_markdown = "\n\n".join(parts)

    doc_title = doc.metadata.title or doc.source
    # Estimate read time from word count (~200 words/min)
    word_count = len(note_markdown.split())
    estimated_read_time_min = max(1, round(word_count / 200))

    # Build per-section snippet: heading + first 2-3 sentences of the section body
    def _snippet(md: str, max_sentences: int = 3) -> str:
        sentences = []
        for line in md.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sentences.append(line)
            if len(sentences) >= max_sentences:
                break
        return " ".join(sentences)

    snippet_parts = []
    for section, md in zip(sections, section_markdowns):
        snippet_parts.append(f"## {section.heading}\n{_snippet(md)}")
    section_snippets = "\n\n".join(snippet_parts)

    provider = _MODEL_REGISTRY[model]["provider"]
    call_fn = _PROVIDER_DISPATCH[provider]

    system = ASSEMBLY_PROMPT.format(
        doc_title=doc_title,
        section_snippets=section_snippets,
        estimated_read_time_min=estimated_read_time_min,
    )

    t0 = time.perf_counter()
    try:
        raw, input_tokens, output_tokens = call_fn(
            model, system, "Generate metadata JSON."
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        cost_usd = _compute_cost(model, input_tokens, output_tokens)

        log_api_call(
            model=model,
            stage="note_generation_assembly",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=True,
            error=None,
        )

        try:
            metadata = json.loads(_strip_markdown_fence(raw))
            if not isinstance(metadata, dict):
                raise ValueError("Assembly response must be a JSON object")
        except (json.JSONDecodeError, ValueError) as parse_exc:
            LOGGER.warning("Assembly JSON parse failed: %s", parse_exc)
            metadata = {}

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        LOGGER.warning("Assembly call failed: %s", exc)
        log_api_call(
            model=model,
            stage="note_generation_assembly",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        metadata = {}

    return {
        "title": metadata.get("title", doc.source),
        "summary": metadata.get("summary", ""),
        "note_markdown": note_markdown,
        "key_concepts": metadata.get("key_concepts", []),
        "difficulty_level": metadata.get("difficulty_level", "intermediate"),
        "estimated_read_time_min": metadata.get("estimated_read_time_min", 5),
        "schema_version": SECTION_PROMPT_VERSION,
        "confidence": metadata.get("confidence", 0.5),
        "errors": metadata.get("errors", []),
    }


def generate_note_sectioned(
    doc: Document,
    model: str = "gpt-4o-mini",
    max_blocks_per_section: int = 40,
) -> dict[str, Any]:
    """Generate a structured study note using section-based splitting.

    For large documents, extracts section headings from the document's
    TOC structure and generates notes per-section. This guarantees every
    section in the TOC appears in the generated note.

    For small documents (<=_MAX_BLOCKS non-noise blocks) or documents
    with no detectable sections, delegates to ``generate_note()``.

    Args:
        doc: Source document with populated blocks.
        model: Model used for all LLM calls (per-section and assembly).
            Defaults to ``"gpt-4o-mini"``.
        max_blocks_per_section: Maximum blocks sent per section call.

    Returns:
        dict matching NoteGenerationOutput schema.

    Raises:
        ValueError: If ``model`` is not in SUPPORTED_LLM_MODELS.
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {SUPPORTED_LLM_MODELS}"
        )

    from llm.section_splitter import extract_sections, group_blocks_by_section

    sections = extract_sections(doc)

    # Count non-noise blocks for threshold check
    non_noise_count = sum(1 for b in doc.blocks if not is_noise_block(b))

    # Adaptive: small docs or no sections → single-call path
    if not sections or non_noise_count <= _MAX_BLOCKS:
        LOGGER.info(
            "generate_note_sectioned: delegating to generate_note "
            "(sections=%d, non_noise_blocks=%d, threshold=%d)",
            len(sections),
            non_noise_count,
            _MAX_BLOCKS,
        )
        return generate_note(doc, model=model)

    sections = group_blocks_by_section(doc, sections)

    # Filter out sections with no blocks after grouping
    sections = [s for s in sections if s.blocks]

    if not sections:
        return generate_note(doc, model=model)

    LOGGER.info(
        "generate_note_sectioned: %d sections for document '%s' (%d blocks)",
        len(sections),
        doc.source,
        len(doc.blocks),
    )

    # Generate per-section notes in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc_title = doc.metadata.title or doc.source
    total = len(sections)

    def _call(args: tuple) -> tuple[int, str]:
        idx, section = args
        md = _generate_section_note(
            section=section,
            doc_title=doc_title,
            section_idx=idx,
            total_sections=total,
            model=model,
            max_blocks=max_blocks_per_section,
        )
        return idx, md

    _SECTION_TIMEOUT = 120  # seconds per individual future result
    _TOTAL_TIMEOUT = _SECTION_TIMEOUT * 2 + 30  # safety net for as_completed

    # Detect Streamlit context: ThreadPoolExecutor deadlocks when Streamlit's
    # script runner holds a reentrant lock that worker threads cannot acquire.
    # Simple check: if streamlit is imported we're in a streamlit process.
    _in_streamlit = "streamlit" in sys.modules

    results: dict[int, str] = {}

    if _in_streamlit:
        # Sequential fallback to avoid ThreadPoolExecutor deadlock in Streamlit
        LOGGER.info("generate_note_sectioned: running sequentially (Streamlit context)")
        for idx, section in enumerate(sections, start=1):
            try:
                md = _generate_section_note(
                    section=section,
                    doc_title=doc_title,
                    section_idx=idx,
                    total_sections=total,
                    model=model,
                    max_blocks=max_blocks_per_section,
                )
            except Exception as exc:
                LOGGER.warning("Section %d/%d failed: %s", idx, total, exc)
                md = f"> ⚠️ 섹션 {idx} 생성 실패: {exc}"
            results[idx] = md
            LOGGER.info("Section %d/%d done", idx, total)
    else:
        with ThreadPoolExecutor(max_workers=min(total, 10)) as pool:
            futures = {pool.submit(_call, (idx, sec)): idx for idx, sec in enumerate(sections, start=1)}
            try:
                for fut in as_completed(futures, timeout=_TOTAL_TIMEOUT):
                    try:
                        idx, md = fut.result(timeout=_SECTION_TIMEOUT)
                    except Exception as exc:
                        idx = futures[fut]
                        LOGGER.warning("Section %d/%d failed or timed out: %s", idx, total, exc)
                        md = f"> ⚠️ 섹션 {idx} 생성 실패: {exc}"
                    results[idx] = md
                    LOGGER.info("Section %d/%d done", idx, total)
            except TimeoutError:
                LOGGER.warning("as_completed global timeout — %d/%d sections collected", len(results), total)
                for idx in range(1, total + 1):
                    results.setdefault(idx, f"> ⚠️ 섹션 {idx} 타임아웃")

    section_markdowns: list[str] = [results[i] for i in range(1, total + 1)]

    # Assemble and extract metadata
    result = _assemble_sections(sections, section_markdowns, doc, model)
    doc.status = ProcessingStatus.NOTE_GENERATED
    return result


__all__ = ["generate_note", "generate_note_sectioned", "SUPPORTED_LLM_MODELS"]
