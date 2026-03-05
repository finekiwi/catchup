"""Image parser that maps VLM JSON outputs into shared Document blocks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

from models.document import (
    Block,
    BlockMetadata,
    BlockType,
    Document,
    DocumentFormat,
    ImageType,
    ProcessingInfo,
    ProcessingStatus,
    generate_document_id,
)
from parsers.schemas.vlm_outputs import CodeVLMOutput, DiagramVLMOutput, TextVLMOutput, VLMOutputBase
from prompts.vlm_code import PROMPT as VLM_CODE_PROMPT
from prompts.vlm_diagram import PROMPT as VLM_DIAGRAM_PROMPT
from prompts.vlm_text import PROMPT as VLM_TEXT_PROMPT
from utils.logging import log_api_call

LOGGER = logging.getLogger(__name__)

VLMInferFn = Callable[[str, str], str]
VLMParsedOutput = CodeVLMOutput | DiagramVLMOutput | TextVLMOutput

_PROMPT_BY_IMAGE_TYPE: dict[ImageType, str] = {
    ImageType.CODE_SCREENSHOT: VLM_CODE_PROMPT,
    ImageType.DIAGRAM: VLM_DIAGRAM_PROMPT,
    ImageType.TEXT_CAPTURE: VLM_TEXT_PROMPT,
    ImageType.EQUATION: VLM_TEXT_PROMPT,
    ImageType.OTHER: VLM_TEXT_PROMPT,
}


def parse_image(
    file_path: str,
    image_type: ImageType,
    vlm_infer: VLMInferFn,
    *,
    model_name: str = "unknown-vlm",
    retry_count: int = 1,
) -> Document:
    """
    Parse one image file through VLM and map output to shared Document schema.

    Note:
        VLM JSON schema and Block schema are intentionally separated.
        Mapping is handled by `map_vlm_output_to_block`.
    """
    start_time = time.perf_counter()
    source_name = Path(file_path).name
    document_id = _safe_document_id(file_path)

    response_error: str | None = None
    parsed_output: VLMParsedOutput | None = None
    raw_response = ""

    prompt = _PROMPT_BY_IMAGE_TYPE.get(image_type, VLM_TEXT_PROMPT)

    for attempt in range(retry_count + 1):
        current_prompt = prompt if attempt == 0 else _build_retry_prompt(raw_response=raw_response, image_type=image_type)
        call_start = time.perf_counter()
        success = False
        error: str | None = None

        try:
            raw_response = vlm_infer(file_path, current_prompt)
            parsed_output = _parse_vlm_output(raw_response=raw_response, image_type=image_type)
            success = True
            response_error = None
            break
        except Exception as exc:  # noqa: BLE001
            response_error = str(exc)
            error = response_error
            LOGGER.warning("VLM parse attempt failed for %s (attempt=%s): %s", file_path, attempt + 1, exc)
        finally:
            latency_ms = (time.perf_counter() - call_start) * 1000
            log_api_call(
                model=model_name,
                stage="image_parsing",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                success=success,
                error=error,
            )

    blocks: list[Block] = []
    if parsed_output is not None:
        blocks.append(map_vlm_output_to_block(image_type=image_type, payload=parsed_output, order=0, image_path=file_path))

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    processing = ProcessingInfo(
        parser_model="image_parser_v1.1",
        vlm_model=model_name,
        latency_ms=total_latency_ms,
    )

    if response_error and not blocks:
        LOGGER.error("Failed to parse image %s. Returning fallback document. error=%s", file_path, response_error)

    return Document(
        id=document_id,
        source=source_name,
        format=DocumentFormat.IMAGE,
        blocks=blocks,
        processing=processing,
        status=ProcessingStatus.PARSED,
    )


def map_vlm_output_to_block(
    image_type: ImageType,
    payload: VLMParsedOutput,
    *,
    order: int,
    image_path: str | None,
) -> Block:
    """
    Convert validated VLM output payload into the internal Block schema.

    This is an explicit mapper layer:
    VLM JSON schema != Block / BlockMetadata schema.
    """
    metadata = BlockMetadata(image_type=image_type, confidence=payload.confidence)
    quality_note = _build_quality_note(payload)

    if isinstance(payload, CodeVLMOutput):
        metadata.language = payload.language
        metadata.caption = _join_notes(payload.description, quality_note)
        return Block(
            type=BlockType.CODE,
            content=_normalize_escaped_text(payload.code_markdown or payload.code),
            order=order,
            metadata=metadata,
            image_path=image_path,
        )

    if isinstance(payload, DiagramVLMOutput):
        metadata.caption = _join_notes(payload.title, quality_note)
        return Block(
            type=BlockType.FIGURE,
            content=_diagram_to_text(payload),
            order=order,
            metadata=metadata,
            image_path=image_path,
        )

    metadata.caption = _join_notes(payload.title, quality_note)
    return Block(
        type=BlockType.TEXT,
        content=payload.content,
        order=order,
        metadata=metadata,
        image_path=image_path,
    )


def _parse_vlm_output(raw_response: str, image_type: ImageType) -> VLMParsedOutput:
    """Parse and validate VLM JSON response by image type."""
    payload_dict = _parse_json_dict(raw_response)

    if image_type == ImageType.CODE_SCREENSHOT:
        return CodeVLMOutput.model_validate(payload_dict)
    if image_type == ImageType.DIAGRAM:
        return DiagramVLMOutput.model_validate(payload_dict)
    return TextVLMOutput.model_validate(payload_dict)


def _parse_json_dict(raw_response: str) -> dict:
    """Extract JSON object from model output, tolerating markdown fences."""
    cleaned = _strip_markdown_fence(raw_response).strip()
    if not cleaned:
        raise ValueError("empty VLM response")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in VLM response")

    candidate = cleaned[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON must be an object")
    return parsed


def _strip_markdown_fence(text: str) -> str:
    """Remove a single outer markdown fence if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _diagram_to_text(payload: DiagramVLMOutput) -> str:
    """Convert diagram payload into readable structured text content."""
    lines: list[str] = []

    if payload.title:
        lines.append(f"Title: {payload.title}")
    if payload.diagram_type:
        lines.append(f"Type: {payload.diagram_type}")
    if payload.description:
        lines.append(payload.description)

    if payload.components:
        lines.append("Components:")
        for item in payload.components:
            role = f": {item.role}" if item.role else ""
            lines.append(f"- {item.name}{role}")

    if payload.relationships:
        lines.append("Relationships:")
        for relation in payload.relationships:
            label = f" ({relation.label})" if relation.label else ""
            lines.append(f"- {relation.from_component} -> {relation.to_component}{label}")

    if payload.flow_summary:
        lines.append("Flow:")
        lines.append(payload.flow_summary)

    return "\n".join(lines).strip()


def _build_quality_note(payload: VLMOutputBase) -> str | None:
    """Build quality metadata text from truncation/errors."""
    notes: list[str] = []
    if payload.has_truncation:
        notes.append("truncation=true")
    if payload.errors:
        notes.append(f"errors={'; '.join(payload.errors)}")
    return " | ".join(notes) if notes else None


def _join_notes(first: str | None, second: str | None) -> str | None:
    """Join two optional text notes into one caption field."""
    parts = [value for value in (first, second) if value]
    return " | ".join(parts) if parts else None


def _build_retry_prompt(raw_response: str, image_type: ImageType) -> str:
    """Prompt used for one-step JSON repair retry."""
    return (
        "Your previous response was not valid JSON for the required schema.\n"
        f"Image type: {image_type.value}\n"
        "Return ONLY a valid JSON object that matches the required fields.\n"
        "Do not include markdown fences or explanations.\n"
        f"Previous response:\n{raw_response}"
    )


def _normalize_escaped_text(value: str) -> str:
    """Normalize escaped newline/tab sequences for markdown/code strings."""
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _safe_document_id(file_path: str) -> str:
    """Generate document id with fallback if file read fails."""
    try:
        return generate_document_id(file_path)
    except Exception:  # noqa: BLE001
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]


__all__ = ["map_vlm_output_to_block", "parse_image"]
