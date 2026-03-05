"""Image parser: VLM-based classification + analysis → Document."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

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
from prompts.vlm_classify import PROMPT as CLASSIFY_PROMPT
from prompts.vlm_code import PROMPT as VLM_CODE_PROMPT
from prompts.vlm_diagram import PROMPT as VLM_DIAGRAM_PROMPT
from prompts.vlm_text import PROMPT as VLM_TEXT_PROMPT
from vlm.client import call_vlm

LOGGER = logging.getLogger(__name__)

VLMParsedOutput = CodeVLMOutput | DiagramVLMOutput | TextVLMOutput

_IMAGE_TYPE_MAP: dict[str, ImageType] = {
    "code_screenshot": ImageType.CODE_SCREENSHOT,
    "diagram": ImageType.DIAGRAM,
    "text_capture": ImageType.TEXT_CAPTURE,
    "equation": ImageType.EQUATION,
    "other": ImageType.OTHER,
}

_PROMPT_BY_IMAGE_TYPE: dict[ImageType, str] = {
    ImageType.CODE_SCREENSHOT: VLM_CODE_PROMPT,
    ImageType.DIAGRAM: VLM_DIAGRAM_PROMPT,
    ImageType.TEXT_CAPTURE: VLM_TEXT_PROMPT,
    ImageType.EQUATION: VLM_TEXT_PROMPT,
    ImageType.OTHER: VLM_TEXT_PROMPT,
}


def parse_image(file_path: str, model: str = "gpt-4o-mini") -> Document:
    """
    Parse one image file through VLM (classify then analyze) and return a Document.

    Two VLM calls are made:
        1. Classification: determine image type (code/diagram/text/equation/other)
        2. Analysis: type-specific structured extraction

    Args:
        file_path: Path to the image file (JPEG / PNG / GIF / WebP).
        model: VLM model identifier. Defaults to "gpt-4o-mini".

    Returns:
        Document with one Block from analysis, or empty blocks on VLM failure.
        On JSON parse failure, raw VLM response is stored as a TEXT block.
    """
    source_name = Path(file_path).name
    document_id = _safe_document_id(file_path)

    # Step 1: classify
    image_type = _classify_image(file_path, model)

    # Step 2: analyze
    analysis_prompt = _PROMPT_BY_IMAGE_TYPE[image_type]
    result = call_vlm(model, file_path, analysis_prompt, stage="image_analysis")

    blocks: list[Block] = []
    if result.success:
        try:
            parsed = _parse_vlm_output(result.content, image_type)
            blocks.append(map_vlm_output_to_block(image_type=image_type, payload=parsed, order=0, image_path=file_path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("VLM JSON parse failed for %s, using raw fallback: %s", file_path, exc)
            blocks.append(
                Block(
                    type=BlockType.TEXT,
                    content=result.content,
                    order=0,
                    metadata=BlockMetadata(image_type=image_type),
                    image_path=file_path,
                )
            )
    else:
        LOGGER.error("VLM analysis call failed for %s: %s", file_path, result.error)

    processing = ProcessingInfo(
        parser_model="image_parser_v2.0",
        vlm_model=model,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )

    return Document(
        id=document_id,
        source=source_name,
        format=DocumentFormat.IMAGE,
        blocks=blocks,
        processing=processing,
        status=ProcessingStatus.PARSED,
    )


def _classify_image(file_path: str, model: str) -> ImageType:
    """Call VLM to classify image type. Falls back to OTHER on any failure."""
    result = call_vlm(model, file_path, CLASSIFY_PROMPT, stage="image_classify")
    if not result.success:
        LOGGER.warning("Classification VLM call failed for %s, defaulting to OTHER", file_path)
        return ImageType.OTHER
    try:
        payload = json.loads(_strip_markdown_fence(result.content).strip())
        type_str = payload.get("image_type", "other")
        return _IMAGE_TYPE_MAP.get(type_str, ImageType.OTHER)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Classification JSON parse failed for %s: %s, defaulting to OTHER", file_path, exc)
        return ImageType.OTHER


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

    # TextVLMOutput
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


def _parse_json_dict(raw_response: str) -> dict[str, Any]:
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
