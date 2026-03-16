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
from parsers.schemas.vlm_outputs import (
    CodeVLMOutput,
    DiagramVLMOutput,
    TextVLMOutput,
)
from prompts.vlm_classify import PROMPT as CLASSIFY_PROMPT
from prompts import vlm_code, vlm_diagram, vlm_text
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

_PROMPT_GETTER_BY_IMAGE_TYPE = {
    ImageType.CODE_SCREENSHOT: vlm_code.get_prompt,
    ImageType.DIAGRAM: vlm_diagram.get_prompt,
    ImageType.TEXT_CAPTURE: vlm_text.get_prompt,
    ImageType.EQUATION: vlm_text.get_prompt,
    ImageType.OTHER: vlm_text.get_prompt,
}

# Backward compat: static prompt map (Korean default) used by figure_enricher
_PROMPT_BY_IMAGE_TYPE: dict[ImageType, str] = {
    k: fn("ko") for k, fn in _PROMPT_GETTER_BY_IMAGE_TYPE.items()
}


def parse_image(file_path: str, model: str = "gpt-4o-mini", language: str = "ko") -> Document:
    """
    Parse one image file through VLM (classify then analyze) and return a Document.

    Two VLM calls are made:
        1. Classification: determine image type (code/diagram/text/equation/other)
        2. Analysis: type-specific structured extraction

    Args:
        file_path: Path to the image file (JPEG / PNG / GIF / WebP).
        model: VLM model identifier. Defaults to "gpt-4o-mini".
        language: Output language for VLM descriptions ("ko" or "en").

    Returns:
        Document with one Block from analysis, or empty blocks on VLM failure.
        On JSON parse failure, raw VLM response is stored as a TEXT block.
    """
    source_name = Path(file_path).name
    document_id = _safe_document_id(file_path)

    # Step 1: classify
    image_type = classify_image(file_path, model)

    # Step 2: analyze
    analysis_prompt = _PROMPT_GETTER_BY_IMAGE_TYPE[image_type](language)
    result = call_vlm(model, file_path, analysis_prompt, stage="image_analysis")

    blocks: list[Block] = []
    if result.success:
        try:
            parsed = parse_vlm_output(result.content, image_type)
            blocks.append(
                map_vlm_output_to_block(
                    image_type=image_type, payload=parsed, order=0, image_path=file_path
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "VLM JSON parse failed for %s, using raw fallback: %s", file_path, exc
            )
            blocks.append(
                Block(
                    type=BlockType.TEXT,
                    content=_json_to_plain_text(result.content),
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


def classify_image(file_path: str, model: str) -> ImageType:
    """Call VLM to classify image type. Falls back to OTHER on any failure."""
    result = call_vlm(model, file_path, CLASSIFY_PROMPT, stage="image_classify")
    if not result.success:
        LOGGER.warning(
            "Classification VLM call failed for %s, defaulting to OTHER", file_path
        )
        return ImageType.OTHER
    try:
        payload = json.loads(_strip_markdown_fence(result.content).strip())
        type_str = payload.get("image_type", "other")
        return _IMAGE_TYPE_MAP.get(type_str, ImageType.OTHER)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Classification JSON parse failed for %s: %s, defaulting to OTHER",
            file_path,
            exc,
        )
        return ImageType.OTHER


# Internal alias for backward compat
_classify_image = classify_image


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

    if isinstance(payload, CodeVLMOutput):
        metadata.language = payload.language
        metadata.caption = payload.description or None
        return Block(
            type=BlockType.CODE,
            content=_normalize_escaped_text(payload.code_markdown or payload.code),
            order=order,
            metadata=metadata,
            image_path=image_path,
        )

    if isinstance(payload, DiagramVLMOutput):
        metadata.caption = payload.title or None
        return Block(
            type=BlockType.FIGURE,
            content=_diagram_to_text(payload),
            order=order,
            metadata=metadata,
            image_path=image_path,
        )

    # TextVLMOutput
    metadata.caption = payload.title or None
    return Block(
        type=BlockType.TEXT,
        content=payload.content,
        order=order,
        metadata=metadata,
        image_path=image_path,
    )


def parse_vlm_output(raw_response: str, image_type: ImageType) -> VLMParsedOutput:
    """Parse and validate VLM JSON response by image type."""
    payload_dict = _parse_json_dict(raw_response)

    if image_type == ImageType.CODE_SCREENSHOT:
        return CodeVLMOutput.model_validate(payload_dict)
    if image_type == ImageType.DIAGRAM:
        return DiagramVLMOutput.model_validate(payload_dict)
    return TextVLMOutput.model_validate(payload_dict)


# Internal alias for backward compat
_parse_vlm_output = parse_vlm_output


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
            lines.append(
                f"- {relation.from_component} -> {relation.to_component}{label}"
            )

    if payload.flow_summary:
        lines.append("Flow:")
        lines.append(payload.flow_summary)

    return "\n".join(lines).strip()



def _normalize_escaped_text(value: str) -> str:
    """Normalize escaped newline/tab sequences for markdown/code strings."""
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _json_to_plain_text(raw: str) -> str:
    """Convert a JSON string to plain text for LLM consumption.

    When the VLM returns JSON that doesn't match the expected schema, this
    function converts it to a readable key-value text format so that downstream
    LLMs receive natural language instead of raw JSON.
    Falls back to the original string if parsing fails.
    """
    import json as _json

    try:
        data = _json.loads(_strip_markdown_fence(raw).strip())
    except (ValueError, TypeError):
        return raw

    if not isinstance(data, dict):
        return raw

    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _safe_document_id(file_path: str) -> str:
    """Generate document id with fallback if file read fails."""
    try:
        return generate_document_id(file_path)
    except Exception:  # noqa: BLE001
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]


__all__ = ["classify_image", "map_vlm_output_to_block", "parse_image", "parse_vlm_output"]
