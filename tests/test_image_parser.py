"""Unit tests for image parser and VLM -> Block mapping layer."""

from __future__ import annotations

import json
from pathlib import Path

from models.document import BlockType, DocumentFormat, ImageType, ProcessingStatus
from parsers.image_parser import map_vlm_output_to_block, parse_image
from parsers.schemas.vlm_outputs import DiagramVLMOutput


def _write_fake_image(path: Path) -> None:
    """Create a tiny fake image-like file for hashing input."""
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")


def test_parse_image_code_screenshot_maps_to_code_block(tmp_path: Path, monkeypatch) -> None:
    """Code screenshot VLM output should map to CODE block with metadata."""
    image_path = tmp_path / "code.png"
    _write_fake_image(image_path)

    log_calls: list[dict] = []
    monkeypatch.setattr("parsers.image_parser.log_api_call", lambda **kwargs: log_calls.append(kwargs))

    def fake_vlm_infer(file_path: str, prompt: str) -> str:
        assert file_path == str(image_path)
        assert "code_markdown" in prompt
        return json.dumps(
            {
                "schema_version": "v1.1.0",
                "language": "python",
                "code": "print('hello')",
                "code_markdown": "```python\\nprint('hello')\\n```",
                "description": "출력 예시",
                "has_truncation": False,
                "confidence": 0.91,
                "errors": [],
            }
        )

    document = parse_image(
        file_path=str(image_path),
        image_type=ImageType.CODE_SCREENSHOT,
        vlm_infer=fake_vlm_infer,
        model_name="gpt-4o-mini",
    )

    assert document.format == DocumentFormat.IMAGE
    assert document.status == ProcessingStatus.PARSED
    assert len(document.blocks) == 1
    block = document.blocks[0]
    assert block.type == BlockType.CODE
    assert block.content == "```python\nprint('hello')\n```"
    assert block.metadata.language == "python"
    assert block.metadata.image_type == ImageType.CODE_SCREENSHOT
    assert block.metadata.confidence == 0.91
    assert block.metadata.caption == "출력 예시"
    assert len(log_calls) == 1
    assert log_calls[0]["stage"] == "image_parsing"
    assert log_calls[0]["success"] is True


def test_parse_image_retries_once_after_invalid_json(tmp_path: Path) -> None:
    """Parser should retry once and recover when first response is invalid."""
    image_path = tmp_path / "retry.png"
    _write_fake_image(image_path)

    responses = iter(
        [
            "not json at all",
            json.dumps(
                {
                    "schema_version": "v1.1.0",
                    "text_type": "lecture_slide",
                    "title": "강의 슬라이드",
                    "content": "## 제목\\n내용",
                    "key_points": ["포인트 1"],
                    "has_math": False,
                    "has_truncation": False,
                    "confidence": 0.77,
                    "errors": [],
                }
            ),
        ]
    )

    def fake_vlm_infer(_: str, __: str) -> str:
        return next(responses)

    document = parse_image(
        file_path=str(image_path),
        image_type=ImageType.TEXT_CAPTURE,
        vlm_infer=fake_vlm_infer,
        retry_count=1,
    )

    assert len(document.blocks) == 1
    assert document.blocks[0].type == BlockType.TEXT
    assert "## 제목" in document.blocks[0].content


def test_parse_image_returns_fallback_document_on_persistent_failure(tmp_path: Path) -> None:
    """Parser should return empty-block fallback document after retry exhaustion."""
    image_path = tmp_path / "broken.png"
    _write_fake_image(image_path)

    def always_invalid(_: str, __: str) -> str:
        return "```json\n{broken}\n```"

    document = parse_image(
        file_path=str(image_path),
        image_type=ImageType.DIAGRAM,
        vlm_infer=always_invalid,
        retry_count=1,
    )

    assert document.format == DocumentFormat.IMAGE
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []


def test_map_vlm_output_to_block_for_diagram() -> None:
    """Diagram payload should map to FIGURE block with structured text."""
    payload = DiagramVLMOutput.model_validate(
        {
            "schema_version": "v1.1.0",
            "diagram_type": "flowchart",
            "title": "파이프라인",
            "description": "전체 흐름도",
            "components": [{"name": "A", "role": "입력"}, {"name": "B", "role": "출력"}],
            "relationships": [{"from": "A", "to": "B", "label": "next"}],
            "flow_summary": "A에서 B로 이동",
            "has_truncation": False,
            "confidence": 0.88,
            "errors": [],
        }
    )

    block = map_vlm_output_to_block(
        image_type=ImageType.DIAGRAM,
        payload=payload,
        order=0,
        image_path="/tmp/diagram.png",
    )

    assert block.type == BlockType.FIGURE
    assert "Components:" in block.content
    assert "A -> B (next)" in block.content
    assert block.metadata.image_type == ImageType.DIAGRAM
    assert block.metadata.confidence == 0.88
