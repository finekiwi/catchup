"""Unit tests for image parser: VLM classification + analysis → Block mapping."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from models.document import BlockType, DocumentFormat, ImageType, ProcessingStatus
from parsers.image_parser import map_vlm_output_to_block, parse_image
from parsers.schemas.vlm_outputs import DiagramVLMOutput
from vlm.client import VLMResult


def _fake_image(path: Path) -> None:
    """Create a minimal fake image file for document ID hashing."""
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")


def _vlm_result(
    content: str, success: bool = True, error: str | None = None
) -> VLMResult:
    return VLMResult(content=content, model="gpt-4o-mini", success=success, error=error)


def _classify(image_type: str) -> VLMResult:
    return _vlm_result(json.dumps({"image_type": image_type, "confidence": 0.95}))


def _code_analysis() -> VLMResult:
    return _vlm_result(
        json.dumps(
            {
                "schema_version": "v1.1.0",
                "language": "python",
                "code": "print('hello')",
                "code_markdown": "```python\nprint('hello')\n```",
                "description": "출력 예시",
                "has_truncation": False,
                "confidence": 0.91,
                "errors": [],
            }
        )
    )


def _diagram_analysis() -> VLMResult:
    return _vlm_result(
        json.dumps(
            {
                "schema_version": "v1.1.0",
                "diagram_type": "flowchart",
                "title": "파이프라인",
                "description": "전체 흐름도",
                "components": [
                    {"name": "A", "role": "입력"},
                    {"name": "B", "role": "출력"},
                ],
                "relationships": [{"from": "A", "to": "B", "label": "next"}],
                "flow_summary": "A에서 B로 이동",
                "has_truncation": False,
                "confidence": 0.88,
                "errors": [],
            }
        )
    )


def _text_analysis() -> VLMResult:
    return _vlm_result(
        json.dumps(
            {
                "schema_version": "v1.1.0",
                "text_type": "lecture_slide",
                "title": "강의 슬라이드",
                "content": "## 제목\n내용",
                "key_points": ["포인트 1"],
                "has_math": False,
                "has_truncation": False,
                "confidence": 0.85,
                "errors": [],
            }
        )
    )


def test_parse_image_code_screenshot_maps_to_code_block(tmp_path: Path) -> None:
    """Code screenshot → BlockType.CODE with language and code content."""
    image_path = tmp_path / "code.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("code_screenshot"), _code_analysis()],
    ):
        doc = parse_image(str(image_path))

    assert doc.format == DocumentFormat.IMAGE
    assert doc.status == ProcessingStatus.PARSED
    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.type == BlockType.CODE
    assert "print('hello')" in block.content
    assert block.metadata.language == "python"
    assert block.metadata.image_type == ImageType.CODE_SCREENSHOT
    assert block.metadata.confidence == 0.91


def test_parse_image_diagram_maps_to_figure_block(tmp_path: Path) -> None:
    """Diagram → BlockType.FIGURE with structured component/relationship text."""
    image_path = tmp_path / "diagram.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("diagram"), _diagram_analysis()],
    ):
        doc = parse_image(str(image_path))

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.type == BlockType.FIGURE
    assert "Components:" in block.content
    assert block.metadata.image_type == ImageType.DIAGRAM


def test_parse_image_text_capture_maps_to_text_block(tmp_path: Path) -> None:
    """Text capture → BlockType.TEXT with markdown content."""
    image_path = tmp_path / "text.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("text_capture"), _text_analysis()],
    ):
        doc = parse_image(str(image_path))

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.type == BlockType.TEXT
    assert "제목" in block.content
    assert block.metadata.image_type == ImageType.TEXT_CAPTURE


def test_parse_image_classification_failure_defaults_to_other(tmp_path: Path) -> None:
    """Classification JSON parse failure → OTHER image type → vlm_text prompt used."""
    image_path = tmp_path / "unknown.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_vlm_result("not json"), _text_analysis()],
    ):
        doc = parse_image(str(image_path))

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.type == BlockType.TEXT
    assert block.metadata.image_type == ImageType.OTHER


def test_parse_image_analysis_json_failure_uses_raw_fallback(tmp_path: Path) -> None:
    """Analysis JSON parse failure → raw VLM response stored as TEXT block."""
    image_path = tmp_path / "broken.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[
            _classify("code_screenshot"),
            _vlm_result("This image shows some code."),
        ],
    ):
        doc = parse_image(str(image_path))

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.type == BlockType.TEXT
    assert block.content == "This image shows some code."
    assert block.metadata.image_type == ImageType.CODE_SCREENSHOT


def test_parse_image_vlm_analysis_failure_returns_empty_blocks(tmp_path: Path) -> None:
    """VLM analysis call failure (success=False) → Document with empty blocks."""
    image_path = tmp_path / "fail.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[
            _classify("code_screenshot"),
            _vlm_result("", success=False, error="API timeout"),
        ],
    ):
        doc = parse_image(str(image_path))

    assert doc.format == DocumentFormat.IMAGE
    assert doc.status == ProcessingStatus.PARSED
    assert doc.blocks == []


def test_parse_image_calls_vlm_twice_for_classify_and_analyze(tmp_path: Path) -> None:
    """Both classify (image_classify) and analyze (image_analysis) VLM calls are made."""
    image_path = tmp_path / "two_calls.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("text_capture"), _text_analysis()],
    ) as mock_vlm:
        parse_image(str(image_path))

    assert mock_vlm.call_count == 2
    stages = [c.kwargs.get("stage") for c in mock_vlm.call_args_list]
    assert "image_classify" in stages
    assert "image_analysis" in stages


def test_parse_image_document_fields(tmp_path: Path) -> None:
    """Document id, source, format, and processing fields are correctly set."""
    image_path = tmp_path / "fields.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("text_capture"), _text_analysis()],
    ):
        doc = parse_image(str(image_path), model="gpt-4o-mini")

    assert doc.id  # non-empty hash
    assert doc.source == "fields.png"
    assert doc.format == DocumentFormat.IMAGE
    assert doc.processing.vlm_model == "gpt-4o-mini"


def test_parse_image_uses_preprocessed_path_and_attaches_metadata(tmp_path: Path) -> None:
    """Analysis should use the preprocessed image while blocks keep preprocessing metadata."""
    image_path = tmp_path / "preprocess.png"
    processed_path = tmp_path / "preprocess_preprocessed.png"
    _fake_image(image_path)
    _fake_image(processed_path)
    preprocess_meta = {
        "original_size": (2000, 1500),
        "processed_size": (1600, 1200),
        "original_tokens_openai": 765,
        "processed_tokens_openai": 765,
        "token_reduction_pct": 0.0,
        "format": "PNG",
        "resized": True,
    }

    with (
        patch(
            "parsers.image_parser.preprocess_image",
            return_value=(str(processed_path), preprocess_meta),
        ),
        patch(
            "parsers.image_parser.call_vlm",
            side_effect=[_classify("text_capture"), _text_analysis()],
        ) as mock_vlm,
    ):
        doc = parse_image(str(image_path))

    assert mock_vlm.call_args_list[1].args[1] == str(processed_path)
    assert doc.blocks[0].metadata.preprocess == preprocess_meta
    assert doc.model_dump(mode="json")["blocks"][0]["metadata"]["preprocess"] == {
        **preprocess_meta,
        "original_size": [2000, 1500],
        "processed_size": [1600, 1200],
    }


def test_parse_image_preprocess_failure_falls_back_to_original(tmp_path: Path) -> None:
    """Preprocessing failures should not block analysis of the original image."""
    image_path = tmp_path / "fallback.png"
    _fake_image(image_path)

    with (
        patch(
            "parsers.image_parser.preprocess_image",
            side_effect=ValueError("bad image"),
        ),
        patch(
            "parsers.image_parser.call_vlm",
            side_effect=[_classify("text_capture"), _text_analysis()],
        ) as mock_vlm,
    ):
        doc = parse_image(str(image_path))

    assert mock_vlm.call_args_list[1].args[1] == str(image_path)
    assert doc.blocks[0].metadata.preprocess is None


def test_parse_image_removes_transient_preprocessed_file(tmp_path: Path) -> None:
    """Transient preprocessed siblings should be removed after analysis completes."""
    image_path = tmp_path / "cleanup.png"
    processed_path = tmp_path / "cleanup_preprocessed.png"
    _fake_image(image_path)
    _fake_image(processed_path)

    with (
        patch(
            "parsers.image_parser.preprocess_image",
            return_value=(str(processed_path), {"format": "PNG", "resized": True}),
        ),
        patch(
            "parsers.image_parser.call_vlm",
            side_effect=[_classify("text_capture"), _text_analysis()],
        ),
    ):
        parse_image(str(image_path))

    assert not processed_path.exists()


def test_parse_image_chart_classification_uses_chart_branch(tmp_path: Path) -> None:
    """Chart classifications should map to ImageType.CHART and the diagram parser path."""
    image_path = tmp_path / "chart.png"
    _fake_image(image_path)

    with patch(
        "parsers.image_parser.call_vlm",
        side_effect=[_classify("chart"), _diagram_analysis()],
    ):
        doc = parse_image(str(image_path))

    assert len(doc.blocks) == 1
    assert doc.blocks[0].type == BlockType.FIGURE
    assert doc.blocks[0].metadata.image_type == ImageType.CHART


def test_map_vlm_output_to_block_for_diagram() -> None:
    """Diagram payload maps to FIGURE block with structured component/relationship text."""
    payload = DiagramVLMOutput.model_validate(
        {
            "schema_version": "v1.1.0",
            "diagram_type": "flowchart",
            "title": "파이프라인",
            "description": "전체 흐름도",
            "components": [
                {"name": "A", "role": "입력"},
                {"name": "B", "role": "출력"},
            ],
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
