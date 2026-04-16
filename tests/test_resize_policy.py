from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import eval.resize_policy as resize_policy_module
from eval.resize_policy import (
    aggregate_rows,
    build_condition_artifacts,
    discover_manifest_rows,
    evaluate_quality,
    select_pdf_bucket,
    write_manifest_csv,
)
from models.document import ImageType


def _make_image(path: Path, *, size: tuple[int, int] = (1200, 800), mode: str = "RGB") -> None:
    image = Image.new(mode, size, color=(255, 255, 255))
    image.save(path)


def test_discover_manifest_rows_and_write_csv(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "golden_resize"
    code_dir = dataset_dir / "code_screenshot"
    chart_dir = dataset_dir / "chart"
    code_dir.mkdir(parents=True)
    chart_dir.mkdir(parents=True)

    manual_image = code_dir / "manual_python_slide.png"
    auto_image = chart_dir / "pdf_auto__deep_learning_ch3__p2__o7.png"
    _make_image(manual_image)
    _make_image(auto_image)

    rows = discover_manifest_rows(dataset_dir)

    assert [row.image_id for row in rows] == [
        "manual_python_slide",
        "pdf_auto__deep_learning_ch3__p2__o7",
    ]
    assert rows[0].source_kind == "manual_capture"
    assert rows[1].source_kind == "pdf_auto"
    assert rows[1].source_doc == "deep_learning_ch3"

    manifest_path = write_manifest_csv(rows, dataset_dir / "manifest.csv")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "image_id,image_path,image_type,source_kind,source_doc,filename" in manifest_text
    assert "pdf_auto__deep_learning_ch3__p2__o7" in manifest_text


def test_select_pdf_bucket_prefers_chart_label(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    _make_image(image_path)

    bucket = select_pdf_bucket(
        image_path,
        label_hint="chart",
        classify_model="gpt-4.1-nano",
    )

    assert bucket == ImageType.CHART


def test_select_pdf_bucket_falls_back_to_diagram_on_classifier_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "figure.png"
    _make_image(image_path)
    monkeypatch.setattr(
        resize_policy_module,
        "classify_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing key")),
    )

    bucket = select_pdf_bucket(
        image_path,
        label_hint="picture",
        classify_model="gpt-4.1-nano",
    )

    assert bucket == ImageType.DIAGRAM


def test_build_condition_artifacts_returns_three_conditions(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    _make_image(image_path, size=(2200, 1400))

    artifacts = build_condition_artifacts(str(image_path), ImageType.DIAGRAM)

    assert [artifact.condition for artifact in artifacts] == [
        "before",
        "after_uniform",
        "after_adaptive",
    ]
    assert artifacts[0].image_path == str(image_path)
    assert artifacts[1].preprocess["processed_tokens_openai"] <= artifacts[0].preprocess["processed_tokens_openai"]
    assert artifacts[2].preprocess["processed_tokens_openai"] <= artifacts[0].preprocess["processed_tokens_openai"]


def test_evaluate_quality_code_and_text() -> None:
    code_payload = json.dumps(
        {
            "schema_version": "v1.2.0",
            "language": "python",
            "code": "print('hello')",
            "code_markdown": "```python\\nprint('hello')\\n```",
            "description": "짧은 설명",
            "has_truncation": False,
            "confidence": 0.9,
            "errors": [],
        }
    )
    text_payload = json.dumps(
        {
            "schema_version": "v1.2.0",
            "text_type": "lecture_slide",
            "title": "제목",
            "content": "충분히 긴 텍스트 본문이 있어서 quality check를 통과합니다.",
            "key_points": ["a"],
            "has_math": False,
            "has_truncation": False,
            "confidence": 0.8,
            "errors": [],
        }
    )

    code_metrics = evaluate_quality(code_payload, ImageType.CODE_SCREENSHOT)
    text_metrics = evaluate_quality(text_payload, ImageType.TEXT_CAPTURE)

    assert code_metrics["json_parse_success"] is True
    assert code_metrics["required_fields_ok"] is True
    assert text_metrics["schema_validation_success"] is True
    assert text_metrics["required_fields_ok"] is True


def test_aggregate_rows_by_condition_and_type() -> None:
    rows = [
        {
            "image_type": "diagram",
            "condition": "before",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001,
            "latency_ms": 700,
            "json_parse_success": True,
            "required_fields_ok": True,
            "empty_output": False,
            "has_truncation": False,
        },
        {
            "image_type": "diagram",
            "condition": "before",
            "input_tokens": 120,
            "output_tokens": 40,
            "cost_usd": 0.0012,
            "latency_ms": 900,
            "json_parse_success": False,
            "required_fields_ok": False,
            "empty_output": True,
            "has_truncation": True,
        },
        {
            "image_type": "diagram",
            "condition": "after_adaptive",
            "input_tokens": 60,
            "output_tokens": 35,
            "cost_usd": 0.0007,
            "latency_ms": 500,
            "json_parse_success": True,
            "required_fields_ok": True,
            "empty_output": False,
            "has_truncation": False,
        },
    ]

    aggregates = aggregate_rows(rows)
    before = next(item for item in aggregates["by_condition"] if item["condition"] == "before")
    adaptive = next(item for item in aggregates["by_condition"] if item["condition"] == "after_adaptive")

    assert before["count"] == 2
    assert before["avg_input_tokens"] == 110.0
    assert before["json_parse_success_rate"] == 0.5
    assert adaptive["avg_latency_ms"] == 500.0
