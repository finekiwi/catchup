"""Lightweight integration test for image parser -> note generator flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llm.note_generator as note_gen_module
from parsers.image_parser import parse_image
from vlm.client import VLMResult

generate_note = pytest.importorskip("llm.note_generator").generate_note


def test_image_to_note_pipeline_with_mock_models(tmp_path: Path, monkeypatch) -> None:
    """Image parse output should flow into note generation successfully."""
    image_path = tmp_path / "lecture.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    classify_result = VLMResult(
        content=json.dumps({"image_type": "text_capture", "confidence": 0.95}),
        model="gpt-4o-mini",
        success=True,
    )
    analysis_result = VLMResult(
        content=json.dumps({
            "schema_version": "v1.1.0",
            "text_type": "lecture_slide",
            "title": "미분 개요",
            "content": "## 미분\n변화율을 다룬다.",
            "key_points": ["변화율"],
            "has_math": False,
            "has_truncation": False,
            "confidence": 0.9,
            "errors": [],
        }),
        model="gpt-4o-mini",
        success=True,
    )

    with patch("parsers.image_parser.call_vlm", side_effect=[classify_result, analysis_result]):
        parsed_doc = parse_image(file_path=str(image_path))
    assert len(parsed_doc.blocks) == 1

    llm_note_response = json.dumps(
        {
            "schema_version": "v1.1.0",
            "title": "미분 학습노트",
            "summary": "미분의 핵심을 정리한다.",
            "note_markdown": "## 핵심\n- 변화율",
            "key_concepts": ["미분", "변화율"],
            "difficulty_level": "beginner",
            "estimated_read_time_min": 2,
            "confidence": 0.86,
            "errors": [],
        }
    )

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = llm_note_response
    mock_resp.usage.prompt_tokens = 80
    mock_resp.usage.completion_tokens = 120
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(note_gen_module, "openai", MagicMock(OpenAI=lambda: mock_client))
    monkeypatch.setattr(note_gen_module, "log_api_call", lambda **kw: None)

    result = generate_note(parsed_doc)
    assert "## 핵심" in result["note_markdown"]
    assert "변화율" in result["key_concepts"]
