"""Unit tests for LLM note generator (CU-06)."""

from __future__ import annotations

import json

import llm.note_generator as note_gen_module
from models.document import Block, BlockMetadata, BlockType, Document, DocumentFormat, ProcessingStatus
from llm.note_generator import generate_note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_document() -> Document:
    """Create a minimal sample document for testing."""
    return Document(
        id="doc-test-01",
        source="linear_algebra.pdf",
        format=DocumentFormat.PDF,
        blocks=[
            Block(type=BlockType.TEXT, content="선형대수의 핵심 개념은 벡터와 행렬입니다.", order=0),
            Block(
                type=BlockType.CODE,
                content="import numpy as np\nA = np.array([[1, 2], [3, 4]])",
                order=1,
                metadata=BlockMetadata(language="python"),
            ),
        ],
    )


def _valid_llm_response(**overrides) -> dict:
    """Build a valid LLM JSON response dict."""
    base = {
        "schema_version": "v1.1.0",
        "title": "선형대수 학습 노트",
        "summary": "벡터와 행렬의 기초 개념을 정리한 노트입니다.",
        "note_markdown": "## 선형대수 기초\n\n- **벡터**: 방향과 크기를 가진 양\n- **행렬**: 벡터의 집합",
        "key_concepts": ["벡터", "행렬", "선형변환"],
        "difficulty_level": "beginner",
        "estimated_read_time_min": 5,
        "confidence": 0.92,
        "errors": [],
    }
    base.update(overrides)
    return base


def _patch_call_openai(monkeypatch, doc_dict: dict, input_tokens: int = 120, output_tokens: int = 300) -> None:
    """Patch the openai entry in _PROVIDER_DISPATCH to return (raw_json, input_tokens, output_tokens).

    Patching the dispatch dict item (not the module attribute) is required because
    _PROVIDER_DISPATCH stores function references captured at import time.
    """
    raw = json.dumps(doc_dict)
    monkeypatch.setitem(note_gen_module._PROVIDER_DISPATCH, "openai", lambda model, system, user: (raw, input_tokens, output_tokens))


def _patch_call_openai_raise(monkeypatch, exc: Exception) -> None:
    """Patch the openai dispatch entry to raise an exception."""
    def _raise(model: str, system: str, user: str) -> tuple:
        raise exc
    monkeypatch.setitem(note_gen_module._PROVIDER_DISPATCH, "openai", _raise)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_note_success_all_fields_present(monkeypatch) -> None:
    """Successful JSON response must contain all required fields."""
    doc = _sample_document()
    _patch_call_openai(monkeypatch, _valid_llm_response())
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    result = generate_note(doc)

    required_keys = {
        "title", "summary", "note_markdown", "key_concepts",
        "difficulty_level", "estimated_read_time_min",
        "schema_version", "confidence", "errors",
    }
    assert required_keys.issubset(result.keys())
    assert result["title"] == "선형대수 학습 노트"
    assert result["schema_version"] == "v1.1.0"


def test_generate_note_markdown_contains_markdown_content(monkeypatch) -> None:
    """note_markdown field must contain actual markdown content."""
    doc = _sample_document()
    _patch_call_openai(monkeypatch, _valid_llm_response())
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    result = generate_note(doc)

    assert "##" in result["note_markdown"]
    assert len(result["note_markdown"]) > 0


def test_generate_note_key_concepts_in_valid_range(monkeypatch) -> None:
    """key_concepts must be a list with 0 to 10 items."""
    doc = _sample_document()
    _patch_call_openai(monkeypatch, _valid_llm_response(key_concepts=["벡터", "행렬", "내적", "외적"]))
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    result = generate_note(doc)

    assert isinstance(result["key_concepts"], list)
    assert 0 <= len(result["key_concepts"]) <= 10


def test_generate_note_json_parse_failure_returns_fallback(monkeypatch) -> None:
    """JSON parse failure must return fallback dict with raw_response in note_markdown."""
    doc = _sample_document()
    monkeypatch.setitem(note_gen_module._PROVIDER_DISPATCH, "openai", lambda model, system, user: ("이건 JSON이 아닙니다.", 100, 20))

    log_calls: list[dict] = []
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: log_calls.append(kw))

    result = generate_note(doc)

    assert result["title"] == doc.source
    assert result["note_markdown"] == "이건 JSON이 아닙니다."
    assert result["errors"]  # fallback must populate errors list
    assert result["confidence"] == 0.0


def test_generate_note_api_failure_returns_fallback_and_logs_error(monkeypatch) -> None:
    """API call failure must return fallback dict and call log_api_call with success=False."""
    doc = _sample_document()
    _patch_call_openai_raise(monkeypatch, RuntimeError("connection timeout"))

    log_calls: list[dict] = []
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: log_calls.append(kw))

    result = generate_note(doc)

    assert result["title"] == doc.source
    assert result["note_markdown"] == ""
    assert len(log_calls) == 1
    assert log_calls[0]["success"] is False
    assert "connection timeout" in log_calls[0]["error"]


def test_generate_note_empty_blocks_document(monkeypatch) -> None:
    """Document with no blocks must process without error."""
    doc = Document(
        id="empty-doc",
        source="empty.pdf",
        format=DocumentFormat.PDF,
        blocks=[],
    )
    _patch_call_openai(
        monkeypatch,
        _valid_llm_response(title="빈 문서 노트", key_concepts=[], note_markdown="## 내용 없음\n"),
    )
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    result = generate_note(doc)

    assert isinstance(result, dict)
    assert "note_markdown" in result


def test_generate_note_sets_status_note_generated(monkeypatch) -> None:
    """On success, doc.status must be updated to NOTE_GENERATED."""
    doc = _sample_document()
    assert doc.status != ProcessingStatus.NOTE_GENERATED

    _patch_call_openai(monkeypatch, _valid_llm_response())
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    generate_note(doc)

    assert doc.status == ProcessingStatus.NOTE_GENERATED


def test_generate_note_failure_adds_note_generation_failed_tag(monkeypatch) -> None:
    """On failure, 'note_generation_failed' tag must be added to doc.metadata.tags."""
    doc = _sample_document()
    _patch_call_openai_raise(monkeypatch, ValueError("invalid api key"))
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    generate_note(doc)

    assert "note_generation_failed" in doc.metadata.tags


def test_generate_note_json_parse_failure_adds_tag(monkeypatch) -> None:
    """JSON parse failure must also add 'note_generation_failed' tag."""
    doc = _sample_document()
    monkeypatch.setitem(note_gen_module._PROVIDER_DISPATCH, "openai", lambda model, system, user: ("not json at all", 50, 10))
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    generate_note(doc)

    assert "note_generation_failed" in doc.metadata.tags


def test_generate_note_calls_log_api_call_on_success(monkeypatch) -> None:
    """log_api_call must be invoked with correct stage on successful note generation."""
    doc = _sample_document()
    _patch_call_openai(monkeypatch, _valid_llm_response())

    log_calls: list[dict] = []
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: log_calls.append(kw))

    generate_note(doc, model="gpt-4o-mini")

    assert len(log_calls) == 1
    assert log_calls[0]["stage"] == "note_generation"
    assert log_calls[0]["model"] == "gpt-4o-mini"
    assert log_calls[0]["success"] is True


def test_generate_note_failure_does_not_set_note_generated_status(monkeypatch) -> None:
    """On failure, doc.status must NOT be changed to NOTE_GENERATED."""
    doc = _sample_document()
    original_status = doc.status

    _patch_call_openai_raise(monkeypatch, OSError("network error"))
    monkeypatch.setattr("llm.note_generator.log_api_call", lambda **kw: None)

    generate_note(doc)

    assert doc.status == original_status
    assert doc.status != ProcessingStatus.NOTE_GENERATED
