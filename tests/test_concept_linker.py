"""Tests for llm/concept_linker.py — concept linking pipeline."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_resp(content: str, input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Build a minimal mock of an OpenAI chat completion response."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    return resp


def _norm_list(items: list[dict]) -> str:
    """Serialize a list of dicts for LLM normalize response."""
    return json.dumps(items, ensure_ascii=False)


# ---------------------------------------------------------------------------
# normalize_concepts
# ---------------------------------------------------------------------------


def test_normalize_concepts_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """normalize_concepts should return one dict per input concept."""
    from llm import concept_linker

    payload = [{"raw": "역전파", "canonical": "backpropagation", "aliases": ["backprop", "역전파"], "definition": "신경망 가중치를 업데이트하는 알고리즘"}]
    mock_resp = _make_openai_resp(json.dumps(payload))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    with patch("openai.OpenAI", return_value=mock_client):
        result = concept_linker.normalize_concepts(["역전파"], "gpt-4o-mini")

    assert len(result) == 1
    assert result[0]["canonical"] == "backpropagation"


def test_normalize_concepts_empty() -> None:
    """normalize_concepts with empty input should return empty list without API call."""
    from llm import concept_linker

    result = concept_linker.normalize_concepts([], "gpt-4o-mini")
    assert result == []


def test_normalize_concepts_json_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """All 4 required keys must be present in each result dict."""
    from llm import concept_linker

    payload = [
        {"raw": "gradient descent", "canonical": "gradient descent", "aliases": [], "definition": "손실함수를 최소화하는 최적화 알고리즘"},
        {"raw": "relu", "canonical": "relu", "aliases": ["ReLU", "rectified linear unit"], "definition": "음수를 0으로 만드는 활성화 함수"},
    ]
    mock_resp = _make_openai_resp(json.dumps(payload))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    with patch("openai.OpenAI", return_value=mock_client):
        result = concept_linker.normalize_concepts(["gradient descent", "relu"], "gpt-4o-mini")

    assert len(result) == 2
    for item in result:
        assert "raw" in item
        assert "canonical" in item
        assert "aliases" in item
        assert "definition" in item


# ---------------------------------------------------------------------------
# embed_and_store_concepts
# ---------------------------------------------------------------------------


def test_embed_and_store_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_and_store_concepts should call upsert on the collection for each concept."""
    from llm import concept_linker

    fake_collection = MagicMock()
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)
    monkeypatch.setattr(concept_linker, "_get_openai_embedding", lambda text: ([0.1] * 1536, 5))
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    concepts = [
        {"concept_name": "역전파", "canonical_name": "backpropagation", "aliases": ["backprop"], "definition": "가중치 업데이트 알고리즘", "id": 1},
        {"concept_name": "활성화 함수", "canonical_name": "activation function", "aliases": [], "definition": "비선형 변환 함수", "id": 2},
    ]
    concept_linker.embed_and_store_concepts("doc-1", concepts)

    assert fake_collection.upsert.call_count == 2


# ---------------------------------------------------------------------------
# find_exact_matches
# ---------------------------------------------------------------------------


def test_find_exact_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_exact_matches should return pairs with confidence=1.0 for canonical matches."""
    from llm import concept_linker

    existing = [
        {
            "id": 99,
            "document_id": "doc-other",
            "concept_name": "역전파",
            "canonical_name": "backpropagation",
            "aliases": [],
            "definition": "가중치 업데이트 알고리즘",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    monkeypatch.setattr(concept_linker, "get_all_concepts", lambda exclude_document_id=None: existing)

    concepts = [
        {"id": 1, "concept_name": "역전파", "canonical_name": "backpropagation", "aliases": [], "definition": ""}
    ]
    matches = concept_linker.find_exact_matches("doc-new", concepts)

    assert len(matches) == 1
    assert matches[0]["confidence_score"] == 1.0
    assert matches[0]["relationship_type"] == "same_concept"
    assert matches[0]["concept_id_a"] == 1
    assert matches[0]["concept_id_b"] == 99


# ---------------------------------------------------------------------------
# find_similar_concepts
# ---------------------------------------------------------------------------


def test_find_similar_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concepts below the similarity threshold must be excluded."""
    from llm import concept_linker

    fake_collection = MagicMock()
    fake_collection.count.return_value = 5
    # Return a hit with distance=0.30 → similarity=0.70 (below 0.80 threshold)
    fake_collection.query.return_value = {
        "metadatas": [[{"document_id": "doc-other", "canonical_name": "relu", "concept_name": "relu"}]],
        "distances": [[0.30]],
    }
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)
    monkeypatch.setattr(concept_linker, "_get_openai_embedding", lambda text: ([0.1] * 1536, 5))
    monkeypatch.setattr(concept_linker, "get_all_concepts", lambda exclude_document_id=None: [])

    concepts = [{"id": 1, "concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": "활성화 함수"}]
    result = concept_linker.find_similar_concepts("doc-new", concepts, already_matched_pairs=set(), threshold=0.80)

    assert result == []


def test_find_similar_same_doc_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hits from the same document must be excluded."""
    from llm import concept_linker

    fake_collection = MagicMock()
    fake_collection.count.return_value = 3
    # Return a hit from the same document with high similarity
    fake_collection.query.return_value = {
        "metadatas": [[{"document_id": "doc-new", "canonical_name": "relu", "concept_name": "relu"}]],
        "distances": [[0.05]],  # similarity=0.95
    }
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)
    monkeypatch.setattr(concept_linker, "_get_openai_embedding", lambda text: ([0.1] * 1536, 5))
    monkeypatch.setattr(concept_linker, "get_all_concepts", lambda exclude_document_id=None: [])

    concepts = [{"id": 1, "concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": "활성화 함수"}]
    result = concept_linker.find_similar_concepts("doc-new", concepts, already_matched_pairs=set(), threshold=0.80)

    assert result == []


def test_find_similar_first_doc_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ChromaDB collection is empty, find_similar_concepts returns empty."""
    from llm import concept_linker

    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)

    concepts = [{"id": 1, "concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": ""}]
    result = concept_linker.find_similar_concepts("doc-new", concepts, already_matched_pairs=set())

    assert result == []
    fake_collection.query.assert_not_called()


# ---------------------------------------------------------------------------
# label_relationships
# ---------------------------------------------------------------------------


def test_label_relationships_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """label_relationships should keep pairs with a valid relationship_type."""
    from llm import concept_linker

    payload = {"relationship_type": "prerequisite", "description": "역전파를 이해하려면 미분을 알아야 한다"}
    mock_resp = _make_openai_resp(json.dumps(payload))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    pair = {
        "concept_a": {"canonical_name": "backpropagation", "aliases": [], "definition": "가중치 업데이트", "document_id": "doc-a"},
        "concept_b": {"canonical_name": "calculus", "aliases": [], "definition": "미분적분학", "document_id": "doc-b"},
        "concept_id_a": 1,
        "concept_id_b": 2,
        "confidence_score": 0.85,
        "relationship_type": "",
        "relationship_desc": "",
    }

    with patch("openai.OpenAI", return_value=mock_client):
        result = concept_linker.label_relationships([pair], "gpt-4o-mini")

    assert len(result) == 1
    assert result[0]["relationship_type"] == "prerequisite"
    assert "역전파" in result[0]["relationship_desc"]


def test_label_relationships_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """label_relationships should drop pairs where LLM returns null relationship_type."""
    from llm import concept_linker

    payload = {"relationship_type": None, "description": ""}
    mock_resp = _make_openai_resp(json.dumps(payload))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    pair = {
        "concept_a": {"canonical_name": "dog", "aliases": [], "definition": "개", "document_id": "doc-a"},
        "concept_b": {"canonical_name": "compiler", "aliases": [], "definition": "컴파일러", "document_id": "doc-b"},
        "concept_id_a": 1,
        "concept_id_b": 2,
        "confidence_score": 0.82,
        "relationship_type": "",
        "relationship_desc": "",
    }

    with patch("openai.OpenAI", return_value=mock_client):
        result = concept_linker.label_relationships([pair], "gpt-4o-mini")

    assert result == []


# ---------------------------------------------------------------------------
# link_concepts (full pipeline)
# ---------------------------------------------------------------------------


def test_link_concepts_full_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """link_concepts should call save_concepts and save_concept_links."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CATCHUP_CHROMA_PATH", str(tmp_path / "chroma"))

    from llm import concept_linker

    # Mock all external calls
    normalize_payload = [
        {"raw": "역전파", "canonical": "backpropagation", "aliases": ["backprop"], "definition": "가중치 업데이트 알고리즘"}
    ]
    mock_resp = _make_openai_resp(json.dumps(normalize_payload))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    monkeypatch.setattr(concept_linker, "_get_openai_embedding", lambda text: ([0.0] * 1536, 5))
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    fake_collection.get.return_value = {"ids": []}
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)

    with patch("openai.OpenAI", return_value=mock_client):
        connections = concept_linker.link_concepts("doc-x", ["역전파"], "gpt-4o-mini")

    # No other docs exist → no links, but save_concepts was called (verified via DB)
    from db.sqlite import get_concepts_for_document

    saved = get_concepts_for_document("doc-x")
    assert len(saved) == 1
    assert saved[0]["canonical_name"] == "backpropagation"
    assert isinstance(connections, list)


def test_link_concepts_preserves_existing_on_save_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Existing concepts must not be deleted when save_concepts fails (probe-before-delete safety)."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CATCHUP_CHROMA_PATH", str(tmp_path / "chroma"))

    from llm import concept_linker
    from db.sqlite import save_concepts, get_concepts_for_document

    # Pre-populate existing valid concepts for the document
    save_concepts(
        "doc-safe",
        [{"concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": "activation function"}],
    )

    normalize_payload = [
        {"raw": "역전파", "canonical": "backpropagation", "aliases": [], "definition": "가중치 업데이트 알고리즘"}
    ]
    monkeypatch.setattr(concept_linker, "log_api_call", MagicMock())

    # Make the probe save_concepts (step 2) return empty — simulates SQLite write failure
    original_save = concept_linker.save_concepts
    call_count = {"n": 0}

    def failing_save(doc_id: str, rows: list) -> list[int]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return []  # probe save fails
        return original_save(doc_id, rows)

    monkeypatch.setattr(concept_linker, "save_concepts", failing_save)
    monkeypatch.setattr(concept_linker, "normalize_concepts", lambda *a, **kw: normalize_payload)
    monkeypatch.setattr(concept_linker, "delete_document_concepts", MagicMock())

    result = concept_linker.link_concepts("doc-safe", ["역전파"], "gpt-4o-mini")

    # Pipeline must abort early — delete must never have been called
    concept_linker.delete_document_concepts.assert_not_called()  # type: ignore[attr-defined]
    assert result == []

    # Original concepts must still be present (delete was never reached)
    still_saved = get_concepts_for_document("doc-safe")
    assert len(still_saved) == 1
    assert still_saved[0]["canonical_name"] == "relu"


# ---------------------------------------------------------------------------
# delete_document_concepts
# ---------------------------------------------------------------------------


def test_delete_document_concepts(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """delete_document_concepts should call SQLite delete and ChromaDB delete."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    from llm import concept_linker

    fake_collection = MagicMock()
    fake_collection.get.return_value = {"ids": ["doc-y:0", "doc-y:1"]}
    monkeypatch.setattr(concept_linker, "_get_concepts_collection", lambda: fake_collection)

    # Save a concept first
    from db.sqlite import save_concepts

    save_concepts("doc-y", [{"concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": ""}])

    concept_linker.delete_document_concepts("doc-y")

    # Verify ChromaDB delete was called
    fake_collection.delete.assert_called_once()

    # Verify SQLite concepts are gone
    from db.sqlite import get_concepts_for_document

    assert get_concepts_for_document("doc-y") == []
