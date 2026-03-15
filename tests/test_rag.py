"""Unit tests for RAG Q&A pipeline (CU-08)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

import chromadb

import rag.qa_chain as qa_module
from models.document import Block, BlockMetadata, BlockType, Document, DocumentFormat
from rag.qa_chain import QAResult, SourceBlock, index_document, query
from utils.models import LLMResponse

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR = [0.1] * 1536  # text-embedding-3-small output dimension


def _make_ephemeral_collection():
    """Create an isolated in-memory ChromaDB collection for testing.

    Uses a unique name per call so tests never share state even when
    EphemeralClient reuses a process-level singleton.
    """
    try:
        client = chromadb.EphemeralClient()
    except AttributeError:
        client = chromadb.Client()
    unique_name = f"test_rag_{uuid.uuid4().hex}"
    return client.get_or_create_collection(unique_name)


def _sample_document(doc_id: str = "doc-test-01") -> Document:
    return Document(
        id=doc_id,
        source="intro_ml.pdf",
        format=DocumentFormat.PDF,
        blocks=[
            Block(
                type=BlockType.TEXT,
                content="Gradient descent is an optimization algorithm used to minimize loss.",
                order=0,
                metadata=BlockMetadata(page=1),
            ),
            Block(
                type=BlockType.TEXT,
                content="The learning rate controls the step size during optimization.",
                order=1,
                metadata=BlockMetadata(page=2),
            ),
        ],
    )


def _patch_embedding(monkeypatch, vector=None, tokens=50):
    """Patch _get_openai_embedding to return a deterministic fake vector."""
    v = vector if vector is not None else FAKE_VECTOR
    monkeypatch.setattr(qa_module, "_get_openai_embedding", lambda text: (v, tokens))


def _patch_llm(monkeypatch, answer="Test answer.", input_tokens=100, output_tokens=50):
    """Patch the shared LLM dispatcher to return a canned answer."""
    monkeypatch.setattr(
        qa_module,
        "call_llm",
        lambda model, messages, max_tokens=2048: LLMResponse(
            content=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _patch_log(monkeypatch):
    """Suppress log_api_call file I/O."""
    monkeypatch.setattr("rag.qa_chain.log_api_call", lambda **kw: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_index_and_query_returns_source_blocks(monkeypatch):
    """Normal case: index a document then query returns answer with populated source_blocks."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_llm(
        monkeypatch,
        answer="Gradient descent optimizes the loss [intro_ml.pdf, page 1].",
    )
    _patch_log(monkeypatch)

    doc = _sample_document()
    index_document(doc)

    result = query("What is gradient descent?")

    assert isinstance(result, QAResult)
    assert result.answer != ""
    assert len(result.source_blocks) > 0
    assert result.source_blocks[0].source == "intro_ml.pdf"
    assert result.source_blocks[0].document_id == "doc-test-01"
    assert isinstance(result.source_blocks[0], SourceBlock)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.latency_ms >= 0


def test_query_empty_db_returns_appropriate_message(monkeypatch):
    """Empty DB: querying without any indexed documents returns informative answer, no exception."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    result = query("What is machine learning?")

    assert isinstance(result, QAResult)
    assert len(result.source_blocks) == 0
    assert len(result.answer) > 0
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_duplicate_indexing_does_not_store_blocks_twice(monkeypatch):
    """Calling index_document twice with the same document_id must not duplicate stored blocks."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    doc = _sample_document()
    index_document(doc)
    count_after_first = collection.count()

    index_document(doc)  # second call — should be skipped entirely
    count_after_second = collection.count()

    assert count_after_first == count_after_second
    assert count_after_first == 2  # both blocks of the document


def test_query_no_search_results_returns_no_relevant_docs_message(monkeypatch):
    """When vector search returns no IDs, the answer indicates no relevant content found."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: mock_collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    result = query("completely unrelated question xyz123")

    assert isinstance(result, QAResult)
    assert len(result.source_blocks) == 0
    assert "관련 문서를 찾지 못했습니다" in result.answer


def test_llm_api_failure_returns_fallback_answer(monkeypatch):
    """LLM API failure during query must return a fallback answer without raising."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    doc = _sample_document()
    index_document(doc)

    def _raise(model, system, user):
        raise RuntimeError("API timeout")

    monkeypatch.setattr(
        qa_module,
        "call_llm",
        lambda model, messages, max_tokens=2048: _raise(model, "", ""),
    )

    result = query("What is gradient descent?")

    assert isinstance(result, QAResult)
    assert result.answer != ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_embedding_failure_skips_block_and_indexes_remaining(monkeypatch):
    """If embedding fails for one block, that block is skipped and the rest are stored."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_log(monkeypatch)

    call_count = 0

    def _flaky_embed(text: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("embedding API error")
        return FAKE_VECTOR, 50

    monkeypatch.setattr(qa_module, "_get_openai_embedding", _flaky_embed)

    doc = _sample_document()  # 2 blocks
    index_document(doc)

    # block order=0 embedding failed (skipped), block order=1 succeeded
    assert collection.count() == 1


def test_source_block_content_preview_is_truncated_to_200_chars(monkeypatch):
    """content_preview in SourceBlock must be at most 200 characters even for long blocks."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_llm(monkeypatch)
    _patch_log(monkeypatch)

    long_content = "A" * 500
    doc = Document(
        id="doc-long",
        source="long.pdf",
        format=DocumentFormat.PDF,
        blocks=[Block(type=BlockType.TEXT, content=long_content, order=0)],
    )
    index_document(doc)

    result = query("test question")

    assert len(result.source_blocks) > 0
    assert len(result.source_blocks[0].content_preview) <= 200


def test_query_unsupported_model_raises_value_error():
    """query() with an unregistered model must raise ValueError, not silently fall through to OpenAI."""
    with pytest.raises(ValueError, match="Unsupported model"):
        query("any question", model="gpt-99-nonexistent")


def test_index_document_reindexes_on_version_mismatch(monkeypatch):
    """Stale chunks (old indexing_version) must be deleted and re-indexed."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    doc = _sample_document()

    # Pre-populate with old-version metadata
    old_meta = {"document_id": doc.id, "block_order": 0, "indexing_version": "v_old"}
    collection.upsert(
        ids=["stale-id"],
        documents=["stale content"],
        metadatas=[old_meta],
        embeddings=[FAKE_VECTOR],
    )
    assert collection.count() == 1

    # index_document should detect version mismatch, delete stale chunk, then re-index
    index_document(doc)

    stored = collection.get(where={"document_id": doc.id}, include=["metadatas"])
    stored_versions = {m.get("indexing_version") for m in stored["metadatas"]}
    assert "stale-id" not in stored["ids"]
    assert stored_versions == {qa_module.INDEXING_VERSION}
    assert len(stored["ids"]) == len(doc.blocks)


def test_index_document_reindexes_when_version_missing(monkeypatch):
    """Chunks indexed before versioning (no indexing_version field) must be re-indexed."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    doc = _sample_document()

    # Pre-populate without any indexing_version field (pre-versioning data)
    old_meta = {"document_id": doc.id, "block_order": 0}
    collection.upsert(
        ids=["legacy-id"],
        documents=["legacy content"],
        metadatas=[old_meta],
        embeddings=[FAKE_VECTOR],
    )
    assert collection.count() == 1

    # index_document should detect missing version, delete legacy chunk, then re-index
    index_document(doc)

    stored = collection.get(where={"document_id": doc.id}, include=["metadatas"])
    assert "legacy-id" not in stored["ids"], "Legacy chunk must be replaced"
    stored_versions = {m.get("indexing_version") for m in stored["metadatas"]}
    assert stored_versions == {qa_module.INDEXING_VERSION}


def test_index_document_skips_when_version_matches(monkeypatch):
    """Document already indexed at the current version must not be re-indexed."""
    collection = _make_ephemeral_collection()
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: collection)
    _patch_embedding(monkeypatch)
    _patch_log(monkeypatch)

    doc = _sample_document()
    index_document(doc)
    count_after_first = collection.count()

    # Second call with same INDEXING_VERSION must be a no-op
    index_document(doc)
    assert collection.count() == count_after_first


def test_query_filters_to_current_document(monkeypatch):
    """When document_id is provided, retrieval must be restricted to that document."""
    meta = {
        "document_id": "doc-current",
        "source": "guardrails.pdf",
        "block_order": 0,
        "block_type": "text",
        "page": 3,
    }
    content = "Prompt injection is a representative jailbreak risk."
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_collection.get.return_value = {
        "ids": ["doc-current:0", "doc-current:1", "doc-current:2"]
    }
    mock_collection.query.return_value = {
        "ids": [["doc-current:0"]],
        "documents": [[content]],
        "metadatas": [[meta]],
        "distances": [[0.01]],
    }
    monkeypatch.setattr(qa_module, "_get_rag_collection", lambda: mock_collection)
    monkeypatch.setattr(
        qa_module,
        "_expand_with_adjacent_blocks",
        lambda *args, **kwargs: [(meta, content)],
    )
    _patch_embedding(monkeypatch)
    _patch_llm(
        monkeypatch, answer="Prompt injection is covered [guardrails.pdf, page 3]."
    )
    _patch_log(monkeypatch)

    result = query("prompt injection이 뭐야?", document_id="doc-current")

    assert result.source_blocks[0].document_id == "doc-current"
    assert mock_collection.query.call_args.kwargs["where"] == {
        "document_id": {"$eq": "doc-current"}
    }
