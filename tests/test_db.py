"""Tests for SQLite and ChromaDB helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from db import chroma as chroma_db
from db import sqlite as sqlite_db
from models.document import (
    Block,
    BlockMetadata,
    BlockType,
    Document,
    DocumentFormat,
    DocumentMetadata,
    ProcessingStatus,
)


def _build_document(doc_id: str = "doc-1", source: str = "sample.pdf") -> Document:
    """Create a sample document for DB tests."""
    return Document(
        id=doc_id,
        source=source,
        format=DocumentFormat.PDF,
        blocks=[
            Block(
                type=BlockType.TEXT,
                content="설명 블록",
                order=0,
                image_path=f"data/figures/{doc_id}/0.png",
                metadata=BlockMetadata(page=3),
            )
        ],
        metadata=DocumentMetadata(
            title="Sample Title", tags=["math", "ml"], total_pages=10
        ),
        created_at=datetime(2026, 3, 3, tzinfo=timezone.utc),
    )


def test_sqlite_crud_flow(tmp_path, monkeypatch) -> None:
    """SQLite should support save -> get -> update_status -> list flow."""
    sqlite_path = tmp_path / "sqlite" / "catchup.db"
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(sqlite_path))

    document = _build_document()
    sqlite_db.save_document(document)

    saved = sqlite_db.get_document(document.id)
    assert saved is not None
    assert saved.id == document.id
    assert saved.metadata.title == "Sample Title"

    sqlite_db.update_status(document.id, ProcessingStatus.NOTE_GENERATED)
    fetched = sqlite_db.get_document(document.id)
    assert fetched is not None
    assert fetched.status == ProcessingStatus.NOTE_GENERATED

    all_documents = sqlite_db.list_documents()
    assert len(all_documents) == 1
    assert all_documents[0].id == document.id


def test_sqlite_upsert_on_duplicate_document_id(tmp_path, monkeypatch) -> None:
    """Duplicate document IDs should be handled with upsert behavior."""
    sqlite_path = tmp_path / "sqlite" / "catchup.db"
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(sqlite_path))

    first = _build_document(doc_id="duplicate-doc", source="first.pdf")
    sqlite_db.save_document(first)

    second = Document(
        id="duplicate-doc",
        source="second.pdf",
        format=DocumentFormat.PDF,
        status=ProcessingStatus.CONCEPTS_EXTRACTED,
        blocks=[
            Block(
                type=BlockType.CODE,
                content="print('updated')",
                order=1,
                image_path="data/figures/duplicate-doc/1.png",
                metadata=BlockMetadata(page=5, language="python"),
            )
        ],
        metadata=DocumentMetadata(title="Updated Title", tags=["updated"]),
        created_at=datetime(2026, 3, 3, tzinfo=timezone.utc),
    )
    sqlite_db.save_document(second)

    fetched = sqlite_db.get_document("duplicate-doc")
    assert fetched is not None
    assert fetched.source == "second.pdf"
    assert fetched.status == ProcessingStatus.CONCEPTS_EXTRACTED
    assert fetched.metadata.title == "Updated Title"
    assert len(fetched.blocks) == 1
    assert fetched.blocks[0].image_path == "data/figures/duplicate-doc/1.png"
    assert len(sqlite_db.list_documents()) == 1


def test_sqlite_roundtrip_preserves_blocks_and_image_paths(tmp_path, monkeypatch) -> None:
    """Stored library documents must keep blocks so inline figures can render later."""
    sqlite_path = tmp_path / "sqlite" / "catchup.db"
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(sqlite_path))

    document = _build_document(doc_id="doc-with-figure", source="figure.pdf")
    sqlite_db.save_document(document)

    fetched = sqlite_db.get_document(document.id)
    assert fetched is not None
    assert len(fetched.blocks) == 1
    assert fetched.blocks[0].content == "설명 블록"
    assert fetched.blocks[0].image_path == "data/figures/doc-with-figure/0.png"
    assert fetched.blocks[0].metadata.page == 3


def test_sqlite_creates_db_file_automatically(tmp_path, monkeypatch) -> None:
    """SQLite DB file should be created automatically when missing."""
    sqlite_path = tmp_path / "missing-dir" / "catchup.db"
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(sqlite_path))

    assert not sqlite_path.exists()
    assert sqlite_db.list_documents() == []
    assert sqlite_path.exists()


def test_notes_save_and_get(tmp_path, monkeypatch) -> None:
    """save_note -> get_note should round-trip the result dict."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    result = {"title": "Test Note", "note_markdown": "## Hello\nWorld"}
    sqlite_db.save_note("doc-1", "abc123hash", result, "gpt-4o", "gpt-4o")

    fetched = sqlite_db.get_note("doc-1", "gpt-4o", "gpt-4o")
    assert fetched is not None
    assert fetched["result"] == result
    assert fetched["file_hash"] == "abc123hash"
    assert fetched["vlm_model"] == "gpt-4o"
    assert fetched["llm_model"] == "gpt-4o"
    assert fetched["is_image"] is False
    assert fetched["updated_at"]  # non-empty


def test_notes_upsert(tmp_path, monkeypatch) -> None:
    """Saving with the same PK should overwrite the previous note."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    sqlite_db.save_note("doc-1", "hash1", {"v": 1}, "gpt-4o", "gpt-4o")
    sqlite_db.save_note("doc-1", "hash1", {"v": 2}, "gpt-4o", "gpt-4o")

    fetched = sqlite_db.get_note("doc-1", "gpt-4o", "gpt-4o")
    assert fetched is not None
    assert fetched["result"] == {"v": 2}


def test_notes_list_for_document(tmp_path, monkeypatch) -> None:
    """list_notes_for_document should filter by document_id."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    sqlite_db.save_note("doc-A", "h1", {"a": 1}, "vlm1", "llm1")
    sqlite_db.save_note("doc-A", "h1", {"a": 2}, "vlm2", "llm2")
    sqlite_db.save_note("doc-B", "h2", {"b": 1}, "vlm1", "llm1")

    notes_a = sqlite_db.list_notes_for_document("doc-A")
    assert len(notes_a) == 2

    notes_b = sqlite_db.list_notes_for_document("doc-B")
    assert len(notes_b) == 1
    assert notes_b[0]["result"] == {"b": 1}


def test_notes_delete(tmp_path, monkeypatch) -> None:
    """delete_note should remove the note, get_note returns None after."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    sqlite_db.save_note("doc-1", "h", {"x": 1}, "vlm", "llm")
    assert sqlite_db.get_note("doc-1", "vlm", "llm") is not None

    sqlite_db.delete_note("doc-1", "vlm", "llm")
    assert sqlite_db.get_note("doc-1", "vlm", "llm") is None


class _FakeCollection:
    """Simple in-memory fake Chroma collection."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def delete(self, where: dict[str, Any]) -> None:
        """Delete records matching where filter (supports doc_id key)."""
        doc_id = where.get("doc_id")
        if doc_id:
            keys_to_delete = [
                k
                for k, v in self.records.items()
                if v["metadata"].get("doc_id") == doc_id
            ]
            for key in keys_to_delete:
                del self.records[key]

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Store records by ID."""
        for record_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            self.records[record_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def query(
        self, query_embeddings: list[list[float]], n_results: int, include: list[str]
    ) -> dict[str, list[list[Any]]]:
        """Return deterministic mock query payload."""
        del query_embeddings
        del include

        # NOTE: real ChromaDB returns by similarity score; this fake returns alphabetically by ID.
        # Update when switching to real embeddings.
        selected_ids = sorted(self.records.keys())[:n_results]
        selected_documents = [
            self.records[item_id]["document"] for item_id in selected_ids
        ]
        selected_metadatas = [
            self.records[item_id]["metadata"] for item_id in selected_ids
        ]
        selected_distances = [0.0 for _ in selected_ids]
        return {
            "ids": [selected_ids],
            "documents": [selected_documents],
            "metadatas": [selected_metadatas],
            "distances": [selected_distances],
        }


def test_chroma_reingest_removes_stale_vectors(monkeypatch) -> None:
    """Re-ingesting a doc with fewer blocks must not leave stale old vectors queryable."""
    fake_collection = _FakeCollection()
    monkeypatch.setattr(chroma_db, "_get_collection", lambda: fake_collection)

    # First ingest: 3 blocks
    blocks_v1 = [
        Block(type=BlockType.TEXT, content="Block A", order=0),
        Block(type=BlockType.TEXT, content="Block B", order=1),
        Block(type=BlockType.TEXT, content="Block C", order=2),
    ]
    chroma_db.store_embeddings("doc-reingest", blocks_v1)
    assert len(fake_collection.records) == 3

    # Re-ingest same doc_id with only 1 block — stale vectors must be gone
    blocks_v2 = [
        Block(type=BlockType.TEXT, content="Block A updated", order=0),
    ]
    chroma_db.store_embeddings("doc-reingest", blocks_v2)

    assert len(fake_collection.records) == 1, (
        "Stale vectors from first ingest must be removed"
    )
    assert "doc-reingest:0" in fake_collection.records
    assert fake_collection.records["doc-reingest:0"]["document"] == "Block A updated"


def test_chroma_store_and_search(monkeypatch) -> None:
    """Chroma helper should store block embeddings and return search results."""
    fake_collection = _FakeCollection()
    monkeypatch.setattr(chroma_db, "_get_collection", lambda: fake_collection)

    blocks = [
        Block(type=BlockType.TEXT, content="Linear algebra fundamentals", order=0),
        Block(type=BlockType.TEXT, content="Python data structures and lists", order=1),
    ]
    chroma_db.store_embeddings("doc-1", blocks)

    results = chroma_db.search("python", n_results=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc-1:0"
    assert results[1]["id"] == "doc-1:1"
    assert results[1]["metadata"]["doc_id"] == "doc-1"
    assert results[1]["metadata"]["type"] == BlockType.TEXT.value


# ---------------------------------------------------------------------------
# Concepts CRUD tests (CU-17)
# ---------------------------------------------------------------------------


def test_save_get_concepts_roundtrip(tmp_path, monkeypatch) -> None:
    """save_concepts → get_concepts_for_document should preserve all fields."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    concepts = [
        {"concept_name": "역전파", "canonical_name": "backpropagation", "aliases": ["backprop", "역전파"], "definition": "가중치 업데이트 알고리즘"},
        {"concept_name": "활성화 함수", "canonical_name": "activation function", "aliases": [], "definition": "비선형 변환"},
    ]
    ids = sqlite_db.save_concepts("doc-1", concepts)
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)

    fetched = sqlite_db.get_concepts_for_document("doc-1")
    assert len(fetched) == 2

    bp = next(c for c in fetched if c["canonical_name"] == "backpropagation")
    assert bp["document_id"] == "doc-1"
    assert bp["concept_name"] == "역전파"
    assert "backprop" in bp["aliases"]
    assert bp["definition"] == "가중치 업데이트 알고리즘"


def test_concept_links_join(tmp_path, monkeypatch) -> None:
    """save_concept_links → get_concept_links_for_document should return correct source/target."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    # Create two documents in SQLite so JOIN with documents table works
    doc_a = _build_document("doc-a", "lecture_a.pdf")
    doc_b = _build_document("doc-b", "lecture_b.pdf")
    sqlite_db.save_document(doc_a)
    sqlite_db.save_document(doc_b)

    ids_a = sqlite_db.save_concepts("doc-a", [{"concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": "활성화 함수"}])
    ids_b = sqlite_db.save_concepts("doc-b", [{"concept_name": "activation", "canonical_name": "relu", "aliases": ["relu"], "definition": "비선형 함수"}])

    sqlite_db.save_concept_links([
        {
            "concept_id_a": ids_a[0],
            "concept_id_b": ids_b[0],
            "confidence_score": 1.0,
            "relationship_type": "same_concept",
            "relationship_desc": "",
        }
    ])

    links_for_a = sqlite_db.get_concept_links_for_document("doc-a")
    assert len(links_for_a) == 1
    link = links_for_a[0]
    assert link["source_canonical_name"] == "relu"
    assert link["target_document_id"] == "doc-b"
    assert link["confidence_score"] == 1.0
    assert link["relationship_type"] == "same_concept"

    # Also verify from doc-b perspective
    links_for_b = sqlite_db.get_concept_links_for_document("doc-b")
    assert len(links_for_b) == 1
    assert links_for_b[0]["target_document_id"] == "doc-a"


def test_delete_concepts_cascade(tmp_path, monkeypatch) -> None:
    """delete_concepts_for_document should remove links and concepts."""
    monkeypatch.setenv("CATCHUP_SQLITE_PATH", str(tmp_path / "test.db"))

    ids_a = sqlite_db.save_concepts("doc-a", [{"concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": ""}])
    ids_b = sqlite_db.save_concepts("doc-b", [{"concept_name": "relu", "canonical_name": "relu", "aliases": [], "definition": ""}])

    sqlite_db.save_concept_links([
        {"concept_id_a": ids_a[0], "concept_id_b": ids_b[0], "confidence_score": 1.0, "relationship_type": "same_concept", "relationship_desc": ""}
    ])

    # Verify link exists
    assert len(sqlite_db.get_concept_links_for_document("doc-a")) == 1

    # Delete doc-a concepts → link should cascade
    sqlite_db.delete_concepts_for_document("doc-a")

    assert sqlite_db.get_concepts_for_document("doc-a") == []
    assert sqlite_db.get_concept_links_for_document("doc-a") == []
    # doc-b concepts should be unaffected
    assert len(sqlite_db.get_concepts_for_document("doc-b")) == 1
