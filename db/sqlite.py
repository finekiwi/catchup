"""SQLite persistence layer for documents and generated notes."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from models.document import Document, DocumentMetadata, ProcessingStatus

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_SQLITE_PATH = "data/catchup.db"
_NOTE_RESULT_VERSION = "v2"
_NOTE_RESULT_VERSION_KEY = "_note_result_version"
_DOCUMENT_JSON_COLUMN = "document_json"


def _sqlite_path() -> Path:
    """Return SQLite database path, optionally overridden by environment variable."""
    return Path(os.getenv("CATCHUP_SQLITE_PATH", DEFAULT_SQLITE_PATH))


def _initialize_schema(connection: sqlite3.Connection) -> None:
    """Create required tables if they do not exist."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            format TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            document_id TEXT NOT NULL,
            file_hash TEXT NOT NULL DEFAULT '',
            vlm_model TEXT NOT NULL DEFAULT '',
            llm_model TEXT NOT NULL DEFAULT '',
            result_json TEXT NOT NULL,
            is_image INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (document_id, vlm_model, llm_model)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            concept_name TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            aliases TEXT NOT NULL DEFAULT '[]',
            definition TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(document_id, concept_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id_a INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            concept_id_b INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            confidence_score REAL NOT NULL,
            relationship_type TEXT NOT NULL DEFAULT '',
            relationship_desc TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(concept_id_a, concept_id_b)
        )
        """
    )
    _ensure_column(
        connection,
        table="documents",
        column=_DOCUMENT_JSON_COLUMN,
        column_type="TEXT NOT NULL DEFAULT ''",
    )
    connection.commit()


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    column_type: str,
) -> None:
    """Add a missing SQLite column for lightweight schema migrations."""
    existing = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column in existing:
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _connect() -> Optional[sqlite3.Connection]:
    """Create a SQLite connection with initialized schema."""
    database_path = _sqlite_path()
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        _initialize_schema(connection)
        return connection
    except (OSError, sqlite3.Error):
        LOGGER.exception("Failed to connect SQLite database at path=%s", database_path)
        return None


def _row_to_document(row: sqlite3.Row) -> Optional[Document]:
    """Convert a DB row into a Document model."""
    metadata_raw = row["metadata"] or "{}"
    try:
        metadata_dict = json.loads(metadata_raw)
    except json.JSONDecodeError:
        LOGGER.warning("Invalid metadata JSON for document id=%s", row["id"])
        metadata_dict = {}

    document_raw = (
        row[_DOCUMENT_JSON_COLUMN]
        if _DOCUMENT_JSON_COLUMN in row.keys()
        else ""
    )
    if document_raw:
        try:
            document_dict = json.loads(document_raw)
            document_dict.update(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "format": row["format"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "metadata": metadata_dict or document_dict.get("metadata", {}),
                }
            )
            return Document.model_validate(document_dict)
        except Exception:
            LOGGER.exception("Failed to deserialize full document id=%s", row["id"])

    try:
        metadata = DocumentMetadata.model_validate(metadata_dict)
        return Document.model_validate(
            {
                "id": row["id"],
                "source": row["source"],
                "format": row["format"],
                "status": row["status"],
                "created_at": row["created_at"],
                "metadata": metadata.model_dump(mode="json"),
                "blocks": [],
            }
        )
    except Exception:
        LOGGER.exception("Failed to deserialize document id=%s", row["id"])
        return None


def save_document(doc: Document) -> None:
    """Save a full document into SQLite using upsert semantics."""
    connection = _connect()
    if connection is None:
        return

    metadata_json = json.dumps(doc.metadata.model_dump(mode="json"), ensure_ascii=False)
    document_json = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False)
    try:
        connection.execute(
            """
            INSERT INTO documents (
                id, source, format, status, created_at, metadata, document_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                format = excluded.format,
                status = excluded.status,
                metadata = excluded.metadata,
                document_json = excluded.document_json
            """,
            (
                doc.id,
                doc.source,
                doc.format.value,
                doc.status.value,
                doc.created_at.isoformat() if doc.created_at else datetime.now(timezone.utc).isoformat(),
                metadata_json,
                document_json,
            ),
        )
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to save document id=%s", doc.id)
    finally:
        connection.close()


def get_document(doc_id: str) -> Optional[Document]:
    """Get a document by ID from SQLite."""
    connection = _connect()
    if connection is None:
        return None

    try:
        row = connection.execute(
            """
            SELECT id, source, format, status, created_at, metadata, document_json
            FROM documents
            WHERE id = ?
            """,
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)
    except sqlite3.Error:
        LOGGER.exception("Failed to fetch document id=%s", doc_id)
        return None
    finally:
        connection.close()


def delete_document(doc_id: str) -> None:
    """Delete a document and all its associated notes from SQLite."""
    connection = _connect()
    if connection is None:
        return

    try:
        # Delete notes before documents — no FOREIGN KEY ... ON DELETE CASCADE is defined,
        # so the ordering must be explicit to preserve referential integrity.
        connection.execute("DELETE FROM notes WHERE document_id = ?", (doc_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to delete document id=%s", doc_id)
    finally:
        connection.close()


def update_status(doc_id: str, status: ProcessingStatus) -> None:
    """Update processing status for a document."""
    connection = _connect()
    if connection is None:
        return

    try:
        connection.execute(
            """
            UPDATE documents
            SET status = ?
            WHERE id = ?
            """,
            (status.value, doc_id),
        )
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to update status for document id=%s", doc_id)
    finally:
        connection.close()


def list_documents(limit: int = 20) -> list[Document]:
    """List saved documents, ordered by most recent first.

    Args:
        limit: Maximum number of documents to return.
    """
    connection = _connect()
    if connection is None:
        return []

    try:
        rows = connection.execute(
            """
            SELECT id, source, format, status, created_at, metadata, document_json
            FROM documents
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        LOGGER.exception("Failed to list documents")
        return []
    finally:
        connection.close()

    documents: list[Document] = []
    for row in rows:
        document = _row_to_document(row)
        if document is not None:
            documents.append(document)
    return documents


# ---------------------------------------------------------------------------
# Notes CRUD
# ---------------------------------------------------------------------------


def save_note(
    document_id: str,
    file_hash: str,
    result: dict,
    vlm_model: str,
    llm_model: str,
    is_image: bool = False,
) -> None:
    """Upsert a note (analysis result) for a document + model combination."""
    connection = _connect()
    if connection is None:
        return

    persisted_result = dict(result)
    persisted_result[_NOTE_RESULT_VERSION_KEY] = _NOTE_RESULT_VERSION
    result_json = json.dumps(persisted_result, ensure_ascii=False)
    now = datetime.now(timezone.utc).isoformat()
    try:
        connection.execute(
            """
            INSERT INTO notes (document_id, file_hash, vlm_model, llm_model, result_json, is_image, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, vlm_model, llm_model) DO UPDATE SET
                file_hash = excluded.file_hash,
                result_json = excluded.result_json,
                is_image = excluded.is_image,
                updated_at = excluded.updated_at
            """,
            (
                document_id,
                file_hash,
                vlm_model,
                llm_model,
                result_json,
                int(is_image),
                now,
            ),
        )
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to save note for document_id=%s", document_id)
    finally:
        connection.close()


def get_note(document_id: str, vlm_model: str, llm_model: str) -> Optional[dict]:
    """Retrieve a specific note by document + model combination.

    Returns a dict with keys: result, file_hash, vlm_model, llm_model, is_image, updated_at.
    """
    connection = _connect()
    if connection is None:
        return None

    try:
        row = connection.execute(
            """
            SELECT result_json, file_hash, vlm_model, llm_model, is_image, updated_at
            FROM notes
            WHERE document_id = ? AND vlm_model = ? AND llm_model = ?
            """,
            (document_id, vlm_model, llm_model),
        ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"])
        if result.get(_NOTE_RESULT_VERSION_KEY) != _NOTE_RESULT_VERSION:
            return None
        result.pop(_NOTE_RESULT_VERSION_KEY, None)
        return {
            "result": result,
            "file_hash": row["file_hash"],
            "vlm_model": row["vlm_model"],
            "llm_model": row["llm_model"],
            "is_image": bool(row["is_image"]),
            "updated_at": row["updated_at"],
        }
    except (sqlite3.Error, json.JSONDecodeError):
        LOGGER.exception("Failed to get note for document_id=%s", document_id)
        return None
    finally:
        connection.close()


def list_notes_for_document(document_id: str) -> list[dict]:
    """List all notes for a given document, newest first."""
    connection = _connect()
    if connection is None:
        return []

    try:
        rows = connection.execute(
            """
            SELECT result_json, file_hash, vlm_model, llm_model, is_image, updated_at
            FROM notes
            WHERE document_id = ?
            ORDER BY updated_at DESC
            """,
            (document_id,),
        ).fetchall()
        results = []
        for row in rows:
            try:
                result = json.loads(row["result_json"])
                if result.get(_NOTE_RESULT_VERSION_KEY) != _NOTE_RESULT_VERSION:
                    continue
                result.pop(_NOTE_RESULT_VERSION_KEY, None)
                results.append(
                    {
                        "result": result,
                        "file_hash": row["file_hash"],
                        "vlm_model": row["vlm_model"],
                        "llm_model": row["llm_model"],
                        "is_image": bool(row["is_image"]),
                        "updated_at": row["updated_at"],
                    }
                )
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Skipping note with invalid JSON for document_id=%s", document_id
                )
        return results
    except sqlite3.Error:
        LOGGER.exception("Failed to list notes for document_id=%s", document_id)
        return []
    finally:
        connection.close()


def delete_note(document_id: str, vlm_model: str, llm_model: str) -> None:
    """Delete a specific note by document + model combination."""
    connection = _connect()
    if connection is None:
        return

    try:
        connection.execute(
            """
            DELETE FROM notes
            WHERE document_id = ? AND vlm_model = ? AND llm_model = ?
            """,
            (document_id, vlm_model, llm_model),
        )
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to delete note for document_id=%s", document_id)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Concepts CRUD
# ---------------------------------------------------------------------------


def save_concepts(document_id: str, concepts: list[dict]) -> list[int]:
    """Upsert concepts for a document and return their IDs.

    Each concept dict must contain: concept_name, canonical_name, aliases (list), definition.
    Uses INSERT OR REPLACE semantics on (document_id, concept_name) UNIQUE constraint.

    Args:
        document_id: Document.id owning these concepts.
        concepts: List of concept dicts with required keys.

    Returns:
        List of inserted/replaced row IDs in insertion order.
    """
    connection = _connect()
    if connection is None:
        return []

    now = datetime.now(timezone.utc).isoformat()
    ids: list[int] = []
    try:
        for concept in concepts:
            aliases_json = json.dumps(concept.get("aliases") or [], ensure_ascii=False)
            cursor = connection.execute(
                """
                INSERT OR REPLACE INTO concepts
                    (document_id, concept_name, canonical_name, aliases, definition, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    concept["concept_name"],
                    concept["canonical_name"],
                    aliases_json,
                    concept.get("definition") or "",
                    now,
                ),
            )
            ids.append(cursor.lastrowid)
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to save concepts for document_id=%s", document_id)
    finally:
        connection.close()

    return ids


def _parse_concept_row(row: sqlite3.Row) -> dict:
    """Convert a concepts table row into a plain dict with parsed aliases."""
    aliases_raw = row["aliases"] or "[]"
    try:
        aliases = json.loads(aliases_raw)
    except json.JSONDecodeError:
        aliases = []
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "concept_name": row["concept_name"],
        "canonical_name": row["canonical_name"],
        "aliases": aliases,
        "definition": row["definition"],
        "created_at": row["created_at"],
    }


def get_concepts_for_document(document_id: str) -> list[dict]:
    """Return all concepts belonging to a document.

    Args:
        document_id: Document.id to query.

    Returns:
        List of concept dicts with id, document_id, concept_name, canonical_name,
        aliases (parsed list), definition, created_at.
    """
    connection = _connect()
    if connection is None:
        return []

    try:
        rows = connection.execute(
            "SELECT id, document_id, concept_name, canonical_name, aliases, definition, created_at "
            "FROM concepts WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        return [_parse_concept_row(row) for row in rows]
    except sqlite3.Error:
        LOGGER.exception("Failed to get concepts for document_id=%s", document_id)
        return []
    finally:
        connection.close()


def get_all_concepts(exclude_document_id: str | None = None) -> list[dict]:
    """Return all concepts, optionally excluding one document's concepts.

    Args:
        exclude_document_id: If provided, skip concepts belonging to this document.

    Returns:
        List of concept dicts with id, document_id, concept_name, canonical_name,
        aliases (parsed list), definition, created_at.
    """
    connection = _connect()
    if connection is None:
        return []

    try:
        if exclude_document_id is not None:
            rows = connection.execute(
                "SELECT id, document_id, concept_name, canonical_name, aliases, definition, created_at "
                "FROM concepts WHERE document_id != ?",
                (exclude_document_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id, document_id, concept_name, canonical_name, aliases, definition, created_at "
                "FROM concepts"
            ).fetchall()
        return [_parse_concept_row(row) for row in rows]
    except sqlite3.Error:
        LOGGER.exception("Failed to get all concepts (exclude=%s)", exclude_document_id)
        return []
    finally:
        connection.close()


def save_concept_links(links: list[dict]) -> None:
    """Upsert concept links with normalized pair ordering.

    Pair ordering is normalized: always stores (min(a,b), max(a,b)) to ensure
    uniqueness regardless of which side is "source" vs "target".

    Each link dict must contain: concept_id_a, concept_id_b, confidence_score,
    relationship_type, relationship_desc.

    Args:
        links: List of link dicts to upsert.
    """
    connection = _connect()
    if connection is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    try:
        for link in links:
            id_a = link["concept_id_a"]
            id_b = link["concept_id_b"]
            # Normalize pair so (min, max) is always stored
            low, high = min(id_a, id_b), max(id_a, id_b)
            connection.execute(
                """
                INSERT OR REPLACE INTO concept_links
                    (concept_id_a, concept_id_b, confidence_score, relationship_type, relationship_desc, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    low,
                    high,
                    link["confidence_score"],
                    link.get("relationship_type") or "",
                    link.get("relationship_desc") or "",
                    now,
                ),
            )
        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to save concept links")
    finally:
        connection.close()


def get_concept_links_for_document(document_id: str) -> list[dict]:
    """Return concept links where either endpoint belongs to document_id.

    JOINs with documents table to resolve target document title. For each link,
    determines which side is the "source" (belongs to document_id) and which is
    the "target" (belongs to another document).

    Args:
        document_id: Document.id whose concept links to retrieve.

    Returns:
        List of link dicts with keys: concept_id_a, concept_id_b, confidence_score,
        relationship_type, relationship_desc, source_concept_name, source_canonical_name,
        target_concept_name, target_canonical_name, target_document_id, target_document_title.
    """
    connection = _connect()
    if connection is None:
        return []

    try:
        rows = connection.execute(
            """
            SELECT
                cl.concept_id_a,
                cl.concept_id_b,
                cl.confidence_score,
                cl.relationship_type,
                cl.relationship_desc,
                ca.concept_name   AS concept_name_a,
                ca.canonical_name AS canonical_name_a,
                ca.document_id    AS document_id_a,
                cb.concept_name   AS concept_name_b,
                cb.canonical_name AS canonical_name_b,
                cb.document_id    AS document_id_b,
                da.source         AS source_a,
                db.source         AS source_b
            FROM concept_links cl
            JOIN concepts ca ON ca.id = cl.concept_id_a
            JOIN concepts cb ON cb.id = cl.concept_id_b
            LEFT JOIN documents da ON da.id = ca.document_id
            LEFT JOIN documents db ON db.id = cb.document_id
            WHERE ca.document_id = ? OR cb.document_id = ?
            """,
            (document_id, document_id),
        ).fetchall()

        results: list[dict] = []
        for row in rows:
            doc_id_a = row["document_id_a"]
            doc_id_b = row["document_id_b"]

            # Determine which side is source (current doc) vs target (other doc)
            if doc_id_a == document_id:
                source_concept_name = row["concept_name_a"]
                source_canonical = row["canonical_name_a"]
                target_concept_name = row["concept_name_b"]
                target_canonical = row["canonical_name_b"]
                target_doc_id = doc_id_b
                target_doc_title = row["source_b"] or doc_id_b
            else:
                source_concept_name = row["concept_name_b"]
                source_canonical = row["canonical_name_b"]
                target_concept_name = row["concept_name_a"]
                target_canonical = row["canonical_name_a"]
                target_doc_id = doc_id_a
                target_doc_title = row["source_a"] or doc_id_a

            results.append(
                {
                    "concept_id_a": row["concept_id_a"],
                    "concept_id_b": row["concept_id_b"],
                    "confidence_score": row["confidence_score"],
                    "relationship_type": row["relationship_type"],
                    "relationship_desc": row["relationship_desc"],
                    "source_concept_name": source_concept_name,
                    "source_canonical_name": source_canonical,
                    "target_concept_name": target_concept_name,
                    "target_canonical_name": target_canonical,
                    "target_document_id": target_doc_id,
                    "target_document_title": target_doc_title,
                }
            )
        return results
    except sqlite3.Error:
        LOGGER.exception("Failed to get concept links for document_id=%s", document_id)
        return []
    finally:
        connection.close()


def delete_concepts_for_document(document_id: str) -> None:
    """Delete all concept links and concepts for a document (cascading).

    Deletes links first (referential integrity), then the concepts themselves.

    Args:
        document_id: Document.id whose concepts and links to remove.
    """
    connection = _connect()
    if connection is None:
        return

    try:
        # Collect concept IDs belonging to this document
        rows = connection.execute(
            "SELECT id FROM concepts WHERE document_id = ?", (document_id,)
        ).fetchall()
        concept_ids = [row["id"] for row in rows]

        if concept_ids:
            placeholders = ",".join("?" * len(concept_ids))
            connection.execute(
                f"DELETE FROM concept_links WHERE concept_id_a IN ({placeholders}) "
                f"OR concept_id_b IN ({placeholders})",
                concept_ids + concept_ids,
            )
            connection.execute(
                f"DELETE FROM concepts WHERE id IN ({placeholders})",
                concept_ids,
            )

        connection.commit()
    except sqlite3.Error:
        LOGGER.exception("Failed to delete concepts for document_id=%s", document_id)
    finally:
        connection.close()
