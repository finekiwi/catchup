"""SQLite persistence layer for document metadata."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from models.document import Document, DocumentMetadata, ProcessingStatus

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_SQLITE_PATH = "data/catchup.db"


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
    connection.commit()


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
    """Save document metadata into SQLite using upsert semantics."""
    connection = _connect()
    if connection is None:
        return

    metadata_json = json.dumps(doc.metadata.model_dump(mode="json"), ensure_ascii=False)
    try:
        connection.execute(
            """
            INSERT INTO documents (id, source, format, status, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                format = excluded.format,
                status = excluded.status,
                created_at = excluded.created_at,
                metadata = excluded.metadata
            """,
            (
                doc.id,
                doc.source,
                doc.format.value,
                doc.status.value,
                doc.created_at.isoformat(),
                metadata_json,
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
            SELECT id, source, format, status, created_at, metadata
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


def get_document_by_hash(doc_id: str) -> Optional[Document]:
    """Get a document by hash ID for cache checks."""
    return get_document(doc_id)


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


def list_documents() -> list[Document]:
    """List all saved documents."""
    connection = _connect()
    if connection is None:
        return []

    try:
        rows = connection.execute(
            """
            SELECT id, source, format, status, created_at, metadata
            FROM documents
            ORDER BY created_at DESC
            """
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
