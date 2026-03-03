"""ChromaDB vector storage and search utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv

from models.document import Block

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "catchup_blocks"
EMBEDDING_DIMENSION = 64


def _chroma_path() -> Path:
    """Return Chroma persist path, optionally overridden by environment variable."""
    return Path(os.getenv("CATCHUP_CHROMA_PATH", DEFAULT_CHROMA_PATH))


def _build_client() -> Any:
    """Create a Chroma client compatible with multiple versions."""
    path = _chroma_path()
    path.mkdir(parents=True, exist_ok=True)

    try:
        return chromadb.PersistentClient(path=str(path))
    except AttributeError:
        from chromadb.config import Settings

        return chromadb.Client(Settings(is_persistent=True, persist_directory=str(path)))


def _get_collection() -> Optional[Any]:
    """Get or create CatchUp block collection."""
    try:
        client = _build_client()
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        LOGGER.exception("Failed to initialize Chroma collection at path=%s", _chroma_path())
        return None


def _embed_text(text: str) -> list[float]:
    """
    Build a deterministic embedding vector from text.

    This avoids external embedding API dependencies while keeping search behavior deterministic.
    """
    vector = [0.0] * EMBEDDING_DIMENSION
    if not text:
        return vector

    for index, byte in enumerate(text.encode("utf-8")):
        vector[index % EMBEDDING_DIMENSION] += float(byte)

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def store_embeddings(doc_id: str, blocks: list[Block]) -> None:
    """Store per-block embeddings for one document."""
    if not blocks:
        return

    collection = _get_collection()
    if collection is None:
        return

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for block in blocks:
        ids.append(f"{doc_id}:{block.order}")
        documents.append(block.content)
        block_metadata = block.metadata.model_dump(mode="json", exclude_none=True)
        block_metadata.update(
            {
                "doc_id": doc_id,
                "order": block.order,
                "type": block.type.value,
            }
        )
        metadatas.append(block_metadata)
        embeddings.append(_embed_text(block.content))

    try:
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        else:
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    except Exception:
        LOGGER.exception("Failed to store Chroma embeddings for document id=%s", doc_id)


def search(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Search similar blocks by query text."""
    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[_embed_text(query)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        LOGGER.exception("Failed to search Chroma for query")
        return []

    ids_nested = results.get("ids") or [[]]
    documents_nested = results.get("documents") or [[]]
    metadatas_nested = results.get("metadatas") or [[]]
    distances_nested = results.get("distances") or [[]]

    ids = ids_nested[0] if ids_nested else []
    documents = documents_nested[0] if documents_nested else []
    metadatas = metadatas_nested[0] if metadatas_nested else []
    distances = distances_nested[0] if distances_nested else []

    formatted_results: list[dict[str, Any]] = []
    for index, result_id in enumerate(ids):
        formatted_results.append(
            {
                "id": result_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return formatted_results

