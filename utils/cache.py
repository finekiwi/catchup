"""Document parse cache: serialize/deserialize Document to disk by content hash.

Usage:
    from utils.cache import load_cached_parse, save_cached_parse

    cached = load_cached_parse(pdf_path)
    if cached is not None:
        return cached

    doc = parse_pdf(pdf_path)          # expensive
    save_cached_parse(pdf_path, doc)
    return doc

Cache location: data/parsed/{sha256_prefix}.json  (configurable via PARSE_CACHE_DIR env var)
Invalidation:   file content hash — if the PDF changes, the old cache entry is ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from models.document import Document

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/parsed")
_PARSE_CACHE_VERSION = "v2"
_DOCLING_CACHE_VERSION = "v2"


def _cache_dir() -> Path:
    """Return the cache directory, respecting PARSE_CACHE_DIR env override."""
    return Path(os.environ.get("PARSE_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))


def _file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(file_path: Path) -> Path:
    """Return the cache JSON path for a given source file."""
    digest = _file_sha256(file_path)
    return _cache_dir() / f"{digest[:16]}.json"


def load_cached_parse(file_path: Path) -> Optional[Document]:
    """Load a cached Document for file_path if a valid cache entry exists.

    Validity is determined by SHA-256 of file contents embedded in the cache.
    Returns None on cache miss or any read/decode error.

    Args:
        file_path: Path to the source document (PDF, ipynb, etc.).

    Returns:
        Deserialized Document, or None if no valid cache entry exists.
    """
    try:
        path = _cache_path(file_path)
        if not path.exists():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))

        # Verify stored hash matches current file contents
        stored_hash = raw.get("_cache_hash")
        current_hash = _file_sha256(file_path)
        if stored_hash != current_hash:
            LOGGER.debug("Cache hash mismatch for %s — invalidating", file_path.name)
            return None
        if raw.get("_cache_version") != _PARSE_CACHE_VERSION:
            LOGGER.debug("Cache version mismatch for %s — invalidating", file_path.name)
            return None

        doc = Document.model_validate(raw["document"])
        LOGGER.info("Cache hit: %s (%s)", file_path.name, path.name)
        return doc

    except Exception as exc:
        LOGGER.debug("Cache load failed for %s: %s", file_path, exc)
        return None


def save_cached_parse(file_path: Path, document: Document) -> None:
    """Persist a parsed Document to the cache directory.

    Stores the file's SHA-256 hash alongside the serialized Document so future
    calls to load_cached_parse can detect stale entries.

    Args:
        file_path: Path to the original source file.
        document: Parsed Document to cache.
    """
    try:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        path = _cache_path(file_path)
        payload = {
            "_cache_hash": _file_sha256(file_path),
            "_cache_version": _PARSE_CACHE_VERSION,
            "_source": str(file_path),
            "document": document.model_dump(mode="json"),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Cache saved: %s → %s", file_path.name, path.name)

    except Exception as exc:
        LOGGER.warning("Cache save failed for %s: %s", file_path, exc)


def _docling_cache_path(file_path: Path) -> Path:
    """Return the cache JSON path for the raw DoclingDocument of a given source file."""
    digest = _file_sha256(file_path)
    return _cache_dir() / f"{digest[:16]}_docling.json"


def load_docling_doc(file_path: Path) -> Optional[Any]:
    """Load a cached DoclingDocument for file_path if a valid cache entry exists.

    Returns None on cache miss, hash mismatch, or any read/decode error.

    Args:
        file_path: Path to the source PDF.

    Returns:
        DoclingDocument instance, or None if no valid cache entry exists.
    """
    try:
        from docling_core.types.doc import DoclingDocument

        path = _docling_cache_path(file_path)
        if not path.exists():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))

        stored_hash = raw.get("_cache_hash")
        current_hash = _file_sha256(file_path)
        if stored_hash != current_hash:
            LOGGER.debug(
                "Docling cache hash mismatch for %s — invalidating", file_path.name
            )
            return None

        # Invalidate caches that were saved without generate_picture_images=True.
        # Version "v2" indicates the cache was built with picture image generation enabled.
        if raw.get("_cache_version") != _DOCLING_CACHE_VERSION:
            LOGGER.debug(
                "Docling cache version mismatch for %s — invalidating (no picture images)",
                file_path.name,
            )
            return None

        doc = DoclingDocument.model_validate(raw["docling_document"])
        LOGGER.info("Docling cache hit: %s (%s)", file_path.name, path.name)
        return doc

    except Exception as exc:
        LOGGER.debug("Docling cache load failed for %s: %s", file_path, exc)
        return None


def save_docling_doc(file_path: Path, dl_doc: Any) -> None:
    """Persist a raw DoclingDocument to the cache directory.

    Args:
        file_path: Path to the original source PDF.
        dl_doc: DoclingDocument instance to cache.
    """
    try:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        path = _docling_cache_path(file_path)
        payload = {
            "_cache_hash": _file_sha256(file_path),
            "_cache_version": _DOCLING_CACHE_VERSION,
            "_source": str(file_path),
            "docling_document": dl_doc.model_dump(mode="json"),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Docling cache saved: %s → %s", file_path.name, path.name)

    except Exception as exc:
        LOGGER.warning("Docling cache save failed for %s: %s", file_path, exc)


def _chunks_cache_path(file_path: Path) -> Path:
    """Return the cache JSON path for rechunked flat blocks of a given source file."""
    digest = _file_sha256(file_path)
    return _cache_dir() / f"{digest[:16]}_chunks.json"


def load_cached_chunks(file_path: Path) -> Optional[list[tuple[str, dict]]]:
    """Load cached rechunk output for file_path if a valid entry exists.

    Used for ipynb flat-block rechunking where no DoclingDocument is available.
    Returns None on cache miss, hash mismatch, or decode error.

    Args:
        file_path: Path to the source .ipynb file.

    Returns:
        List of (chunk_text, metadata_dict) tuples, or None if no valid cache entry.
    """
    try:
        path = _chunks_cache_path(file_path)
        if not path.exists():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))

        stored_hash = raw.get("_cache_hash")
        current_hash = _file_sha256(file_path)
        if stored_hash != current_hash:
            LOGGER.debug(
                "Chunks cache hash mismatch for %s — invalidating", file_path.name
            )
            return None

        chunks = [(item[0], item[1]) for item in raw["chunks"]]
        LOGGER.info("Chunks cache hit: %s (%s)", file_path.name, path.name)
        return chunks

    except Exception as exc:
        LOGGER.debug("Chunks cache load failed for %s: %s", file_path, exc)
        return None


def save_cached_chunks(file_path: Path, chunks: list[tuple[str, dict]]) -> None:
    """Persist rechunked flat-block output to the cache directory.

    Args:
        file_path: Path to the original source .ipynb file.
        chunks: List of (chunk_text, metadata_dict) tuples from rechunk_blocks().
    """
    try:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        path = _chunks_cache_path(file_path)
        payload = {
            "_cache_hash": _file_sha256(file_path),
            "_source": str(file_path),
            "chunks": [[text, meta] for text, meta in chunks],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("Chunks cache saved: %s → %s", file_path.name, path.name)

    except Exception as exc:
        LOGGER.warning("Chunks cache save failed for %s: %s", file_path, exc)


def load_docling_doc_by_id(doc_id: str) -> Optional[Any]:
    """Load cached DoclingDocument by document ID (SHA-256 prefix).

    Cache files are named ``{doc_id}_docling.json`` where *doc_id* is the
    SHA-256 prefix computed at parse time.  Since the ID **is** the hash
    prefix, no hash re-verification is needed — we just look up the file
    directly.

    Args:
        doc_id: Document ID (SHA-256 hex prefix, typically 16 chars).

    Returns:
        DoclingDocument instance, or None if no cache entry or decode error.
    """
    try:
        from docling_core.types.doc import DoclingDocument

        path = _cache_dir() / f"{doc_id}_docling.json"
        if not path.exists():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("_cache_version") != _DOCLING_CACHE_VERSION:
            LOGGER.debug("Docling cache version mismatch for %s — invalidating", doc_id)
            return None
        doc = DoclingDocument.model_validate(raw["docling_document"])
        LOGGER.info("Docling cache hit by id: %s (%s)", doc_id, path.name)
        return doc

    except Exception as exc:
        LOGGER.debug("Docling cache load by id failed for %s: %s", doc_id, exc)
        return None


__all__ = [
    "load_cached_parse",
    "save_cached_parse",
    "load_docling_doc",
    "load_docling_doc_by_id",
    "save_docling_doc",
    "load_cached_chunks",
    "save_cached_chunks",
]
