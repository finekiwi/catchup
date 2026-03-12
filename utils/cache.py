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
from typing import Optional

from models.document import Document

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/parsed")


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
            "_source": str(file_path),
            "document": document.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
            LOGGER.debug("Docling cache hash mismatch for %s — invalidating", file_path.name)
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
            "_source": str(file_path),
            "docling_document": dl_doc.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Docling cache saved: %s → %s", file_path.name, path.name)

    except Exception as exc:
        LOGGER.warning("Docling cache save failed for %s: %s", file_path, exc)


__all__ = ["load_cached_parse", "save_cached_parse", "load_docling_doc", "save_docling_doc"]
