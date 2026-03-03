"""PDF parser that converts Docling output into CatchUp Document objects."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

try:
    from langchain_community.document_loaders import DoclingLoader
except ImportError:  # pragma: no cover - handled by fallback test behavior
    DoclingLoader = None  # type: ignore[assignment]

from models.document import (
    Block,
    BlockMetadata,
    BlockType,
    Document,
    DocumentFormat,
    ProcessingStatus,
    generate_document_id,
)

LOGGER = logging.getLogger(__name__)

_TEXT_TYPES = {"text", "paragraph", "markdown", "title", "heading"}
_TABLE_TYPES = {"table"}
_FIGURE_TYPES = {"figure", "image", "picture", "diagram", "chart"}
_CODE_TYPES = {"code"}
_EQUATION_TYPES = {"equation", "formula", "math"}


def parse_pdf(file_path: str) -> Document:
    """
    Parse a PDF with DoclingLoader and return the intermediate Document schema.

    On parser failure, this function returns a fallback Document with empty blocks
    while keeping core metadata fields populated.
    """
    document = Document(
        id=_safe_document_id(file_path=file_path),
        source=Path(file_path).name,
        format=DocumentFormat.PDF,
        status=ProcessingStatus.PARSED,
    )

    if DoclingLoader is None:
        LOGGER.error("DoclingLoader is unavailable; returning fallback document for %s", file_path)
        return document

    try:
        loader = _create_docling_loader(file_path=file_path)
        loaded_docs = loader.load()
        document.blocks = _to_blocks(loaded_docs=loaded_docs)
        return document
    except Exception:
        LOGGER.exception("Failed to parse PDF %s; returning fallback document", file_path)
        return document


def _create_docling_loader(file_path: str) -> Any:
    """Create DoclingLoader while tolerating minor constructor signature differences."""
    try:
        return DoclingLoader(file_path=file_path)
    except TypeError:
        return DoclingLoader(file_path)  # type: ignore[misc]


def _to_blocks(loaded_docs: Any) -> list[Block]:
    """Convert Docling loader output into ordered Block objects."""
    blocks: list[Block] = []

    if not isinstance(loaded_docs, list):
        loaded_docs = [loaded_docs]

    for doc in loaded_docs:
        default_page = _extract_page(doc)
        nested_elements = _extract_nested_elements(doc)

        if nested_elements:
            for element in nested_elements:
                block = _to_single_block(element=element, order=len(blocks), fallback_page=default_page)
                if block is not None:
                    blocks.append(block)
            continue

        block = _to_single_block(element=doc, order=len(blocks), fallback_page=default_page)
        if block is not None:
            blocks.append(block)

    return blocks


def _to_single_block(element: Any, order: int, fallback_page: int | None) -> Block | None:
    """Convert a single Docling element to a Block. Returns None if no usable content exists."""
    block_type = _extract_block_type(element)
    content = _extract_content(element)

    if not content:
        if block_type == BlockType.TABLE:
            content = "[table]"
        elif block_type == BlockType.FIGURE:
            content = "[figure]"
        else:
            return None

    page = _extract_page(element)
    metadata = BlockMetadata(page=page if page is not None else fallback_page)

    caption = _extract_caption(element)
    if caption is not None:
        metadata.caption = caption

    image_path_value = _get_first_present(element, "image_path", "image_uri", "image_file")
    image_path = str(image_path_value) if image_path_value is not None else None

    return Block(
        type=block_type,
        content=content,
        order=order,
        metadata=metadata,
        image_path=image_path,
    )


def _extract_nested_elements(element: Any) -> list[Any]:
    """Extract nested element collections from Docling output if present."""
    nested = _get_first_present(element, "elements", "blocks", "items")
    if isinstance(nested, list):
        return nested
    return []


def _extract_block_type(element: Any) -> BlockType:
    """Infer BlockType from type/category metadata in Docling output."""
    raw_type = _get_first_present(
        element,
        "type",
        "block_type",
        "category",
        "element_type",
    )
    if raw_type is None:
        raw_type = _extract_metadata(element).get("type")

    normalized = str(raw_type).strip().lower() if raw_type is not None else ""

    if normalized in _TABLE_TYPES:
        return BlockType.TABLE
    if normalized in _FIGURE_TYPES:
        return BlockType.FIGURE
    if normalized in _CODE_TYPES:
        return BlockType.CODE
    if normalized in _EQUATION_TYPES:
        return BlockType.EQUATION
    if normalized in _TEXT_TYPES:
        return BlockType.TEXT

    return BlockType.TEXT


def _extract_content(element: Any) -> str:
    """Extract text content from a Docling element."""
    raw_content = _get_first_present(
        element,
        "content",
        "text",
        "page_content",
        "markdown",
        "html",
    )
    if raw_content is None:
        metadata = _extract_metadata(element)
        raw_content = metadata.get("text")

    if raw_content is None:
        return ""

    content = str(raw_content).strip()
    return content


def _extract_caption(element: Any) -> str | None:
    """Extract optional caption metadata from a Docling element."""
    raw_caption = _get_first_present(element, "caption", "title", "label")
    if raw_caption is None:
        raw_caption = _extract_metadata(element).get("caption")
    if raw_caption is None:
        return None
    return str(raw_caption).strip() or None


def _extract_page(element: Any) -> int | None:
    """Extract page number from a Docling element, if available."""
    raw_page = _get_first_present(element, "page", "page_number", "page_no")
    if raw_page is None:
        metadata = _extract_metadata(element)
        raw_page = metadata.get("page")
        if raw_page is None:
            raw_page = metadata.get("page_number")

    if raw_page is None:
        return None

    try:
        page = int(raw_page)
    except (TypeError, ValueError):
        return None

    return page if page > 0 else None


def _extract_metadata(element: Any) -> dict[str, Any]:
    """Return metadata dict from Docling output object or dict."""
    if isinstance(element, dict):
        metadata = element.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    metadata = getattr(element, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _get_first_present(element: Any, *keys: str) -> Any:
    """Return first non-None value among candidate keys from dict/object."""
    for key in keys:
        if isinstance(element, dict) and key in element and element[key] is not None:
            return element[key]

        value = getattr(element, key, None)
        if value is not None:
            return value

    return None


def _safe_document_id(file_path: str) -> str:
    """Generate robust document id with path-hash fallback when file read fails."""
    try:
        return generate_document_id(file_path=file_path)
    except Exception:
        LOGGER.exception("Failed to hash source file %s; using path hash fallback", file_path)
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]
