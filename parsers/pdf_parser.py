"""PDF parser that converts Docling output into CatchUp Document objects."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

try:
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem, TextItem
except ImportError:  # pragma: no cover
    DocumentConverter = None  # type: ignore[assignment,misc]
    ConversionStatus = None  # type: ignore[assignment]
    DocItemLabel = None  # type: ignore[assignment]
    TextItem = None  # type: ignore[assignment]
    TableItem = None  # type: ignore[assignment]
    PictureItem = None  # type: ignore[assignment]
    InputFormat = None  # type: ignore[assignment]
    PdfPipelineOptions = None  # type: ignore[assignment]
    PdfFormatOption = None  # type: ignore[assignment]

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

_LABEL_TO_BLOCK_TYPE: dict[str, BlockType] = {
    DocItemLabel.TEXT.value if DocItemLabel else "text": BlockType.TEXT,
    DocItemLabel.TITLE.value if DocItemLabel else "title": BlockType.TEXT,
    DocItemLabel.SECTION_HEADER.value
    if DocItemLabel
    else "section_header": BlockType.TEXT,
    DocItemLabel.LIST_ITEM.value if DocItemLabel else "list_item": BlockType.TEXT,
    DocItemLabel.CAPTION.value if DocItemLabel else "caption": BlockType.TEXT,
    DocItemLabel.FOOTNOTE.value if DocItemLabel else "footnote": BlockType.TEXT,
    DocItemLabel.PAGE_HEADER.value if DocItemLabel else "page_header": BlockType.TEXT,
    DocItemLabel.PAGE_FOOTER.value if DocItemLabel else "page_footer": BlockType.TEXT,
    DocItemLabel.PARAGRAPH.value if DocItemLabel else "paragraph": BlockType.TEXT,
    DocItemLabel.REFERENCE.value if DocItemLabel else "reference": BlockType.TEXT,
    DocItemLabel.TABLE.value if DocItemLabel else "table": BlockType.TABLE,
    DocItemLabel.PICTURE.value if DocItemLabel else "picture": BlockType.FIGURE,
    DocItemLabel.CHART.value if DocItemLabel else "chart": BlockType.FIGURE,
    DocItemLabel.FORMULA.value if DocItemLabel else "formula": BlockType.EQUATION,
    DocItemLabel.CODE.value if DocItemLabel else "code": BlockType.CODE,
}


def parse_pdf(file_path: str) -> Document:
    """
    Parse a PDF with Docling DocumentConverter and return the intermediate Document schema.

    Results are cached to data/parsed/ by file content hash. On subsequent calls with
    the same file, the cached Document is returned immediately without re-running Docling.
    On parser failure, returns a fallback Document with empty blocks while keeping
    core metadata fields populated.
    """
    from utils.cache import load_cached_parse, save_cached_parse, save_docling_doc

    source_path = Path(file_path)
    cached = load_cached_parse(source_path)
    if cached is not None:
        return cached

    start_time = time.perf_counter()
    document = Document(
        id=_safe_document_id(file_path=file_path),
        source=source_path.name,
        format=DocumentFormat.PDF,
        status=ProcessingStatus.PARSED,
    )
    document.processing.parser_model = "docling"

    try:
        if DocumentConverter is None:
            LOGGER.error(
                "docling is unavailable; returning fallback document for %s", file_path
            )
            _mark_parse_failed(document=document)
            return document

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(file_path, raises_on_error=False)

        if ConversionStatus is not None and result.status == ConversionStatus.FAILURE:
            LOGGER.error("Docling conversion failed for %s", file_path)
            _mark_parse_failed(document=document)
            return document

        document.blocks = _to_blocks(result.document)
        if not document.blocks:
            _mark_parse_failed(document=document)
        else:
            save_cached_parse(source_path, document)
            save_docling_doc(source_path, result.document)
        return document

    except Exception:
        LOGGER.exception(
            "Failed to parse PDF %s; returning fallback document", file_path
        )
        _mark_parse_failed(document=document)
        return document
    finally:
        document.processing.latency_ms = (time.perf_counter() - start_time) * 1000


def _to_blocks(doc: object) -> list[Block]:
    """Convert a DoclingDocument into ordered Block objects."""
    blocks: list[Block] = []

    for item, _level in doc.iterate_items():
        label_str = (
            item.label.value if hasattr(item.label, "value") else str(item.label)
        )
        block_type = _LABEL_TO_BLOCK_TYPE.get(label_str, BlockType.TEXT)
        page = _extract_page(item)

        if isinstance(item, TableItem):
            content = (
                item.export_to_markdown()
                if hasattr(item, "export_to_markdown")
                else "[table]"
            )
            if not content:
                content = "[table]"
        elif isinstance(item, PictureItem):
            content = "[figure]"
        elif isinstance(item, TextItem):
            content = item.text or ""
        else:
            content = getattr(item, "text", None) or ""

        if not content:
            continue

        caption = _extract_caption(item)
        metadata = BlockMetadata(page=page, caption=caption)
        blocks.append(
            Block(
                type=block_type, content=content, order=len(blocks), metadata=metadata
            )
        )

    return blocks


def _extract_page(item: object) -> int | None:
    """Extract page number from item provenance."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(first, "page_no", None)
    if page_no is None:
        return None
    try:
        return int(page_no)
    except (TypeError, ValueError):
        return None


def _extract_caption(item: object) -> str | None:
    """Extract caption text from picture or table items."""
    captions = getattr(item, "captions", None)
    if not captions:
        return None
    first = captions[0] if isinstance(captions, list) else captions
    text = getattr(first, "text", None)
    return str(text).strip() or None if text else None


def _safe_document_id(file_path: str) -> str:
    """Generate robust document id with path-hash fallback when file read fails."""
    try:
        return generate_document_id(file_path=file_path)
    except Exception:
        LOGGER.exception(
            "Failed to hash source file %s; using path hash fallback", file_path
        )
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]


def _mark_parse_failed(document: Document) -> None:
    """Mark parsing failure in document tags without duplicating entries."""
    if "parse_failed" not in document.metadata.tags:
        document.metadata.tags.append("parse_failed")
