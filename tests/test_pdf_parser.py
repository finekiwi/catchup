"""Unit tests for PDF parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from models.document import (
    BlockType,
    DocumentFormat,
    ProcessingStatus,
    generate_document_id,
)
from parsers import pdf_parser


def _write_fake_pdf(path: Path, content: bytes = b"%PDF-1.4\n%fake\n") -> None:
    """Write a tiny fake PDF-like file for hashing and parser input."""
    path.write_bytes(content)


def _make_text_item(
    text: str, label_value: str = "text", page_no: int = 1
) -> MagicMock:
    """Build a mock TextItem."""
    item = MagicMock(spec=["label", "prov", "captions", "text"])
    item.label = MagicMock()
    item.label.value = label_value
    item.text = text
    prov = MagicMock()
    prov.page_no = page_no
    item.prov = [prov]
    item.captions = []
    return item


def _make_table_item(markdown: str = "|a|b|", page_no: int = 1) -> MagicMock:
    """Build a mock TableItem."""
    from docling_core.types.doc import TableItem

    item = MagicMock(spec=TableItem)
    item.label = MagicMock()
    item.label.value = "table"
    item.export_to_markdown = Mock(return_value=markdown)
    prov = MagicMock()
    prov.page_no = page_no
    item.prov = [prov]
    item.captions = []
    return item


def _make_picture_item(page_no: int = 1) -> MagicMock:
    """Build a mock PictureItem."""
    from docling_core.types.doc import PictureItem

    item = MagicMock(spec=PictureItem)
    item.label = MagicMock()
    item.label.value = "picture"
    prov = MagicMock()
    prov.page_no = page_no
    item.prov = [prov]
    item.captions = []
    return item


def _make_conversion_result(items: list, status_value: str = "success") -> MagicMock:
    """Build a mock ConversionResult."""
    doc = MagicMock()
    doc.iterate_items = Mock(return_value=[(item, 0) for item in items])
    result = MagicMock()
    result.document = doc
    result.status = MagicMock()
    result.status.value = status_value
    from docling.datamodel.base_models import ConversionStatus

    result.status = ConversionStatus.SUCCESS
    return result


def test_parse_pdf_extracts_text_block(tmp_path: Path) -> None:
    """Parses text content from Docling output into a text block."""
    pdf_path = tmp_path / "sample.pdf"
    _write_fake_pdf(pdf_path)

    item = _make_text_item("This is page one text.", page_no=1)
    mock_result = _make_conversion_result([item])

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.return_value = mock_result
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert document.id == generate_document_id(str(pdf_path))
    assert document.source == "sample.pdf"
    assert document.format == DocumentFormat.PDF
    assert document.status == ProcessingStatus.PARSED
    assert len(document.blocks) == 1
    assert document.blocks[0].type == BlockType.TEXT
    assert document.blocks[0].content == "This is page one text."
    assert document.blocks[0].metadata.page == 1
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_handles_multiple_pages_and_block_types(tmp_path: Path) -> None:
    """Preserves page metadata and block types for multi-page data."""
    pdf_path = tmp_path / "multi_page.pdf"
    _write_fake_pdf(pdf_path)

    text_item = _make_text_item("Overview text", page_no=1)
    table_item = _make_table_item("|a|b|", page_no=2)
    figure_item = _make_picture_item(page_no=3)
    mock_result = _make_conversion_result([text_item, table_item, figure_item])

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.return_value = mock_result
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert [block.type for block in document.blocks] == [
        BlockType.TEXT,
        BlockType.TABLE,
        BlockType.FIGURE,
    ]
    assert [block.metadata.page for block in document.blocks] == [1, 2, 3]
    assert [block.order for block in document.blocks] == [0, 1, 2]
    assert document.blocks[2].content == "[figure]"
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_empty_output_returns_parse_failed(tmp_path: Path) -> None:
    """Returns a valid fallback document when Docling returns no elements."""
    pdf_path = tmp_path / "empty.pdf"
    _write_fake_pdf(pdf_path)

    mock_result = _make_conversion_result([])

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.return_value = mock_result
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert document.id == generate_document_id(str(pdf_path))
    assert document.source == "empty.pdf"
    assert document.format == DocumentFormat.PDF
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []
    assert "parse_failed" in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_failure_returns_fallback_document(tmp_path: Path) -> None:
    """Returns fallback document when Docling parsing raises an error."""
    pdf_path = tmp_path / "broken.pdf"
    _write_fake_pdf(pdf_path, content=b"not-a-real-pdf")

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.side_effect = RuntimeError("parse failed")
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert document.id == generate_document_id(str(pdf_path))
    assert document.source == "broken.pdf"
    assert document.format == DocumentFormat.PDF
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []
    assert "parse_failed" in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_marks_failure_when_docling_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    """Returns fallback document and parse_failed tag when DocumentConverter is unavailable."""
    pdf_path = tmp_path / "no_loader.pdf"
    _write_fake_pdf(pdf_path)

    monkeypatch.setattr(pdf_parser, "DocumentConverter", None)

    document = pdf_parser.parse_pdf(str(pdf_path))

    assert document.id == generate_document_id(str(pdf_path))
    assert document.source == "no_loader.pdf"
    assert document.format == DocumentFormat.PDF
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []
    assert "parse_failed" in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_zero_byte_file_returns_parse_failed(tmp_path: Path) -> None:
    """Treats zero-byte input as parse failure when no blocks are produced."""
    pdf_path = tmp_path / "zero_byte.pdf"
    _write_fake_pdf(pdf_path, content=b"")

    mock_result = _make_conversion_result([])

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.return_value = mock_result
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert document.id == generate_document_id(str(pdf_path))
    assert document.source == "zero_byte.pdf"
    assert document.format == DocumentFormat.PDF
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []
    assert "parse_failed" in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_unknown_label_defaults_to_text(tmp_path: Path) -> None:
    """Maps unknown Docling labels to BlockType.TEXT."""
    pdf_path = tmp_path / "unknown_block_type.pdf"
    _write_fake_pdf(pdf_path)

    item = _make_text_item(
        "Unknown typed content", label_value="unknown_type_xyz", page_no=1
    )
    mock_result = _make_conversion_result([item])

    with patch("parsers.pdf_parser.DocumentConverter") as mock_cls:
        mock_cls.return_value.convert.return_value = mock_result
        document = pdf_parser.parse_pdf(str(pdf_path))

    assert len(document.blocks) == 1
    assert document.blocks[0].type == BlockType.TEXT
    assert document.blocks[0].content == "Unknown typed content"
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0
