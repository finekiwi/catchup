"""Unit tests for PDF parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from models.document import BlockType, DocumentFormat, ProcessingStatus, generate_document_id
from parsers import pdf_parser


def _write_fake_pdf(path: Path, content: bytes = b"%PDF-1.4\n%fake\n") -> None:
    """Write a tiny fake PDF-like file for hashing and parser input."""
    path.write_bytes(content)


def test_parse_pdf_extracts_text_block(monkeypatch, tmp_path: Path) -> None:
    """Parses text content from Docling output into a text block."""
    pdf_path = tmp_path / "sample.pdf"
    _write_fake_pdf(pdf_path)

    loader_instance = Mock()
    loader_instance.load.return_value = [
        {
            "page_content": "This is page one text.",
            "metadata": {"type": "text", "page": 1},
        }
    ]
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

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
    loader_class.assert_called_once_with(file_path=str(pdf_path))
    loader_instance.load.assert_called_once()


def test_parse_pdf_handles_multiple_pages_and_block_types(monkeypatch, tmp_path: Path) -> None:
    """Preserves page metadata and block types for multi-page data."""
    pdf_path = tmp_path / "multi_page.pdf"
    _write_fake_pdf(pdf_path)

    loader_instance = Mock()
    loader_instance.load.return_value = [
        {
            "page_content": "Overview text",
            "metadata": {"type": "text", "page": 1},
        },
        {
            "metadata": {"page": 2},
            "elements": [
                {"type": "table", "content": "|a|b|", "metadata": {"page": 2}},
                {"type": "image", "metadata": {"page": 3}, "image_path": "/tmp/fig-1.png"},
            ],
        },
    ]
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

    document = pdf_parser.parse_pdf(str(pdf_path))

    assert [block.type for block in document.blocks] == [BlockType.TEXT, BlockType.TABLE, BlockType.FIGURE]
    assert [block.metadata.page for block in document.blocks] == [1, 2, 3]
    assert [block.order for block in document.blocks] == [0, 1, 2]
    assert document.blocks[2].content == "[figure]"
    assert document.blocks[2].image_path == "/tmp/fig-1.png"
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


def test_parse_pdf_empty_loader_output_returns_empty_blocks(monkeypatch, tmp_path: Path) -> None:
    """Returns a valid fallback document when Docling returns no elements."""
    pdf_path = tmp_path / "empty.pdf"
    _write_fake_pdf(pdf_path)

    loader_instance = Mock()
    loader_instance.load.return_value = []
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

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


def test_parse_pdf_failure_returns_fallback_document(monkeypatch, tmp_path: Path) -> None:
    """Returns fallback document when Docling parsing raises an error."""
    pdf_path = tmp_path / "broken.pdf"
    _write_fake_pdf(pdf_path, content=b"not-a-real-pdf")

    loader_instance = Mock()
    loader_instance.load.side_effect = RuntimeError("parse failed")
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

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


def test_parse_pdf_marks_failure_when_docling_loader_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    """Returns fallback document and parse_failed tag when DoclingLoader import is unavailable."""
    pdf_path = tmp_path / "no_loader.pdf"
    _write_fake_pdf(pdf_path)

    monkeypatch.setattr(pdf_parser, "DoclingLoader", None)

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


def test_parse_pdf_zero_byte_file_returns_parse_failed(monkeypatch, tmp_path: Path) -> None:
    """Treats zero-byte input as parse failure when no blocks are produced."""
    pdf_path = tmp_path / "zero_byte.pdf"
    _write_fake_pdf(pdf_path, content=b"")

    loader_instance = Mock()
    loader_instance.load.return_value = []
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

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


def test_parse_pdf_unknown_block_type_defaults_to_text(monkeypatch, tmp_path: Path) -> None:
    """Maps unknown Docling block types to BlockType.TEXT."""
    pdf_path = tmp_path / "unknown_block_type.pdf"
    _write_fake_pdf(pdf_path)

    loader_instance = Mock()
    loader_instance.load.return_value = [
        {
            "metadata": {"type": "unknown_type_xyz", "page": 1},
            "content": "Unknown typed content",
        }
    ]
    loader_class = Mock(return_value=loader_instance)
    monkeypatch.setattr(pdf_parser, "DoclingLoader", loader_class)

    document = pdf_parser.parse_pdf(str(pdf_path))

    assert len(document.blocks) == 1
    assert document.blocks[0].type == BlockType.TEXT
    assert document.blocks[0].content == "Unknown typed content"
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "docling"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0
