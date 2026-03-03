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
