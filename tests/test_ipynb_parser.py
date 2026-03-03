"""Tests for ipynb parser."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.document import BlockType, DocumentFormat, ProcessingStatus
from parsers.ipynb_parser import parse_ipynb


def _touch_notebook_file(tmp_path: Path, name: str = "sample.ipynb") -> Path:
    """Create a notebook file so document id generation can read bytes."""
    file_path = tmp_path / name
    file_path.write_text("{}", encoding="utf-8")
    return file_path


def _notebook(cells: list[dict[str, object]]) -> SimpleNamespace:
    """Build a notebook-like object with a cells attribute."""
    return SimpleNamespace(cells=cells)


def _markdown_cell(source: str) -> dict[str, object]:
    """Create a markdown cell payload."""
    return {"cell_type": "markdown", "source": source}


def _code_cell(source: str, outputs: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Create a code cell payload."""
    return {"cell_type": "code", "source": source, "outputs": outputs or []}


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_basic_cells(mock_read, tmp_path: Path) -> None:
    """Parses markdown and code cells into separate blocks."""
    notebook = _notebook(cells=[_markdown_cell("## Intro"), _code_cell("print('hello')")])
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    mock_read.assert_called_once_with(str(file_path), as_version=4)
    assert document.format == DocumentFormat.IPYNB
    assert document.status == ProcessingStatus.PARSED
    assert document.metadata.total_cells == 2
    assert [block.type for block in document.blocks] == [BlockType.TEXT, BlockType.CODE]
    assert [block.order for block in document.blocks] == [0, 1]
    assert document.blocks[0].metadata.cell_index == 0
    assert document.blocks[0].metadata.cell_type == "markdown"
    assert document.blocks[1].metadata.cell_index == 1
    assert document.blocks[1].metadata.cell_type == "code"
    assert document.blocks[1].metadata.language == "python"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_code_output_text_blocks(mock_read, tmp_path: Path) -> None:
    """Converts text outputs to output text blocks."""
    notebook = _notebook(
        cells=[
            _code_cell(
                "x = 42\nx",
                outputs=[
                    {"output_type": "stream", "name": "stdout", "text": "hello\n"},
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": "42"},
                        "execution_count": 1,
                        "metadata": {},
                    },
                ],
            )
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.TEXT, BlockType.TEXT]
    assert [block.order for block in document.blocks] == [0, 1, 2]
    assert all(block.metadata.cell_index == 0 for block in document.blocks)
    assert document.blocks[1].metadata.cell_type == "output"
    assert document.blocks[2].metadata.cell_type == "output"
    assert document.blocks[1].content == "hello\n"
    assert document.blocks[2].content == "42"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_image_output_creates_figure_block(mock_read, tmp_path: Path) -> None:
    """Converts image outputs to figure blocks."""
    notebook = _notebook(
        cells=[
            _code_cell(
                "plot()",
                outputs=[
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "iVBORw0KGgoAAAANSUhEUgAAAAUA", "text/plain": "<Figure>"},
                        "metadata": {},
                    }
                ],
            )
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.FIGURE]
    assert document.blocks[1].metadata.cell_type == "output"
    assert "image/png" in document.blocks[1].content


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_empty_notebook(mock_read, tmp_path: Path) -> None:
    """Returns a valid Document for empty notebooks."""
    notebook = _notebook(cells=[])
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert document.format == DocumentFormat.IPYNB
    assert document.blocks == []
    assert document.metadata.total_cells == 0


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_broken_file_returns_fallback_document(mock_read, tmp_path: Path) -> None:
    """Returns empty-block fallback document when parsing fails."""
    mock_read.side_effect = ValueError("broken notebook")
    file_path = _touch_notebook_file(tmp_path, "broken.ipynb")

    document = parse_ipynb(str(file_path))

    assert document.format == DocumentFormat.IPYNB
    assert document.status == ProcessingStatus.PARSED
    assert document.blocks == []
    assert document.metadata.total_cells == 0
    assert document.source == "broken.ipynb"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_metadata_cell_index_and_type(mock_read, tmp_path: Path) -> None:
    """Tracks cell index/type metadata across cell and output blocks."""
    notebook = _notebook(
        cells=[
            _markdown_cell("A"),
            _code_cell(
                "print('x')",
                outputs=[{"output_type": "stream", "name": "stdout", "text": "x\n"}],
            ),
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    metadata_pairs = [(block.metadata.cell_index, block.metadata.cell_type) for block in document.blocks]
    assert metadata_pairs == [(0, "markdown"), (1, "code"), (1, "output")]
