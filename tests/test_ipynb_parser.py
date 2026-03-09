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


def _notebook(cells: list[dict[str, object]], metadata: dict[str, object] | None = None) -> SimpleNamespace:
    """Build a notebook-like object with a cells attribute."""
    return SimpleNamespace(cells=cells, metadata=metadata or {})


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
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "nbformat"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


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
    assert "parse_failed" not in document.metadata.tags
    assert document.processing.parser_model == "nbformat"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0


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
    assert "parse_failed" in document.metadata.tags
    assert document.processing.parser_model == "nbformat"
    assert document.processing.latency_ms is not None
    assert document.processing.latency_ms >= 0
    assert len(document.id) == 16


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


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_error_output_traceback_to_text_block(mock_read, tmp_path: Path) -> None:
    """Converts error output traceback into text blocks."""
    notebook = _notebook(
        cells=[
            _code_cell(
                "1/0",
                outputs=[
                    {
                        "output_type": "error",
                        "traceback": ["Traceback...\n", "ZeroDivisionError: division by zero"],
                        "ename": "ZeroDivisionError",
                        "evalue": "division by zero",
                    }
                ],
            )
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.TEXT]
    assert "ZeroDivisionError" in document.blocks[1].content
    assert document.blocks[1].metadata.cell_type == "output"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_error_output_without_traceback_uses_ename_evalue(mock_read, tmp_path: Path) -> None:
    """Uses ename/evalue fallback when traceback is missing."""
    notebook = _notebook(
        cells=[
            _code_cell(
                "1/0",
                outputs=[
                    {
                        "output_type": "error",
                        "ename": "ZeroDivisionError",
                        "evalue": "division by zero",
                    }
                ],
            )
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.TEXT]
    assert document.blocks[1].content == "ZeroDivisionError: division by zero"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_error_output_without_details_uses_generic_text(mock_read, tmp_path: Path) -> None:
    """Uses a generic text fallback when error details are missing."""
    notebook = _notebook(cells=[_code_cell("x", outputs=[{"output_type": "error"}])])
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.TEXT]
    assert document.blocks[1].content == "error output (details unavailable)"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_unknown_output_type_is_ignored_and_order_is_continuous(mock_read, tmp_path: Path) -> None:
    """Skips unsupported outputs while keeping block order continuous."""
    notebook = _notebook(
        cells=[
            _code_cell(
                "x = 42\nx",
                outputs=[
                    {"output_type": "stream", "text": "ok\n"},
                    {"output_type": "custom_widget", "payload": {"a": 1}},
                    {"output_type": "execute_result", "data": {"text/plain": "42"}},
                ],
            )
        ]
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert [block.type for block in document.blocks] == [BlockType.CODE, BlockType.TEXT, BlockType.TEXT]
    assert [block.order for block in document.blocks] == [0, 1, 2]
    assert document.blocks[1].content == "ok\n"
    assert document.blocks[2].content == "42"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_uses_language_info_over_kernelspec(mock_read, tmp_path: Path) -> None:
    """Prefers language_info.name over kernelspec.language for code metadata."""
    notebook = _notebook(
        cells=[_code_cell("println(1)")],
        metadata={"language_info": {"name": "julia"}, "kernelspec": {"language": "python"}},
    )
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert document.blocks[0].type == BlockType.CODE
    assert document.blocks[0].metadata.language == "julia"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_uses_kernelspec_language_when_language_info_missing(mock_read, tmp_path: Path) -> None:
    """Uses kernelspec language when language_info metadata is absent."""
    notebook = _notebook(cells=[_code_cell("x <- 1")], metadata={"kernelspec": {"language": "r"}})
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert document.blocks[0].type == BlockType.CODE
    assert document.blocks[0].metadata.language == "r"


@patch("parsers.ipynb_parser.nbformat.read")
def test_parse_ipynb_defaults_language_to_python_without_metadata(mock_read, tmp_path: Path) -> None:
    """Defaults code language to python when notebook metadata is missing."""
    notebook = SimpleNamespace(cells=[_code_cell("print('x')")])
    mock_read.return_value = notebook
    file_path = _touch_notebook_file(tmp_path)

    document = parse_ipynb(str(file_path))

    assert document.blocks[0].type == BlockType.CODE
    assert document.blocks[0].metadata.language == "python"
