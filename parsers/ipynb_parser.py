"""Parser for Jupyter Notebook files."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

try:
    import nbformat
except ModuleNotFoundError:
    class _NBFormatProxy:
        """Fallback proxy to keep parser import-safe without nbformat installed."""

        @staticmethod
        def read(*args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError("nbformat is required to parse .ipynb files")

    nbformat = _NBFormatProxy()  # type: ignore[assignment]

from models.document import (
    Block,
    BlockMetadata,
    BlockType,
    Document,
    DocumentFormat,
    DocumentMetadata,
    ProcessingInfo,
    ProcessingStatus,
    generate_document_id,
)

logger = logging.getLogger(__name__)

TEXT_MIME_PRIORITY = ("text/plain", "text/markdown")
IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/svg+xml")


def _safe_document_id(file_path: str) -> str:
    """Generate a stable document id and fallback if file read fails."""
    try:
        return generate_document_id(file_path)
    except Exception:
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]


def _normalize_text(value: Any) -> str:
    """Convert notebook values to normalized string content."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "".join(str(item) for item in value)
    else:
        text = str(value)
    return text


def _strip_js_object_placeholders(text: str) -> str:
    """Remove JavaScript object placeholders from Jupyter rich display output text.

    Only called for display_data / execute_result text payloads where
    ``[object Object]`` artifacts appear when the kernel serialises non-string
    rich output.  Not applied to markdown cells or code source to avoid
    corrupting notebooks that legitimately reference the literal string.
    """
    return re.sub(r",?\[object Object\],?", "", text)


def _extract_text_from_data(data: dict[str, Any]) -> str:
    """Extract text content from rich output data payloads."""
    for mime in TEXT_MIME_PRIORITY:
        if mime in data:
            return _strip_js_object_placeholders(_normalize_text(data[mime]))
    return ""


def _get_nested_value(container: Any, key: str) -> Any:
    """Get value from dict-like or object-like containers."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _extract_notebook_language(notebook: Any) -> str:
    """Extract notebook language with a python fallback."""
    metadata = _get_nested_value(notebook, "metadata")

    language_info = _get_nested_value(metadata, "language_info")
    language_name = _normalize_text(_get_nested_value(language_info, "name")).strip()
    if language_name:
        return language_name

    kernelspec = _get_nested_value(metadata, "kernelspec")
    kernelspec_language = _normalize_text(_get_nested_value(kernelspec, "language")).strip()
    if kernelspec_language:
        return kernelspec_language

    return "python"


def _mark_parse_failed(document: Document) -> None:
    """Mark parse failure in metadata tags without duplicate entries."""
    if "parse_failed" not in document.metadata.tags:
        document.metadata.tags.append("parse_failed")


def _extract_output_block(output: dict[str, Any], cell_index: int, order: int) -> Block | None:
    """Convert one code cell output into a single Block when supported."""
    output_type = str(output.get("output_type", ""))
    metadata = BlockMetadata(cell_index=cell_index, cell_type="output")

    if output_type == "stream":
        text = _normalize_text(output.get("text"))
        if not text:
            return None
        return Block(type=BlockType.TEXT, content=text, order=order, metadata=metadata)

    if output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        if not isinstance(data, dict):
            return None

        image_mimes = [mime for mime in IMAGE_MIME_TYPES if mime in data]
        if image_mimes:
            mime_list = ", ".join(image_mimes)
            return Block(
                type=BlockType.FIGURE,
                content=f"[notebook image output: {mime_list}]",
                order=order,
                metadata=metadata,
            )

        text = _extract_text_from_data(data)
        if text:
            return Block(type=BlockType.TEXT, content=text, order=order, metadata=metadata)
        return None

    if output_type == "error":
        traceback_lines = output.get("traceback", [])
        traceback_text = _normalize_text(traceback_lines)
        if traceback_text:
            return Block(type=BlockType.TEXT, content=traceback_text, order=order, metadata=metadata)
        ename = _normalize_text(output.get("ename")).strip()
        evalue = _normalize_text(output.get("evalue")).strip()
        if ename and evalue:
            error_text = f"{ename}: {evalue}"
        elif ename:
            error_text = ename
        elif evalue:
            error_text = evalue
        else:
            error_text = "error output (details unavailable)"
        return Block(type=BlockType.TEXT, content=error_text, order=order, metadata=metadata)

    return None


def parse_ipynb(file_path: str) -> Document:
    """
    Parse a Jupyter notebook file into the shared Document schema.

    Results are cached to data/parsed/ by file content hash. On subsequent calls
    with the same file, the cached Document is returned immediately without re-parsing.

    Args:
        file_path: Path to a `.ipynb` file.

    Returns:
        Parsed Document. If parsing fails, returns a fallback Document
        with empty blocks.
    """
    from utils.cache import load_cached_parse, save_cached_parse

    source_path = Path(file_path)
    cached = load_cached_parse(source_path)
    if cached is not None:
        return cached

    start_time = time.perf_counter()
    source_name = source_path.name
    document_id = _safe_document_id(file_path)
    document = Document(
        id=document_id,
        source=source_name,
        format=DocumentFormat.IPYNB,
        blocks=[],
        metadata=DocumentMetadata(total_cells=0),
        processing=ProcessingInfo(parser_model="nbformat"),
        status=ProcessingStatus.PARSED,
    )

    try:
        notebook = nbformat.read(file_path, as_version=4)
        blocks: list[Block] = []
        order = 0
        notebook_language = _extract_notebook_language(notebook)
        cells = list(getattr(notebook, "cells", []))

        for cell_index, cell in enumerate(cells):
            cell_type = str(cell.get("cell_type", ""))

            if cell_type == "markdown":
                markdown_content = _normalize_text(cell.get("source"))
                blocks.append(
                    Block(
                        type=BlockType.TEXT,
                        content=markdown_content,
                        order=order,
                        metadata=BlockMetadata(cell_index=cell_index, cell_type="markdown"),
                    )
                )
                order += 1
                continue

            if cell_type == "code":
                code_content = _normalize_text(cell.get("source"))
                blocks.append(
                    Block(
                        type=BlockType.CODE,
                        content=code_content,
                        order=order,
                        metadata=BlockMetadata(cell_index=cell_index, cell_type="code", language=notebook_language),
                    )
                )
                order += 1

                for output in cell.get("outputs", []):
                    output_block = _extract_output_block(output, cell_index, order)
                    if output_block is None:
                        continue
                    blocks.append(output_block)
                    order += 1

        document.blocks = blocks
        document.metadata.total_cells = len(cells)

    except Exception:
        logger.exception("Failed to parse ipynb file: %s", file_path)
        document.blocks = []
        document.metadata.total_cells = 0
        _mark_parse_failed(document)
    finally:
        document.processing.latency_ms = (time.perf_counter() - start_time) * 1000

    save_cached_parse(source_path, document)
    return document
