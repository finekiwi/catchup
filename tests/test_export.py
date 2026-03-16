"""Tests for utils/export.py — interleave_figures_into_sections, export_markdown/pdf/docx."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from models.document import Block, BlockMetadata, BlockType
from utils.export import (
    export_docx,
    export_markdown,
    export_pdf,
    interleave_figures_into_sections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_block(page: int, image_path: str, caption: str = "") -> Block:
    """Build a minimal FIGURE block for export tests."""
    return Block(
        type=BlockType.FIGURE,
        content="",
        order=page * 10,
        image_path=image_path,
        metadata=BlockMetadata(page=page, caption=caption or None),
    )


def _png_bytes() -> bytes:
    """Return minimal valid PNG bytes (4×4 grey image)."""
    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), color=(128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


_SAMPLE_MD = """\
## 서론

소개 내용입니다.

## 핵심 개념

핵심 내용입니다.

## 결론

마무리 내용입니다."""


# ---------------------------------------------------------------------------
# interleave_figures_into_sections
# ---------------------------------------------------------------------------


def test_interleave_no_figures_returns_sections_only() -> None:
    """With no figures, items should be only section entries."""
    items = interleave_figures_into_sections(_SAMPLE_MD, [])
    assert all(i["type"] == "section" for i in items)
    assert len(items) == 3


def test_interleave_section_count_preserved() -> None:
    """Section count should equal the number of ## headings in the note."""
    items = interleave_figures_into_sections(_SAMPLE_MD, [])
    section_items = [i for i in items if i["type"] == "section"]
    assert len(section_items) == 3


def test_interleave_figures_inserted_between_sections(tmp_path: Path) -> None:
    """Figures should appear as 'figure' items interleaved with sections."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))

    items = interleave_figures_into_sections(_SAMPLE_MD, [block])
    fig_items = [i for i in items if i["type"] == "figure"]
    assert len(fig_items) == 1, "One figure block should produce one figure item"

    # Figure must appear after at least one section item
    fig_pos = next(idx for idx, i in enumerate(items) if i["type"] == "figure")
    assert fig_pos > 0, "Figure should be interleaved after a section, not at position 0"


def test_interleave_caption_fallback(tmp_path: Path) -> None:
    """When block has no caption, fallback should include page number."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=5, image_path=str(img), caption="")

    items = interleave_figures_into_sections(_SAMPLE_MD, [block])
    fig_items = [i for i in items if i["type"] == "figure"]
    assert "5" in fig_items[0]["caption"], "Fallback caption should include page number"


def test_interleave_missing_image_excluded(tmp_path: Path) -> None:
    """Blocks pointing to non-existent paths should be excluded from output."""
    block = _make_block(page=1, image_path=str(tmp_path / "missing.png"))
    items = interleave_figures_into_sections(_SAMPLE_MD, [block])
    assert all(i["type"] != "figure" for i in items), "Missing-file figure should be skipped"


# ---------------------------------------------------------------------------
# export_markdown
# ---------------------------------------------------------------------------


def test_export_markdown_no_figures() -> None:
    """With no figures, output should be vanilla markdown starting with # title."""
    result = export_markdown(_SAMPLE_MD, "테스트 노트", [])
    assert result.startswith("# 테스트 노트")
    assert "## 서론" in result
    assert "data:image" not in result


def test_export_markdown_base64_inline_at_correct_position(tmp_path: Path) -> None:
    """Figure image should appear as base64 data URI after the matching section."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))

    result = export_markdown(_SAMPLE_MD, "노트 제목", [block])

    assert "data:image/png;base64," in result, "Base64 image must be present in output"
    # Section heading should appear before the image
    img_pos = result.index("data:image")
    section_pos = result.index("## 서론")
    assert section_pos < img_pos, "Section heading must precede its inline image"


def test_export_markdown_title_prefix(tmp_path: Path) -> None:
    """Output must begin with '# <title>'."""
    result = export_markdown(_SAMPLE_MD, "My Title", [])
    assert result.startswith("# My Title\n\n")


# ---------------------------------------------------------------------------
# export_pdf
# ---------------------------------------------------------------------------


def test_export_pdf_returns_valid_pdf_header(tmp_path: Path) -> None:
    """PDF output should start with the %PDF magic bytes."""
    result = export_pdf(_SAMPLE_MD, "PDF 테스트", [])
    assert result[:4] == b"%PDF", "PDF output must begin with %PDF header"


def test_export_pdf_includes_image_data(tmp_path: Path) -> None:
    """PDF should embed the figure as an image XObject (xhtml2pdf re-encodes with ASCII85)."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))

    result = export_pdf(_SAMPLE_MD, "PDF with image", [block])
    assert result[:4] == b"%PDF"
    # xhtml2pdf converts the image to a PDF XObject — check for image stream markers
    assert b"/Subtype /Image" in result, "PDF should contain an image XObject stream"
    assert b"/Width 4" in result, "Image dimensions should match the 4×4 test PNG"


# ---------------------------------------------------------------------------
# export_docx
# ---------------------------------------------------------------------------


def test_export_docx_returns_valid_docx_header() -> None:
    """DOCX is a ZIP archive — output should start with the PK magic bytes."""
    result = export_docx(_SAMPLE_MD, "DOCX 테스트", [])
    assert result[:2] == b"PK", "DOCX output must start with ZIP/PK header"


def test_export_docx_valid_zip_structure() -> None:
    """DOCX output should be parseable as a ZIP file."""
    result = export_docx(_SAMPLE_MD, "DOCX test", [])
    buf = io.BytesIO(result)
    assert zipfile.is_zipfile(buf), "DOCX bytes should form a valid ZIP archive"


def test_export_docx_includes_image(tmp_path: Path) -> None:
    """DOCX archive should contain a media/ entry for the embedded figure image."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))

    result = export_docx(_SAMPLE_MD, "DOCX with image", [block])

    buf = io.BytesIO(result)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    media_files = [n for n in names if n.startswith("word/media/")]
    assert media_files, "DOCX should contain at least one media file for the figure image"
