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
    _build_section_page_ranges,
    _place_figures_page_based,
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
# Helpers for doc-aware tests
# ---------------------------------------------------------------------------


def _make_doc(body_blocks: list[tuple[int, int, str]]) -> MagicMock:
    """Build a minimal mock Document.

    Args:
        body_blocks: list of (order, page, content) for TEXT blocks.
            Blocks whose content starts with a section-number pattern
            (e.g. "4.1.2 Activation Functions") will be treated as
            section headers by _build_section_page_ranges.
    """
    blocks = []
    for order, page, content in body_blocks:
        b = MagicMock()
        b.order = order
        b.image_path = None
        b.metadata.page = page
        b.content = content
        blocks.append(b)
    doc = MagicMock()
    doc.blocks = blocks
    return doc


# ---------------------------------------------------------------------------
# _build_section_page_ranges
# ---------------------------------------------------------------------------


def test_build_section_page_ranges_header_matching_exact() -> None:
    """Section-header blocks should set exact page boundaries per note section."""
    # Doc: "4.1 Intro" starts on page 1, "4.2 Details" starts on page 5
    doc = _make_doc([
        (1, 1, "4.1 Introduction"),
        (2, 2, "body text page 2"),
        (3, 3, "body text page 3"),
        (4, 5, "4.2 Details"),
        (5, 6, "body text page 6"),
        (6, 8, "body text page 8"),
    ])
    headings = ["## 4.1 Introduction", "## 4.2 Details"]
    ranges = _build_section_page_ranges(doc, 2, section_headings=headings)

    assert ranges[0] == (1, 4), "4.1 should cover pages 1–4 (before 4.2 on page 5)"
    assert ranges[1][0] == 5, "4.2 should start on page 5"


def test_build_section_page_ranges_unmatched_heading_fallback() -> None:
    """Headings without a doc section number should fall back to even-split."""
    doc = _make_doc([
        (1, 1, "body text"),
        (2, 3, "body text"),
        (3, 6, "body text"),
        (4, 9, "body text"),
    ])
    headings = ["## 서론", "## 결론"]  # no section numbers → fallback
    ranges = _build_section_page_ranges(doc, 2, section_headings=headings)

    assert len(ranges) == 2
    # Both ranges should be valid (min_page, max_page) tuples
    for lo, hi in ranges:
        assert lo <= hi


def test_build_section_page_ranges_no_headings_even_split() -> None:
    """Without section_headings, falls back to even body-block split."""
    doc = _make_doc([
        (1, 1, "text"), (2, 2, "text"),
        (3, 5, "text"), (4, 6, "text"),
    ])
    ranges = _build_section_page_ranges(doc, 2)  # no section_headings
    assert len(ranges) == 2
    # First half should be earlier pages than second half
    assert ranges[0][0] <= ranges[1][0]


def test_build_section_page_ranges_empty_doc_returns_sentinel() -> None:
    """A doc with no body blocks should return (1, 9999) sentinels."""
    doc = _make_doc([])
    ranges = _build_section_page_ranges(doc, 3)
    assert ranges == [(1, 9999)] * 3


# ---------------------------------------------------------------------------
# _place_figures_page_based
# ---------------------------------------------------------------------------


def test_place_figures_exact_range_match(tmp_path: Path) -> None:
    """A figure whose page falls within a range should match that section."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=3, image_path=str(img))

    # section 0: pages 1-2, section 1: pages 3-5, section 2: pages 6-9
    ranges = [(1, 2), (3, 5), (6, 9)]
    result = _place_figures_page_based([block], ranges)

    assert 1 in result, "Figure on page 3 should map to section 1 (pages 3–5)"
    assert result[1][0] is block


def test_place_figures_nearest_midpoint_fallback(tmp_path: Path) -> None:
    """A figure whose page is outside all ranges should use nearest midpoint."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=10, image_path=str(img))

    # No range covers page 10; nearest midpoint: section 1 mid=(6+8)/2=7
    ranges = [(1, 2), (6, 8)]
    result = _place_figures_page_based([block], ranges)

    assert 1 in result, "Page 10 is nearest to section 1 midpoint (7)"


def test_place_figures_none_page_goes_to_last_section(tmp_path: Path) -> None:
    """Figures without a page number should be appended to the last section."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))
    block.metadata.page = None  # override page to None

    ranges = [(1, 3), (4, 6), (7, 9)]
    result = _place_figures_page_based([block], ranges)

    assert 2 in result, "No-page figure should go to last section (index 2)"


# ---------------------------------------------------------------------------
# interleave_figures_into_sections — doc-aware path
# ---------------------------------------------------------------------------


def test_interleave_with_doc_places_figure_in_header_matched_section(tmp_path: Path) -> None:
    """With doc provided, figure should land in the section whose page range matches."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    # Figure is on page 3 — should land in 4.1.2 section (pages 2-4)
    block = _make_block(page=3, image_path=str(img))

    doc = _make_doc([
        (1, 1, "4.1 Introduction"),
        (2, 2, "body"),
        (3, 2, "4.1.2 Detail"),   # section header on page 2
        (4, 3, "body page 3"),
        (5, 4, "body page 4"),
        (6, 5, "4.2 Next"),       # next section starts page 5
        (7, 6, "body page 6"),
    ])
    md = "## 4.1 Introduction\n\nbody\n\n## 4.1.2 Detail\n\nbody\n\n## 4.2 Next\n\nbody"
    items = interleave_figures_into_sections(md, [block], doc=doc)

    fig_items = [i for i in items if i["type"] == "figure"]
    assert len(fig_items) == 1

    # The figure item must come after the "4.1.2 Detail" section item
    fig_pos = next(idx for idx, i in enumerate(items) if i["type"] == "figure")
    preceding_sections = [i["markdown"] for i in items[:fig_pos] if i["type"] == "section"]
    assert any("4.1.2" in s for s in preceding_sections), (
        "Figure on page 3 should be placed after the 4.1.2 section"
    )


def test_interleave_with_doc_none_uses_interpolation_fallback(tmp_path: Path) -> None:
    """Without doc, interleave should still work via legacy interpolation."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))

    items = interleave_figures_into_sections(_SAMPLE_MD, [block], doc=None)
    fig_items = [i for i in items if i["type"] == "figure"]
    assert len(fig_items) == 1


# ---------------------------------------------------------------------------
# export_markdown / export_pdf / export_docx — doc param forwarded
# ---------------------------------------------------------------------------


def test_export_markdown_with_doc_does_not_raise(tmp_path: Path) -> None:
    """export_markdown(doc=...) should succeed without raising."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))
    doc = _make_doc([(1, 1, "body")])

    result = export_markdown(_SAMPLE_MD, "title", [block], doc=doc)
    assert result.startswith("# title")
    assert "data:image/png;base64," in result


def test_export_docx_with_doc_does_not_raise(tmp_path: Path) -> None:
    """export_docx(doc=...) should return valid DOCX bytes without raising."""
    img = tmp_path / "fig.png"
    img.write_bytes(_png_bytes())
    block = _make_block(page=1, image_path=str(img))
    doc = _make_doc([(1, 1, "body")])

    result = export_docx(_SAMPLE_MD, "title", [block], doc=doc)
    assert result[:2] == b"PK"


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
