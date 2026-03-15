"""Unit tests for llm/section_splitter.py (CU-14)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from models.document import Block, BlockMetadata, BlockType, Document, DocumentFormat
from llm.section_splitter import (
    SectionInfo,
    extract_sections,
    extract_sections_ipynb,
    extract_sections_pdf,
    group_blocks_by_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_block(order: int, content: str, *, cell_type: str | None = None) -> Block:
    """Build a TEXT block with optional cell_type metadata."""
    meta = BlockMetadata(cell_type=cell_type) if cell_type else BlockMetadata()
    return Block(type=BlockType.TEXT, content=content, order=order, metadata=meta)


def _code_block(order: int, content: str) -> Block:
    """Build a CODE block."""
    return Block(
        type=BlockType.CODE,
        content=content,
        order=order,
        metadata=BlockMetadata(language="python"),
    )


def _doc_with_blocks(
    blocks: list[Block], fmt: DocumentFormat = DocumentFormat.PDF
) -> Document:
    """Build a minimal Document from a block list."""
    return Document(id="test-doc-id", source="test.pdf", format=fmt, blocks=blocks)


def _fake_docling_doc(items: list[tuple[SimpleNamespace, int]]):
    """Build a fake DoclingDocument that yields items from iterate_items()."""
    doc = SimpleNamespace()
    doc.iterate_items = lambda: iter(items)
    return doc


def _make_text_item(label_value: str, text: str) -> SimpleNamespace:
    """Build a fake TextItem-like object."""
    item = SimpleNamespace()
    item.label = SimpleNamespace(value=label_value)
    item.text = text
    return item


# ---------------------------------------------------------------------------
# extract_sections_pdf
# ---------------------------------------------------------------------------


def test_extract_sections_pdf_from_docling_cache() -> None:
    """SECTION_HEADER and TITLE items at levels <= 2 should produce SectionInfo entries."""
    items = [
        (_make_text_item("section_header", "1. 소개"), 1),
        (_make_text_item("text", "본문 내용입니다."), 0),
        (_make_text_item("section_header", "1.1 배경"), 2),
        (_make_text_item("text", "배경 설명입니다."), 0),
        (_make_text_item("section_header", "2. 본론"), 1),
        (_make_text_item("text", "본론 내용입니다."), 0),
    ]
    dl_doc = _fake_docling_doc(items)

    with patch("utils.cache.load_docling_doc_by_id", return_value=dl_doc):
        sections = extract_sections_pdf("test-doc-id", max_level=2)

    assert len(sections) == 3
    assert sections[0].heading == "1. 소개"
    assert sections[0].level == 1
    assert sections[0].start_block_order == 0
    assert sections[1].heading == "1.1 배경"
    assert sections[1].level == 2
    assert sections[2].heading == "2. 본론"
    assert sections[2].level == 1


def test_extract_sections_pdf_max_level_filtering() -> None:
    """H3 headings should be excluded when max_level=2."""
    items = [
        (_make_text_item("section_header", "1. 개요"), 1),
        (_make_text_item("section_header", "1.1 세부"), 2),
        (_make_text_item("section_header", "1.1.1 상세"), 3),  # should be excluded
        (_make_text_item("text", "내용"), 0),
    ]
    dl_doc = _fake_docling_doc(items)

    with patch("utils.cache.load_docling_doc_by_id", return_value=dl_doc):
        sections = extract_sections_pdf("test-doc-id", max_level=2)

    headings = [s.heading for s in sections]
    assert "1. 개요" in headings
    assert "1.1 세부" in headings
    assert "1.1.1 상세" not in headings


def test_extract_sections_pdf_h1_only_fallback() -> None:
    """Document with only H1 headings should still split on H1."""
    items = [
        (_make_text_item("title", "Introduction"), 1),
        (_make_text_item("text", "Some text"), 0),
        (_make_text_item("title", "Methods"), 1),
        (_make_text_item("text", "More text"), 0),
    ]
    dl_doc = _fake_docling_doc(items)

    with patch("utils.cache.load_docling_doc_by_id", return_value=dl_doc):
        sections = extract_sections_pdf("test-doc-id", max_level=2)

    assert len(sections) == 2
    assert sections[0].heading == "Introduction"
    assert sections[1].heading == "Methods"


def test_extract_sections_pdf_fallback_heuristic() -> None:
    """No Docling cache → heuristic heading detection on blocks."""
    blocks = [
        _text_block(0, "CHAPTER 1 Git 기초"),
        _text_block(
            1, "Git은 분산 버전 관리 시스템으로, 소프트웨어 개발에 널리 사용됩니다."
        ),
        _text_block(2, "CHAPTER 2 브랜치"),
        _text_block(3, "브랜치는 독립적인 작업 흐름을 만들 수 있게 해줍니다."),
    ]
    doc = _doc_with_blocks(blocks)

    with patch("utils.cache.load_docling_doc_by_id", return_value=None):
        sections = extract_sections(doc, max_level=2)

    assert len(sections) == 2
    assert sections[0].heading == "CHAPTER 1 Git 기초"
    assert sections[1].heading == "CHAPTER 2 브랜치"


# ---------------------------------------------------------------------------
# extract_sections_ipynb
# ---------------------------------------------------------------------------


def test_extract_sections_ipynb_markdown_headings() -> None:
    """Markdown cells with # headings should produce sections."""
    blocks = [
        _text_block(0, "# Introduction\nSome intro text", cell_type="markdown"),
        _code_block(1, "import numpy as np"),
        _text_block(2, "## Data Loading\nLoad the dataset", cell_type="markdown"),
        _code_block(3, "df = pd.read_csv('data.csv')"),
        _text_block(4, "## Analysis\nAnalyze the data", cell_type="markdown"),
    ]
    doc = _doc_with_blocks(blocks, fmt=DocumentFormat.IPYNB)

    sections = extract_sections_ipynb(doc, max_level=2)

    assert len(sections) == 3
    assert sections[0].heading == "Introduction"
    assert sections[0].level == 1
    assert sections[1].heading == "Data Loading"
    assert sections[1].level == 2
    assert sections[2].heading == "Analysis"
    assert sections[2].level == 2


# ---------------------------------------------------------------------------
# group_blocks_by_section
# ---------------------------------------------------------------------------


def test_group_blocks_by_section_correct_assignment() -> None:
    """Blocks should be mapped to the correct sections by order range."""
    blocks = [
        _text_block(0, "Section 1 content " * 10),
        _text_block(1, "Section 1 more " * 10),
        _text_block(2, "Section 1 extra " * 10),
        _text_block(3, "Section 1 detail " * 10),
        _text_block(4, "Section 1 final " * 10),
        _text_block(5, "Section 2 content " * 10),
        _text_block(6, "Section 2 more " * 10),
        _text_block(7, "Section 2 extra " * 10),
        _text_block(8, "Section 2 detail " * 10),
        _text_block(9, "Section 2 final " * 10),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="Section 1", level=1, start_block_order=0, end_block_order=5
        ),
        SectionInfo(
            heading="Section 2", level=1, start_block_order=5, end_block_order=None
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=3)

    assert len(result) == 2
    assert len(result[0].blocks) == 5
    assert len(result[1].blocks) == 5


def test_group_blocks_preamble() -> None:
    """Blocks before the first heading should go into a preamble section."""
    blocks = [
        _text_block(0, "Preamble content that is long enough to not be noise " * 5),
        _text_block(1, "More preamble content that is long enough " * 5),
        _text_block(2, "Even more preamble content " * 5),
        _text_block(3, "Extra preamble " * 10),
        _text_block(4, "Final preamble " * 10),
        _text_block(5, "Section 1 heading content " * 10),
        _text_block(6, "Section 1 body " * 10),
        _text_block(7, "Section 1 more body " * 10),
        _text_block(8, "Section 1 even more " * 10),
        _text_block(9, "Section 1 final " * 10),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="Section 1", level=1, start_block_order=5, end_block_order=None
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=3)

    assert result[0].heading == "서론"
    assert len(result[0].blocks) == 5
    assert result[1].heading == "Section 1"


def test_group_blocks_short_section_merge() -> None:
    """Heuristic (from_toc=False) sections with fewer than min_blocks blocks should merge."""
    blocks = [
        _text_block(0, "Short section content"),  # only 1 block — will merge
        _text_block(1, "Next section content " * 10),
        _text_block(2, "Next section body " * 10),
        _text_block(3, "Next section more " * 10),
        _text_block(4, "Next section extra " * 10),
        _text_block(5, "Next section final " * 10),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="Tiny Section", level=1, start_block_order=0, end_block_order=1, from_toc=False
        ),
        SectionInfo(
            heading="Big Section", level=1, start_block_order=1, end_block_order=None, from_toc=False
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=5)

    # Tiny section should be merged into Big Section
    assert len(result) == 1
    assert result[0].heading == "Big Section"
    # Merged section should contain both sections' blocks plus heading block
    assert len(result[0].blocks) >= 5


def test_group_blocks_toc_section_with_enough_blocks_survives() -> None:
    """from_toc=True sections with enough blocks are never absorbed into others."""
    blocks = [_text_block(i, f"Content block {i} " * 10) for i in range(12)]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="3.1 First Section", level=1, start_block_order=0, end_block_order=6, from_toc=True
        ),
        SectionInfo(
            heading="3.2 Second Section", level=1, start_block_order=6, end_block_order=None, from_toc=True
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=5)

    # Both have >= 5 blocks — both must survive independently
    assert len(result) == 2
    assert result[0].heading == "3.1 First Section"
    assert result[1].heading == "3.2 Second Section"


def test_group_blocks_toc_section_too_short_merges_into_next() -> None:
    """from_toc=True section with < min_blocks merges into the next section (step C)."""
    blocks = [
        _text_block(0, "Tiny content"),  # 1 block only
        _text_block(1, "Big section content " * 10),
        _text_block(2, "Big section body " * 10),
        _text_block(3, "Big section more " * 10),
        _text_block(4, "Big section extra " * 10),
        _text_block(5, "Big section final " * 10),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="3.1 Tiny TOC Section", level=1, start_block_order=0, end_block_order=1, from_toc=True
        ),
        SectionInfo(
            heading="3.2 Big Section", level=1, start_block_order=1, end_block_order=None, from_toc=True
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=5)

    # 3.1 has only 1 block — merges into 3.2
    assert len(result) == 1
    assert result[0].heading == "3.2 Big Section"


def test_extract_sections_empty_doc() -> None:
    """Empty document should produce empty sections list."""
    doc = _doc_with_blocks([], fmt=DocumentFormat.PDF)

    with patch("utils.cache.load_docling_doc_by_id", return_value=None):
        sections = extract_sections(doc)

    assert sections == []


def test_extract_sections_image_format() -> None:
    """IMAGE format should return empty list (no sections)."""
    doc = _doc_with_blocks([], fmt=DocumentFormat.IMAGE)

    sections = extract_sections(doc)

    assert sections == []


def test_group_preserves_pre_toc_content() -> None:
    """Blocks in from_toc=False sections before the first structural TOC section must not be discarded."""
    blocks = [
        _text_block(0, "Document title text " * 10),  # belongs to the TITLE section
        _text_block(1, "Introduction paragraph " * 10),  # also pre-toc
        _text_block(2, "Chapter body content " * 10),
        _text_block(3, "Chapter body more " * 10),
        _text_block(4, "Chapter body final " * 10),
    ]
    doc = _doc_with_blocks(blocks)

    # A TITLE section (from_toc=False) appears before the first structural section
    sections = [
        SectionInfo(
            heading="My Document Title",
            level=1,
            start_block_order=0,
            end_block_order=2,
            from_toc=False,
        ),
        SectionInfo(
            heading="CHAPTER 1 Introduction",
            level=1,
            start_block_order=2,
            end_block_order=None,
            from_toc=True,
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    # Pre-toc blocks (blocks 0 and 1) must be present somewhere in the output
    all_block_orders = {b.order for s in result for b in s.blocks}
    assert 0 in all_block_orders, "Pre-TOC block 0 must not be discarded"
    assert 1 in all_block_orders, "Pre-TOC block 1 must not be discarded"
