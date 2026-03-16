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


def _text_block(
    order: int,
    content: str,
    *,
    cell_type: str | None = None,
    page: int | None = None,
) -> Block:
    """Build a TEXT block with optional cell_type and page metadata."""
    meta = BlockMetadata(cell_type=cell_type, page=page)
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


def test_group_blocks_drops_structural_heading_without_body_like_blocks() -> None:
    """Structural headings backed only by TOC fragments should be removed."""
    blocks = [
        _text_block(0, "CHAPTER 1 헬로 파이썬", page=2),
        _text_block(1, "| 1.1 소개 ........ 25 |", page=2),
        _text_block(2, "3.1 신경망 소개", page=3),
        _text_block(3, "이 절에서는 신경망의 기본 흐름을 예제와 함께 설명합니다.", page=3),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo(
            heading="CHAPTER 1 헬로 파이썬",
            level=1,
            start_block_order=0,
            end_block_order=2,
            from_toc=True,
        ),
        SectionInfo(
            heading="3.1 신경망 소개",
            level=1,
            start_block_order=2,
            end_block_order=None,
            from_toc=True,
        ),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["3.1 신경망 소개"]


def test_group_blocks_keeps_short_korean_body_paragraph() -> None:
    """A real Korean paragraph around 45-50 chars must count as body content."""
    short_body = "이 절에서는 배열 곱의 계산 흐름을 간단한 예제로 설명합니다. 핵심 단계만 먼저 살펴봅시다."
    assert 45 <= len(short_body) < 55

    blocks = [
        _text_block(0, "3.3 행렬의 곱", page=10),
        _text_block(1, short_body, page=10),
        _text_block(2, "3.4 다음 절", page=11),
        _text_block(3, "다음 절에서는 실제 코드 예제와 출력 결과를 함께 살펴봅니다.", page=11),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("3.3 행렬의 곱", 1, 0, 2, from_toc=True),
        SectionInfo("3.4 다음 절", 1, 2, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["3.3 행렬의 곱", "3.4 다음 절"]


def test_group_blocks_keeps_sentence_like_mid_length_korean_text() -> None:
    """Sentence-like Korean text in the 25-44 char range should survive."""
    short_sentence = "이 절은 경사하강법의 핵심 직관만 짧게 설명합니다."
    assert 25 <= len(short_sentence) < 45

    blocks = [
        _text_block(0, "4.1 경사하강법", page=20),
        _text_block(1, short_sentence, page=20),
        _text_block(2, "4.2 다음 주제", page=21),
        _text_block(3, "다음 절에서는 실제 업데이트 식을 설명합니다.", page=21),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("4.1 경사하강법", 1, 0, 2, from_toc=True),
        SectionInfo("4.2 다음 주제", 1, 2, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["4.1 경사하강법", "4.2 다음 주제"]


def test_group_blocks_rejects_short_numeric_toc_fragment() -> None:
    """Numeric TOC fragments like 6.1 must not validate a structural heading."""
    blocks = [
        _text_block(0, "CHAPTER 6 학습 관련 기술들", page=5),
        _text_block(1, "6.1", page=5),
        _text_block(2, "3.2 활성화 함수", page=6),
        _text_block(3, "활성화 함수는 입력 신호를 다음 층으로 전달하기 전에 변환합니다.", page=6),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("CHAPTER 6 학습 관련 기술들", 1, 0, 2, from_toc=True),
        SectionInfo("3.2 활성화 함수", 1, 2, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["3.2 활성화 함수"]


def test_group_blocks_trims_front_matter_when_toc_precedes_first_body_section() -> None:
    """Cover/TOC pages before the first real body page should not leak into the preamble."""
    blocks = [
        _text_block(0, "표지 설명이 길게 이어집니다. " * 6, page=1),
        _text_block(1, "출판 정보가 자세히 이어집니다. " * 6, page=1),
        _text_block(2, "CONTENTS", page=2),
        _text_block(3, "CHAPTER 1 헬로 파이썬", page=2),
        _text_block(4, "| 1.1 소개 ........ 25 |", page=2),
        _text_block(5, "이 장에서는 신경망의 큰 흐름을 먼저 설명합니다.", page=3),
        _text_block(6, "3.1 신경망 소개", page=3),
        _text_block(7, "신경망은 앞 장의 퍼셉트론과 공통점이 많지만 활성화 함수에서 중요한 차이가 있습니다.", page=3),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("표지", 1, 0, 3, from_toc=False),
        SectionInfo("CHAPTER 1 헬로 파이썬", 1, 3, 5, from_toc=True),
        SectionInfo("Chapter opener", 1, 5, 6, from_toc=False),
        SectionInfo("3.1 신경망 소개", 1, 6, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["서론", "3.1 신경망 소개"]
    preamble_orders = [block.order for block in result[0].blocks]
    assert preamble_orders == [5]
    all_orders = {block.order for section in result for block in section.blocks}
    assert 0 not in all_orders
    assert 1 not in all_orders
    assert 2 not in all_orders
    assert 3 not in all_orders
    assert 4 not in all_orders


def test_group_blocks_without_toc_like_pages_keeps_legacy_preamble_behavior() -> None:
    """Ordinary intro pages without TOC signals should still form a preamble."""
    blocks = [
        _text_block(0, "문서 소개가 길게 이어집니다. " * 6, page=1),
        _text_block(1, "학습 목표를 자세히 설명합니다. " * 6, page=1),
        _text_block(2, "1. 시작하기", page=2),
        _text_block(3, "이 절에서는 전체 학습 흐름과 이후 절의 연결 관계를 소개합니다.", page=2),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("1. 시작하기", 1, 2, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert [section.heading for section in result] == ["서론", "1. 시작하기"]
    assert [block.order for block in result[0].blocks] == [0, 1]


def test_group_blocks_preserves_first_real_section_on_same_page_as_chapter_opener() -> None:
    """Trim should keep content on the first real body page even before the heading block."""
    blocks = [
        _text_block(0, "CONTENTS", page=2),
        _text_block(1, "CHAPTER 1 헬로 파이썬", page=2),
        _text_block(2, "| 1.1 소개 ........ 25 |", page=2),
        _text_block(3, "이번 장에서는 핵심 개념을 짧게 미리 정리합니다.", page=3),
        _text_block(4, "3.1 신경망 소개", page=3),
        _text_block(5, "신경망은 입력과 출력을 연결하는 여러 층으로 구성됩니다.", page=3),
    ]
    doc = _doc_with_blocks(blocks)
    sections = [
        SectionInfo("CHAPTER 1 헬로 파이썬", 1, 1, 3, from_toc=True),
        SectionInfo("Chapter opener", 1, 3, 4, from_toc=False),
        SectionInfo("3.1 신경망 소개", 1, 4, None, from_toc=True),
    ]

    result = group_blocks_by_section(doc, sections, min_blocks_per_section=1)

    assert result[0].heading == "서론"
    assert [block.order for block in result[0].blocks] == [3]
    assert result[1].heading == "3.1 신경망 소개"


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
