"""Section-aware splitting for large documents.

Pure data-processing module — no LLM calls. Extracts TOC structure from
DoclingDocument cache (PDF) or markdown headings (ipynb) and groups
document blocks by section.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from models.document import Block, BlockType, Document, DocumentFormat
from llm.block_filter import is_noise_block, _HEADING_PATTERN, _SECTION_NUM_PATTERN

LOGGER = logging.getLogger(__name__)

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_NORMALIZED_WS_RE = re.compile(r"\s+")
_TOC_LEADER_RE = re.compile(r"(?:\.{4,}|-{4,}|·{4,}|•{4,})")
_SHORT_SECTION_FRAGMENT_RE = re.compile(r"^\d+(?:\.\d+){1,4}$")
_TABLE_ROW_RE = re.compile(r"\|.+\|")
_SENTENCE_END_RE = re.compile(r'(?:다\.|[.?!])(?:["\')\]]+)?$')

# Patterns that look like figure/table captions or code snippets — NOT structural TOC entries
_CAPTION_RE = re.compile(
    r"^(그림|표|Figure|Table|Fig\.?|Tab\.?)\s*[\d\-]",
    re.IGNORECASE,
)
# Structural heading: numbered section (3.1 ...) or chapter/part keyword
_STRUCTURAL_RE = re.compile(
    r"^(\d+[\.\d]*\s+\S|CHAPTER|PART|SECTION|장|절|부록|Appendix|Chapter|Part)",
    re.IGNORECASE,
)


def _is_structural_heading(text: str) -> bool:
    """Return True if ``text`` looks like a real TOC entry (not a figure/table caption)."""
    t = _normalized_text(text)
    if _CAPTION_RE.match(t):
        return False
    if _STRUCTURAL_RE.match(t):
        return True
    # Accept if it matches the existing heading/section-num patterns used in block_filter
    if _HEADING_PATTERN.match(t) or _SECTION_NUM_PATTERN.match(t):
        return True
    return False


def _normalized_text(text: str) -> str:
    """Normalize Docling text for heuristic checks."""
    return _NORMALIZED_WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def _looks_sentence_like(text: str) -> bool:
    """Return True if text looks like a short natural-language sentence."""
    return bool(_SENTENCE_END_RE.search(text))


def _looks_like_toc_fragment(text: str) -> bool:
    """Return True for navigation fragments commonly found on TOC pages."""
    normalized = _normalized_text(text)
    if not normalized:
        return False
    if _TOC_LEADER_RE.search(normalized):
        return True
    if _TABLE_ROW_RE.search(normalized):
        return True
    if _SHORT_SECTION_FRAGMENT_RE.match(normalized):
        return True
    if _is_structural_heading(normalized) and "\n" not in normalized and not _looks_sentence_like(normalized):
        return True
    return False


def _is_body_like_block(block: Block) -> bool:
    """Return True when a block looks like actual body content, not navigation."""
    if is_noise_block(block):
        return False

    if block.type == BlockType.TEXT:
        text = _normalized_text(block.content)
        if not text:
            return False
        if _looks_like_toc_fragment(text):
            return False
        if "|" in text and _TABLE_ROW_RE.search(text):
            return False
        if len(text) >= 45:
            return True
        if len(text) >= 25 and _looks_sentence_like(text):
            return True
        return False

    if block.type in {BlockType.CODE, BlockType.TABLE, BlockType.FIGURE}:
        return True

    return False


def _blocks_in_range(blocks: list[Block], start: int, end: int | None) -> list[Block]:
    """Return blocks whose order falls within [start, end)."""
    return [
        block
        for block in blocks
        if block.order >= start and (end is None or block.order < end)
    ]


def _recompute_end_block_orders(sections: list["SectionInfo"]) -> None:
    """Set end_block_order based on the next surviving section."""
    for index, section in enumerate(sections):
        if index + 1 < len(sections):
            section.end_block_order = sections[index + 1].start_block_order
        else:
            section.end_block_order = None


def _section_has_body_like_followers(section: "SectionInfo", raw_blocks: list[Block]) -> bool:
    """Return True when a structural heading is followed by body-like content."""
    for block in raw_blocks:
        if block.order <= section.start_block_order:
            continue
        if _is_body_like_block(block):
            return True
    return False


def _page_is_toc_like(blocks_on_page: list[Block]) -> bool:
    """Return True when a page is dominated by TOC/navigation fragments."""
    toc_fragments = 0
    has_body_like = False

    for block in blocks_on_page:
        if block.type != BlockType.TEXT:
            if _is_body_like_block(block):
                has_body_like = True
            continue

        text = _normalized_text(block.content)
        if not text:
            continue
        if text.upper() == "CONTENTS":
            return True
        if _looks_like_toc_fragment(text):
            toc_fragments += 1
        elif _is_body_like_block(block):
            has_body_like = True

    return toc_fragments >= 3 or (toc_fragments >= 2 and not has_body_like)


def _section_pages(raw_blocks: list[Block]) -> set[int]:
    """Collect known page numbers for blocks in a section."""
    return {
        block.metadata.page
        for block in raw_blocks
        if block.metadata.page is not None
    }


def _all_blocks_before_page(raw_blocks: list[Block], page: int) -> bool:
    """Return True when every block in a range belongs to pages before ``page``."""
    pages = _section_pages(raw_blocks)
    return bool(pages) and all(block_page < page for block_page in pages)


@dataclass
class SectionInfo:
    """Represents one section extracted from a document's heading structure."""

    heading: str  # e.g. "3.1 Git 기초"
    level: int  # 1 = top-level, 2 = sub-section
    start_block_order: int  # first block.order in this section
    end_block_order: int | None = None  # exclusive upper bound (None = end of doc)
    blocks: list[Block] = field(default_factory=list)
    from_toc: bool = True  # True = extracted from Docling TOC or ipynb headings; False = heuristic


# ---------------------------------------------------------------------------
# PDF path: extract from DoclingDocument cache
# ---------------------------------------------------------------------------


def extract_sections_pdf(doc_id: str, max_level: int = 2) -> list[SectionInfo]:
    """Extract section headings from cached DoclingDocument.

    Iterates ``doc.iterate_items()`` to find ``SECTION_HEADER`` / ``TITLE``
    items and maps them to block order indices by mirroring the same
    skip-conditions as ``parsers/pdf_parser._to_blocks()``.

    Falls back to heuristic heading detection on blocks if DoclingDocument
    cache is unavailable.

    Args:
        doc_id: Document ID (SHA-256 prefix) used to look up cache.
        max_level: Maximum heading level to split on (1 = H1 only,
            2 = H1 + H2, etc.). Default 2.

    Returns:
        List of SectionInfo with ``heading``, ``level``, and
        ``start_block_order`` populated. ``blocks`` are NOT filled yet —
        use ``group_blocks_by_section`` for that.
    """
    from utils.cache import load_docling_doc_by_id

    dl_doc = load_docling_doc_by_id(doc_id)
    if dl_doc is None:
        LOGGER.debug(
            "No DoclingDocument cache for %s — cannot extract sections from Docling",
            doc_id,
        )
        return []

    try:
        from docling_core.types.doc import (
            DocItemLabel,
            PictureItem,
            TableItem,
            TextItem,
        )
    except ImportError:
        LOGGER.debug("docling_core not available — cannot extract sections")
        return []

    heading_labels = {DocItemLabel.SECTION_HEADER.value, DocItemLabel.TITLE.value}

    sections: list[SectionInfo] = []
    block_order = 0  # mirrors parsers/pdf_parser._to_blocks order counter

    for item, level in dl_doc.iterate_items():
        label_str = (
            item.label.value if hasattr(item.label, "value") else str(item.label)
        )

        # Determine content the same way pdf_parser._to_blocks does.
        # Pass dl_doc to export_to_markdown so Docling versions that require
        # document context for table export (e.g. for cross-references) work
        # correctly — mirrors parsers/pdf_parser._to_blocks line 144.
        if isinstance(item, TableItem):
            content = (
                item.export_to_markdown(dl_doc)
                if hasattr(item, "export_to_markdown")
                else "[table]"
            )
            if not content:
                content = "[table]"
        elif isinstance(item, PictureItem):
            content = "[figure]"
        elif isinstance(item, TextItem):
            content = item.text or ""
        else:
            content = getattr(item, "text", None) or ""

        if not content:
            continue  # mirroring the skip in _to_blocks

        if label_str in heading_labels:
            heading_level = level if isinstance(level, int) and level >= 1 else 1
            if heading_level <= max_level:
                heading_text = content.strip()
                sections.append(
                    SectionInfo(
                        heading=heading_text,
                        level=heading_level,
                        start_block_order=block_order,
                        from_toc=_is_structural_heading(heading_text),
                    )
                )

        block_order += 1

    # If document has only H1 headings and max_level >= 2, that's fine — they're already included.
    # But if no sections found at all with max_level=2, retry with higher max_level
    if not sections and max_level == 2:
        return extract_sections_pdf(doc_id, max_level=6)

    # Set end_block_order for each section
    for i in range(len(sections) - 1):
        sections[i].end_block_order = sections[i + 1].start_block_order
    # Last section: end_block_order stays None (extends to end of doc)

    return sections


# ---------------------------------------------------------------------------
# ipynb path: extract from markdown cell headings
# ---------------------------------------------------------------------------


def extract_sections_ipynb(doc: Document, max_level: int = 2) -> list[SectionInfo]:
    """Extract section headings from ipynb markdown cells.

    Scans TEXT blocks where ``metadata.cell_type == "markdown"`` for
    ``# Heading`` patterns and returns matching sections.

    Args:
        doc: Parsed Document with blocks from ipynb parser.
        max_level: Maximum heading level to split on. Default 2.

    Returns:
        List of SectionInfo ordered by block position.
    """
    sections: list[SectionInfo] = []

    for block in doc.blocks:
        if block.type != BlockType.TEXT:
            continue
        if block.metadata.cell_type != "markdown":
            continue

        for match in _MARKDOWN_HEADING_RE.finditer(block.content):
            hashes = match.group(1)
            heading_text = match.group(2).strip()
            level = len(hashes)
            if level <= max_level and heading_text:
                sections.append(
                    SectionInfo(
                        heading=heading_text,
                        level=level,
                        start_block_order=block.order,
                    )
                )
                break  # one heading per block — use the first one

    if not sections and max_level == 2:
        return extract_sections_ipynb(doc, max_level=6)

    for i in range(len(sections) - 1):
        sections[i].end_block_order = sections[i + 1].start_block_order

    return sections


# ---------------------------------------------------------------------------
# Heuristic fallback: detect headings from block content patterns
# ---------------------------------------------------------------------------


def _extract_sections_heuristic(doc: Document, max_level: int = 2) -> list[SectionInfo]:
    """Fallback heading detection using content patterns.

    Used when DoclingDocument cache is unavailable for PDFs. Detects
    headings via CHAPTER/PART patterns and standalone short uppercase lines.

    Args:
        doc: Parsed Document.
        max_level: Maximum heading level (only level-1 detected heuristically).

    Returns:
        List of SectionInfo.
    """
    sections: list[SectionInfo] = []

    for block in doc.blocks:
        if block.type != BlockType.TEXT:
            continue
        content = block.content.strip()
        if not content or "\n" in content:
            continue
        if len(content) > 120:
            continue

        if _HEADING_PATTERN.match(content) or _SECTION_NUM_PATTERN.match(content):
            sections.append(
                SectionInfo(
                    heading=content,
                    level=1,
                    start_block_order=block.order,
                    from_toc=False,
                )
            )

    for i in range(len(sections) - 1):
        sections[i].end_block_order = sections[i + 1].start_block_order

    return sections


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------


def extract_sections(doc: Document, max_level: int = 2) -> list[SectionInfo]:
    """Extract sections from a document, dispatching by format.

    - PDF → ``extract_sections_pdf(doc.id)`` with heuristic fallback
    - IPYNB → ``extract_sections_ipynb(doc)``
    - IMAGE → empty list (no sections)

    Args:
        doc: Parsed Document.
        max_level: Maximum heading level to split on. Default 2.

    Returns:
        List of SectionInfo.
    """
    if doc.format == DocumentFormat.PDF:
        sections = extract_sections_pdf(doc.id, max_level=max_level)
        if not sections:
            sections = _extract_sections_heuristic(doc, max_level=max_level)
        return sections
    elif doc.format == DocumentFormat.IPYNB:
        return extract_sections_ipynb(doc, max_level=max_level)
    else:
        return []


# ---------------------------------------------------------------------------
# Group blocks by section
# ---------------------------------------------------------------------------


def group_blocks_by_section(
    doc: Document,
    sections: list[SectionInfo],
    min_blocks_per_section: int = 5,
) -> list[SectionInfo]:
    """Assign document blocks to sections and merge short sections.

    Blocks before the first heading become a "preamble" section.
    Each section's blocks are noise-filtered. Sections with fewer than
    ``min_blocks_per_section`` non-noise blocks are merged into the next
    section (or previous if last).

    Args:
        doc: Parsed Document with blocks.
        sections: Section list from ``extract_sections``.
        min_blocks_per_section: Minimum non-noise block count per section
            before merge. Default 5.

    Returns:
        Updated sections list with ``blocks`` populated. May have fewer
        sections than input due to merges.
    """
    if not sections:
        return sections

    # Build a mutable copy so we don't modify the caller's data.
    # ``from_toc`` here means "structural-looking heading", not confirmed
    # provenance from a TOC page.
    working: list[SectionInfo] = []
    for s in sections:
        working.append(
            SectionInfo(
                heading=s.heading,
                level=s.level,
                start_block_order=s.start_block_order,
                end_block_order=s.end_block_order,
                blocks=[],
                from_toc=s.from_toc,
            )
        )

    # Step A: assign raw own-range blocks to each candidate so structural
    # headings can be validated against actual follower content.
    raw_ranges: dict[int, list[Block]] = {}
    for section in working:
        raw_ranges[section.start_block_order] = _blocks_in_range(
            doc.blocks,
            section.start_block_order,
            section.end_block_order,
        )

    has_page_metadata = any(block.metadata.page is not None for block in doc.blocks)

    # Step B: validate structural PDF headings. Headings that never lead into
    # body-like content are likely TOC/menu artefacts and should not become
    # note sections. Skip this path when page metadata is unavailable because
    # the front-matter/TOC heuristics rely on real PDF pagination.
    validated: list[SectionInfo] = []
    for section in working:
        raw_blocks = raw_ranges[section.start_block_order]
        if (
            has_page_metadata
            and section.from_toc
            and _is_structural_heading(section.heading)
            and not _section_has_body_like_followers(section, raw_blocks)
        ):
            continue
        validated.append(section)

    if not validated:
        return []

    _recompute_end_block_orders(validated)

    # Step C: detect whether front-matter trimming should activate.
    first_structural = next(
        (
            section
            for section in validated
            if section.from_toc and _is_structural_heading(section.heading)
        ),
        None,
    )
    first_body_page = None
    if has_page_metadata and first_structural is not None:
        heading_pages = _section_pages(raw_ranges[first_structural.start_block_order])
        if heading_pages:
            first_body_page = min(heading_pages)

    trim_front_matter = False
    if first_body_page is not None:
        blocks_by_page: dict[int, list[Block]] = {}
        for block in doc.blocks:
            page = block.metadata.page
            if page is None or page >= first_body_page:
                continue
            blocks_by_page.setdefault(page, []).append(block)
        trim_front_matter = any(
            _page_is_toc_like(page_blocks)
            for _, page_blocks in sorted(blocks_by_page.items())
        )

    if trim_front_matter and first_body_page is not None:
        validated = [
            section
            for section in validated
            if not _all_blocks_before_page(
                raw_ranges[section.start_block_order],
                first_body_page,
            )
        ]
        if not validated:
            return []
        _recompute_end_block_orders(validated)
        first_structural = next(
            (
                section
                for section in validated
                if section.from_toc and _is_structural_heading(section.heading)
            ),
            None,
        )

    # Step D: build the optional preamble only after structural validation and
    # optional front-matter trim so early TOC pages do not leak into "서론".
    has_toc = any(section.from_toc for section in validated)
    anchor_section = first_structural if has_toc and first_structural is not None else validated[0]
    anchor_order = anchor_section.start_block_order
    preamble_blocks = [
        block
        for block in doc.blocks
        if block.order < anchor_order
        and not is_noise_block(block)
        and (
            not trim_front_matter
            or block.metadata.page is None
            or (
                first_body_page is not None and block.metadata.page >= first_body_page
            )
        )
    ]

    result: list[SectionInfo] = []
    if preamble_blocks:
        result.append(
            SectionInfo(
                heading="서론",
                level=1,
                start_block_order=0,
                end_block_order=anchor_order,
                blocks=preamble_blocks,
                from_toc=False,
            )
        )

    first_structural_start = first_structural.start_block_order if first_structural else None
    for section in validated:
        if (
            has_toc
            and not section.from_toc
            and first_structural_start is not None
            and section.start_block_order < first_structural_start
        ):
            continue
        section.blocks = [
            block
            for block in _blocks_in_range(doc.blocks, section.start_block_order, section.end_block_order)
            if not is_noise_block(block)
        ]
        result.append(section)

    # Step E: determine merge strategy based on whether real TOC sections exist.
    #
    # If from_toc=True sections are present (Docling / ipynb headings), absorb all
    # from_toc=False sections into the nearest preceding from_toc=True section.
    # Figure captions, code-snippet headings, and other artefacts are never
    # top-level sections in this mode.
    #
    # If no from_toc=True sections exist (pure heuristic fallback), treat all
    # sections as independent and apply the classic short-section merge.
    has_toc = any(s.from_toc for s in result if s.heading != "서론")

    if has_toc:
        # Structural sections now define the main note skeleton. Any remaining
        # non-structural sections are treated as subordinate content and folded
        # into the nearest preceding structural section.
        toc_sections: list[SectionInfo] = []
        last_structural: SectionInfo | None = None
        for section in result:
            if section.heading == "서론":
                toc_sections.append(section)
            elif section.from_toc:
                toc_sections.append(section)
                last_structural = section
            else:
                if last_structural is None:
                    if toc_sections and toc_sections[0].heading == "서론":
                        toc_sections[0].blocks.extend(section.blocks)
                    else:
                        toc_sections.append(section)
                    continue
                if section.blocks:
                    heading_block = Block(
                        type=BlockType.TEXT,
                        content=f"### {section.heading}",
                        order=section.start_block_order,
                    )
                    last_structural.blocks.extend([heading_block] + section.blocks)

        work = toc_sections
    else:
        # Pure heuristic — all sections are fair game for classic merging
        work = result

    # Step C: merge sections with too few blocks into the next one
    merged: list[SectionInfo] = []
    i = 0
    while i < len(work):
        current = work[i]
        if len(current.blocks) < min_blocks_per_section and i + 1 < len(work):
            next_section = work[i + 1]
            heading_block = Block(
                type=BlockType.TEXT,
                content=f"### {current.heading}",
                order=current.start_block_order,
            )
            next_section.blocks = [heading_block] + current.blocks + next_section.blocks
            next_section.start_block_order = current.start_block_order
            i += 1
        else:
            merged.append(current)
            i += 1

    # Merge the last section backward if still too short
    if len(merged) >= 2 and len(merged[-1].blocks) < min_blocks_per_section:
        last = merged.pop()
        heading_block = Block(
            type=BlockType.TEXT,
            content=f"### {last.heading}",
            order=last.start_block_order,
        )
        merged[-1].blocks.extend([heading_block] + last.blocks)
        merged[-1].end_block_order = last.end_block_order

    return merged


__all__ = [
    "SectionInfo",
    "extract_sections",
    "extract_sections_pdf",
    "extract_sections_ipynb",
    "group_blocks_by_section",
]
