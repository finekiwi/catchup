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
    t = text.strip()
    if _CAPTION_RE.match(t):
        return False
    if _STRUCTURAL_RE.match(t):
        return True
    # Accept if it matches the existing heading/section-num patterns used in block_filter
    if _HEADING_PATTERN.match(t) or _SECTION_NUM_PATTERN.match(t):
        return True
    return False


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

        # Determine content the same way pdf_parser._to_blocks does
        if isinstance(item, TableItem):
            content = (
                item.export_to_markdown()
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

    # Build a mutable copy so we don't modify the caller's data
    result: list[SectionInfo] = []
    for s in sections:
        result.append(
            SectionInfo(
                heading=s.heading,
                level=s.level,
                start_block_order=s.start_block_order,
                end_block_order=s.end_block_order,
                blocks=[],
                from_toc=s.from_toc,
            )
        )

    # Preamble: blocks before first section heading
    first_order = result[0].start_block_order
    preamble_blocks = [
        b for b in doc.blocks if b.order < first_order and not is_noise_block(b)
    ]
    if preamble_blocks:
        preamble = SectionInfo(
            heading="서론",
            level=1,
            start_block_order=0,
            end_block_order=first_order,
            blocks=preamble_blocks,
            from_toc=False,  # preamble is never a TOC entry
        )
        result.insert(0, preamble)

    # Step A: assign own-range blocks to all sections
    for section in result:
        if section.blocks:  # preamble already populated
            continue
        start = section.start_block_order
        end = section.end_block_order
        section.blocks = [
            b
            for b in doc.blocks
            if b.order >= start
            and (end is None or b.order < end)
            and not is_noise_block(b)
        ]

    # Step B: determine merge strategy based on whether real TOC sections exist.
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
        # Absorb from_toc=False into preceding from_toc=True.
        # Preamble (heading="서론", from_toc=False) is always kept standalone.
        toc_sections: list[SectionInfo] = []
        for section in result:
            if section.from_toc:
                toc_sections.append(section)
            elif section.heading == "서론":
                toc_sections.insert(0, section)  # preamble always first
            elif toc_sections:
                parent = toc_sections[-1]
                if section.blocks:
                    heading_block = Block(
                        type=BlockType.TEXT,
                        content=f"### {section.heading}",
                        order=section.start_block_order,
                    )
                    parent.blocks.extend([heading_block] + section.blocks)
            # from_toc=False before any toc section (e.g. book-cover text): discard

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
