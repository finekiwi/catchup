"""Noise-block filtering for document blocks.

Extracted from note_generator.py so that both note_generator and
section_splitter can share the same filtering logic without circular imports.
"""

from __future__ import annotations

import re

from models.document import Block, BlockType

_MIN_FIGURE_CONTENT_LEN = 20  # figure blocks shorter than this carry no useful text
_NOISE_TEXT_MAX_LEN = (
    60  # text blocks below this length are candidates for noise filtering
)
_NOISE_DOT_RATIO = 0.3  # if >30% of chars are dots/dashes, treat as TOC/page-num line
_HEADING_MAX_LEN = 80  # standalone chapter/part/section headings below this length
_CHAPTER_INTRO_MAX_LEN = (
    450  # chapter/appendix intro blurbs below this length are noise
)

# Compiled pattern for purely numeric or dot-leader lines (e.g. "........... 54", "3.1  ......53")
_NOISE_PATTERN = re.compile(r"^[\s\d\.\-\·\•·]+$")
# Standalone chapter/part/appendix heading (e.g. "CHAPTER 3 Git 기초", "PART II IDE 활용", "부록 B GitLab")
_HEADING_PATTERN = re.compile(
    r"^(CHAPTER|PART|chapter|part|부록|Appendix|APPENDIX)\s+\S", re.IGNORECASE
)
_SECTION_NUM_PATTERN = re.compile(r"^\d{1,2}\s+\S")


def is_noise_block(block: Block) -> bool:
    """Return True if the block carries no substantive content.

    Filters out:
    - Empty/very-short figure blocks (no VLM description)
    - Short text blocks that are page numbers, TOC dot-leaders, headers/footers
      e.g. ".......... 54" or "3.1 기본 명령어 ............ 53"
    - Standalone chapter/part/appendix headings with no body text
      e.g. "CHAPTER 7 Visual Studio에서의 Git 사용법", "부록 B GitLab"
    - Chapter/appendix intro blurbs: first line is a chapter-level marker and
      total content is short (< _CHAPTER_INTRO_MAX_LEN chars), e.g.
      "CHAPTER 5 Git의 다양한 활용법\n\n다양한 IDE가 Git을 통합해 관리할 수 있도록..."
      These are structural summaries, not the actual substantive section content.
    """
    content = block.content.strip()
    if block.type == BlockType.FIGURE:
        return len(content) < _MIN_FIGURE_CONTENT_LEN
    if block.type == BlockType.TEXT:
        if len(content) < _NOISE_TEXT_MAX_LEN:
            # Purely numeric/dot-leader line
            if _NOISE_PATTERN.match(content):
                return True
            # High dot/dash ratio (TOC lines like "기본 명령어 ........... 53")
            dot_count = content.count(".") + content.count("·") + content.count("-")
            if len(content) > 0 and dot_count / len(content) > _NOISE_DOT_RATIO:
                return True
        # Standalone CHAPTER/PART/부록 heading with no body
        if len(content) < _HEADING_MAX_LEN and "\n" not in content:
            if _HEADING_PATTERN.match(content) or _SECTION_NUM_PATTERN.match(content):
                return True
        # Multi-line chapter/appendix intro blurb: first line is a chapter-level marker
        # and total block is short — these are structural summaries, not learning content
        if len(content) < _CHAPTER_INTRO_MAX_LEN:
            first_line = content.split("\n")[0].strip()
            if _HEADING_PATTERN.match(first_line):
                return True
    return False


__all__ = [
    "is_noise_block",
    "_HEADING_PATTERN",
    "_SECTION_NUM_PATTERN",
]
