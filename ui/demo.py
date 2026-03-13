"""Streamlit demo for CatchUp: upload → parse → note generation."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run ui/demo.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import markdown as md_lib  # noqa: E402
import streamlit as st  # noqa: E402

from llm.note_editor import NoteEditResult, edit_section  # noqa: E402
from llm.note_editor import _split_sections as _split_note_sections  # noqa: E402
from llm.note_generator import SUPPORTED_LLM_MODELS, generate_note  # noqa: E402
from parsers.image_parser import parse_image  # noqa: E402
from parsers.ipynb_parser import parse_ipynb  # noqa: E402
from parsers.pdf_parser import parse_pdf  # noqa: E402
from rag import index_document, query as rag_query  # noqa: E402
from vlm.client import SUPPORTED_MODELS  # noqa: E402

# Keys injected by the LLM schema but not rendered as note content
_NOTE_INTERNAL_KEYS: frozenset[str] = frozenset(
    {"schema_version", "confidence", "errors", "starter prompts"}
)

# ---------------------------------------------------------------------------
# Global CSS — light theme styles for all custom components
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """\
<style>
/* === CatchUp Warm & Soft Palette === */

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-card {
    background: #FDF8F3;
    border: 1px solid #E5D9CD;
    border-radius: 12px;
    padding: 1.3em 1em;
    text-align: center;
    box-shadow: 0 1px 3px rgba(61,46,36,0.06);
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #C4553A;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #7A6555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.4em;
}

/* ── Block type badges ─────────────────────────────────────────────────── */
.block-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* ── Concept tags (5-color rotation) ─────────────────────────────────── */
.concept-tag {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 18px;
    font-size: 0.83rem;
    font-weight: 500;
    margin: 4px 5px 4px 0;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.concept-tag-0 { background: #E8D5C4; color: #7A5A3E; }
.concept-tag-1 { background: #D4DDD0; color: #4A6342; }
.concept-tag-2 { background: #D6D4E0; color: #5A5470; }
.concept-tag-3 { background: #DBDCE8; color: #4A4C66; }
.concept-tag-4 { background: #E8D8D4; color: #6B4A42; }

/* ── Note content area ─────────────────────────────────────────────────── */
.note-wrapper {
    background: #FDF8F3;
    border: 1px solid #E5D9CD;
    border-radius: 14px;
    padding: 2em 2.5em;
    margin-top: 0.8em;
    max-width: 860px;
    box-shadow: 0 1px 4px rgba(61,46,36,0.06);
}
/* Edit mode: sections flow as one document, no per-section card */
.note-section {
    background: #FDF8F3;
    padding: 0.5em 2.5em;
    max-width: 860px;
}
hr.note-sep {
    border: none;
    border-top: 1px solid #EDE3D9;
    margin: 0.2em 2.5em;
    max-width: 860px;
}
.note-content {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    line-height: 1.8;
    color: #3D2E24;
}
.note-content h1 { font-size: 1.35rem; font-weight: 700; margin: 1.5em 0 0.5em; border-bottom: 1px solid #E5D9CD; padding-bottom: 0.3em; color: #3D2E24; }
.note-content h2 { font-size: 1.15rem; font-weight: 700; margin: 1.3em 0 0.4em; color: #3D2E24; }
.note-content h3 { font-size: 1.02rem; font-weight: 600; margin: 1.1em 0 0.3em; color: #7A6555; }
.note-content h4, .note-content h5, .note-content h6 { font-size: 0.95rem; font-weight: 600; margin: 1em 0 0.3em; color: #7A6555; }
.note-content p { margin: 0.5em 0; font-size: 0.93rem; }
.note-content ul, .note-content ol { margin: 0.4em 0 0.4em 1.5em; }
.note-content li { margin: 0.25em 0; font-size: 0.93rem; }
.note-content code {
    background: #F2DDD6;
    color: #A8432C;
    padding: 0.15em 0.45em;
    border-radius: 4px;
    font-size: 0.85rem;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
.note-content pre {
    background: #F5EDE4;
    border: 1px solid #E5D9CD;
    border-radius: 10px;
    padding: 1.4em 1.5em 1.2em 1.5em;
    overflow-x: auto;
    margin: 1em 0;
    position: relative;
}
.note-content pre::before {
    content: '';
    position: absolute;
    top: 13px; left: 14px;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #B84233;
    box-shadow: 15px 0 0 #C4883A, 30px 0 0 #5B8C5A;
}
.note-content pre code {
    background: none;
    padding: 0;
    font-size: 0.82rem;
    color: #3D2E24;
    display: block;
    padding-top: 0.8em;
}
.note-content blockquote {
    border-left: 3px solid #EACFC5;
    padding: 0.5em 1.2em;
    margin: 0.8em 0;
    background: #F2DDD6;
    color: #7A6555;
    border-radius: 0 8px 8px 0;
}
.note-content hr { border: none; border-top: 1px solid #E5D9CD; margin: 1.5em 0; }
.note-content strong { font-weight: 700; color: #3D2E24; }
.note-content a { color: #C4553A; text-decoration: none; }
.note-content a:hover { text-decoration: underline; }

/* ── Summary card ──────────────────────────────────────────────────────── */
.summary-card {
    background: #F5EDE4;
    border: 1px solid #EACFC5;
    border-left: 3px solid #EACFC5;
    border-radius: 10px;
    padding: 1em 1.4em;
    margin: 0.6em 0 1em;
    color: #7A6555;
    font-size: 0.93rem;
    line-height: 1.7;
}

/* ── Image learning workspace ─────────────────────────────────────────── */
.image-note-card {
    background: linear-gradient(180deg, #FDF8F3 0%, #F8F1E8 100%);
    border: 1px solid #E5D9CD;
    border-radius: 18px;
    padding: 1.2em;
    box-shadow: 0 8px 24px rgba(61, 46, 36, 0.10);
}
.image-note-lead {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 1em;
}
.image-note-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A89585;
    margin-bottom: 0.45em;
}
.image-note-title {
    font-family: 'DM Serif Display', 'Times New Roman', serif;
    font-size: 1.35rem;
    line-height: 1.2;
    color: #3D2E24;
    margin: 0;
}
.image-note-copy {
    color: #7A6555;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 0.5em 0 0;
}
.image-note-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 11px;
    border-radius: 999px;
    background: #F2DDD6;
    color: #A8432C;
    border: 1px solid #EACFC5;
    font-size: 0.76rem;
    font-weight: 700;
    white-space: nowrap;
}
.image-preview-shell {
    background: #F5EDE4;
    border: 1px solid #E5D9CD;
    border-radius: 14px;
    overflow: hidden;
}
.image-preview-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #E5D9CD;
    background: rgba(255, 255, 255, 0.55);
}
.image-preview-toolbar-label {
    color: #3D2E24;
    font-size: 0.8rem;
    font-weight: 700;
}
.image-preview-toolbar-value {
    color: #A8432C;
    font-size: 0.76rem;
    font-weight: 700;
}
.image-preview-scroll {
    max-height: 620px;
    overflow: auto;
    background:
        linear-gradient(180deg, #FCF7F1 0%, #F5EDE4 100%);
}
.image-preview-frame {
    padding: 14px;
    background:
        radial-gradient(circle at top left, rgba(196, 85, 58, 0.08), transparent 32%);
    min-width: 100%;
}
.image-preview-frame img {
    max-width: none;
    height: auto;
    display: block;
    border-radius: 12px;
    border: 1px solid #E5D9CD;
    background: #FFFFFF;
}
.image-preview-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-top: 1px solid #E5D9CD;
    background: rgba(255, 255, 255, 0.52);
}
.image-preview-name {
    color: #3D2E24;
    font-size: 0.86rem;
    font-weight: 600;
    word-break: break-word;
}
.image-preview-hint {
    color: #7A6555;
    font-size: 0.76rem;
    text-align: right;
}
.chat-suggestion-card {
    margin: 0.75em 0 1em;
    background: #F2DDD6;
    border: 1px solid #EACFC5;
    border-radius: 16px;
    padding: 1em 1.1em 1.05em;
}
.chat-suggestion-title {
    color: #A8432C;
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 0.25em;
}
.chat-suggestion-copy {
    color: #7A6555;
    font-size: 0.8rem;
    line-height: 1.55;
    margin-bottom: 0.65em;
}
.chat-suggestion-list {
    margin: 0;
    padding-left: 1.05em;
    color: #3D2E24;
}
.chat-suggestion-list li {
    margin: 0.28em 0;
    line-height: 1.55;
    font-size: 0.86rem;
}
.lightbox-stage {
    max-height: 74vh;
    overflow: auto;
    padding: 14px;
    border-radius: 16px;
    border: 1px solid #E5D9CD;
    background:
        radial-gradient(circle at top left, rgba(196, 85, 58, 0.10), transparent 26%),
        linear-gradient(180deg, #FCF7F1 0%, #F5EDE4 100%);
}
.lightbox-stage img {
    max-width: none;
    height: auto;
    display: block;
    border-radius: 14px;
    border: 1px solid #E5D9CD;
    background: #FFFFFF;
    box-shadow: 0 8px 24px rgba(61, 46, 36, 0.10);
}

/* ── Meta badges ───────────────────────────────────────────────────────── */
.meta-badge {
    display: inline-block;
    background: #F5EDE4;
    border: 1px solid #E5D9CD;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: #7A6555;
    margin-right: 8px;
}

/* ── Sidebar branding ──────────────────────────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 0.5em 0 0.8em;
}
.sidebar-brand .logo {
    font-size: 2.0rem;
    font-weight: 800;
    color: #C4553A;
    letter-spacing: -0.02em;
}
.sidebar-brand .tagline {
    font-size: 0.82rem;
    color: #A89585;
    margin-top: 0.3em;
    letter-spacing: 0.03em;
}

/* ── Pipeline step indicator ───────────────────────────────────────────── */
.step-indicator {
    display: flex;
    flex-direction: column;
    gap: 0;
    padding: 0.2em 0;
}
.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-circle {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
    border: 2px solid #E5D9CD;
    color: #A89585;
    background: #FDF8F3;
}
.step-circle.done {
    background: #5B8C5A;
    border-color: #5B8C5A;
    color: #FFFFFF;
}
.step-circle.active {
    background: #C4553A;
    border-color: #C4553A;
    color: #FFFFFF;
}
.step-label {
    font-size: 0.80rem;
    color: #A89585;
}
.step-label.done { color: #5B8C5A; font-weight: 600; }
.step-label.active { color: #C4553A; font-weight: 600; }
.step-connector {
    width: 2px;
    height: 14px;
    margin-left: 11px;
    background: #E5D9CD;
}
.step-connector.done { background: #5B8C5A; }

/* ── File uploader icon color ─────────────────────────────────────────── */
[data-testid="stFileUploader"] svg {
    color: #C4553A !important;
    fill: #C4553A !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    border-color: #EACFC5;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #C4553A;
    background: #F2DDD6;
}

/* ── Global accent override ──────────────────────────────────────────── */
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #FAF4ED !important;
}

/* Buttons */
.stButton > button[kind="primary"],
button[kind="primary"] {
    background-color: #C4553A !important;
    border-color: #C4553A !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"],
button[kind="primary"] {
    transition: filter 0.2s, transform 0.15s, box-shadow 0.2s;
}
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(196, 85, 58, 0.25) !important;
}
.stButton > button[kind="primary"]:active,
button[kind="primary"]:active {
    filter: brightness(0.95) !important;
    transform: translateY(0);
}
.stButton > button[kind="primary"]:focus:not(:active) {
    box-shadow: 0 0 0 0.2rem rgba(196, 85, 58, 0.35) !important;
}

/* Download button */
.stDownloadButton > button {
    border-color: #C4553A !important;
    color: #C4553A !important;
}
.stDownloadButton > button:hover {
    background-color: #F2DDD6 !important;
    border-color: #A8432C !important;
    color: #A8432C !important;
}

/* Toggle */
[data-testid="stToggle"] label span[data-checked="true"] {
    background-color: #C4553A !important;
}

/* Tabs — active underline */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #C4553A !important;
    height: 3px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #C4553A !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="false"] {
    color: #A89585 !important;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #EDE4DA !important;
}

/* Chat input focus */
[data-testid="stChatInput"] textarea:focus {
    border-color: #C4553A !important;
    box-shadow: 0 0 0 1px #C4553A !important;
}
[data-testid="stChatInputSubmitButton"] button,
[data-testid="stChatInputSubmitButton"] svg {
    color: #C4553A !important;
    fill: #C4553A !important;
}

/* Chat message assistant avatar */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background-color: #C4553A !important;
}

/* Status widget */
[data-testid="stStatus"] summary svg {
    color: #C4553A !important;
}
details[data-testid="stStatus"][open] > summary {
    border-color: #C4553A !important;
}

/* Spinner */
.stSpinner > div > div {
    border-top-color: #C4553A !important;
}

/* Selectbox / multiselect */
[data-baseweb="select"] [data-baseweb="input"] {
    border-color: #E5D9CD !important;
}
[data-baseweb="select"] [data-baseweb="input"]:focus-within {
    border-color: #C4553A !important;
    box-shadow: 0 0 0 1px #C4553A !important;
}
[data-baseweb="select"] svg {
    color: #C4553A !important;
    fill: #C4553A !important;
}
[data-baseweb="popover"] li[aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #F2DDD6 !important;
    color: #A8432C !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background-color: #F0E6DA !important;
}
[data-testid="stSelectbox"] > div > div {
    border-color: #E5D9CD !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #C4553A !important;
    box-shadow: 0 0 0 1px #C4553A !important;
}

/* Expander */
[data-testid="stExpander"] summary:hover svg {
    color: #C4553A !important;
}

/* Text input / textarea focus */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #C4553A !important;
    box-shadow: 0 0 0 1px #C4553A !important;
}

/* Link & anchor color */
a { color: #C4553A !important; }
a:hover { color: #A8432C !important; }

/* Toast */
[data-testid="stToast"] {
    border-left-color: #C4553A !important;
}

/* ── Panel height sync ───────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
}

/* ── Chat UI — Claude.ai inspired ──────────────────────────────────────── */
/* Hide default avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }

/* User message: right-aligned bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
    gap: 0;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: #EAD5C8;
    border-radius: 18px 18px 4px 18px;
    padding: 0.55em 1em;
    max-width: 78%;
    margin-left: auto;
    color: #3D2E24;
    font-size: 0.9rem;
}

/* Assistant message: no bubble, clean text */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: transparent;
    border: none;
    padding: 0.3em 0;
    font-size: 0.9rem;
    color: #3D2E24;
}
</style>
"""

# Block type → badge color mapping
_BLOCK_TYPE_COLORS: dict[str, str] = {
    "text": "#5B8C5A",
    "code": "#5A7B8C",
    "table": "#C4883A",
    "figure": "#C4553A",
    "equation": "#5A5470",
    "heading": "#4A6342",
}
_DEFAULT_BADGE_COLOR = "#A89585"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_metric_card(value: str | int, label: str) -> str:
    """Return HTML for a styled metric card."""
    return (
        f'<div class="metric-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


def _render_block_type_badges(type_counts: dict[str, int]) -> str:
    """Return HTML for colored block type badges."""
    badges = []
    for btype, count in sorted(type_counts.items()):
        color = _BLOCK_TYPE_COLORS.get(btype, _DEFAULT_BADGE_COLOR)
        bg = color + "1A"  # ~10% opacity hex
        badges.append(
            f'<span class="block-badge" style="background:{bg};color:{color};border:1px solid {color}33;">'
            f"{btype} {count}"
            f"</span>"
        )
    return " ".join(badges)


def _render_concept_tags(concepts: list[str]) -> str:
    """Return HTML for concept tag badges with 5-color rotation."""
    tags = "".join(
        f'<span class="concept-tag concept-tag-{i % 5}">{c}</span>'
        for i, c in enumerate(concepts)
    )
    return f'<div style="margin: 0.4em 0 0.8em;">{tags}</div>'


# ---------------------------------------------------------------------------
# Note markdown normalization (LLM output → clean markdown string)
# ---------------------------------------------------------------------------
def _normalize_note_markdown(note_md: str | dict) -> str:
    """Convert note_markdown to readable markdown.

    When the LLM returns a JSON structure instead of plain markdown,
    this function converts it into proper markdown.
    Plain markdown strings pass through unchanged.
    """
    if isinstance(note_md, dict):
        return _dict_to_markdown(note_md)

    if not isinstance(note_md, str):
        return str(note_md)

    stripped = note_md.strip()
    if not stripped.startswith("{"):
        return note_md

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return _regex_extract_sections(stripped)

    if isinstance(data, dict):
        return _dict_to_markdown(data)
    return note_md


def _dict_to_markdown(data: dict) -> str:
    """Convert a dict to markdown, handling multiple LLM output patterns."""
    # Case 1: {"sections": [{"title": ..., "content": ...}]}
    sections = data.get("sections", [])
    if isinstance(sections, list) and sections:
        lines: list[str] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = section.get("title", "")
            content = section.get("content", "")
            if title:
                lines.append(f"## {title}")
                lines.append("")
            if content:
                lines.append(content)
                lines.append("")
        if lines:
            return "\n".join(lines).strip()

    # Case 2: {"section1": {"title": ..., "content": ...}, ...}
    if data and all(
        isinstance(v, dict) and "title" in v and "content" in v for v in data.values()
    ):
        lines = []
        for section in data.values():
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(str(section["content"]))
            lines.append("")
        return "\n".join(lines).strip()

    # Case 3 & 4: heading keys / arbitrary dict
    lines = []
    for key, value in data.items():
        if str(key).strip().lower() in _NOTE_INTERNAL_KEYS:
            continue
        if key.startswith("#"):
            lines.append(key)
        else:
            lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, str):
            lines.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    lines.append(f"- {item}")
                elif isinstance(item, dict):
                    parts = [f"**{k}:** {v}" for k, v in item.items()]
                    lines.append(f"- {' | '.join(parts)}")
                else:
                    lines.append(f"- {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                lines.append(f"- **{k}:** {v}")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines).strip()


def _regex_extract_sections(text: str) -> str:
    """Fallback: extract 'key': 'value' pairs via regex when json.loads fails."""
    internal_keys = _NOTE_INTERNAL_KEYS
    pairs = re.findall(r'"(#{1,3}\s+[^"]+?)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not pairs:
        pairs = re.findall(r'"([^"]+?)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not pairs:
        return text

    lines: list[str] = []
    for key, value in pairs:
        if key.strip().lower() in internal_keys:
            continue
        if key.startswith("#"):
            lines.append(key)
        else:
            lines.append(f"## {key}")
        lines.append("")
        cleaned = value.replace("\\n", "\n").replace('\\"', '"')
        lines.append(cleaned)
        lines.append("")
    return "\n".join(lines).strip()


def _downshift_headings(md_text: str) -> str:
    """Shift markdown headings down by 2 levels, skipping content inside code fences."""
    _HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)")
    in_code_fence = False
    result_lines = []
    for line in md_text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
        if not in_code_fence:
            line = _HEADING_RE.sub(lambda m: "#" * min(len(m.group(1)) + 2, 6) + " " + m.group(2), line)
        result_lines.append(line)
    return "\n".join(result_lines)


def _render_note_html(note_md: str) -> str:
    """Convert note markdown to scoped HTML for consistent rendering."""
    clean_md = _downshift_headings(_normalize_note_markdown(note_md))
    html_body = md_lib.markdown(clean_md, extensions=["fenced_code", "tables", "nl2br"])
    return f'<div class="note-wrapper"><div class="note-content">\n{html_body}\n</div></div>'


def _render_note_section_html(note_md: str) -> str:
    """Render a single section without card border (used in edit mode)."""
    clean_md = _downshift_headings(_normalize_note_markdown(note_md))
    html_body = md_lib.markdown(clean_md, extensions=["fenced_code", "tables", "nl2br"])
    return f'<div class="note-section"><div class="note-content">\n{html_body}\n</div></div>'


def _image_data_uri(image_bytes: bytes, suffix: str) -> str:
    """Encode image bytes into a browser-safe data URI."""
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix.lower(), "image/png")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _render_image_preview_card(
    file_name: str, image_bytes: bytes, suffix: str, zoom_percent: int
) -> str:
    """Render the uploaded image inside a styled preview card."""
    image_uri = _image_data_uri(image_bytes, suffix)
    safe_name = html.escape(file_name)
    return (
        '<div class="image-note-card">'
        '<div class="image-note-lead">'
        "<div>"
        '<div class="image-note-kicker">Image-First Study Mode</div>'
        f'<h3 class="image-note-title">{safe_name}</h3>'
        '<p class="image-note-copy">'
        "이 이미지는 학습 노트 대신 원본 화면을 보면서 바로 질문하는 흐름에 최적화되어 있습니다."
        "</p>"
        "</div>"
        '<div class="image-note-pill">Q&A Ready</div>'
        "</div>"
        '<div class="image-preview-shell">'
        '<div class="image-preview-toolbar">'
        '<div class="image-preview-toolbar-label">원본 미리보기</div>'
        f'<div class="image-preview-toolbar-value">{zoom_percent}% 확대</div>'
        "</div>"
        '<div class="image-preview-scroll">'
        '<div class="image-preview-frame">'
        f'<img src="{image_uri}" alt="{safe_name}" style="width: {zoom_percent}%;" />'
        "</div>"
        "</div>"
        '<div class="image-preview-meta">'
        f'<div class="image-preview-name">{safe_name}</div>'
        '<div class="image-preview-hint">확대 후 스크롤해서 작은 글씨와 세부 영역을 확인하세요.</div>'
        "</div>"
        "</div>"
        "</div>"
    )


def _render_lightbox_image(
    file_name: str, image_bytes: bytes, suffix: str, zoom_percent: int
) -> str:
    """Render the full-size image viewer content."""
    image_uri = _image_data_uri(image_bytes, suffix)
    safe_name = html.escape(file_name)
    return (
        '<div class="lightbox-stage">'
        f'<img src="{image_uri}" alt="{safe_name}" style="width: {zoom_percent}%;" />'
        "</div>"
    )


@st.dialog("원본 이미지 확대 보기", width="large")
def _show_image_lightbox(
    file_name: str, image_bytes: bytes, suffix: str, key_prefix: str
) -> None:
    """Open a large modal image viewer for detailed inspection."""
    st.caption(
        "작은 글씨나 세부 UI 요소를 확인할 수 있도록 크게 띄웠습니다. 확대 후 스크롤해서 살펴보세요."
    )
    zoom_percent = st.slider(
        "확대 배율",
        min_value=100,
        max_value=400,
        value=220,
        step=10,
        key=f"{key_prefix}_lightbox_zoom",
    )
    st.markdown(
        _render_lightbox_image(file_name, image_bytes, suffix, zoom_percent),
        unsafe_allow_html=True,
    )
    if st.button("닫기", use_container_width=True, key=f"{key_prefix}_lightbox_close"):
        st.rerun()


def _extract_image_topic(doc: "Document") -> str | None:
    """Extract a short topic hint from image-derived blocks."""
    ignored_prefixes = ("title:", "type:", "components:", "relationships:", "flow:")
    rejected_openers = (
        "이 코드는",
        "이 이미지는",
        "이 화면은",
        "this code",
        "this image",
        "this screen",
    )

    for block in doc.blocks:
        candidates: list[str] = []
        if block.metadata.caption:
            candidates.append(block.metadata.caption.split("|", 1)[0].strip())
        candidates.extend(line.strip() for line in block.content.splitlines())

        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" -:|")
            if not cleaned:
                continue
            if cleaned.lower().startswith(ignored_prefixes):
                continue
            if cleaned.lower().startswith(rejected_openers):
                continue
            if len(cleaned) < 6:
                continue
            if len(cleaned.split()) > 6:
                continue
            if any(token in cleaned for token in ("'", '"', "`", "{", "}", "=", ";")):
                continue
            if cleaned.endswith(
                ("입니다.", "입니다", "합니다.", "합니다", "다.", "요.")
            ):
                continue
            if len(cleaned) > 32:
                continue
            return cleaned
    return None


def _is_code_like_topic(topic: str | None) -> bool:
    """Heuristically detect code identifiers such as classes/functions/snippets."""
    if not topic:
        return False

    stripped = topic.strip()
    if stripped.startswith(("class ", "def ", "function ", "interface ")):
        return True
    if any(token in stripped for token in ("()", "::", "->")):
        return True
    if "_" in stripped:
        return True
    words = stripped.split()
    if any(word[:1].isupper() and not word.isupper() for word in words):
        return True
    return False


def _build_image_question_suggestions(doc: "Document") -> list[str]:
    """Create image-aware starter questions from parsed image hints."""
    topic = _extract_image_topic(doc)
    topic_is_code_like = _is_code_like_topic(topic)
    first_block = doc.blocks[0] if doc.blocks else None
    image_type = None
    if first_block and first_block.metadata.image_type is not None:
        image_type = first_block.metadata.image_type.value
    plain_text = " ".join(
        filter(
            None,
            [
                *(block.content for block in doc.blocks),
                *(block.metadata.caption or "" for block in doc.blocks),
            ],
        )
    ).lower()

    suggestions: list[str] = []
    if any(
        keyword in plain_text
        for keyword in (
            "guardrail",
            "가드레일",
            "prompt injection",
            "프롬프트 인젝션",
            "jailbreak",
            "정책",
            "policy",
        )
    ):
        if topic:
            if topic_is_code_like:
                suggestions.append(
                    f"'{topic}'가 guardrail 흐름에서 어떤 역할을 하는지 설명해줘"
                )
            else:
                suggestions.append(f"이 이미지에서 '{topic}'가 왜 중요한지 설명해줘")
        suggestions.extend(
            [
                "이 자료에서 guardrail 핵심 원칙이 뭐야?",
                "어떤 위험이나 공격 시나리오를 막으려는 거야?",
                "구현 단계나 체크 포인트를 순서대로 설명해줘",
            ]
        )
    elif image_type == "code_screenshot":
        if topic:
            if topic_is_code_like:
                suggestions.append(f"'{topic}'의 역할을 쉽게 설명해줘")
            else:
                suggestions.append(f"'{topic}' 코드가 하는 일을 쉽게 설명해줘")
        suggestions.extend(
            [
                "이 코드의 실행 흐름을 단계별로 설명해줘",
                "버그나 위험해 보이는 부분이 어디야?",
                "핵심 함수와 변수 역할을 요약해줘",
            ]
        )
    elif image_type == "diagram":
        if topic:
            if topic_is_code_like:
                suggestions.append(
                    f"'{topic}'가 다이어그램에서 어디에 연결되는지 설명해줘"
                )
            else:
                suggestions.append(
                    f"'{topic}' 흐름을 처음 보는 사람도 이해하게 설명해줘"
                )
        suggestions.extend(
            [
                "각 구성요소 역할을 순서대로 설명해줘",
                "입력부터 출력까지 어떻게 연결되는지 설명해줘",
                "병목이나 위험 신호가 있는 부분이 어디야?",
            ]
        )
    elif image_type == "text_capture":
        if topic:
            if topic_is_code_like:
                suggestions.append(f"'{topic}'가 문맥상 무엇을 가리키는지 설명해줘")
            else:
                suggestions.append(f"'{topic}'가 핵심적으로 말하는 내용을 요약해줘")
        suggestions.extend(
            [
                "중요 개념 3가지를 뽑아서 설명해줘",
                "이 내용을 초보자도 이해하게 풀어서 설명해줘",
                "시험이나 면접 대비용으로 핵심만 정리해줘",
            ]
        )
    elif image_type == "equation":
        if topic:
            if topic_is_code_like:
                suggestions.append(f"'{topic}' 표기가 무엇을 뜻하는지 설명해줘")
            else:
                suggestions.append(f"'{topic}' 수식이 의미하는 바를 설명해줘")
        suggestions.extend(
            [
                "이 수식의 각 항이 무슨 뜻인지 알려줘",
                "언제 쓰는 식인지 예시와 함께 설명해줘",
                "직관적으로 이해할 수 있게 풀어줘",
            ]
        )
    else:
        if topic:
            if topic_is_code_like:
                suggestions.append(
                    f"'{topic}'가 이 화면에서 어떤 역할을 하는지 설명해줘"
                )
            else:
                suggestions.append(f"이 이미지에서 '{topic}'가 왜 중요한지 설명해줘")
        suggestions.extend(
            [
                "이 화면의 핵심 개념이 뭐야?",
                "중요해 보이는 부분을 위에서 아래로 설명해줘",
                "질문해야 할 포인트를 먼저 짚어줘",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) == 3:
            break
    return deduped


def _build_document_question_suggestions(doc: "Document", result: dict) -> list[str]:
    """Create note/document-aware starter questions for PDF and ipynb uploads."""
    title = (result.get("title") or "").strip()
    summary = (result.get("summary") or "").strip()
    key_concepts = [
        str(item).strip()
        for item in (result.get("key_concepts") or [])
        if str(item).strip()
    ]
    type_counts = Counter(block.type.value for block in doc.blocks)

    suggestions: list[str] = []
    if key_concepts:
        primary = key_concepts[0]
        suggestions.append(f"'{primary}' 개념을 예시와 함께 설명해줘")
        suggestions.append(f"이 자료에서 '{primary}'가 왜 중요한지 설명해줘")
    if title and title != doc.source:
        suggestions.append(f"'{title}' 문서를 3줄로 요약해줘")
    elif summary:
        suggestions.append("이 자료의 핵심 내용을 3줄로 요약해줘")
    else:
        suggestions.append("이 자료의 핵심 개념 3가지를 뽑아줘")

    if doc.format.value == "ipynb" or type_counts.get("code", 0) > 0:
        suggestions.append("코드 흐름을 위에서 아래 순서대로 설명해줘")
    elif type_counts.get("table", 0) > 0:
        suggestions.append("표에 나온 정보를 어떻게 해석하면 되는지 설명해줘")
    else:
        suggestions.append("중요한 부분부터 읽는 순서를 추천해줘")

    suggestions.append("시험이나 면접 대비용으로 꼭 기억할 포인트만 정리해줘")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) == 3:
            break
    return deduped


def _render_chat_suggestion_card(suggestions: list[str], copy_text: str) -> str:
    """Render a compact Q&A suggestion card above the chat panel."""
    items = "".join(f"<li>{html.escape(question)}</li>" for question in suggestions)
    return (
        '<div class="chat-suggestion-card">'
        '<div class="chat-suggestion-title">질문 예시</div>'
        f'<div class="chat-suggestion-copy">{html.escape(copy_text)}</div>'
        f'<ol class="chat-suggestion-list">{items}</ol>'
        "</div>"
    )


def _serialize_source_blocks(source_blocks: list) -> list[dict]:
    """Convert retrieved source blocks into session-safe dicts."""
    serialized: list[dict] = []
    for block in source_blocks:
        if isinstance(block, dict):
            serialized.append(block)
        else:
            serialized.append(block.model_dump())
    return serialized


def _render_source_block_expanders(source_blocks: list[dict]) -> None:
    """Render deduplicated source block expanders under one assistant answer."""
    if not source_blocks:
        return

    st.caption("참조 블록")
    seen: set[str] = set()
    for src in source_blocks:
        page = src.get("page")
        cell_index = src.get("cell_index")
        loc = (
            f"page {page}"
            if page is not None
            else (f"cell {cell_index}" if cell_index is not None else "")
        )
        dedup_key = f"{src.get('source', '')}:{loc}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        exp_label = (
            f"📄 {src.get('source', 'unknown')}"
            + (f" · {loc}" if loc else "")
            + f"  `{src.get('block_type', '')}`"
        )
        with st.expander(exp_label, expanded=False):
            st.caption(f"block_order: {src.get('block_order', 0)}")
            st.text(str(src.get("content_preview", "")))


def _render_note_editor_panel(result: dict, llm_model: str, chat_height: int) -> None:
    """Render the note editor chatbot panel (✏️ 노트 수정 tab).

    Shows a section selectbox (synced with inline ✏️ clicks) and a chat interface
    for multi-turn natural-language section editing.  The edited section is shown
    as a preview in the chat; the note is only updated when the user clicks [적용].
    """
    note_markdown = result.get("note_markdown", "")
    raw_md = _normalize_note_markdown(note_markdown) if note_markdown else ""
    sections = _split_note_sections(raw_md) if raw_md else []
    section_headings = [h for h, _ in sections if h]

    if not section_headings:
        st.info("수정할 섹션이 없습니다. 먼저 노트를 생성해주세요.")
        return

    st.markdown("#### ✏️ 노트 수정")
    st.caption(f"섹션을 선택하고 수정 지시를 입력하세요 · LLM: `{llm_model}`")

    # Section selectbox — auto-updated when user clicks inline ✏️.
    # We set st.session_state["edit_section_selectbox"] directly in the ✏️ handler
    # because Streamlit ignores index= when the key is already in session state.
    selected = st.selectbox(
        "수정할 섹션",
        options=section_headings,
        key="edit_section_selectbox",
    )
    st.session_state["selected_edit_section"] = selected

    # Show current section preview (first 120 chars)
    section_map = {h: b for h, b in sections}
    preview = section_map.get(selected, "")[:120].replace("\n", " ")
    if preview:
        st.caption(f"현재 내용: {preview}…")

    # Session state init
    if "edit_chat_messages" not in st.session_state:
        st.session_state["edit_chat_messages"] = []
    if "note_edit_history" not in st.session_state:
        st.session_state["note_edit_history"] = []

    # Chat container
    chat_container = st.container(height=chat_height)
    with chat_container:
        for msg in st.session_state["edit_chat_messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("is_preview"):
                    st.markdown("**수정 미리보기:**")
                st.markdown(msg["content"])

        # Process pending edit
        if _pending_edit := st.session_state.pop("_pending_edit", None):
            instruction = _pending_edit["instruction"]
            section_heading = _pending_edit["section"]
            with st.chat_message("assistant"):
                with st.spinner("수정 중..."):
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["edit_chat_messages"]
                        if m["role"] in ("user", "assistant")
                        and not m.get("is_preview")
                    ]
                    edit_result: NoteEditResult = edit_section(
                        full_markdown=raw_md,
                        section_heading=section_heading,
                        instruction=instruction,
                        model=llm_model,
                        history=history,
                    )
            if edit_result.success:
                st.session_state["edit_pending_markdown"] = edit_result.edited_markdown
                preview_content = f"**{section_heading}** 섹션 수정 결과:\n\n{edit_result.edited_section_body}"
                st.session_state["edit_chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": preview_content,
                        "is_preview": True,
                    }
                )
            else:
                error_msg = f"수정 실패: {edit_result.error}"
                st.session_state["edit_chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "is_preview": False,
                    }
                )
            st.rerun()

    # Apply / Cancel buttons (only show when there's a pending edit)
    if st.session_state.get("edit_pending_markdown"):
        col_apply, col_cancel = st.columns(2)
        with col_apply:
            if st.button("✅ 적용", use_container_width=True, type="primary"):
                # Push undo entry (max 10)
                history_entry = {
                    "markdown_before": raw_md,
                    "section": st.session_state.get("selected_edit_section", ""),
                    "instruction": st.session_state["edit_chat_messages"][-2]["content"]
                    if len(st.session_state["edit_chat_messages"]) >= 2
                    else "",
                }
                edit_history = st.session_state["note_edit_history"]
                edit_history.append(history_entry)
                if len(edit_history) > 10:
                    edit_history.pop(0)
                # Update note
                result["note_markdown"] = st.session_state.pop("edit_pending_markdown")
                st.session_state["edit_chat_messages"] = []
                st.rerun()
        with col_cancel:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state.pop("edit_pending_markdown", None)
                st.session_state["edit_chat_messages"] = []
                st.rerun()

    # Edit instruction input
    if edit_instruction := st.chat_input(
        "수정 지시를 입력하세요 (예: 코드 예제 추가해줘)"
    ):
        st.session_state["edit_chat_messages"].append(
            {"role": "user", "content": edit_instruction}
        )
        st.session_state["_pending_edit"] = {
            "instruction": edit_instruction,
            "section": st.session_state.get(
                "selected_edit_section", section_headings[0]
            ),
        }
        st.rerun()


def _render_qa_panel(
    doc: "Document", result: dict, llm_model: str, is_image: bool, chat_height: int
) -> None:
    """Render the Q&A area with optional image-specific starter prompts."""
    st.markdown("#### 💬 Q&A")
    qa_subject = "이미지" if is_image else "문서"
    st.caption(f"{qa_subject}에 대해 질문하세요 · LLM: `{llm_model}`")
    if is_image:
        st.markdown(
            _render_chat_suggestion_card(
                _build_image_question_suggestions(doc),
                "이미지 유형과 추출된 구조를 기준으로 바로 물어볼 수 있는 질문입니다.",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _render_chat_suggestion_card(
                _build_document_question_suggestions(doc, result),
                "노트와 문서 구조를 기준으로 바로 이어서 물어볼 수 있는 질문입니다.",
            ),
            unsafe_allow_html=True,
        )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    chat_container = st.container(height=chat_height)
    with chat_container:
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_source_block_expanders(msg.get("source_blocks", []))

        if _pending := st.session_state.pop("_pending_chat", None):
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    try:
                        _chat_result = rag_query(
                            _pending, model=llm_model, document_id=doc.id
                        )
                        _reply = _chat_result.answer
                        _source_blocks = _serialize_source_blocks(
                            _chat_result.source_blocks
                        )
                    except Exception as _exc:
                        _reply = f"오류가 발생했습니다: {_exc}"
                        _source_blocks = []
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": _reply,
                    "source_blocks": _source_blocks,
                }
            )
            st.rerun()

    if user_input := st.chat_input("질문을 입력하세요"):
        st.session_state["chat_messages"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state["_pending_chat"] = user_input
        st.rerun()


# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="CatchUp - 학습자료 자동 구조화",
    page_icon="🍅",
    layout="wide",
)

# Inject global CSS
st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="logo">🍅 CatchUp</div>'
        '<div class="tagline">학습자료 자동 구조화 파이프라인</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.expander("모델 설정", expanded=True):
        vlm_model = st.selectbox("VLM 모델 (이미지)", options=SUPPORTED_MODELS, index=0)
        llm_model = st.selectbox(
            "LLM 모델 (노트/Q&A)", options=SUPPORTED_LLM_MODELS, index=0
        )

# ===================================================================
# FILE UPLOAD
# ===================================================================
uploaded_file = st.file_uploader(
    "학습자료를 업로드하세요",
    type=["pdf", "ipynb", "png", "jpg", "jpeg", "webp"],
    help="PDF, Jupyter Notebook, 이미지 (PNG/JPG/JPEG/WEBP) 지원",
)

suffix = os.path.splitext(uploaded_file.name)[1] if uploaded_file else ""
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
is_image = suffix.lower() in IMAGE_SUFFIXES

with st.sidebar:
    st.markdown("---")

    # Pipeline step indicator
    st.caption("파이프라인")
    steps = st.session_state.get("_pipeline_steps", {})
    step_defs = [("upload", "파일 업로드"), ("parse", "파싱")]
    if not is_image:
        step_defs.append(("note", "노트 생성"))
    html_parts = ['<div class="step-indicator">']
    for i, (step_name, label) in enumerate(step_defs):
        done = steps.get(step_name, False)
        # determine active: first undone step after at least one done step
        prev_done = i == 0 or steps.get(step_defs[i - 1][0], False)
        active = (
            not done
            and prev_done
            and i > 0
            or (i == 0 and not done and any(steps.values()))
        )
        circle_cls = "done" if done else ("active" if active else "")
        label_cls = circle_cls
        icon = "✓" if done else str(i + 1)
        html_parts.append(
            f'<div class="step-row">'
            f'<div class="step-circle {circle_cls}">{icon}</div>'
            f'<div class="step-label {label_cls}">{label}</div>'
            f"</div>"
        )
        if i < len(step_defs) - 1:
            connector_cls = "done" if done else ""
            html_parts.append(f'<div class="step-connector {connector_cls}"></div>')
    html_parts.append("</div>")
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

    st.markdown("---")

    if st.button("캐시 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.caption("Built with Streamlit + OpenAI / Anthropic / Google")

if uploaded_file is None:
    # Landing state
    st.markdown("#### 지원 형식")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("📄 **PDF**\nDocling 기반 구조 추출")
    with c2:
        st.markdown("📓 **Jupyter Notebook**\n코드/텍스트 블록 분석")
    with c3:
        st.markdown("🖼️ **이미지**\nVLM 기반 분류 + 분석")
    st.stop()

# Update pipeline
st.session_state.setdefault("_pipeline_steps", {})["upload"] = True

st.markdown(f"**{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

# ===================================================================
# ANALYSIS PIPELINE
# ===================================================================
file_bytes = uploaded_file.read()
file_hash = hashlib.sha256(file_bytes).hexdigest()
cache_key = f"result_{file_hash}_{vlm_model}_{llm_model}"
doc_cache_key = f"doc_{file_hash}_{vlm_model}"

if not st.button("분석 시작", type="primary", use_container_width=False):
    if cache_key not in st.session_state:
        st.stop()

if cache_key in st.session_state:
    doc = st.session_state[doc_cache_key]
    result = st.session_state[cache_key]
    is_image = st.session_state.get(f"is_image_{cache_key}", False)
    if not st.session_state.get(f"_toast_shown_{cache_key}"):
        st.toast("캐시된 결과를 불러왔습니다", icon="⚡")
        st.session_state[f"_toast_shown_{cache_key}"] = True
else:
    tmp_path: str | None = None
    doc = None

    with st.status("분석 진행 중...", expanded=True) as status:
        # Step 1: Parse
        parse_status_label = (
            "1/1 — 이미지 분석 중..." if is_image else "1/2 — 파일 파싱 중..."
        )
        status.update(label=parse_status_label, state="running")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            if suffix.lower() == ".pdf":
                doc = parse_pdf(tmp_path)
            elif suffix.lower() == ".ipynb":
                doc = parse_ipynb(tmp_path)
            else:
                try:
                    doc = parse_image(tmp_path, model=vlm_model)
                except Exception as exc:
                    if any(kw in str(exc).lower() for kw in ("api", "key", "auth")):
                        st.error("API 키를 .env에 설정해주세요")
                    else:
                        st.error(f"이미지 파싱 실패: {exc}")
                    st.stop()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if doc is None:
            st.error("파일 처리 중 오류가 발생했습니다.")
            st.stop()

        doc.source = uploaded_file.name
        st.session_state.setdefault("_pipeline_steps", {})["parse"] = True

        parse_failed = not doc.blocks or "parse_failed" in doc.metadata.tags
        if parse_failed:
            status.update(label="파싱 실패", state="error")
            st.stop()

        st.write(f"파싱 완료 — {doc.block_count}개 블록 추출")

        # Step 2: Generate note
        if not is_image:
            status.update(label="2/2 — 학습 노트 생성 중...", state="running")
            try:
                result = generate_note(doc, model=llm_model)
            except Exception as exc:
                st.error(f"노트 생성 실패: {exc}")
                st.stop()

            st.session_state.setdefault("_pipeline_steps", {})["note"] = True
        else:
            result = {}
        status.update(label="분석 완료!", state="complete", expanded=False)

    # Cache
    st.session_state[doc_cache_key] = doc
    st.session_state[cache_key] = result
    st.session_state[f"is_image_{cache_key}"] = is_image

# ===================================================================
# RAG INDEXING — run once per document (session-cached)
# ===================================================================
_indexed_key = f"indexed_{doc.id}"
if not st.session_state.get(_indexed_key):
    try:
        index_document(doc)
        st.session_state[_indexed_key] = True
    except Exception as _idx_exc:
        st.warning(f"RAG 인덱싱 실패: {_idx_exc}")

# ===================================================================
# RESULTS — Navigation
# ===================================================================
active_tab = st.radio(
    "탭 선택",
    ["📊 파싱 결과", "📝 학습 노트"],
    horizontal=True,
    key="active_tab",
    label_visibility="collapsed",
)

# ─── Tab 1: Parsing results ───────────────────────────────────────────
if active_tab == "📊 파싱 결과":
    type_counts = Counter(block.type.value for block in doc.blocks)

    # Metric cards
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            _render_metric_card(doc.block_count, "총 블록 수"), unsafe_allow_html=True
        )
    with mc2:
        st.markdown(
            _render_metric_card(doc.format.value.upper(), "문서 형식"),
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            _render_metric_card(doc.status.value, "처리 상태"), unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Block type badges
    st.markdown("**블록 구성**")
    st.markdown(_render_block_type_badges(type_counts), unsafe_allow_html=True)

    # Block detail expander
    with st.expander("블록 상세 보기", expanded=False):
        for i, block in enumerate(doc.blocks[:30]):
            content_preview = block.content[:120].replace("\n", " ")
            st.text(f"[{i}] {block.type.value}: {content_preview}")
        if len(doc.blocks) > 30:
            st.caption(f"... 외 {len(doc.blocks) - 30}개 블록")

# ─── Tab 2: Study note + Q&A (side by side) ──────────────────────────
if active_tab == "📝 학습 노트":
    title = result.get("title") or doc.source
    summary = result.get("summary")
    note_markdown = result.get("note_markdown")
    key_concepts = result.get("key_concepts") or []
    difficulty = result.get("difficulty_level")
    read_time = result.get("estimated_read_time_min")
    errors = result.get("errors") or []

    if errors:
        for err in errors:
            st.warning(str(err))

    if not is_image:
        # Title
        st.markdown(f"### {title}")

        # Meta badges
        meta_html = ""
        if difficulty:
            meta_html += f'<span class="meta-badge">난이도: {difficulty}</span>'
        if read_time:
            meta_html += f'<span class="meta-badge">읽기 시간: {read_time}분</span>'
        if meta_html:
            st.markdown(meta_html, unsafe_allow_html=True)

        # Summary card
        if summary:
            st.markdown(
                f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True
            )

        # Key concepts as tags
        if key_concepts:
            st.markdown("**핵심 개념**")
            st.markdown(_render_concept_tags(key_concepts), unsafe_allow_html=True)

    col_content, col_chat = st.columns([1.12, 1], gap="large")

    with col_content:
        if is_image:
            preview_controls, preview_action = st.columns([3, 1])
            with preview_controls:
                zoom_percent = st.slider(
                    "이미지 확대",
                    min_value=100,
                    max_value=300,
                    value=160,
                    step=10,
                    key=f"image_zoom_{cache_key}",
                    help="작은 글씨가 있는 이미지라면 확대 후 스크롤해서 확인하세요.",
                )
            with preview_action:
                st.caption("")
                if st.button(
                    "전체화면 보기",
                    use_container_width=True,
                    key=f"open_image_lightbox_{cache_key}",
                ):
                    _show_image_lightbox(
                        uploaded_file.name, file_bytes, suffix, cache_key
                    )
            st.markdown(
                _render_image_preview_card(
                    uploaded_file.name, file_bytes, suffix, zoom_percent
                ),
                unsafe_allow_html=True,
            )
        elif note_markdown:
            raw_md = _normalize_note_markdown(note_markdown)
            file_stem = Path(doc.source).stem

            # Toolbar row: edit toggle + undo button
            toolbar_col, undo_col = st.columns([3, 1])
            with toolbar_col:
                edit_mode = st.toggle(
                    "✏️ 편집 모드", value=False, key="note_edit_toggle"
                )
            with undo_col:
                edit_history = st.session_state.get("note_edit_history", [])
                if st.button(
                    "↩ 되돌리기",
                    disabled=len(edit_history) == 0,
                    use_container_width=True,
                    key="note_undo_btn",
                ):
                    last = edit_history.pop()
                    result["note_markdown"] = last["markdown_before"]
                    st.rerun()

            # Shared panel height — note scroll container and chat_height are
            # calibrated so total visual heights approximately match.
            # Note: toolbar(~55px) + PANEL_H + download(~45px) ≈ chat header(~155px) + chat_height
            _NOTE_PANEL_H = 700

            note_scroll = st.container(height=_NOTE_PANEL_H)
            with note_scroll:
                if edit_mode:
                    # Section-level edit mode: flow as one document, ✏️ per section
                    sections = _split_note_sections(raw_md)
                    first_rendered = True
                    for _sec_heading, _sec_body in sections:
                        content_md = (
                            f"{_sec_heading}\n\n{_sec_body}".strip() if _sec_heading else _sec_body
                        )
                        if not content_md:
                            continue
                        if not first_rendered:
                            st.markdown('<hr class="note-sep">', unsafe_allow_html=True)
                        st.markdown(_render_note_section_html(content_md), unsafe_allow_html=True)
                        if _sec_heading:
                            if st.button(
                                "✏️",
                                key=f"edit_sec_{_sec_heading}",
                                help=f"'{_sec_heading}' 섹션 수정",
                            ):
                                st.session_state["selected_edit_section"] = _sec_heading
                                # Directly set selectbox key — Streamlit ignores
                                # index= when the key already exists in session state
                                st.session_state["edit_section_selectbox"] = _sec_heading
                                st.session_state["active_right_panel"] = "✏️ 노트 수정"
                                st.rerun()
                        first_rendered = False
                else:
                    # Pure read view: single note block (Claude Artifact MD preview style)
                    st.markdown(_render_note_html(raw_md), unsafe_allow_html=True)
            download_md = raw_md

            full_md = f"# {title}\n\n{download_md}"
            st.download_button(
                label="📥 마크다운 다운로드",
                data=full_md,
                file_name=f"{file_stem}_note.md",
                mime="text/markdown",
            )
        else:
            st.info("노트 내용이 없습니다.")

    with col_chat:
        if is_image:
            # Image mode: Q&A only, no note editor
            _render_qa_panel(doc, result, llm_model, is_image=True, chat_height=640)
        else:
            # Non-image mode: Q&A tab + Note editor tab
            right_panel = st.radio(
                "right_panel",
                ["💬 Q&A", "✏️ 노트 수정"],
                horizontal=True,
                key="active_right_panel",
                label_visibility="collapsed",
            )
            if right_panel == "💬 Q&A":
                _render_qa_panel(
                    doc, result, llm_model, is_image=False, chat_height=600
                )
            else:
                _render_note_editor_panel(result, llm_model, chat_height=420)
