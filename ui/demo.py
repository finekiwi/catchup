"""Streamlit demo for CatchUp: upload → parse → note generation."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root is on sys.path when launched via `streamlit run ui/demo.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import markdown as md_lib  # noqa: E402
import streamlit as st  # noqa: E402
import pyperclip  # noqa: E402

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from models.document import Document

from llm.note_editor import NoteEditResult, edit_section  # noqa: E402
from llm.note_editor import _split_sections as _split_note_sections  # noqa: E402
from llm.note_generator import SUPPORTED_LLM_MODELS, generate_note_sectioned  # noqa: E402
from utils.logging import langfuse_session  # noqa: E402
from parsers.image_parser import parse_image  # noqa: E402
from parsers.ipynb_parser import parse_ipynb  # noqa: E402
from parsers.pdf_parser import parse_pdf  # noqa: E402
from db.sqlite import (  # noqa: E402
    delete_document,
    get_document,
    get_note,
    list_documents,
    list_notes_for_document,
    save_document,
    save_note,
)
from rag import (  # noqa: E402
    delete_document_index,
    has_document_vectors,
    index_document,
    query as rag_query,
)
from vlm.client import SUPPORTED_MODELS  # noqa: E402

# Keys injected by the LLM schema but not rendered as note content
_NOTE_INTERNAL_KEYS: frozenset[str] = frozenset(
    {"schema_version", "confidence", "errors", "starter prompts", "_note_result_version"}
)
_ANALYSIS_CACHE_VERSION = "v2"

# Compiled regex for heading downshift — hoisted to avoid repeated compilation
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)")


def _result_cache_key(file_hash: str, vlm_model: str, llm_model: str) -> str:
    """Return the session cache key for saved note output."""
    return f"result_{_ANALYSIS_CACHE_VERSION}_{file_hash}_{vlm_model}_{llm_model}"


def _doc_cache_key(file_hash: str, vlm_model: str) -> str:
    """Return the session cache key for parsed documents."""
    return f"doc_{_ANALYSIS_CACHE_VERSION}_{file_hash}_{vlm_model}"

# ---------------------------------------------------------------------------
# Global CSS — light theme styles for all custom components
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """\
<style>
/* === CatchUp Warm & Soft Palette === */

/* (chat_input bottom pinning handled via Python spacer injection) */

/* ── Follow-up question pill buttons ─────────────────────────────────── */
.followup-btns > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] button {
    background: #F5EDE4 !important;
    border: 1px solid #E5D9CD !important;
    color: #7A6555 !important;
    font-size: 0.83rem !important;
    min-height: 32px !important;
    border-radius: 16px !important;
    text-align: left !important;
    padding: 4px 14px !important;
}
.followup-btns > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] button:hover {
    background: #EACFC5 !important;
    border-color: #C4553A !important;
    color: #A8432C !important;
}

/* ── Compact download trigger button ─────────────────────────────────── */
.dl-compact > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] button {
    min-height: 36px !important;
    padding: 4px 10px !important;
    font-size: 0.85rem !important;
    border-color: #C4553A !important;
    color: #C4553A !important;
}
.dl-compact > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] button:hover {
    background: #F2DDD6 !important;
    border-color: #A8432C !important;
    color: #A8432C !important;
}

/* ── Section action buttons (✏️ + ↩): side by side, no gap ────────────── */
.sec-action-btns > div[data-testid="stVerticalBlock"] {
    flex-direction: row !important;
    gap: 2px !important;
    align-items: center !important;
}
.sec-action-btns button {
    min-height: 30px !important;
    padding: 2px 8px !important;
    line-height: 1 !important;
}

/* ── Note / chat column layout breathing room ──────────────────────────── */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stVerticalBlockBorderWrapper"] {
    margin-right: 1rem;
}

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
    margin: 0.8em auto 0;
    max-width: 860px;
    box-shadow: 0 1px 4px rgba(61,46,36,0.06);
}
/* Edit mode: sections flow as one document, no per-section card */
.note-section {
    background: #FDF8F3;
    padding: 0.5em 2.5em;
    max-width: 860px;
    margin: 0 auto;
}
hr.note-sep {
    border: none;
    border-top: 1px solid #EDE3D9;
    margin: 0.2em auto;
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
    min-height: 38px !important;
    width: 100% !important;
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
/* Note panel: warm background fills full column height */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    background: #FDF8F3;
    border-radius: 14px;
    padding: 0 0.5rem;
}

/* ── Chat UI — Claude.ai inspired ──────────────────────────────────────── */
.chat-user-msg-wrap {
    display: flex;
    justify-content: flex-end;
    margin: 2em 0 0.8em 0;
}
.chat-user-msg {
    background: #EAD5C8;
    border-radius: 18px 18px 4px 18px;
    padding: 0.6em 1.1em;
    max-width: 78%;
    display: inline-block;
    color: #3D2E24;
    font-size: 0.9rem;
    line-height: 1.6;
    word-break: break-word;
}
.chat-assistant-msg {
    background: transparent;
    padding: 0.4em 0 2em;
    font-size: 0.9rem;
    color: #3D2E24;
    line-height: 1.7;
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
    in_code_fence = False
    result_lines = []
    for line in md_text.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
        if not in_code_fence:
            line = _HEADING_RE.sub(
                lambda m: "#" * min(len(m.group(1)) + 2, 6) + " " + m.group(2), line
            )
        result_lines.append(line)
    return "\n".join(result_lines)


def _deduplicate_headings(md: str) -> str:
    """Remove duplicate consecutive ## headings with no body between them.

    The LLM occasionally emits the same ## heading twice in a row (once with
    no body, once with content). This pass keeps only the last occurrence of
    each run of identical headings, preserving the one that has content after it.
    """
    lines = md.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## ") and not line.startswith("### "):
            # Collect consecutive identical headings with only blank lines between
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or lines[j] == line):
                if lines[j] == line:
                    i = j  # advance to the last duplicate
                j += 1
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def _render_note_html(note_md: str) -> str:
    """Convert note markdown to scoped HTML for consistent rendering."""
    clean_md = _downshift_headings(_normalize_note_markdown(_deduplicate_headings(note_md)))
    # nl2br intentionally excluded: it injects <br> inside <pre><code> blocks which
    # causes Streamlit's react-markdown code renderer to receive an array of nodes
    # instead of a plain string, producing [object Object] for each line.
    html_body = md_lib.markdown(clean_md, extensions=["fenced_code", "tables"])
    return f'<div class="note-wrapper"><div class="note-content">\n{html_body}\n</div></div>'


def _inline_figure_body_start_page(doc: "Document") -> int | None:
    """Return the first body page that should be eligible for inline note figures."""
    if getattr(getattr(doc, "format", None), "value", None) != "pdf":
        return None

    try:
        from llm.section_splitter import extract_sections, group_blocks_by_section

        sections = extract_sections(doc)
        if not sections:
            return None
        grouped_sections = group_blocks_by_section(doc, sections)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Inline figure page filter skipped: %s", exc)
        return None

    for section in grouped_sections:
        section_pages = [
            block.metadata.page
            for block in section.blocks
            if block.metadata.page is not None
        ]
        if section_pages:
            return min(section_pages)

    return None


def _filter_inline_figure_blocks(doc: "Document", fig_blocks: list) -> list:
    """Drop front-matter image blocks so they do not render in the study note."""
    first_body_page = _inline_figure_body_start_page(doc)
    if first_body_page is None:
        return fig_blocks

    filtered = [
        block
        for block in fig_blocks
        if block.metadata.page is None or block.metadata.page >= first_body_page
    ]
    skipped = len(fig_blocks) - len(filtered)
    if skipped > 0:
        LOGGER.info(
            "Filtered %d inline figure blocks before first body page %d",
            skipped,
            first_body_page,
        )
    return filtered


def _render_note_with_figures(raw_md: str, fig_blocks: list) -> None:
    """Render note markdown section-by-section, injecting figure images between sections.

    Each figure is placed after the section whose page position best matches
    the figure's page number. Same-page figures are spread across adjacent sections
    to avoid stacking.

    Args:
        raw_md: Note markdown text (without title prefix).
        fig_blocks: List of Block objects with type==FIGURE and image_path set.
    """
    sections = _split_note_sections(raw_md)  # list of (heading, body) tuples
    n = len(sections)
    if n == 0:
        st.markdown(_render_note_html(raw_md), unsafe_allow_html=True)
        return

    pages = [b.metadata.page for b in fig_blocks]
    valid_pages = [p for p in pages if p is not None]
    page_min = min(valid_pages) if valid_pages else 1
    page_max = max(valid_pages) if valid_pages else 1
    page_range = max(page_max - page_min, 1)

    section_figs: dict[int, list] = defaultdict(list)
    page_counters: dict = defaultdict(int)
    for b in fig_blocks:
        p = b.metadata.page if b.metadata.page is not None else page_max
        ratio = (p - page_min) / page_range
        base_idx = min(int(ratio * n), n - 1)
        count = page_counters[b.metadata.page]
        page_counters[b.metadata.page] += 1
        idx = min(base_idx + count, n - 1)
        section_figs[idx].append(b)

    for i, (heading, body) in enumerate(sections):
        # Skip heading-only sections with no body and no figures — these are
        # LLM duplicates where the same ## heading was emitted twice in a row.
        if heading and not body and not section_figs.get(i):
            continue
        section_md = f"{heading}\n\n{body}".strip() if heading else body
        if section_md:
            st.markdown(_render_note_section_html(section_md), unsafe_allow_html=True)
        for b in section_figs.get(i, []):
            img_path = b.image_path
            if img_path and Path(img_path).exists():
                caption = b.metadata.caption or ""
                _, _col_img, _ = st.columns([1, 4, 1])
                with _col_img:
                    st.image(img_path, caption=caption, width="stretch")


def _render_note_section_html(note_md: str) -> str:
    """Render a single section without card border (used in edit mode)."""
    clean_md = _downshift_headings(_normalize_note_markdown(_deduplicate_headings(note_md)))
    html_body = md_lib.markdown(clean_md, extensions=["fenced_code", "tables"])
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


def _render_qa_notice(msg: str, *, is_loading: bool = False) -> None:
    """Render a palette-styled notice in the Q&A panel.

    Uses info palette (#DDE8ED / #5A7B8C) for loading states and
    warning palette (#F5EBDB / #C4883A) for disabled / missing-vector states.
    """
    if is_loading:
        bg, border, color, icon = "#DDE8ED", "#5A7B8C", "#3A5260", "⏳"
    else:
        bg, border, color, icon = "#F5EBDB", "#C4883A", "#7A5020", "ℹ️"
    st.markdown(
        f'<div style="background:{bg};border-left:3px solid {border};border-radius:8px;'
        f'padding:12px 16px;font-size:0.85rem;color:{color};line-height:1.5;">'
        f"{icon}&nbsp;{html.escape(msg)}</div>",
        unsafe_allow_html=True,
    )


def _render_source_block_expanders(source_blocks: list[dict]) -> None:
    """Render deduplicated source block expanders under one assistant answer."""
    if not source_blocks:
        return

    # 1. figure sources with a valid image_path: render inline above expanders
    seen_images: set[str] = set()
    for src in source_blocks:
        img_path = src.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue
        if img_path in seen_images:
            continue
        seen_images.add(img_path)
        page = src.get("page")
        caption = f"그림 (page {page})" if page is not None else "그림"
        st.image(img_path, caption=caption, use_container_width=True)

    # 2. all source blocks: collapsed expanders (image already rendered above)
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


def _parse_followup_suggestions(answer: str) -> tuple[str, list[str]]:
    """Split answer text from follow-up suggestion block appended by the RAG prompt.

    Returns (clean_answer, suggestions) where suggestions is a list of up to 3 strings.
    If no block is found, suggestions is empty and answer is returned unchanged.
    """
    match = re.search(r"\n?---SUGGESTIONS---\n(.*?)\n---END---", answer, re.DOTALL)
    if not match:
        return answer, []
    clean = answer[: match.start()].rstrip()
    raw_lines = [
        line.strip() for line in match.group(1).strip().splitlines() if line.strip()
    ]
    suggestions = [re.sub(r"^\d+[.)]\s*", "", line) for line in raw_lines]
    return clean, [s for s in suggestions if s][:3]


def _replace_section_body(full_md: str, heading_idx: int, new_body: str) -> str:
    """Return full_md with the body of the nth named section replaced by new_body.

    heading_idx is a 0-based index into the sub-list of sections that have headings,
    matching the position returned by _render_note_editor_panel's section selectbox.
    Using an index (rather than heading text) avoids ambiguity when a document
    contains duplicate ## headings.
    """
    sections = _split_note_sections(full_md)
    parts: list[str] = []
    named_count = 0
    for h, b in sections:
        if h:
            body = new_body if named_count == heading_idx else b
            named_count += 1
        else:
            body = b
        parts.append(f"{h}\n\n{body}".strip() if h else body)
    return "\n\n".join(parts)


def _render_note_editor_panel(
    result: dict, llm_model: str, chat_height: int, doc_key: str = ""
) -> None:
    """Render the note editor panel (✏️ 노트 수정 tab).

    Supports two editing modes selectable via a toggle:
    - 💬 챗봇: section-level chat-driven editing with preview/apply flow.
    - ⌨️ 직접 편집: full-note text area for direct markdown editing.

    Args:
        result: The note generation result dict held in session state.
        llm_model: Default LLM model from the sidebar.
        chat_height: Height in pixels for the chat container.
        doc_key: A stable identifier for the current document (e.g. doc.id).
                 Used to namespace session-state keys so that two documents
                 with identical section headings (e.g. "## 개요") never share
                 chat history or undo stacks.
    """
    note_markdown = result.get("note_markdown", "")
    raw_md = _normalize_note_markdown(note_markdown) if note_markdown else ""
    sections = _split_note_sections(raw_md) if raw_md else []
    section_headings = [h for h, _ in sections if h]

    if not section_headings:
        st.info("수정할 섹션이 없습니다. 먼저 노트를 생성해주세요.")
        return

    # Session-state key prefix scoped to this document
    _sk = f"editor_{doc_key}_" if doc_key else "editor_"
    edit_mode_options = ["💬 챗봇", "⌨️ 직접 편집"]
    edit_method_key = f"{_sk}note_editor_method"
    edit_method_pref_key = f"{_sk}note_editor_method_pref"
    edit_method_default = st.session_state.get(
        edit_method_pref_key, edit_mode_options[0]
    )
    edit_method_index = (
        edit_mode_options.index(edit_method_default)
        if edit_method_default in edit_mode_options
        else 0
    )

    # Mode toggle
    edit_method = st.radio(
        "edit_method",
        edit_mode_options,
        horizontal=True,
        index=edit_method_index,
        key=edit_method_key,
        label_visibility="collapsed",
    )
    st.session_state[edit_method_pref_key] = edit_method

    # ── Direct markdown editing mode ────────────────────────────────────────
    if edit_method == "⌨️ 직접 편집":
        undo_key = f"{_sk}note_section_undo"
        if undo_key not in st.session_state:
            st.session_state[undo_key] = {}

        edited = st.text_area(
            "마크다운 직접 편집",
            value=raw_md,
            height=chat_height,
            key=f"{_sk}direct_edit_textarea",
            label_visibility="collapsed",
        )
        if st.button(
            "✅ 저장",
            use_container_width=True,
            type="primary",
            key=f"{_sk}direct_edit_save",
        ):
            if edited != raw_md:
                # Store full note under special key for direct edits
                undo_stack = st.session_state[undo_key].setdefault("__direct__", [])
                undo_stack.append(raw_md)
                if len(undo_stack) > 10:
                    undo_stack.pop(0)
                result["note_markdown"] = edited
                st.session_state["_note_dirty"] = True
                st.rerun()
        return

    # ── Chatbot editing mode ─────────────────────────────────────────────────
    _editor_cap_col, _editor_model_col = st.columns([3, 2])
    with _editor_cap_col:
        st.caption("섹션을 선택하고 수정 지시를 입력하세요")
    with _editor_model_col:
        editor_model_pref_key = f"{_sk}note_editor_model_pref"
        editor_model_default = st.session_state.get(editor_model_pref_key, llm_model)
        editor_model = st.selectbox(
            "노트 수정 모델",
            options=SUPPORTED_LLM_MODELS,
            index=(
                SUPPORTED_LLM_MODELS.index(editor_model_default)
                if editor_model_default in SUPPORTED_LLM_MODELS
                else 0
            ),
            key=f"{_sk}note_editor_model_select",
            label_visibility="collapsed",
        )
        st.session_state[editor_model_pref_key] = editor_model

    # Use index-based selectbox so duplicate headings (e.g. two "## 개요") are distinguished.
    selected_section_default = st.session_state.get(
        f"{_sk}selected_edit_section_idx", 0
    )
    selected_section_default = min(selected_section_default, len(section_headings) - 1)
    selected_section_idx: int = st.selectbox(  # type: ignore[assignment]
        "수정할 섹션",
        options=list(range(len(section_headings))),
        format_func=lambda i: section_headings[i],
        index=selected_section_default,
        key=f"{_sk}edit_section_selectbox",
    )
    selected = section_headings[selected_section_idx]
    st.session_state[f"{_sk}selected_edit_section"] = selected
    st.session_state[f"{_sk}selected_edit_section_idx"] = selected_section_idx

    # Session state init — chat/undo keyed by integer section index (not heading text)
    chat_key = f"{_sk}edit_chat_messages"
    undo_key = f"{_sk}note_section_undo"
    # Scoped pending-preview keys prevent cross-document leakage
    pending_md_key = f"{_sk}edit_pending_markdown"
    pending_sec_key = f"{_sk}edit_pending_section"
    pending_idx_key = f"{_sk}edit_pending_section_idx"
    if chat_key not in st.session_state or isinstance(st.session_state[chat_key], list):
        st.session_state[chat_key] = {}  # {section_idx: [msg, ...]}
    if undo_key not in st.session_state:
        st.session_state[undo_key] = {}

    # Ensure current section has a message list (keyed by integer index)
    sec_msgs: list = st.session_state[chat_key].setdefault(selected_section_idx, [])

    # Chat container
    chat_container = st.container(height=chat_height)
    with chat_container:
        for msg in sec_msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user-msg-wrap"><div class="chat-user-msg">{html.escape(msg["content"])}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                # Use st.markdown directly — double-processing (md_lib → HTML → st.markdown)
                # causes react-markdown to re-parse * as emphasis nodes → [object Object]
                st.markdown(msg["content"])

        # Process pending edit
        if _pending_edit := st.session_state.pop("_pending_edit", None):
            instruction = _pending_edit["instruction"]
            section_heading = _pending_edit["section"]
            section_idx = _pending_edit["section_idx"]
            cur_msgs = st.session_state[chat_key].setdefault(section_idx, [])
            with st.spinner("수정 중..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in cur_msgs
                    if m["role"] in ("user", "assistant") and not m.get("is_preview")
                ]
                edit_result: NoteEditResult = edit_section(
                    full_markdown=raw_md,
                    section_heading=section_heading,
                    instruction=instruction,
                    model=editor_model,
                    history=history,
                    document_id=doc_key if doc_key else None,
                )
            if edit_result.success:
                st.session_state[pending_md_key] = edit_result.edited_markdown
                st.session_state[pending_sec_key] = section_heading
                st.session_state[pending_idx_key] = section_idx
                preview_content = f"**{section_heading}** 섹션 수정 결과:\n\n{edit_result.edited_section_body}"
                cur_msgs.append(
                    {
                        "role": "assistant",
                        "content": preview_content,
                        "is_preview": True,
                    }
                )
            else:
                cur_msgs.append(
                    {
                        "role": "assistant",
                        "content": f"수정 실패: {edit_result.error}",
                        "is_preview": False,
                    }
                )
            st.rerun()

        # Apply / Cancel buttons inside container (keeps height stable)
        if st.session_state.get(pending_md_key):
            col_apply, col_cancel = st.columns(2)
            with col_apply:
                if st.button("✅ 적용", use_container_width=True, type="primary"):
                    pending_idx = st.session_state.get(pending_idx_key, 0)
                    # Push current body of the pending section to undo stack (max 10)
                    named_sections = [
                        (h, b) for h, b in _split_note_sections(raw_md) if h
                    ]
                    undo_body = (
                        named_sections[pending_idx][1]
                        if pending_idx < len(named_sections)
                        else ""
                    )
                    undo_stack = st.session_state[undo_key].setdefault(pending_idx, [])
                    undo_stack.append(undo_body)
                    if len(undo_stack) > 10:
                        undo_stack.pop(0)
                    result["note_markdown"] = st.session_state.pop(pending_md_key)
                    st.session_state.pop(pending_sec_key, None)
                    st.session_state.pop(pending_idx_key, None)
                    st.session_state["_note_dirty"] = True
                    st.rerun()
            with col_cancel:
                if st.button("❌ 취소", use_container_width=True):
                    st.session_state.pop(pending_md_key, None)
                    st.session_state.pop(pending_sec_key, None)
                    st.session_state.pop(pending_idx_key, None)
                    st.rerun()

    # Chat input OUTSIDE the gray box
    if edit_instruction := st.chat_input(
        "수정 지시를 입력하세요 (예: 코드 예제 추가해줘)"
    ):
        _cur_idx = st.session_state.get(f"{_sk}selected_edit_section_idx", 0)
        st.session_state[chat_key].setdefault(_cur_idx, []).append(
            {"role": "user", "content": edit_instruction}
        )
        st.session_state["_pending_edit"] = {
            "instruction": edit_instruction,
            "section": st.session_state.get(
                f"{_sk}selected_edit_section", section_headings[0]
            ),
            "section_idx": _cur_idx,
        }
        st.rerun()


def _render_qa_panel(
    doc: "Document", result: dict, llm_model: str, is_image: bool, chat_height: int
) -> None:
    """Render the Q&A area with optional image-specific starter prompts."""
    st.markdown("#### 💬 Q&A")
    qa_subject = "이미지" if is_image else "문서"
    _qa_cap_col, _qa_model_col = st.columns([3, 2])
    with _qa_cap_col:
        st.caption(f"{qa_subject}에 대해 질문하세요")
        use_rewrite = st.checkbox(
            "Query Rewriting",
            value=False,
            key="qa_use_rewrite",
            help="한국어 구어체나 약어를 영문 기술 용어로 확장해 검색 정확도를 높입니다.",
        )
    with _qa_model_col:
        qa_model_pref_key = "qa_model_pref"
        qa_model_default = st.session_state.get(qa_model_pref_key, llm_model)
        qa_llm_model = st.selectbox(
            "Q&A 모델",
            options=SUPPORTED_LLM_MODELS,
            index=(
                SUPPORTED_LLM_MODELS.index(qa_model_default)
                if qa_model_default in SUPPORTED_LLM_MODELS
                else 0
            ),
            key="qa_model_select",
            label_visibility="collapsed",
        )
        st.session_state[qa_model_pref_key] = qa_llm_model

    _qa_chat_key = f"chat_messages_{doc.id}"
    if _qa_chat_key not in st.session_state:
        st.session_state[_qa_chat_key] = []

    # Suggestion card shown above the gray box only while no messages
    if not st.session_state[_qa_chat_key]:
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

    chat_container = st.container(height=chat_height)
    with chat_container:
        msgs = st.session_state[_qa_chat_key]
        for msg in msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user-msg-wrap"><div class="chat-user-msg">{html.escape(msg["content"])}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                # Use st.markdown directly — avoid double-processing that causes [object Object]
                st.markdown(msg["content"])
                _render_source_block_expanders(msg.get("source_blocks", []))

        # Follow-up question buttons after the last assistant message
        if (
            msgs
            and msgs[-1]["role"] == "assistant"
            and not st.session_state.get("_pending_chat")
        ):
            followups = msgs[-1].get("followup_suggestions", [])
            if followups:
                last_idx = len(msgs) - 1
                st.markdown('<div class="followup-btns">', unsafe_allow_html=True)
                for fq_idx, fq in enumerate(followups):
                    if st.button(
                        fq, key=f"fq_{last_idx}_{fq_idx}", use_container_width=True
                    ):
                        st.session_state[_qa_chat_key].append(
                            {"role": "user", "content": fq}
                        )
                        st.session_state["_pending_chat"] = fq
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        # Thinking indicator rendered as last item inside the container.
        # st.markdown (HTML) does NOT trigger the two-box split; only st.spinner does.
        if st.session_state.get("_pending_chat"):
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;color:#7A6555;'
                'font-size:0.88rem;padding:6px 4px">'
                '<div style="width:14px;height:14px;border:2px solid #E5D9CD;'
                "border-top-color:#C4553A;border-radius:50%;"
                'animation:qa-spin 0.8s linear infinite;flex-shrink:0"></div>'
                "생각 중...</div>"
                "<style>@keyframes qa-spin{to{transform:rotate(360deg)}}</style>",
                unsafe_allow_html=True,
            )

    # Chat input rendered BEFORE the LLM blocking call so it always renders on
    # every cycle. Streamlit shows a stale-placeholder gray box for any widget
    # that hasn't been reached yet when the script blocks — moving chat_input
    # above the LLM call prevents the second gray box on the first message.
    if user_input := st.chat_input("질문을 입력하세요"):
        st.session_state[_qa_chat_key].append({"role": "user", "content": user_input})
        st.session_state["_pending_chat"] = user_input
        st.rerun()

    if _pending := st.session_state.pop("_pending_chat", None):
        try:
            _chat_result = rag_query(
                _pending,
                model=qa_llm_model,
                document_id=doc.id,
                top_k=8,
                rewrite=use_rewrite,
            )
            _raw_reply = _chat_result.answer
            _reply, _followups = _parse_followup_suggestions(_raw_reply)
            _source_blocks = _serialize_source_blocks(_chat_result.source_blocks)
        except Exception as _exc:
            _reply = f"오류가 발생했습니다: {_exc}"
            _followups = []
            _source_blocks = []
        st.session_state[_qa_chat_key].append(
            {
                "role": "assistant",
                "content": _reply,
                "source_blocks": _source_blocks,
                "followup_suggestions": _followups,
            }
        )
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
        _LLM_HINTS = {
            "gpt-4.1-nano": "빠르고 저렴 (기본값)",
            "gpt-4o-mini": "균형",
            "gpt-4.1-mini": "고품질 · 중간 비용",
            "gpt-4o": "최고 품질 · 고비용",
            "gpt-5-nano": "빠르고 저렴",
            "claude-haiku-4-5-20251001": "빠르고 저렴 (Anthropic)",
            "claude-sonnet-4-6": "고품질 (Anthropic)",
            "gemini-3-flash-preview": "빠르고 저렴 (Google)",
            "gemini-3.1-flash-lite-preview": "최저 비용 (Google)",
            "gemini-3.1-pro-preview": "고품질 (Google)",
        }
        if hint := _LLM_HINTS.get(llm_model):
            st.caption(hint)
        output_language = st.selectbox(
            "노트 언어",
            options=["ko", "en"],
            format_func=lambda x: "한국어" if x == "ko" else "English",
            index=0,
            key="output_language_select",
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

    # ── Recent documents library ──────────────────────────────────────
    _FORMAT_ICON = {"pdf": "📄", "ipynb": "📓", "image": "🖼️"}
    _lib_docs = list_documents(limit=10)
    if _lib_docs:
        st.caption("📚 최근 문서")
        for _ld in _lib_docs:
            _icon = _FORMAT_ICON.get(_ld.format.value, "📄")
            _date_str = _ld.created_at.strftime("%m/%d")
            _label = f"{_icon} {_ld.source}  ({_date_str})"
            _col_load, _col_del = st.columns([5, 1])
            with _col_load:
                if st.button(_label, key=f"lib_{_ld.id}", use_container_width=True):
                    st.session_state["_library_load"] = {"doc_id": _ld.id}
                    st.rerun()
            with _col_del:
                if st.button("🗑️", key=f"lib_del_{_ld.id}", help=f"{_ld.source} 삭제"):
                    # Evict session_state note caches for this document before DB delete
                    for _nr in list_notes_for_document(_ld.id):
                        _evict_key = _result_cache_key(
                            _nr["file_hash"], _nr["vlm_model"], _nr["llm_model"]
                        )
                        st.session_state.pop(_evict_key, None)
                        st.session_state.pop(
                            _doc_cache_key(_nr["file_hash"], _nr["vlm_model"]), None
                        )
                        st.session_state.pop(f"is_image_{_evict_key}", None)
                        st.session_state.pop(f"_toast_shown_{_evict_key}", None)
                    st.session_state.pop(f"indexed_{_ld.id}", None)
                    # If the document being deleted is currently open in library mode,
                    # clear all library pointers so the next rerun returns to landing state
                    # rather than dereferencing missing session keys and crashing.
                    if (
                        st.session_state.get("_library_mode")
                        and st.session_state.get("_library_doc_id") == _ld.id
                    ):
                        for _k in (
                            "_library_mode",
                            "_library_doc_id",
                            "_library_cache_key",
                            "_library_doc_cache_key",
                            "_library_used_vlm",
                            "_library_used_llm",
                            "_library_file_hash",
                        ):
                            st.session_state.pop(_k, None)
                    delete_document_index(
                        _ld.id
                    )  # Remove ChromaDB vectors so re-upload re-indexes
                    delete_document(_ld.id)
                    st.rerun()
        st.markdown("---")

    if st.button("캐시 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.caption("Built with Streamlit + OpenAI / Anthropic / Google")

# ===================================================================
# LIBRARY RESTORATION — load a previously saved document from sidebar
# ===================================================================
_lib_mode = st.session_state.get("_library_mode", False)

if "_library_load" in st.session_state:
    _lib_req = st.session_state.pop("_library_load")
    _lib_doc_id = _lib_req["doc_id"]
    # 1. Try matching current sidebar model selection
    _note_row = get_note(_lib_doc_id, vlm_model, llm_model)
    if _note_row is None:
        # 2. Fallback: most recent note for this document
        _all_notes = list_notes_for_document(_lib_doc_id)
        _note_row = _all_notes[0] if _all_notes else None
    _lib_doc = get_document(_lib_doc_id)

    if _note_row and _lib_doc:
        _fh = _note_row["file_hash"]
        _used_vlm = _note_row["vlm_model"]
        _used_llm = _note_row["llm_model"]
        _ck = _result_cache_key(_fh, _used_vlm, _used_llm)
        _dck = _doc_cache_key(_fh, _used_vlm)
        st.session_state[_ck] = _note_row["result"]
        st.session_state[_dck] = _lib_doc
        st.session_state[f"is_image_{_ck}"] = _note_row["is_image"]
        # Only mark as indexed when ChromaDB actually has vectors for this document.
        # An unconditional True would suppress re-indexing even when embeddings are absent
        # (e.g. after a crash or indexing failure), causing Q&A/note-editor to retrieve nothing.
        if has_document_vectors(_lib_doc.id):
            st.session_state[f"indexed_{_lib_doc.id}"] = True
        st.session_state["_library_mode"] = True
        st.session_state["_library_doc_id"] = _lib_doc.id
        st.session_state["_library_cache_key"] = _ck
        st.session_state["_library_doc_cache_key"] = _dck
        st.session_state["_library_used_vlm"] = _used_vlm
        st.session_state["_library_used_llm"] = _used_llm
        st.session_state["_library_file_hash"] = _fh
        _lib_mode = True

if uploaded_file is None and not _lib_mode:
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

if _lib_mode:
    # Library restoration: load from session_state cache
    cache_key = st.session_state["_library_cache_key"]
    doc_cache_key = st.session_state["_library_doc_cache_key"]
    doc: Document = st.session_state[doc_cache_key]
    result: dict = st.session_state[cache_key]
    is_image = st.session_state.get(f"is_image_{cache_key}", False)
    file_hash = st.session_state.get("_library_file_hash", "")
    _used_vlm = st.session_state.get("_library_used_vlm", vlm_model)
    _used_llm = st.session_state.get("_library_used_llm", llm_model)
    st.markdown(
        f'<div style="background:#F2DDD6;border:1px solid #EACFC5;border-radius:8px;'
        f'padding:0.5em 0.85em;color:#7A6555;font-size:0.82rem;margin-bottom:0.5em">'
        f'📚 라이브러리에서 로드됨 — <b style="color:#3D2E24">{html.escape(doc.source)}</b> '
        f"(모델: {html.escape(_used_vlm)} / {html.escape(_used_llm)})</div>",
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        # Exit library mode only when the user uploads a *different* file.
        # Compare the uploader's file hash against the library document's hash so
        # that a stale file_uploader value (still holding the same file across reruns)
        # does not cancel library mode on every subsequent rerun.
        _uploaded_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        _lib_file_hash = st.session_state.get("_library_file_hash", "")
        if _uploaded_hash != _lib_file_hash:
            for _k in (
                "_library_mode",
                "_library_doc_id",
                "_library_cache_key",
                "_library_doc_cache_key",
                "_library_used_vlm",
                "_library_used_llm",
                "_library_file_hash",
            ):
                st.session_state.pop(_k, None)
            st.rerun()
else:
    # Normal upload flow
    st.session_state.setdefault("_pipeline_steps", {})["upload"] = True
    st.markdown(f"**{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

# ===================================================================
# ANALYSIS PIPELINE (skipped in library mode)
# ===================================================================
if not _lib_mode:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    cache_key = _result_cache_key(file_hash, vlm_model, llm_model)
    doc_cache_key = _doc_cache_key(file_hash, vlm_model)

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

        with st.status("분석 진행 중...", expanded=True) as status, \
                langfuse_session(f"streamlit-{file_hash}"):
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
                    try:
                        from parsers.figure_enricher import enrich_pdf_figures
                        from utils.cache import save_cached_parse

                        doc = enrich_pdf_figures(
                            doc, vlm_model=vlm_model, file_path=tmp_path, language=output_language,
                        )
                        save_cached_parse(Path(tmp_path), doc)
                    except Exception as _enrich_exc:
                        LOGGER.warning("Figure enrichment skipped: %s", _enrich_exc)
                elif suffix.lower() == ".ipynb":
                    doc = parse_ipynb(tmp_path)
                else:
                    try:
                        doc = parse_image(tmp_path, model=vlm_model, language=output_language)
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
                    result = generate_note_sectioned(doc, model=llm_model, language=output_language)
                except Exception as exc:
                    st.error(f"노트 생성 실패: {exc}")
                    st.stop()

                st.session_state.setdefault("_pipeline_steps", {})["note"] = True
            else:
                result = {}
            status.update(label="분석 완료!", state="complete", expanded=True)

        # Cache
        st.session_state[doc_cache_key] = doc
        st.session_state[cache_key] = result
        st.session_state[f"is_image_{cache_key}"] = is_image

        # Persist to SQLite
        save_document(doc)
        save_note(doc.id, file_hash, result, vlm_model, llm_model, is_image)

# ===================================================================
# RESULTS — Study note (default view) + Chat panel
# ===================================================================
_indexed_key = f"indexed_{doc.id}"
# Library mode: check if vectors exist when blocks are unavailable locally
if not st.session_state.get(_indexed_key) and not doc.blocks:
    if has_document_vectors(doc.id):
        st.session_state[_indexed_key] = True
col_content, col_chat = st.columns([1.12, 1], gap="large")

with col_content:
    title = result.get("title") or doc.source
    summary = result.get("summary")
    note_markdown = result.get("note_markdown")
    key_concepts = result.get("key_concepts") or []
    difficulty = result.get("difficulty_level")
    read_time = result.get("estimated_read_time_min")
    errors = result.get("errors") or []

    if errors:
        for err in errors:
            st.info(str(err))

    if is_image:
        if _lib_mode:
            st.info(
                "이미지 미리보기는 원본 파일이 필요합니다. 파일을 다시 업로드하면 미리보기를 확인할 수 있습니다."
            )
        else:
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
        full_md = f"# {title}\n\n{raw_md}"

        # Toolbar row: edit toggle + export popover
        toolbar_col, export_col = st.columns([3, 1])
        with toolbar_col:
            edit_mode = st.toggle("✏️ 편집 모드", value=False, key="note_edit_toggle")
        with export_col:
            from utils.export import export_docx, export_markdown, export_pdf

            _export_fig_blocks = [b for b in doc.blocks if b.image_path and Path(b.image_path).exists()]
            with st.popover("📤 내보내기 ▾", use_container_width=True):
                # Markdown with inline base64 images
                _md_data = export_markdown(raw_md, title, _export_fig_blocks)
                st.download_button(
                    "📥 마크다운 (.md)",
                    data=_md_data,
                    file_name=f"{file_stem}_note.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                # PDF
                try:
                    _pdf_data = export_pdf(raw_md, title, _export_fig_blocks)
                    st.download_button(
                        "📄 PDF (.pdf)",
                        data=_pdf_data,
                        file_name=f"{file_stem}_note.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as _pdf_exc:
                    st.caption(f"PDF 생성 불가: {_pdf_exc}")
                # DOCX
                try:
                    _docx_data = export_docx(raw_md, title, _export_fig_blocks)
                    st.download_button(
                        "📝 Word (.docx)",
                        data=_docx_data,
                        file_name=f"{file_stem}_note.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except Exception as _docx_exc:
                    st.caption(f"DOCX 생성 불가: {_docx_exc}")
                # Clipboard (existing)
                if st.button(
                    "📋 클립보드 복사", use_container_width=True, key="copy_note_btn"
                ):
                    try:
                        pyperclip.copy(_md_data)
                        st.toast("📋 클립보드에 복사됐어요!")
                    except Exception:
                        st.toast("복사 실패")

        # Title + meta outside scroll container
        st.markdown(f"### {title}")
        meta_html = ""
        if difficulty:
            meta_html += f'<span class="meta-badge">난이도: {difficulty}</span>'
        if read_time:
            meta_html += f'<span class="meta-badge">읽기 시간: {read_time}분</span>'
        if meta_html:
            st.markdown(meta_html, unsafe_allow_html=True)
        if summary:
            st.markdown(
                f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True
            )
        if key_concepts:
            st.markdown("**핵심 개념**")
            st.markdown(_render_concept_tags(key_concepts), unsafe_allow_html=True)

        _NOTE_PANEL_H = 840
        note_scroll = st.container(height=_NOTE_PANEL_H)
        with note_scroll:
            if edit_mode:
                # Section-level edit mode: flow as one document, ✏️ per section
                sections = _split_note_sections(raw_md)
                first_rendered = True
                _named_sec_idx = 0  # tracks position among sections that have headings
                for _sec_heading, _sec_body in sections:
                    content_md = (
                        f"{_sec_heading}\n\n{_sec_body}".strip()
                        if _sec_heading
                        else _sec_body
                    )
                    if not content_md:
                        if _sec_heading:
                            _named_sec_idx += 1
                        continue
                    if not first_rendered:
                        st.markdown('<hr class="note-sep">', unsafe_allow_html=True)
                    st.markdown(
                        _render_note_section_html(content_md), unsafe_allow_html=True
                    )
                    if _sec_heading:
                        _cur_named_idx = _named_sec_idx  # capture for closure
                        _, _action_col = st.columns([6, 1])
                        with _action_col:
                            st.markdown(
                                '<div class="sec-action-btns">', unsafe_allow_html=True
                            )
                            if st.button(
                                "✏️",
                                key=f"edit_sec_{_cur_named_idx}",
                                help=f"'{_sec_heading}' 섹션 수정",
                            ):
                                _editor_sk = f"editor_{doc.id}_"
                                st.session_state[
                                    f"{_editor_sk}selected_edit_section"
                                ] = _sec_heading
                                st.session_state[
                                    f"{_editor_sk}selected_edit_section_idx"
                                ] = _cur_named_idx
                                st.session_state[
                                    f"{_editor_sk}edit_section_selectbox"
                                ] = _cur_named_idx
                                st.session_state["active_right_panel"] = "✏️ 노트 수정"
                                st.rerun()
                            _undo_key = f"editor_{doc.id}_note_section_undo"
                            _sec_stack = st.session_state.get(_undo_key, {}).get(
                                _cur_named_idx, []
                            )
                            if st.button(
                                "↩",
                                key=f"undo_sec_{_cur_named_idx}",
                                disabled=len(_sec_stack) == 0,
                                help=f"'{_sec_heading}' 섹션 되돌리기",
                            ):
                                prev_body = _sec_stack.pop()
                                result["note_markdown"] = _replace_section_body(
                                    raw_md, _cur_named_idx, prev_body
                                )
                                st.session_state["_note_dirty"] = True
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        _named_sec_idx += 1
                    first_rendered = False
            else:
                # Pure read view — inject figure images inline when available
                # Filter by image_path (not block type) because enrich_pdf_figures
                # may retype FIGURE blocks to CODE/TEXT after VLM classification.
                fig_blocks = [b for b in doc.blocks if b.image_path]
                fig_blocks = _filter_inline_figure_blocks(doc, fig_blocks)
                if fig_blocks:
                    _render_note_with_figures(raw_md, fig_blocks)
                else:
                    st.markdown(_render_note_html(raw_md), unsafe_allow_html=True)
    else:
        st.info("노트 내용이 없습니다.")

with col_chat:
    _qa_ready = bool(st.session_state.get(_indexed_key))
    _indexing_in_progress = not _qa_ready and bool(doc.blocks) and not is_image and not _lib_mode
    _qa_disabled_msg = (
        "Q&A 인덱싱 중입니다. 잠시 기다려 주세요..."
        if _indexing_in_progress else
        "Q&A를 사용하려면 문서 벡터가 필요합니다. "
        "문서를 다시 분석하거나 라이브러리 인덱스를 확인해주세요."
    )
    if is_image:
        # Image mode: Q&A only, no note editor
        if _qa_ready:
            _render_qa_panel(doc, result, llm_model, is_image=True, chat_height=960)
        else:
            _render_qa_notice(_qa_disabled_msg, is_loading=_indexing_in_progress)
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
            if _qa_ready:
                _render_qa_panel(
                    doc, result, llm_model, is_image=False, chat_height=960
                )
            else:
                _render_qa_notice(_qa_disabled_msg, is_loading=_indexing_in_progress)
        else:
            _render_note_editor_panel(
                result, llm_model, chat_height=837, doc_key=doc.id
            )

# ─── Lazy RAG indexing (runs after note + Q&A columns are rendered) ──
# Indexing is deferred here so the note appears immediately. The Q&A panel
# shows "인덱싱 중..." above until indexing completes and st.rerun() fires.
if not st.session_state.get(_indexed_key) and doc.blocks and not is_image and not _lib_mode:
    try:
        index_document(doc)
        st.session_state[_indexed_key] = True
    except Exception as _idx_exc:
        LOGGER.warning("RAG indexing failed: %s", _idx_exc)
        st.warning(f"RAG 인덱싱 실패: {_idx_exc}")
    st.rerun()

# ─── Persist dirty note edits to SQLite ──────────────────────────────
if st.session_state.pop("_note_dirty", False):
    # Determine the vlm/llm models used for this result
    if _lib_mode:
        _sv_vlm = st.session_state.get("_library_used_vlm", vlm_model)
        _sv_llm = st.session_state.get("_library_used_llm", llm_model)
        _sv_hash = st.session_state.get("_library_file_hash", "")
    else:
        _sv_vlm = vlm_model
        _sv_llm = llm_model
        _sv_hash = file_hash
    save_note(doc.id, _sv_hash, result, _sv_vlm, _sv_llm, is_image)

# ─── Pipeline details (collapsed by default) ─────────────────────────
st.markdown('<div style="height:96px"></div>', unsafe_allow_html=True)
if doc.blocks:
    with st.expander("🔧 파이프라인 상세", expanded=False):
        type_counts = Counter(block.type.value for block in doc.blocks)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                _render_metric_card(doc.block_count, "총 블록 수"),
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                _render_metric_card(doc.format.value.upper(), "문서 형식"),
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                _render_metric_card(doc.status.value, "처리 상태"),
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**블록 구성**")
        st.markdown(_render_block_type_badges(type_counts), unsafe_allow_html=True)
        with st.expander("블록 상세 보기", expanded=False):
            for i, block in enumerate(doc.blocks[:30]):
                content_preview = block.content[:120].replace("\n", " ")
                st.text(f"[{i}] {block.type.value}: {content_preview}")
            if len(doc.blocks) > 30:
                st.caption(f"... 외 {len(doc.blocks) - 30}개 블록")
