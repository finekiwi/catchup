"""Streamlit demo for CatchUp: upload → parse → note generation."""

from __future__ import annotations

import hashlib
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

from llm.note_generator import SUPPORTED_LLM_MODELS, generate_note  # noqa: E402
from parsers.image_parser import parse_image  # noqa: E402
from parsers.ipynb_parser import parse_ipynb  # noqa: E402
from parsers.pdf_parser import parse_pdf  # noqa: E402
from rag import index_document, query as rag_query  # noqa: E402
from vlm.client import SUPPORTED_MODELS  # noqa: E402

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
        f'</div>'
    )


def _render_block_type_badges(type_counts: dict[str, int]) -> str:
    """Return HTML for colored block type badges."""
    badges = []
    for btype, count in sorted(type_counts.items()):
        color = _BLOCK_TYPE_COLORS.get(btype, _DEFAULT_BADGE_COLOR)
        bg = color + "1A"  # ~10% opacity hex
        badges.append(
            f'<span class="block-badge" style="background:{bg};color:{color};border:1px solid {color}33;">'
            f'{btype} {count}'
            f'</span>'
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
        isinstance(v, dict) and "title" in v and "content" in v
        for v in data.values()
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
    pairs = re.findall(r'"(#{1,3}\s+[^"]+?)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not pairs:
        pairs = re.findall(r'"([^"]+?)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not pairs:
        return text

    lines: list[str] = []
    for key, value in pairs:
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
    """Shift markdown headings down by 2 levels (# -> ###, ## -> ####)."""

    def _shift(m: re.Match) -> str:
        hashes = m.group(1)
        rest = m.group(2)
        new_level = min(len(hashes) + 2, 6)
        return "#" * new_level + rest

    return re.sub(r"^(#{1,6})([ \t])", _shift, md_text, flags=re.MULTILINE)


def _render_note_html(note_md: str) -> str:
    """Convert note markdown to scoped HTML for consistent rendering."""
    clean_md = _downshift_headings(_normalize_note_markdown(note_md))
    html_body = md_lib.markdown(clean_md, extensions=["fenced_code", "tables", "nl2br"])
    return f'<div class="note-wrapper"><div class="note-content">\n{html_body}\n</div></div>'


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
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.expander("모델 설정", expanded=True):
        vlm_model = st.selectbox("VLM 모델 (이미지)", options=SUPPORTED_MODELS, index=0)
        llm_model = st.selectbox("LLM 모델 (노트 생성)", options=SUPPORTED_LLM_MODELS, index=0)

    st.markdown("---")

    # Pipeline step indicator
    st.caption("파이프라인")
    steps = st.session_state.get("_pipeline_steps", {})
    step_defs = [("upload", "파일 업로드"), ("parse", "파싱"), ("note", "노트 생성")]
    html_parts = ['<div class="step-indicator">']
    for i, (step_name, label) in enumerate(step_defs):
        done = steps.get(step_name, False)
        # determine active: first undone step after at least one done step
        prev_done = i == 0 or steps.get(step_defs[i - 1][0], False)
        active = not done and prev_done and i > 0 or (i == 0 and not done and any(steps.values()))
        circle_cls = "done" if done else ("active" if active else "")
        label_cls = circle_cls
        icon = "✓" if done else str(i + 1)
        html_parts.append(
            f'<div class="step-row">'
            f'<div class="step-circle {circle_cls}">{icon}</div>'
            f'<div class="step-label {label_cls}">{label}</div>'
            f'</div>'
        )
        if i < len(step_defs) - 1:
            connector_cls = "done" if done else ""
            html_parts.append(f'<div class="step-connector {connector_cls}"></div>')
    html_parts.append('</div>')
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

    st.markdown("---")

    if st.button("캐시 초기화", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.caption("Built with Streamlit + OpenAI / Anthropic / Google")

# ===================================================================
# FILE UPLOAD
# ===================================================================
uploaded_file = st.file_uploader(
    "학습자료를 업로드하세요",
    type=["pdf", "ipynb", "png", "jpg", "jpeg"],
    help="PDF, Jupyter Notebook, 이미지 (PNG/JPG) 지원",
)

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
    st.toast("캐시된 결과를 불러왔습니다", icon="⚡")
else:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_path: str | None = None
    doc = None

    with st.status("분석 진행 중...", expanded=True) as status:
        # Step 1: Parse
        status.update(label="1/2 — 파일 파싱 중...", state="running")
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
        status.update(label="2/2 — 학습 노트 생성 중...", state="running")
        try:
            result = generate_note(doc, model=llm_model)
        except Exception as exc:
            st.error(f"노트 생성 실패: {exc}")
            st.stop()

        st.session_state.setdefault("_pipeline_steps", {})["note"] = True
        status.update(label="분석 완료!", state="complete", expanded=False)

    # Cache
    st.session_state[doc_cache_key] = doc
    st.session_state[cache_key] = result

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
    ["📊 파싱 결과", "📝 학습 노트", "💬 Q&A"],
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
        st.markdown(_render_metric_card(doc.block_count, "총 블록 수"), unsafe_allow_html=True)
    with mc2:
        st.markdown(_render_metric_card(doc.format.value.upper(), "문서 형식"), unsafe_allow_html=True)
    with mc3:
        st.markdown(_render_metric_card(doc.status.value, "처리 상태"), unsafe_allow_html=True)

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
            st.warning(f"노트 생성 경고: {err}")

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
        st.markdown(f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True)

    # Key concepts as tags
    if key_concepts:
        st.markdown("**핵심 개념**")
        st.markdown(_render_concept_tags(key_concepts), unsafe_allow_html=True)

    # ── Two-column layout: Note (left) | Q&A Chat (right) ────────────
    col_note, col_chat = st.columns([3, 2])

    # ── Left column: Note content + edit + download ──────────────────
    with col_note:
        if note_markdown:
            raw_md = _normalize_note_markdown(note_markdown)
            file_stem = Path(doc.source).stem

            # Edit mode toggle
            edit_mode = st.toggle("✏️ 편집 모드", value=False, key="note_edit_toggle")

            if edit_mode:
                edited_md = st.text_area(
                    "마크다운 편집",
                    value=raw_md,
                    height=500,
                    key="note_editor",
                    label_visibility="collapsed",
                )
                st.markdown("**미리보기**")
                st.markdown(_render_note_html(edited_md), unsafe_allow_html=True)
                download_md = edited_md
            else:
                st.markdown(_render_note_html(raw_md), unsafe_allow_html=True)
                download_md = raw_md

            # Download button
            full_md = f"# {title}\n\n{download_md}"
            st.download_button(
                label="📥 마크다운 다운로드",
                data=full_md,
                file_name=f"{file_stem}_note.md",
                mime="text/markdown",
            )
        else:
            st.info("노트 내용이 없습니다.")

    # ── Right column: Q&A Chat ───────────────────────────────────────
    with col_chat:
        st.markdown("#### 💬 Q&A")
        st.caption("질문하거나 노트 수정을 요청하세요")

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        # Scrollable chat area
        chat_container = st.container(height=480)
        with chat_container:
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("질문을 입력하세요"):
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            try:
                _chat_result = rag_query(user_input, model=llm_model)
                _reply = _chat_result.answer
                if _chat_result.source_blocks:
                    _src_lines = []
                    _seen_srcs: set[str] = set()
                    for _src in _chat_result.source_blocks:
                        _loc = (
                            f"page {_src.page}" if _src.page is not None
                            else (f"cell {_src.cell_index}" if _src.cell_index is not None else "")
                        )
                        _dedup_key = f"{_src.source}:{_loc}"
                        if _dedup_key in _seen_srcs:
                            continue
                        _seen_srcs.add(_dedup_key)
                        _src_lines.append(f"- {_src.source}" + (f" ({_loc})" if _loc else ""))
                        if len(_src_lines) >= 3:
                            break
                    _reply += "\n\n**출처:**\n" + "\n".join(_src_lines)
            except Exception as _exc:
                _reply = f"오류가 발생했습니다: {_exc}"
            st.session_state["chat_messages"].append({"role": "assistant", "content": _reply})
            st.rerun()

# ─── Tab 3: RAG Q&A ───────────────────────────────────────────────────
if active_tab == "💬 Q&A":
    st.markdown("#### 💬 문서 기반 Q&A")
    st.caption(f"인덱싱된 문서: **{doc.source}** · LLM: `{llm_model}`")

    if "qa_messages" not in st.session_state:
        st.session_state["qa_messages"] = []

    # Chat history
    qa_container = st.container(height=400)
    with qa_container:
        for _msg in st.session_state["qa_messages"]:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

    # Input
    if qa_input := st.chat_input("문서에 대해 질문하세요", key="qa_tab_input"):
        st.session_state["qa_messages"].append({"role": "user", "content": qa_input})

        try:
            with st.spinner("검색 중..."):
                _qa_result = rag_query(qa_input, model=llm_model)

            _qa_reply = _qa_result.answer
            st.session_state["qa_messages"].append({"role": "assistant", "content": _qa_reply})

            # Store source blocks for display
            st.session_state["qa_last_sources"] = _qa_result.source_blocks
        except Exception as _qa_exc:
            st.error(f"Q&A 처리 중 오류가 발생했습니다: {_qa_exc}")
            st.session_state["qa_last_sources"] = []

        st.rerun()

    # Source blocks from last answer
    _last_sources = st.session_state.get("qa_last_sources", [])
    if _last_sources:
        st.markdown("---")
        st.markdown("**참조 문서 블록**")
        for _src in _last_sources:
            _loc = (
                f"page {_src.page}" if _src.page is not None
                else (f"cell {_src.cell_index}" if _src.cell_index is not None else "")
            )
            _expander_label = f"📄 {_src.source}" + (f" · {_loc}" if _loc else "") + f"  `{_src.block_type}`"
            with st.expander(_expander_label, expanded=False):
                st.caption(f"block_order: {_src.block_order}")
                st.text(_src.content_preview)
