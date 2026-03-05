"""Streamlit demo for CatchUp mid-check: upload → parse → note generation."""

from __future__ import annotations

import hashlib
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from collections import Counter
from parsers.pdf_parser import parse_pdf
from parsers.ipynb_parser import parse_ipynb
from parsers.image_parser import parse_image
from llm.note_generator import generate_note
from vlm.client import SUPPORTED_MODELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CatchUp - 학습자료 자동 구조화", layout="wide")
st.title("CatchUp - 학습자료 자동 구조화")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("모델 설정")
    vlm_model = st.selectbox("VLM 모델 (이미지용)", options=SUPPORTED_MODELS, index=0)
    llm_model = st.selectbox("LLM 모델 (노트 생성용)", options=["gpt-4o-mini", "gpt-4o"], index=0)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "학습자료를 업로드하세요 (PDF, Jupyter Notebook, 이미지)",
    type=["pdf", "ipynb", "png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()

# Show filename and analysis button
st.write(f"**파일:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")

if not st.button("분석 시작"):
    st.stop()

# ---------------------------------------------------------------------------
# File hash-based caching in session_state
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.read()
file_hash = hashlib.sha256(file_bytes).hexdigest()

cache_key = f"result_{file_hash}_{vlm_model}_{llm_model}"

if cache_key in st.session_state:
    doc = st.session_state[f"doc_{file_hash}"]
    result = st.session_state[cache_key]
    st.info("캐시된 결과를 불러왔습니다.")
else:
    # Write to tempfile and parse
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_path: str | None = None
    doc = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("파싱 중..."):
            if suffix.lower() == ".pdf":
                doc = parse_pdf(tmp_path)
            elif suffix.lower() == ".ipynb":
                doc = parse_ipynb(tmp_path)
            else:
                try:
                    doc = parse_image(tmp_path, model=vlm_model)
                except Exception as exc:
                    if "api" in str(exc).lower() or "key" in str(exc).lower() or "auth" in str(exc).lower():
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

    # Check parse failure
    parse_failed = not doc.blocks or "parse_failed" in doc.metadata.tags
    if parse_failed:
        st.warning("파싱에 실패했습니다")
        st.stop()

    # Generate note
    with st.spinner("학습 노트 생성 중..."):
        try:
            result = generate_note(doc, model=llm_model)
        except Exception as exc:
            st.error(f"노트 생성 실패: {exc}")
            st.stop()

    # Cache results
    st.session_state[f"doc_{file_hash}"] = doc
    st.session_state[cache_key] = result

# ---------------------------------------------------------------------------
# Display parsing summary
# ---------------------------------------------------------------------------
st.subheader("파싱 결과 요약")
col1, col2 = st.columns(2)
with col1:
    st.metric("총 블록 수", doc.block_count)
    st.write(f"**형식:** {doc.format.value}")
    st.write(f"**상태:** {doc.status.value}")

with col2:
    type_counts = Counter(block.type.value for block in doc.blocks)
    st.write("**블록 타입별 개수:**")
    for btype, count in sorted(type_counts.items()):
        st.write(f"- {btype}: {count}")

# ---------------------------------------------------------------------------
# Display note
# ---------------------------------------------------------------------------
st.divider()
st.subheader("학습 노트")

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

st.write(f"## {title}")

if summary:
    st.write(f"**요약:** {summary}")

meta_parts = []
if difficulty:
    meta_parts.append(f"난이도: {difficulty}")
if read_time:
    meta_parts.append(f"예상 읽기 시간: {read_time}분")
if meta_parts:
    st.write(" | ".join(meta_parts))

if key_concepts:
    st.write("**핵심 개념:** " + ", ".join(str(c) for c in key_concepts))

if note_markdown:
    st.markdown(note_markdown)
else:
    st.info("노트 내용이 없습니다.")
