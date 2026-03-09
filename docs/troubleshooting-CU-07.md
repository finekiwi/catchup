# Troubleshooting Log — CU-07 Demo

Recorded during mid-check demo (2026-03-05).
Branch: `feature/CU-07-demo-ui`

---

## Issue 1 — DoclingLoader not found in langchain-community

**Status:** Fixed (`066ea68`)

**Symptom**
PDF 업로드 시 `파싱에 실패했습니다` 표시. 터미널에 `ImportError: cannot import name 'DoclingLoader'`.

**Root Cause**
`langchain-community 0.4.1`에는 `DoclingLoader`가 없음. 설치된 `docling 2.76.0`의 네이티브 API와 연결되지 않음.

**Fix**
`parsers/pdf_parser.py`에서 `DoclingLoader` 의존성 제거. `docling.document_converter.DocumentConverter`를 직접 호출하는 방식으로 전면 교체.

```python
# Before
from langchain_community.document_loaders import DoclingLoader
loader = DoclingLoader(file_path=file_path)
docs = loader.load()

# After
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert(file_path, raises_on_error=False)
for item, _ in result.document.iterate_items():
    ...
```

`tests/test_pdf_parser.py`도 `DocumentConverter` mock 방식으로 전면 재작성. 7개 테스트 전부 통과.

---

## Issue 2 — tempfile 경로가 doc.source에 노출

**Status:** Fixed (`791fe5a`)

**Symptom**
학습 노트 제목이 원본 파일명 대신 `/var/folders/.../tmpXXXXX.ipynb` 형태로 표시됨.

**Root Cause**
`streamlit run`으로 실행 시 업로드 파일을 `NamedTemporaryFile`에 저장 후 경로를 파서에 전달. 파서 내부에서 `Path(file_path).name`으로 `doc.source`를 설정하므로 tempfile 이름이 그대로 들어감.

**Fix**
`ui/demo.py`에서 파싱 직후 원본 파일명으로 덮어쓰기.

```python
doc = parse_ipynb(tmp_path)
doc.source = uploaded_file.name  # overwrite tempfile name
```

---

## Issue 3 — LLM이 JSON을 마크다운 펜스로 감싸서 반환

**Status:** Fixed (`791fe5a`)

**Symptom**
`gpt-4o` 사용 시 `note_generation_failed: JSON parse failed` 경고 + fallback으로 raw JSON 전체가 화면에 출력됨.

**Root Cause**
`gpt-4o`는 JSON 응답을 ` ```json ... ``` ` 펜스로 감싸는 경향이 있음. `note_generator.py`에서 `json.loads(raw)`를 바로 호출하므로 파싱 실패.

**Fix**
`llm/note_generator.py`에 `_strip_markdown_fence()` 함수 추가. `json.loads()` 호출 전 적용.

```python
result = json.loads(_strip_markdown_fence(raw))
```

`image_parser.py`의 기존 구현과 동일한 패턴으로 통일.

---

## Issue 4 — note_markdown에 JSON sections 구조가 그대로 반환

**Status:** Partially fixed — demo-level workaround (`2ea6f07`), prompt fix (`2d543d7`)

**Symptom**
`note_markdown` 필드에 마크다운 대신 `{"sections": [{"title": "...", "content": "..."}]}` JSON 객체가 들어옴. `st.markdown()` 렌더링 시 raw JSON 텍스트로 출력.

**Root Cause**
v1.1.0 프롬프트에서 `"note_markdown": "Full markdown note as ONE escaped JSON string"` 지시가 모호함. `gpt-4o`가 `note_markdown` 필드 값을 JSON 객체로 해석해서 삽입.

또한 `gpt-4o`는 `note_markdown`을 파싱된 dict로 반환하는 경우도 있어 (`json.loads` 후 `result["note_markdown"]`이 `str`이 아닌 `dict`).

**Fix (demo-level)**
`ui/demo.py`에 `_normalize_note_markdown()` 추가:
- `dict` 타입 입력 처리 (gpt-4o가 이미 파싱된 객체로 반환하는 경우)
- `{"sections": [...]}` 구조 → `## 제목\n\n내용` 마크다운 변환
- 임의 dict → 키별 섹션 변환

**Fix (prompt)**
`note_generation.py` v1.2.0에서 명시적으로 금지:
> "DO NOT put a JSON object, dict, or any non-markdown structure inside note_markdown"

---

## Issue 5 — 이미지 VLM JSON이 노트에 그대로 포함

**Status:** Fixed (`2ea6f07`)

**Symptom**
이미지 업로드 후 생성된 학습 노트 하단에 `{"문서 흐름": [...], "구성 요소": {...}}` 형태의 원시 JSON이 출력됨.

**Root Cause**
파이프라인 흐름:
1. `image_parser`가 VLM 호출 → `DiagramVLMOutput.model_validate()` 시도
2. VLM이 스키마 무시하고 한국어 키(`"문서 흐름"`, `"구성 요소"`)로 응답 → 검증 실패
3. fallback 경로에서 `result.content` (raw JSON 문자열)를 `Block.content`에 그대로 저장
4. `note_generator`가 그 JSON 문자열을 그대로 `note_markdown`에 삽입

**Fix**
`parsers/image_parser.py` fallback 경로에 `_json_to_plain_text()` 추가. JSON dict를 키-값 평문으로 변환 후 `Block.content`에 저장.

```python
# Before
content=result.content  # raw JSON string

# After
content=_json_to_plain_text(result.content)  # "키:\n  - 값\n  - 값"
```

**Remaining (v2)**
`vlm_diagram.py` 프롬프트에 "JSON 필드명은 스키마 그대로, 번역 금지" 지시 추가 필요. → `VERSION_LOG.md` limitation #3 참조.

---

## Issue 6 — 38블록 ipynb에서 LLM이 원본 코드 그대로 복사

**Status:** Partially improved (prompt v1.2.0/v1.2.1), code-level fix planned

**Symptom**
38블록(코드 24개 + 텍스트 14개) ipynb 분석 시 `note_markdown`에 원본 코드 블록이 그대로 복사되어 출력됨. 요약/합성이 이루어지지 않음.

**Root Cause**
`note_generator._serialize_blocks()`가 최대 50블록을 블록당 최대 2000자로 전달. 코드 24개 × 2000자 = 최대 48,000자 입력. LLM이 유효 컨텍스트 한계에 도달해 합성 대신 복사-붙여넣기로 fallback.

**Prompt Fix (v1.2.0)**
"Synthesize and summarize — do NOT copy-paste block content verbatim"
"For large documents, focus on key concepts and structure. Skip boilerplate setup code."

**Planned Code Fix (v2)**
`note_generator.py`에서 블록 타입별 압축 전략 적용:
- `CODE` 블록: 앞 N줄만 전달 + "N lines omitted" 표시
- 연속 `TEXT` 블록 병합
- 입력 토큰 총량에 명시적 상한 설정

→ `VERSION_LOG.md` limitation #4 참조.

---

## Issue 7 — key_concepts가 원문과 다른 언어로 추출

**Status:** Partially improved (prompt v1.2.0), evaluation pending

**Symptom**
한국어 문서 분석 시 `key_concepts`가 `["Version Control", "Git", "GitHub"]`처럼 영어로 반환됨.

**Root Cause**
v1.1.0 프롬프트에 언어 지시 없음. `"Preserve original language"` 지시가 `note_markdown`에 암묵적으로 적용되지만 `key_concepts`까지 명시적으로 커버하지 않음.

**Prompt Fix (v1.2.0)**
"Extract key_concepts in the SAME language as the source document."
"If the source is Korean, all output fields must be in Korean."

**Planned Eval**
골든셋에 한국어 문서 포함 후 `key_concepts` 언어 일치율 측정 필요.

→ `VERSION_LOG.md` limitation #2 참조.

---

## Summary

| # | Issue | File(s) Changed | Fix Commit | Status |
|---|-------|-----------------|------------|--------|
| 1 | DoclingLoader 미지원 | `parsers/pdf_parser.py`, `tests/test_pdf_parser.py` | `066ea68` | Fixed |
| 2 | doc.source에 tempfile 경로 | `ui/demo.py` | `791fe5a` | Fixed |
| 3 | JSON 마크다운 펜스 미처리 | `llm/note_generator.py` | `791fe5a` | Fixed |
| 4 | note_markdown JSON 구조 반환 | `ui/demo.py`, `prompts/note_generation.py` | `2ea6f07`, `2d543d7` | Fixed (demo) + Prompt |
| 5 | 이미지 VLM JSON → 노트 삽입 | `parsers/image_parser.py` | `2ea6f07` | Fixed (code) |
| 6 | 대용량 ipynb 코드 복사 | `prompts/note_generation.py` | `2d543d7` | Prompt partial, code v2 |
| 7 | key_concepts 언어 불일치 | `prompts/note_generation.py` | `2d543d7` | Prompt partial, eval pending |
