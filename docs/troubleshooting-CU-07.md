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

## Issue 8 — note_generator.py: OpenAI 전용 → 멀티 프로바이더 확장

**Status:** Fixed

**Symptom**
`note_generator.py`가 OpenAI만 지원. Anthropic/Google 모델을 데모 UI에서 선택할 수 없었음.

**Root Cause**
`generate_note()`가 module-level `_openai_client` 캐시와 `openai.OpenAI()` 직접 호출로 하드코딩되어 있었음.

**Fix**
`vlm/client.py`와 동일한 provider dispatch 패턴으로 재구성:
- `_MODEL_REGISTRY`: model_id → `{provider, input, output}` 비용 테이블 (6개 모델)
- `SUPPORTED_LLM_MODELS`: 외부 노출 리스트, `demo.py`에서 import해서 selectbox 자동 확장
- `_call_openai / _call_anthropic / _call_google`: lazy import, 각각 `(raw, input_tokens, output_tokens)` 반환
- `_PROVIDER_DISPATCH`: `{provider: call_fn}` 라우팅 딕셔너리
- `generate_note()`: `_MODEL_REGISTRY[model]["provider"]`로 dispatch, 미지원 모델은 `ValueError` raise

```python
# Before: OpenAI hardcoded
client = openai.OpenAI()
resp = client.chat.completions.create(model=model, messages=[...])

# After: provider dispatch
provider = _MODEL_REGISTRY[model]["provider"]
raw, input_tokens, output_tokens = _PROVIDER_DISPATCH[provider](model, PROMPT, user_content)
```

**Test fix**
기존 테스트가 `_get_client` / `openai.OpenAI` 패치 방식이었음. `_PROVIDER_DISPATCH`는 모듈 로드 시점에 함수 레퍼런스를 캡처하므로 모듈 속성 패치(`monkeypatch.setattr`)로는 dispatch를 우회할 수 없음. `monkeypatch.setitem(note_gen_module._PROVIDER_DISPATCH, "openai", mock_fn)`으로 전면 교체. 62개 테스트 전부 통과.

---

## Issue 9 — demo.py UI 전면 재설계 (다크 테마 + 컴포넌트 시스템)

**Status:** Fixed

**Symptom**
초기 demo.py UI가 단조롭고 정보 밀도가 낮음. 파싱 결과와 노트가 세로로 쭉 나열되고, key_concepts는 raw 리스트로 덤프, 메트릭은 일반 텍스트로 출력됨.

**Root Cause**
MVP 수준 레이아웃 — styled HTML 없이 `st.write()` / `st.metric()` / `st.markdown()` 단순 호출만 사용. 결과 영역이 단일 세로 스크롤로 구성되어 파싱 요약과 노트가 구분되지 않음.

**Fix**
`ui/demo.py` 전면 재설계:

- **전역 CSS** (`_GLOBAL_CSS`): 다크 테마 기반 컴포넌트 스타일 시트. `.metric-card`, `.block-badge`, `.concept-tag`, `.note-wrapper`, `.summary-card`, `.meta-badge`, `.sidebar-brand`, `.pipeline-step` 등 클래스 정의.
- **탭 레이아웃**: `st.tabs(["📊 파싱 결과", "📝 학습 노트"])`로 파싱 요약 / 노트 영역 분리.
- **파싱 요약 탭**: metric cards (`_render_metric_card`), 블록 타입 컬러 배지 (`_render_block_type_badges`), 블록 상세 expander (최대 30개 미리보기).
- **노트 탭**: summary card, meta badges (난이도·읽기시간), key_concepts → `concept-tag` 뱃지 (`_render_concept_tags`), note content는 `.note-wrapper` 내부 렌더링.
- **사이드바**: 브랜드 헤더, 파이프라인 단계 표시기 (`upload → parse → note`), 모델 설정 expander.
- **진행 표시**: `st.spinner` → `st.status(expanded=True)` 2-step 진행 표시. 캐시 히트 시 `st.toast()`.
- **_render_note_html**: `_NOTE_CSS` 인라인 주입 → `note-wrapper` div 재사용으로 변경 (전역 CSS가 담당).

```python
# Before: flat layout
st.write(f"## {title}")
for concept in key_concepts:
    st.markdown(f"- `{concept}`")

# After: styled HTML components
st.markdown(f"### {title}")
st.markdown(_render_concept_tags(key_concepts), unsafe_allow_html=True)
```

---

## Issue 10 — Gemini 모델명 오류 (404 not found)

**Status:** Fixed

**Symptom**
Gemini 모델 선택 시 `404 models/gemini-3-flash is not found for API version v1beta` 오류 발생.

**Root Cause**
`vlm/client.py`와 `llm/note_generator.py`에 `gemini-3-flash` / `gemini-3.1-pro`로 등록되어 있었으나 실제 존재하지 않는 모델명이었음.

**Fix**
두 파일 모두 실제 API 모델명으로 교체:
```python
# Before (존재하지 않음)
"gemini-3-flash", "gemini-3.1-pro"

# After (실제 모델명)
"gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview"
```

---

## Issue 11 — ipynb 코드 블록이 note_markdown에 그대로 복사 → JSON parse fail

**Status:** Fixed

**Symptom**
38-block ipynb (코드 24개) 분석 시 `노트 생성 경고: JSON parse failed` 발생. fallback으로 raw LLM 출력이 렌더링되는데, `note_markdown` 안에 `def detect(cls, text: str):` 같은 raw 코드가 그대로 포함되어 있음.

**Root Cause**
`_serialize_blocks()`가 코드 블록 전체를 LLM에 전달. 코드가 길면 LLM이 note_markdown에 코드 자체를 복사하고, 코드 내 `\n`이 JSON 문자열 안에서 이스케이프 되지 않아 `json.loads()` 실패.

**Fix (1차 — 과도한 제한)**
`_truncate_code(content, max_lines=6)` 추가 — 코드 블록을 앞 6줄로 자르고 `# ... (N lines omitted)` 표시.
→ 부작용: LLM이 코드를 거의 이해 못 해서 note에서 코드 설명 섹션 자체가 사라짐.

**Fix (2차 — 조정)**
`_MAX_CODE_LINES` 6 → 15로 상향. 15줄이면 클래스 정의 + 핵심 메서드 시그니처까지 포함 가능. `[code]` 레이블 유지 (프롬프트에서 이미 코드 설명 지시).

```python
# Before: 전체 코드 전달
content = block.content[:content_limit]  # 최대 1200자

# After: 앞 15줄 + omit 표시
def _truncate_code(content: str, max_lines: int = _MAX_CODE_LINES) -> str:
    code_lines = content.splitlines()
    if len(code_lines) <= max_lines:
        return content
    kept = "\n".join(code_lines[:max_lines])
    return f"{kept}\n# ... ({len(code_lines) - max_lines} lines omitted)"
```

---

## Issue 12 — 데모 UI 색상 테마 불일치 (보라/파랑 잔존)

**Status:** Fixed

**Symptom**
배경을 라이트 테마로 전환했음에도 모델 선택창(selectbox), 사이드바 등 Streamlit 기본 위젯에 보라/파란 회색 색상이 잔존하여 빨강 계열 테마와 어울리지 않음.

**Root Cause**
두 가지 레이어에서 발생:
1. `secondaryBackgroundColor`가 `#F5F5FA` (보라빛 미세 포함) → 사이드바·expander 배경에 노출
2. Streamlit 기본 위젯(selectbox border, dropdown hover, chevron 아이콘)이 자체 파란 계열 색상 사용 — `primaryColor` 설정으로 완전히 제어되지 않음

**Fix**
`.streamlit/config.toml`:
```toml
secondaryBackgroundColor = "#FFF7ED"  # 연한 주황 → 빨강 primary와 조화
```

`ui/demo.py` `_GLOBAL_CSS` 추가:
- `[data-baseweb="select"] [data-baseweb="input"]` → border `#FECACA` (연한 빨강)
- `[data-baseweb="select"] svg` → chevron `#E53935`
- `[data-baseweb="menu"] li:hover` → hover background `#FFF7ED`
- `[data-baseweb="menu"] li[aria-selected="true"]` → 선택 항목 빨강 배경 + 텍스트
- `[data-testid="stSelectbox"] > div > div` → border + focus 링 오버라이드

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
| 8 | note_generator OpenAI 전용 | `llm/note_generator.py`, `ui/demo.py`, `tests/test_note_generator.py`, `tests/test_pipeline_integration.py` | — | Fixed |
| 9 | demo.py UI 단조롭고 정보 밀도 낮음 | `ui/demo.py` | — | Fixed |
| 10 | Gemini 모델명 404 오류 | `vlm/client.py`, `llm/note_generator.py` | — | Fixed |
| 11 | ipynb 코드 블록 → note_markdown 복사 + JSON parse fail | `llm/note_generator.py` | — | Fixed |
| 12 | 데모 UI 색상 테마 불일치 (보라/파랑 잔존) | `ui/demo.py`, `.streamlit/config.toml` | — | Fixed |

---

## Issue #13: Review follow-up — CU-16 persistence / compatibility / chart branch

**Status**: ✅ Fixed (`261d82b`)

**Symptom**:
- adaptive preprocessing metadata가 런타임에는 붙지만 저장 후 다시 불러오면 사라짐
- 기존 note row가 `_note_result_version`이 없다는 이유로 라이브러리에서 읽히지 않음
- 문서를 다시 저장할 때 original `created_at`이 현재 시각으로 덮여 library ordering이 흔들림
- resize policy에 `CHART`용 1024px branch가 정의돼 있어도 실제 분류 결과에서는 도달하지 못함
- 이미지 파싱 중 생성된 `*_preprocessed.*` 파일이 temp/upload 경로에 남음

**Root Cause**:
1. `models/document.py`의 `BlockMetadata`에 `preprocess` 필드가 없어 `doc.model_dump()`/`model_validate()` round-trip 시 메타데이터가 증발함.
2. `db/sqlite.py`가 `_note_result_version == "v2"`만 허용해서 legacy saved note를 전부 버림.
3. `save_document()`가 original `created_at` 의미를 보존하지 않아 재저장 시 library 정렬 기준이 흔들림.
4. `prompts/vlm_classify.py`는 `chart`를 출력하지 않는데 런타임 policy는 `ImageType.CHART`를 기대하고 있었음.
5. `parsers/image_parser.py`가 preprocess 결과 파일을 만들고도 후처리 cleanup을 하지 않았음.

**Fix**:
- `models/document.py`
  - `BlockMetadata.preprocess: Optional[dict[str, Any]]` 추가
- `db/sqlite.py`
  - `_deserialize_note_result()` 추가
  - legacy row (`version` 없음)는 허용
  - unknown future version만 skip
  - `save_document()`는 `doc.created_at`을 그대로 저장하고 upsert 시 `created_at`을 덮어쓰지 않음
- `prompts/vlm_classify.py`
  - `PROMPT_VERSION` `v1.4.0`
  - output enum에 `"chart"` 추가
  - diagram vs chart definition 분리
- `parsers/image_parser.py`
  - chart classification → `ImageType.CHART`
  - chart는 diagram parser branch 사용
  - `_cleanup_preprocessed_image()` 추가로 transient sibling 파일 제거
  - `block.metadata.preprocess` 저장
- `prompts/VERSION_LOG.md`
  - `vlm_classify` v1.4.0 entry 추가

**Regression Tests**:
- `tests/test_db.py`
  - duplicate upsert 후 `created_at` 보존
  - legacy note row read 허용
  - future version row만 skip
- `tests/test_image_parser.py`
  - `preprocess` metadata JSON serialization 확인
  - transient preprocessed file cleanup 확인
  - `chart` classification path 확인
- `tests/test_prompt_contracts.py`
  - classify prompt에 `chart`/`quantitative visualization` 포함 확인

**Takeaway**:
- 런타임에만 존재하는 metadata는 반드시 schema field까지 같이 추가해야 persistence에서 안 사라진다.
- version gate는 migration safety와 backward compatibility를 같이 봐야 한다. `unknown future`만 막고 `legacy known-shape`는 살리는 편이 안전했다.
- policy enum과 classifier prompt가 분리되어 있으면 dead branch가 생긴다. 분류 가능성까지 함께 검증해야 한다.
