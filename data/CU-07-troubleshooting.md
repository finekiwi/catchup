# CU-07 Troubleshooting Record

CU-07: Streamlit mid-check demo (`ui/demo.py`)
Pipeline: upload → parse (pdf/ipynb/image) → `generate_note()` → display

---

## Issue #1: DoclingLoader 미지원 → DocumentConverter 직접 호출로 교체

**Status**: ✅ Fixed

**Symptom**: `parse_pdf()` returned empty blocks or raised import error. Demo에서 "파싱에 실패했습니다" 표시.

**Root Cause**: `DoclingLoader` (LangChain community wrapper)가 설치 환경에서 미지원이거나 빈 결과를 반환. LangChain 래퍼가 실패를 숨겨 디버깅이 어려웠음.

**Fix**: `parsers/pdf_parser.py`에서 `DoclingLoader` 제거, Docling 네이티브 `DocumentConverter` API 직접 호출로 교체.
- `ConversionStatus`로 변환 성공 여부 명시적 체크
- `DocItemLabel` → `BlockType` 매핑 딕셔너리(`_LABEL_TO_BLOCK_TYPE`)로 블록 타입 변환
- `TextItem`, `TableItem`, `PictureItem` 타입별 콘텐츠 추출

**Ref**: `parsers/pdf_parser.py` — Sonnet 구현

---

## Issue #2: tempfile 경로가 doc.source에 들어감

**Status**: ✅ Fixed

**Symptom**: 파싱 결과에 `source: /var/folders/.../tmp5j2izvxn.pdf` 같은 임시 경로가 표시됨. 사용자에게 무의미한 정보.

**Root Cause**: 파서가 tempfile 경로를 `Document.source`에 그대로 저장. `demo.py`에서 원본 파일명으로 덮어쓰는 로직 없었음.

**Fix**: `ui/demo.py`에서 파싱 후 원본 파일명으로 덮어쓰기:
```python
doc.source = uploaded_file.name
```

**Ref**: `ui/demo.py:243` — Sonnet 구현

---

## Issue #3: LLM이 JSON을 마크다운 펜스로 감싸 반환

**Status**: ✅ Fixed

**Symptom**: `generate_note()` JSON 파싱 실패. LLM이 ```` ```json {...} ``` ```` 형태로 응답하여 `json.loads()` 에러.

**Root Cause**: OpenAI 모델이 JSON 출력을 마크다운 코드 펜스로 감싸는 기본 습성. 프롬프트에 "no markdown fences" 지시가 있었으나 모델이 무시.

**Fix**: `llm/note_generator.py`에 `_strip_markdown_fence()` 추가:
```python
def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
```

**Ref**: `llm/note_generator.py` — Sonnet 구현

---

## Issue #4: `note_markdown`에 JSON sections 구조 반환

**Status**: ✅ Fixed (code-level safety net + prompt-level root fix)

**Symptom**: Demo에서 노트 영역에 원시 JSON이 표시됨:
```json
{"sections": [{"title": "개요", "content": "..."}]}
```

**Root Cause**: `note_generation.py` v1.0.0 프롬프트가 `note_markdown` 필드에 "순수 마크다운 문자열"을 강제하지 않음. LLM이 구조화된 JSON으로 응답.

**Observed LLM output variants**:
1. `{"sections": [{"title": ..., "content": ...}]}` — sections 배열
2. `{"section1": {"title": ..., "content": ...}, ...}` — 번호 키
3. `{"## 환경 설정": "...", ...}` — 마크다운 헤딩을 키로 사용
4. 코드 블록의 `{}`가 `json.loads`를 깨뜨림

**Fix (code-level)**: `ui/demo.py`에 `_normalize_note_markdown()` 안전망 추가 (Opus):
- `_dict_to_markdown()` — 4가지 JSON 패턴을 마크다운으로 변환
- `_regex_extract_sections()` — `json.loads` 실패 시 정규식 폴백
- `_downshift_headings()` — 헤딩 레벨 2단계 다운시프트 (# → ###)

**Fix (prompt-level)**: v1.2.0에서 근본 수정:
```
- "note_markdown": MUST be a pure markdown string
- DO NOT put a JSON object, dict, or any non-markdown structure
- Allowed elements: ##/### headings, paragraphs, bullet lists, code fences
```

**Ref**: `ui/demo.py` (Opus), `prompts/note_generation.py` v1.2.0 (Sonnet)

---

## Issue #5: ipynb 38블록 → LLM이 원본 코드 그대로 복사

**Status**: ✅ Fixed in v1.2.0~v1.3.0

**Symptom**: 38블록 ipynb 파일에서 `note_markdown`에 원본 파이썬 코드, `[code]` 블록 라벨, JSON 직렬화 형태가 그대로 포함됨.

**Root Cause**:
1. 모든 블록이 `[{type}] {content}` 형태로 LLM에 전달 — 코드 블록도 원문 그대로
2. 프롬프트에 "코드를 요약하라"는 지시 없음
3. LLM이 컨텍스트 한계에 도달하면 copy-paste로 폴백

**Fix history**:
- v1.2.0: "Synthesize — do NOT copy-paste. DO NOT include raw code — describe it."
- v1.2.2: 섹션당 최소 깊이 요구 (3-6 문장 / 3-5 불릿)
- v1.3.0: 블록 전달량 확대 (`_MAX_BLOCKS` 40→80) + 대형 문서 구조화 지시

**Code-level fixes**:
- `_sample_blocks()`: 균등 샘플링으로 문서 전체 커버 (앞부분만 X)
- `_MAX_CONTENT_LEN_LARGE`: 400→600 (블록당 컨텐츠 확대)

**Ref**: `prompts/note_generation.py` v1.3.0, `llm/note_generator.py`

---

## Issue #6: 이미지 VLM JSON이 노트에 그대로 포함

**Status**: ✅ Partially fixed / 🔧 Prompt v2 pending

**Symptom**: 이미지 파서의 VLM이 한국어 키 JSON (`"문서 흐름"`, `"구성 요소"`)을 반환 → Pydantic 검증 실패 → 원시 JSON이 `Block.content`에 저장 → `generate_note()`에 JSON 문자열이 입력 → `note_markdown`에 JSON 포함.

**Root Cause (chain)**:
1. `vlm_diagram.py` 프롬프트에 "영문 필드명 필수" 지시 없음
2. VLM이 한국어로 키를 번역하여 응답
3. `DiagramVLMOutput.model_validate()` 실패
4. `image_parser.py` 폴백: 원시 JSON → `Block.content`
5. `generate_note()`가 JSON 문자열을 받아 노트에 포함

**Fix (code-level, done)**: `image_parser.py`에 `_json_to_plain_text()` 추가 — 폴백 시 JSON을 readable key-value 텍스트로 변환하여 downstream LLM에 자연어 전달.

**Fix (prompt-level, pending)**: `vlm_diagram.py`에 추가 예정:
```
All JSON field names must be exactly as specified in the schema — do not translate or rename them.
```

**Ref**: `parsers/image_parser.py` (Sonnet), `prompts/vlm_diagram.py` v2 (pending)

---

## Issue #7: `key_concepts` 영어로 나옴

**Status**: ✅ Fixed in v1.2.0

**Symptom**: 한국어 소스 문서에서 `key_concepts`가 영어로 반환됨 (예: "Version Control" → "버전 관리"여야 함).

**Root Cause**: v1.0.0 프롬프트에 `key_concepts` 언어 제약 없음. LLM이 영어를 기본으로 사용.

**Fix**: v1.2.0 프롬프트에 언어 보존 규칙 추가:
```
- Preserve the original language of the source document throughout (title, summary, note_markdown, key_concepts).
- "key_concepts": 0 to 10 concepts extracted in the SAME language as the source document.
```

**Ref**: `prompts/note_generation.py` v1.2.0

---

## Bonus: Streamlit 캐시 미갱신

**Symptom**: 프롬프트/코드 수정 후 동일 파일 재업로드 시 이전 결과 그대로 표시.

**Root Cause**: `st.session_state`가 `file_hash + model`로 캐싱. 코드 변경은 캐시 키에 반영 안 됨.

**Fix**: 사이드바에 "캐시 초기화" 버튼 추가:
```python
if st.button("캐시 초기화"):
    st.session_state.clear()
    st.rerun()
```

---

## Prompt Version History

| Version | Date | Changes | Issues Addressed |
|---------|------|---------|-----------------|
| v1.0.0 | 2026-03-05 | Initial prompt | Baseline |
| v1.1.0 | 2026-03-05 | JSON escaping rules for `note_markdown` | Stability |
| v1.2.0 | 2026-03-05 | Language preservation, pure markdown only, synthesis instruction | #4, #5, #7 |
| v1.2.1 | 2026-03-05 | Heading hierarchy: `##` sections, `###` subsections, `#` forbidden | Font size inconsistency |
| v1.2.2 | 2026-03-05 | Minimum content depth (3-6 sentences/section), code description rules | Over-compression |
| v1.3.0 | 2026-03-05 | Large doc instruction (5-10 sections), block limits doubled | #5 (thin content on large PDFs) |
| v1.4.0 | 2026-03-05 | Section depth 상향 (5-8문장), 최소 2000자, `max_tokens=4096` | 축약 경향 대응 |
| v1.4.1 | 2026-03-05 | 핵심 코드 스니펫 허용 (max 10 lines), code dump 금지 유지 | 코드 중심 자료 학습 효과 향상 |

## Issue #8: `[code:DESCRIBE_ONLY]` 태그가 코드 스니펫 포함을 차단

**Status**: ✅ Fixed

**Symptom**: v1.4.1 프롬프트에서 "핵심 코드 스니펫 포함"을 지시했으나, 실제 학습노트에 코드블록이 전혀 포함되지 않음.

**Root Cause (chain)**:
1. `_serialize_blocks()`에서 코드 블록을 `[code:DESCRIBE_ONLY]`로 태깅
2. LLM이 태그를 "코드 설명만 하라"로 해석 → 코드 스니펫 생략
3. 프롬프트 v1.4.1은 "Include short key code snippets"로 포함을 지시 → 입력 태그와 모순
4. `_MAX_CODE_LINES = 6`으로 코드가 6줄만 전달되어 스니펫 재료 자체 부족

**Fix**:
- `[code:DESCRIBE_ONLY]` → `[code]`로 태그 변경 (프롬프트와 일치)
- `_MAX_CODE_LINES` 6 → 15로 확대 (핵심 로직이 포함될 수 있도록)

**Ref**: `llm/note_generator.py`

---

## Issue #9: UI 다크/라이트 테마 + 브랜딩 색상 통일

**Status**: ✅ Fixed

**Symptom**: 초기 다크 테마 구현 후 사용자가 라이트 모드 선호. 이후 로고를 토마토 레드(`#E53935`)로 변경했으나 나머지 accent가 보라색(`#6C63FF`)으로 불일치.

**Fix**:
1. `.streamlit/config.toml`: `backgroundColor` → `#FFFFFF`, `primaryColor` → `#E53935`
2. `_GLOBAL_CSS` 전체: 보라색(`#6C63FF`, `#5B52E0`, `rgba(108,99,255,...)`) → 토마토 레드(`#E53935`, `#C62828`, `rgba(229,57,53,...)`)로 통일
3. file uploader 아이콘/border도 `#E53935`로 CSS override
4. 사이드바 footer: "Built with Streamlit + OpenAI" → "+ Anthropic / Google" 추가

**Ref**: `.streamlit/config.toml`, `ui/demo.py`

---

## Code-Level Changes

| File | Change | Author |
|------|--------|--------|
| `parsers/pdf_parser.py` | `DoclingLoader` → `DocumentConverter` 직접 호출 | Sonnet |
| `parsers/image_parser.py` | `_json_to_plain_text()` 폴백 추가 | Sonnet |
| `llm/note_generator.py` | `_strip_markdown_fence()`, `_sample_blocks()`, 블록 제한 확대 (40→80), `max_tokens=4096`, `_MAX_CODE_LINES` 6→15, `[code:DESCRIBE_ONLY]` → `[code]` | Sonnet + Opus |
| `prompts/note_generation.py` | v1.0.0 → v1.4.1 (8회 iteration) | Sonnet + Opus |
| `prompts/VERSION_LOG.md` | v1.1.0~v1.4.1 entries + Known Limitations 7건 | Sonnet + Opus |
| `ui/demo.py` | 풀 리디자인: light theme, 토마토 레드 accent, metric cards, concept tags, block badges, st.tabs, st.status, sidebar branding | Opus |
| `.streamlit/config.toml` | 라이트 테마 + primaryColor `#E53935` | Opus |
| `pyproject.toml` | `markdown>=3.7` 의존성 추가 | Opus |
| `SKILL.md` | Should-have에 PDF figure 추출 + ipynb output 이미지 포함 추가 | Opus |

## Iteration Results (38-block ipynb: LLM_046_Guardrails_Evaluation)

| Version | Sections | Depth per section | Code snippets | Key concepts lang |
|---------|----------|-------------------|---------------|-------------------|
| v1.0.0 | 3 | raw JSON dump | N/A (JSON) | English |
| v1.2.0 | 6 | 1 sentence | None | Korean ✅ |
| v1.3.0 | 9 | 1-2 sentences | None | Korean ✅ |
| v1.4.0 | 9 | 3-5 sentences | None | Korean ✅ |
| v1.4.1 | 9 | 3-5 sentences + code | ❌ blocked by DESCRIBE_ONLY | Korean ✅ |
| v1.4.1 (fix) | 9 | 3-5 sentences + code | ✅ (tag fix + 15 lines) | Korean ✅ |

## Remaining Work

1. `vlm_diagram.py` v2: 영문 필드명 강제 규칙 추가 (Issue #6 근본 해결)
2. `_normalize_note_markdown()` 안전망 유지 (프롬프트 안정화되었으나 LLM 불확정성 대비)
3. PDF figure 추출 + 학습노트 포함 (Docling PictureItem → `Block.image_path` → 노트 렌더링)
4. ipynb cell output 이미지 추출 (base64 → 파일 저장 → `Block.image_path` → 노트 렌더링)
5. 다른 문서 타입(PDF, 이미지)으로 코드 스니펫 포함 결과 추가 검증
