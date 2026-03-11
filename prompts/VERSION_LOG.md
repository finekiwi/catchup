# Prompt Version Log

## vlm_code.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.1.0 | 2026-03-03 | Added `schema_version`, `code_markdown`, `errors`, stricter JSON escaping rules, explicit no-guess policy. | Higher parse stability, lower hallucination |

## vlm_diagram.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.1.0 | 2026-03-03 | Added `schema_version`, `has_truncation`, `errors`, nullable title/relationship label policy, no-guess policy. | Better uncertain-input handling |

## vlm_text.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.1.0 | 2026-03-03 | Added `schema_version`, `has_truncation`, `errors`, key points relaxed to 0-5, stricter no-guess policy. | Lower hallucination on sparse text |

## vlm_classify.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-05 | Initial prompt: 5-class image type classification (code_screenshot/diagram/text_capture/equation/other) with confidence score. | Enables auto-classification for 2-call VLM pipeline |

## note_generation.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-05 | Initial prompt: structured study note generation from serialized document blocks. Outputs `title`, `summary`, `note_markdown`, `key_concepts` (0-10), `difficulty_level`, `estimated_read_time_min`, `schema_version`, `confidence`, `errors`. | Baseline |
| v1.1.0 | 2026-03-05 | Added explicit JSON escaping rules for `note_markdown`. | Minor stability improvement |
| v1.2.0 | 2026-03-05 | (1) Explicit language rule: all output fields must match source document language. (2) `note_markdown` strictly forbidden from containing JSON objects or verbatim code — must be pure markdown (headings, paragraphs, bullets, code fences). (3) Synthesis instruction for large documents: summarize, skip boilerplate setup code. (4) `schema_version` bumped to v1.2.0. Fixes limitations #1 #2 #4 #5 from v1 known issues. | Targets JSON-in-markdown bug, English key_concepts on Korean docs, and raw content dump on large ipynb |
| v1.2.1 | 2026-03-05 | Add strict heading hierarchy rule: `##` for main sections, `###` for subsections, `#` (h1) forbidden since title is rendered separately. Fixes inconsistent font sizes observed in Streamlit rendering of large ipynb notes. | Consistent heading levels throughout note_markdown |
| v1.2.2 | 2026-03-05 | Fix over-compression from v1.2.0 "skip boilerplate" instruction. Add minimum content depth requirement: 3-6 sentences or 3-5 bullets per section. Clarify code handling: describe logic/algorithms/design decisions, not raw lines. Restrict "skip" to truly boilerplate lines only (bare imports, assert env checks). | Prevents 1-sentence section summaries; restores substantive note content |
| v1.3.0 | 2026-03-05 | (1) Large document instruction: "organize into 5-10 major sections, each with meaningful depth. Cover the full scope, not just the beginning." (2) Code-level: `_MAX_BLOCKS` 40→80, `_MAX_CONTENT_LEN` 800→1200, `_MAX_CONTENT_LEN_LARGE` 400→600, `_LARGE_DOC_THRESHOLD` 30→40. LLM now sees 2x more blocks with 50% more content per block. | Fixes thin 1-sentence sections on large PDFs (225-block Git textbook produced only surface-level summary) |
| v1.4.0 | 2026-03-05 | (1) Section depth 상향: 5-8문장/4-6불릿 + what/why/how 필수. (2) 코드 설명 상세화: 클래스/함수 역할, 알고리즘, I/O, 디자인 패턴. (3) 최소 길이 강제: `note_markdown` 2000자 이상. (4) Code-level: `max_tokens=4096` 명시. | gpt-4o-mini의 축약 경향 대응 — 짧은 응답 방지 |
| v1.4.1 | 2026-03-05 | 핵심 코드 스니펫 허용: "DO NOT include raw code" → "Include short key code snippets (signatures, core logic) inside code fences — max 10 lines each. Do NOT dump entire blocks verbatim." 학습노트에 클래스/함수 시그니처, 핵심 로직, 사용 예시 포함 가능. | 코드 중심 자료(ipynb)에서 학습 효과 향상 — 설명만으론 부족한 구현 디테일 보완 |

## rag_qa.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-11 | Initial RAG Q&A prompt: context-grounded answer generation with inline source citations. Instructs LLM to cite only from provided context and reply "I don't know" when evidence is insufficient. | Baseline |
| v1.1.0 | 2026-03-11 | Add emotional/conversational query handling: empathize and offer 1-2 relevant topics from context instead of hard "not found" response. Refine fallback message to Korean-only. | Better UX for non-document queries |
| v1.2.0 | 2026-03-11 | Distinguish note modification requests from pure emotional expressions. Note edit requests now redirect user to ✏️ edit mode toggle instead of being misclassified as emotional queries. | Prevents misclassification of functional requests containing emotional words |

## Known Limitations & v2 Roadmap (recorded 2026-03-05, CU-07 demo)

Issues observed during CU-07 mid-check demo.
- ✅ Fixed in v1.2.x prompt
- 🔧 Code-level fix planned for v2
- 📋 See also: `docs/troubleshooting-CU-07.md` for full root cause analysis

| # | Prompt | Symptom | Root Cause | Fix | Target |
|---|--------|---------|------------|-----|--------|
| 1 | `note_generation.py` | `note_markdown` returned as JSON object instead of markdown string | Prompt did not forbid JSON objects in `note_markdown` | ✅ Fixed in v1.2.0: explicit "MUST be pure markdown, DO NOT put JSON object" rule | v1.2.0 |
| 2 | `note_generation.py` | `key_concepts` in English for Korean-language source | No language constraint on `key_concepts` | ✅ Fixed in v1.2.0: "Extract in SAME language as source document" | v1.2.0 |
| 3 | `vlm_diagram.py` | VLM returns Korean-keyed JSON (`"문서 흐름"` etc.) — `DiagramVLMOutput.model_validate()` fails | No instruction to preserve schema field names | 🔧 v2: Add "All JSON field names must match schema exactly — do not translate" | v2 |
| 4 | `note_generation.py` | 38-block ipynb: LLM copies raw code instead of synthesizing | All blocks passed verbatim; LLM hits context limit | ✅ Partial fix in v1.2.0 (synthesis instruction) / 🔧 v2: code-block truncation + token cap in `note_generator.py` | v1.2.0 + v2 |
| 5 | `note_generation.py` | `note_markdown` contains raw JSON / block labels / verbatim code | Prompt did not define what counts as valid markdown output | ✅ Fixed in v1.2.0: allowlist of permitted elements + explicit DO NOT rules | v1.2.0 |
| 6 | — | Heading levels inconsistent in Streamlit (`#`, `##`, `###` mixed) | LLM used arbitrary heading depths | ✅ Fixed in v1.2.1: `##` for sections, `###` for subsections, `#` forbidden | v1.2.1 |
| 7 | `note_generation.py` | 225-block PDF: each section has only 1 sentence | `_MAX_BLOCKS=40` + `_MAX_CONTENT_LEN_LARGE=400` → LLM sees ~16% of document; "skip boilerplate" instruction too aggressive | ✅ Fixed in v1.3.0: block limits doubled, large-doc instruction specifies 5-10 sections with depth | v1.3.0 |
