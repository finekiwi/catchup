# Prompt Version Log

## vlm_code.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.2.0 | 2026-03-16 | Output language selection (ko/en). `PROMPT` → `PROMPT_TEMPLATE` + `get_prompt(language)`. `LANGUAGE_INSTRUCTIONS` dict. Backward compat `PROMPT = get_prompt("ko")`. Removed "original language" instructions — replaced with explicit `{output_language_instruction}`. | Eliminates random Spanish/mixed-language VLM output |
| v1.1.0 | 2026-03-03 | Added `schema_version`, `code_markdown`, `errors`, stricter JSON escaping rules, explicit no-guess policy. | Higher parse stability, lower hallucination |

## vlm_diagram.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.2.0 | 2026-03-16 | Output language selection (ko/en). Same `get_prompt(language)` pattern as vlm_code. Labels preserved as-is, descriptions/flow_summary in chosen language. | Deterministic output language |
| v1.1.0 | 2026-03-03 | Added `schema_version`, `has_truncation`, `errors`, nullable title/relationship label policy, no-guess policy. | Better uncertain-input handling |

## vlm_text.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.2.0 | 2026-03-16 | Output language selection (ko/en). Same `get_prompt(language)` pattern. Content/title/key_points in chosen language. | Deterministic output language |
| v1.1.0 | 2026-03-03 | Added `schema_version`, `has_truncation`, `errors`, key points relaxed to 0-5, stricter no-guess policy. | Lower hallucination on sparse text |

## vlm_classify.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.4.0 | 2026-03-17 | Added explicit `chart` output class and separated chart vs diagram definitions. Charts now cover quantitative plots with axes/legends/series; diagrams remain structure/flow-oriented. | Activates CHART classification in production so the 1024px adaptive resize branch is reachable |
| v1.3.0 | 2026-03-16 | Expanded "other" decorative examples: mascot/cartoon characters, chapter-divider art, title-page artwork, paper-craft decorations, decorative animals — even if they resemble a diagram. | Prevents bee mascots / decorative chapter art from leaking as diagram or text_capture |
| v1.2.0 | 2026-03-16 | Added self-check heuristic ("Would a student learn something from this alone?"). Added exhaustive "other" examples: publisher logos by example (O'Reilly/Hanbit/Packt), book title images (even if technical term in title), chapter divider pages. Clarified text_capture requires "instructional content, not just a title". | Catches title-only images like "Deep Learning from Scratch" text art that v1.1.0 still leaked |
| v1.1.0 | 2026-03-16 | Rewrote definitions to explicitly exclude non-educational content. Added CRITICAL rule: book cover, title page, publisher logo, author photo, decorative illustration → "other" even if text is present. Expanded "diagram" to include graphs/charts with axes, neural network structures. "text_capture" now limited to instructional content only. | Book covers / logos / decorative images that leaked as text_capture or diagram now correctly classified as other |
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
| v1.4.1 | 2026-03-15 | note_generator.py: OpenAI `response_format={"type":"json_object"}` 강제 + JSON 파싱 실패 시 1회 retry (nudge 메시지 추가). 프롬프트 텍스트 변경 없음 — 동작 레이어 수정. | gpt-4o-mini가 JSON 대신 마크다운 텍스트 반환하는 케이스 방어 |
| v1.5.0 | 2026-03-15 | "5-10 major sections" → "Cover EVERY distinct topic, dedicated ## section per topic, TOC headings must all appear"; MINIMUM LENGTH 2000 → 3000자; max_tokens 4096 → 8192. | LLM이 섹션 누락하거나 1060 토큰에서 자발적으로 멈추는 문제 해결 |
| v1.5.1 | 2026-03-15 | No-merge rule 강화: "one heading = one ## section, no exceptions. Do NOT merge adjacent or similar-sounding sections (e.g. 'git commit' and 'git commit -a' are separate sections)." | 유사 섹션 병합으로 인한 TOC 커버리지 누락 방지 (3.5/3.7/3.8 섹션 누락 케이스) |
| v1.6.0 | 2026-03-16 | Output language selection (ko/en). `PROMPT` → `PROMPT_TEMPLATE` + `get_prompt(language)`. Removed implicit "original language" / "source Korean/English" instructions — replaced with explicit `{output_language_instruction}`. `generate_note()` and `generate_note_sectioned()` accept `language` parameter. | Eliminates random Spanish/mixed-language note output; user can choose English notes for English sources |
| v1.5.2 | 2026-03-15 | EXCLUDE auxiliary content rule 추가: 부록(Appendix), 참고문헌, 색인, 챕터 개요 blurb는 노트에 포함 금지. 코드 레벨: `_HEADING_PATTERN`에 `부록/Appendix` 추가 + `_CHAPTER_INTRO_MAX_LEN=450` — 첫 줄이 CHAPTER/부록 패턴인 짧은 multi-line 블록도 노이즈 필터 적용. | "CHAPTER 5 소개합니다…" 블록과 "부록 B GitLab" 섹션이 노트에 포함되는 문제 수정 |

## note_generation_section.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-15 | Initial section-based note generation prompts. SECTION_PROMPT: per-section body generation (raw markdown, no heading). ASSEMBLY_PROMPT: metadata-only JSON extraction from concatenated sectioned notes (title, summary, key_concepts, difficulty, read_time, confidence). | Enables full TOC coverage for large documents via per-section LLM calls |
| v1.1.0 | 2026-03-15 | ASSEMBLY_PROMPT: replaced full `note_markdown` with per-section snippets (`## heading` + first 2-3 sentences of body). `estimated_read_time_min` pre-computed from word count (200 wpm). ~80% token reduction vs full note while preserving summary quality. | Faster assembly call; summary quality maintained via opening sentences per section |
| v1.3.0 | 2026-03-16 | Output language selection (ko/en). SECTION_PROMPT → `get_section_prompt(language)`, ASSEMBLY_PROMPT → `get_assembly_prompt(language)`. Removed implicit "original language" instructions — replaced with `{output_language_instruction}`. Backward compat variables preserved. | Deterministic output language for sectioned notes |
| v1.2.0 | 2026-03-15 | SECTION_PROMPT: add "DO NOT restate the section title in the opening sentence." Code-level: `_strip_leading_heading()` extended to also strip plain-text restatements (normalized match against section heading, handles colons/case). | Eliminates duplicate heading lines when LLM outputs plain-text title instead of markdown `##` |

## eval_judge.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-12 | Initial judge prompt: binary CORRECT/INCORRECT answer quality evaluation. Used by `eval/before_after.py` `_llm_judge_score()`. Judge model must always differ from answer model (anti-self-bias). | Baseline |

## rag_qa.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-11 | Initial RAG Q&A prompt: context-grounded answer generation with inline source citations. Instructs LLM to cite only from provided context and reply "I don't know" when evidence is insufficient. | Baseline |
| v1.1.0 | 2026-03-11 | Add emotional/conversational query handling: empathize and offer 1-2 relevant topics from context instead of hard "not found" response. Refine fallback message to Korean-only. | Better UX for non-document queries |
| v1.2.0 | 2026-03-11 | Distinguish note modification requests from pure emotional expressions. Note edit requests now redirect user to ✏️ edit mode toggle instead of being misclassified as emotional queries. | Prevents misclassification of functional requests containing emotional words |
| v1.2.1 | 2026-03-11 | Fix note modification response: replace English instruction with explicit Korean template to prevent LLM mistranslation ("가능합니다" → "지원하지 않습니다"). | Fixes response polarity bug in note modification guidance |
| v1.2.2 | 2026-03-11 | Add English response template alongside Korean for note modification guidance to prevent confusion when user writes in English. | Covers multilingual users |
| v1.3.0 | 2026-03-13 | Tighten pure-emotion fallback: ask which part of the current document is difficult, and only mention topics explicitly present in retrieved context. | Reduces unrelated topic suggestions when retrieval context is weak or off-target |
| v1.3.1 | 2026-03-13 | Add code-identifier handling: when a relevant code/text block is retrieved, explain the symbol's role from surrounding context instead of falling back to "not found" over minor punctuation/exact-match differences. | Improves answers for code screenshot questions such as functions/classes referenced with quotes or missing punctuation |
| v1.3.2 | 2026-03-13 | Narrow note-modification detection to explicit edit requests only; explanation/summary/simplification requests must remain normal Q&A. | Prevents "설명해줘" style questions from being misrouted to the note-editing fallback |

## note_editor.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-13 | Initial prompt: section-level study note editing via natural-language instruction. Outputs raw markdown body (no heading, no JSON). Includes security guard against prompt injection in user instruction. Multi-turn context supported via section_list + history. | Baseline |
| v1.1.0 | 2026-03-15 | Add `{context_section}` placeholder for RAG-retrieved document chunks. When document_id is provided, `edit_section()` embeds the instruction, retrieves top_k chunks from ChromaDB, and injects them before the section body. New rule: prefer DOCUMENT CONTEXT over LLM parametric knowledge for added examples/facts. | Edit requests like "add a .gitignore example" now use actual document content instead of LLM knowledge. Figure block VLM text (indexed in ChromaDB) also becomes accessible to note editor. |

## rag_qa.py (continued)
| v1.4.0 | 2026-03-13 | Update note-modification response: redirect user to '✏️ 노트 수정' tab instead of edit mode toggle, reflecting new CU-11 note editor UI. | Aligns prompt with new UI affordance |
| v1.5.0 | 2026-03-14 | Add follow-up suggestion block (---SUGGESTIONS---/---END--- delimiters) appended after genuine document answers. UI parses and renders as clickable buttons (NotebookLM-style). Skip for note-mod redirects, emotional responses, and "not found" fallbacks. | Improves discovery of follow-on questions; no impact on main answer quality |
| v1.5.1 | 2026-03-15 | Add indirect-evidence rule: if context touches topic implicitly (e.g. explains life without X), synthesize answer instead of falling back. Fallback reserved for genuinely unrelated context only. | Fixes false "찾을 수 없습니다" on implicit/indirect evidence blocks |

## concept_linking.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-17 | Initial prompts for CU-17 concept linking. CANONICAL_NORMALIZE_PROMPT: batch-normalizes raw concept names into canonical EN + KO aliases + one-line KO definition. RELATIONSHIP_LABEL_PROMPT: classifies semantic relationship between two concepts as implements/extends/prerequisite/application or null. | Enables cross-document concept linking with precision-first approach (null → drop pair) |

## query_rewrite.py
| Version | Date | Change | Quality Impact |
|---------|------|--------|----------------|
| v1.0.0 | 2026-03-15 | Initial prompt: language-agnostic retrieval-friendly expansion. Preserves original text and appends KO→EN translations, EN abbreviation full-names, and compound word decomposition. Returns unchanged query when already retrieval-friendly. Used by `rag/query_rewriter.py` with gpt-4.1-nano, max_tokens=128, temperature=0. | Reduces embedding similarity gap caused by KO/EN vocabulary mismatch and abbreviation ambiguity. |

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
