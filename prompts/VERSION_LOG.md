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
