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

## Known Limitations (v1 — recorded 2026-03-05)

Observed during CU-07 mid-check demo. Candidates for v1.1 prompt iteration.

| # | Prompt | Symptom | Root Cause | Proposed Fix |
|---|--------|---------|------------|--------------|
| 1 | `note_generation.py` | `note_markdown` returned as JSON object (`{"sections": [...]}`) instead of markdown string | No explicit instruction that `note_markdown` must be a plain markdown string | Add "Output `note_markdown` as a plain markdown string, not a JSON object" to prompt |
| 2 | `note_generation.py` | `key_concepts` returns English terms even for Korean-language source documents (e.g. "Version Control" instead of "버전 관리") | Prompt does not specify that concept language should follow source document language | Add "Extract key concepts in the same language as the source document" |
| 3 | `vlm_diagram.py` | VLM ignores schema and returns Korean-keyed JSON (e.g. `"문서 흐름"`, `"구성 요소"`) causing `DiagramVLMOutput.model_validate()` failure | Prompt field names are English but no explicit "respond in English keys only" instruction | Add "All JSON field names must be exactly as specified in the schema — do not translate or rename them" |
| 4 | `note_generation.py` | For large documents (e.g. 38-block ipynb), LLM dumps raw block content into `note_markdown` instead of synthesizing | All 38 blocks are passed verbatim; LLM hits effective context limit and falls back to copy-paste | Add block summarization/compression strategy — e.g. truncate code blocks to first N lines, merge consecutive text blocks, cap total input tokens explicitly |
| 5 | `note_generation.py` | `note_markdown` contains non-markdown content (raw JSON, original code blocks, serialized block format) | Prompt does not explicitly forbid non-markdown content or define what counts as valid output | Add "note_markdown must contain only pure markdown (headings, paragraphs, bullet lists, code fences). Do not include raw JSON, block labels like [code], or unformatted data." |
