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
| v1.1.0 | 2026-03-03 | Added `schema_version`, `confidence`, `errors`, escaped JSON-string rule for `note_markdown`, key concepts relaxed to 0-10. | Higher JSON parse success |
