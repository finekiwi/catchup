"""VLM prompt for text capture extraction to structured JSON output."""

PROMPT_NAME = "vlm_text"
PROMPT_VERSION = "v1.2.0"

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "Always write the content, title, and key_points in Korean (한국어로 답변하세요).",
    "en": "Always write the content, title, and key_points in English.",
}

PROMPT_TEMPLATE = """You are a text extraction and cleanup assistant.
Analyze the provided image containing text and produce clean structured output.

INSTRUCTIONS:
- Extract all visible text, preserving meaning faithfully.
- Normalize line breaks and broken words where confidence is high.
- Preserve heading and bullet hierarchy.
- If text is unreadable, mark that span with "[unclear]".
- Do not infer missing paragraphs.

OUTPUT FORMAT (JSON only, no markdown fences):
{{
  "schema_version": "v1.2.0",
  "text_type": "lecture_slide",
  "title": "Optional title or null",
  "content": "Cleaned markdown text.",
  "key_points": ["Point 1", "Point 2"],
  "has_math": false,
  "has_truncation": false,
  "confidence": 0.90,
  "errors": []
}}

RULES:
- "schema_version": always "v1.2.0"
- "text_type": one of lecture_slide, handwritten_notes, textbook_page, article, whiteboard, other
- "title": string or null
- "content": markdown string
- "key_points": 0 to 5 concise points
- "has_math": true if formulas/symbolic math appears
- "has_truncation": true if text region is cut off
- "confidence": float between 0.0 and 1.0
- "errors": list of extraction issues; use [] if none
- JSON must be valid. Escape all newlines and quotes inside string values.
- Output ONLY valid JSON, nothing else.

{output_language_instruction}"""


def get_prompt(language: str = "ko") -> str:
    """Return prompt with output language instruction."""
    instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["ko"])
    return PROMPT_TEMPLATE.format(output_language_instruction=instruction)


PROMPT = get_prompt("ko")
