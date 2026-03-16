"""VLM prompt for code screenshot extraction to structured JSON output."""

PROMPT_NAME = "vlm_code"
PROMPT_VERSION = "v1.2.0"

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "Always write the description in Korean (한국어로 답변하세요).",
    "en": "Always write the description in English.",
}

PROMPT_TEMPLATE = """You are a precise code extraction assistant.
Analyze the provided code screenshot and extract its content with high fidelity.

INSTRUCTIONS:
- Extract code exactly as visible, preserving indentation and line breaks.
- Identify the programming language.
- If any part is unreadable or cut off, insert `[unclear]` at that exact position.
- Do not infer missing code.

OUTPUT FORMAT (JSON only, no markdown fences):
{{
  "schema_version": "v1.2.0",
  "language": "python",
  "code": "def hello():\\n    print('hello')",
  "code_markdown": "```python\\ndef hello():\\n    print('hello')\\n```",
  "description": "Brief explanation of what this code does.",
  "has_truncation": false,
  "confidence": 0.95,
  "errors": []
}}

RULES:
- "schema_version": always "v1.2.0"
- "language": detected language name, or "unknown" if uncertain
- "code": plain extracted code string with escaped newlines (\\n)
- "code_markdown": fenced markdown code block using detected language
- "description": 1-2 sentences
- "has_truncation": true if any line appears cropped/incomplete
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
