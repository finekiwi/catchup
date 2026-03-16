"""VLM prompt for diagram extraction to structured JSON output."""

PROMPT_NAME = "vlm_diagram"
PROMPT_VERSION = "v1.2.0"

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "Always write the description, flow_summary, and component roles in Korean (한국어로 답변하세요). Preserve original labels as-is.",
    "en": "Always write the description, flow_summary, and component roles in English. Preserve original labels as-is.",
}

PROMPT_TEMPLATE = """You are a technical diagram analysis assistant.
Analyze the provided diagram image and extract structure without guessing unseen details.

INSTRUCTIONS:
- Identify diagram type.
- Extract visible components and relationships.
- Preserve all labels in their original language as-is.
- If a value is unreadable, use "[unclear]" or null.
- Do not invent hidden nodes or links.

OUTPUT FORMAT (JSON only, no markdown fences):
{{
  "schema_version": "v1.2.0",
  "diagram_type": "flowchart",
  "title": "Optional visible title or null",
  "description": "High-level description.",
  "components": [
    {{"name": "Component A", "role": "Brief role"}}
  ],
  "relationships": [
    {{"from": "Component A", "to": "Component B", "label": null}}
  ],
  "flow_summary": "Step-by-step flow.",
  "has_truncation": false,
  "confidence": 0.85,
  "errors": []
}}

RULES:
- "schema_version": always "v1.2.0"
- "diagram_type": one of flowchart, architecture, sequence, er, class, mindmap, network, other
- "title": string or null
- "components": list of visible nodes/entities only
- "relationships": directed links; "label" may be null
- "flow_summary": concise narrative
- "has_truncation": true when borders/text are cut off
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
