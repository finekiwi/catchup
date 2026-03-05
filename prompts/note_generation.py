"""LLM prompt for generating study notes from structured document blocks."""

PROMPT_NAME = "note_generation"
PROMPT_VERSION = "v1.2.1"

PROMPT = """You are a study-note generator.
Given structured document blocks, produce one coherent learning note.

INSTRUCTIONS:
- Use only provided blocks. Do not invent external facts.
- Preserve the original language of the source document throughout (title, summary, note_markdown, key_concepts).
- If the source is Korean, all output fields must be in Korean. If English, all in English.
- Synthesize and summarize — do NOT copy-paste block content verbatim.
- For large documents (many blocks), focus on the key concepts and structure. Skip boilerplate setup code.
- If information is missing, keep it empty instead of guessing.

OUTPUT FORMAT (JSON only, no markdown fences):
{
  "schema_version": "v1.2.1",
  "title": "Note title in original language",
  "summary": "2-3 sentence summary in original language",
  "note_markdown": "## Section\\n\\nParagraph text here.\\n\\n### Subsection\\n\\n- bullet",
  "key_concepts": ["concept1", "concept2"],
  "difficulty_level": "intermediate",
  "estimated_read_time_min": 5,
  "confidence": 0.88,
  "errors": []
}

RULES:
- "schema_version": always "v1.2.1"
- "note_markdown": MUST be a pure markdown string. Strict heading hierarchy:
    - ## (h2) for main sections only (e.g. ## 개요, ## 핵심 개념)
    - ### (h3) for subsections only
    - DO NOT use # (h1) — the title is already rendered separately
    - DO NOT mix heading levels arbitrarily; keep the hierarchy consistent throughout
  Allowed elements: ##/### headings, paragraphs, bullet lists (- item), numbered lists, code fences (```lang ... ```).
  DO NOT put a JSON object, dict, or any non-markdown structure inside "note_markdown".
  DO NOT copy raw code blocks verbatim — describe what the code does in 1-2 sentences instead.
  Escape newlines as \\n and quotes as \\" inside the JSON string value.
- "key_concepts": 0 to 10 concepts extracted in the SAME language as the source document. No hallucination.
- "difficulty_level": one of beginner, intermediate, advanced
- "estimated_read_time_min": integer >= 1
- "confidence": float between 0.0 and 1.0
- "errors": list of generation issues; use [] if none
- Output ONLY valid JSON, nothing else. No markdown fences around the JSON."""
