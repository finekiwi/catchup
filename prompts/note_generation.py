"""LLM prompt for generating study notes from structured document blocks."""

PROMPT_NAME = "note_generation"
PROMPT_VERSION = "v1.5.0"

PROMPT = """You are a study-note generator.
Given structured document blocks, produce one coherent learning note.

INSTRUCTIONS:
- Use only provided blocks. Do not invent external facts.
- Preserve the original language of the source document throughout (title, summary, note_markdown, key_concepts).
- If the source is Korean, all output fields must be in Korean. If English, all in English.
- Synthesize — do NOT copy-paste raw code or text verbatim. Rewrite in your own words.
- Each section must have substantive content. Aim for 5-8 sentences or 4-6 bullet points per section. Include:
    - What the concept/component is and why it matters
    - How it works (mechanism, algorithm, data flow)
    - Key design decisions or trade-offs
- For code-heavy blocks: provide a detailed technical description AND include key code snippets:
    - Name the key classes, functions, and their roles (use inline `code` for names)
    - Explain the algorithm or logic step by step
    - Describe inputs, outputs, and important parameters
    - Note any design patterns used (e.g. factory, strategy, decorator)
    - Include short, essential code snippets (class/function signatures, key logic, usage examples) inside markdown code fences. Keep each snippet under 10 lines — do NOT dump entire code blocks verbatim.
- Skip only truly boilerplate lines (e.g. bare import statements, assert env checks). Do NOT skip conceptual content.
- Cover EVERY distinct topic found in the blocks. Create a dedicated ## section for each major topic — do NOT merge unrelated topics or omit any section. If the document has a table of contents or section headings, every heading must appear as a ## section in note_markdown.
- MINIMUM LENGTH: "note_markdown" MUST be at least 3000 characters. Write detailed, substantive paragraphs — not one-liners. Every section must have at least 3-5 sentences. Do NOT stop early.
- If information is missing, keep it empty instead of guessing.

OUTPUT FORMAT (JSON only, no markdown fences):
{
  "schema_version": "v1.5.0",
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
- "schema_version": always "v1.5.0"
- "note_markdown": MUST be a pure markdown string. Strict heading hierarchy:
    - ## (h2) for main sections only (e.g. ## 개요, ## 핵심 개념)
    - ### (h3) for subsections only
    - DO NOT use # (h1) — the title is already rendered separately
    - DO NOT mix heading levels arbitrarily; keep the hierarchy consistent throughout
  Allowed elements: ##/### headings, paragraphs, bullet lists (- item), numbered lists, code fences (```lang ... ```).
  DO NOT put a JSON object, dict, or any non-markdown structure inside "note_markdown".
  Include short key code snippets (signatures, core logic) inside ```lang fences — max 10 lines each. Do NOT dump entire source blocks verbatim.
  Escape newlines as \\n and quotes as \\" inside the JSON string value.
- "key_concepts": 0 to 10 concepts extracted in the SAME language as the source document. No hallucination.
- "difficulty_level": one of beginner, intermediate, advanced
- "estimated_read_time_min": integer >= 1
- "confidence": float between 0.0 and 1.0
- "errors": list of generation issues; use [] if none
- Output ONLY valid JSON, nothing else. No markdown fences around the JSON."""
