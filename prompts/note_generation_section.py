"""LLM prompts for section-based note generation (CU-14).

Two prompts:
- SECTION_PROMPT: per-section note generation (receives one section's blocks)
- ASSEMBLY_PROMPT: metadata extraction from concatenated sectioned notes
"""

PROMPT_NAME = "note_generation_section"
PROMPT_VERSION = "v1.1.0"

SECTION_PROMPT = """You are a study-note generator working on ONE SECTION of a larger document.

Context:
- Document: "{doc_title}"
- Section: "{section_heading}" (section {section_idx} of {total_sections})

Instructions:
- Preserve the original language of the source document throughout.
- Synthesize — do NOT copy-paste raw code or text verbatim. Rewrite in your own words.
- Write 5-8 sentences or 4-6 bullets of substantive content. Include:
    - What the concept/component is and why it matters
    - How it works (mechanism, algorithm, data flow)
    - Key design decisions or trade-offs
- For code blocks: describe the algorithm + include key snippets (max 10 lines each).
- Use ### for subsections within this section if needed.
- DO NOT include a ## heading — it will be prepended by the assembler.
- Output ONLY raw markdown body text. No JSON wrapper, no markdown fences around the output."""

ASSEMBLY_PROMPT = """You are a study-note metadata extractor.

Document title: "{doc_title}"

Per-section summaries (heading + opening excerpt):
{section_snippets}

OUTPUT FORMAT (JSON only, no markdown fences):
{{
  "schema_version": "v1.0.0",
  "title": "Note title in the same language as the source",
  "summary": "2-3 sentence summary covering the main topics, in the same language as the source",
  "key_concepts": ["concept1", "concept2"],
  "difficulty_level": "intermediate",
  "estimated_read_time_min": {estimated_read_time_min},
  "confidence": 0.88,
  "errors": []
}}

RULES:
- "schema_version": always "v1.0.0"
- "title": concise title capturing the overall document topic
- "summary": 2-3 sentences synthesizing the key ideas across all sections above
- "key_concepts": 0 to 10 key terms in the SAME language as the source
- "difficulty_level": one of beginner, intermediate, advanced
- "estimated_read_time_min": use the value already filled in above (do not change it)
- "confidence": float between 0.0 and 1.0
- "errors": [] unless something went wrong
- Output ONLY valid JSON, nothing else. No markdown fences."""
