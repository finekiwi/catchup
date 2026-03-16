"""LLM prompts for section-based note generation (CU-14).

Two prompts:
- SECTION_PROMPT: per-section note generation (receives one section's blocks)
- ASSEMBLY_PROMPT: metadata extraction from concatenated sectioned notes
"""

PROMPT_NAME = "note_generation_section"
PROMPT_VERSION = "v1.3.0"

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "Write all output in Korean (한국어로 작성하세요).",
    "en": "Write all output in English.",
}

ASSEMBLY_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "All output fields (title, summary, key_concepts) MUST be in Korean (한국어로 작성하세요).",
    "en": "All output fields (title, summary, key_concepts) MUST be in English.",
}

SECTION_PROMPT_TEMPLATE = """You are a study-note generator working on ONE SECTION of a larger document.

Context:
- Document: "{{doc_title}}"
- Section: "{{section_heading}}" (section {{section_idx}} of {{total_sections}})

Instructions:
- Synthesize — do NOT copy-paste raw code or text verbatim. Rewrite in your own words.
- Write 5-8 sentences or 4-6 bullets of substantive content. Include:
    - What the concept/component is and why it matters
    - How it works (mechanism, algorithm, data flow)
    - Key design decisions or trade-offs
- For code blocks: describe the algorithm + include key snippets (max 10 lines each).
- Use ### for subsections within this section if needed.
- DO NOT include a ## heading — it will be prepended by the assembler.
- DO NOT restate the section title in the opening sentence.
- Output ONLY raw markdown body text. No JSON wrapper, no markdown fences around the output.

{output_language_instruction}"""

ASSEMBLY_PROMPT_TEMPLATE = """You are a study-note metadata extractor.

Document title: "{{doc_title}}"

Per-section summaries (heading + opening excerpt):
{{section_snippets}}

OUTPUT FORMAT (JSON only, no markdown fences):
{{{{
  "schema_version": "v1.0.0",
  "title": "Note title",
  "summary": "2-3 sentence summary covering the main topics",
  "key_concepts": ["concept1", "concept2"],
  "difficulty_level": "intermediate",
  "estimated_read_time_min": {{estimated_read_time_min}},
  "confidence": 0.88,
  "errors": []
}}}}

RULES:
- "schema_version": always "v1.0.0"
- "title": concise title capturing the overall document topic
- "summary": 2-3 sentences synthesizing the key ideas across all sections above
- "key_concepts": 0 to 10 key terms
- "difficulty_level": one of beginner, intermediate, advanced
- "estimated_read_time_min": use the value already filled in above (do not change it)
- "confidence": float between 0.0 and 1.0
- "errors": [] unless something went wrong
- Output ONLY valid JSON, nothing else. No markdown fences.

{output_language_instruction}"""


def get_section_prompt(language: str = "ko") -> str:
    """Return SECTION_PROMPT with output language instruction."""
    instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["ko"])
    return SECTION_PROMPT_TEMPLATE.format(output_language_instruction=instruction)


def get_assembly_prompt(language: str = "ko") -> str:
    """Return ASSEMBLY_PROMPT with output language instruction."""
    instruction = ASSEMBLY_LANGUAGE_INSTRUCTIONS.get(language, ASSEMBLY_LANGUAGE_INSTRUCTIONS["ko"])
    return ASSEMBLY_PROMPT_TEMPLATE.format(output_language_instruction=instruction)


# Backward compat
SECTION_PROMPT = get_section_prompt("ko")
ASSEMBLY_PROMPT = get_assembly_prompt("ko")
