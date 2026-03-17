"""Prompts for concept normalization and relationship labeling (CU-17)."""

from __future__ import annotations

PROMPT_NAME = "concept_linking"
VERSION = "v1.1.0"

# ---------------------------------------------------------------------------
# Prompt A — canonical normalization
# ---------------------------------------------------------------------------

CANONICAL_NORMALIZE_PROMPT = """You are a knowledge-graph assistant. Given a list of raw concept names,
normalize each one into a canonical form with aliases and a short definition.

Output a JSON object with key "concepts" containing an array where each element has EXACTLY these keys:
- "raw": the original input string (preserve exactly)
- "canonical": lowercase English canonical name (e.g. "backpropagation", "gradient descent")
- "aliases": list of KO/EN synonyms — include the original Korean name if the input was Korean
- "definition": one-line definition written in Korean (for embedding quality)

Rules:
- canonical must be lowercase English only
- If the input is already English, just lowercase it
- aliases may be empty list [] if no meaningful synonyms exist
- definition must be a single Korean sentence (20-60 characters)
- Output ONLY the raw JSON object {"concepts": [...]} — no markdown fences, no extra text
"""


def get_normalize_prompt(raw_concepts: list[str]) -> str:
    """Return the user message content for concept normalization.

    Args:
        raw_concepts: List of raw concept name strings to normalize.

    Returns:
        User message string containing the JSON-formatted concept list.
    """
    import json

    return f"Normalize these concepts:\n{json.dumps(raw_concepts, ensure_ascii=False)}"


# ---------------------------------------------------------------------------
# Prompt B — relationship labeling
# ---------------------------------------------------------------------------

RELATIONSHIP_LABEL_PROMPT = """You are a knowledge-graph assistant. Given two concepts from different
documents, determine the semantic relationship between them.

Choose ONE of these relationship types, or null if none fits:
- "implements": concept A is a concrete implementation or instantiation of concept B (or vice versa)
- "extends": concept A builds upon or generalizes concept B (or vice versa)
- "prerequisite": understanding concept A requires knowing concept B (or vice versa)
- "application": concept A is an application domain or use case of concept B (or vice versa)

Output ONLY a JSON object with EXACTLY these keys:
- "relationship_type": one of the four strings above, or null
- "description": a single Korean sentence describing the relationship (omit if null)

If none of the four types accurately describe the relationship, set relationship_type to null.
Output ONLY the raw JSON — no markdown fences, no extra text.
"""


def get_label_prompt(source: dict, target: dict) -> str:
    """Return the user message content for relationship labeling.

    Args:
        source: Dict with keys canonical_name, aliases, definition, doc_title for source concept.
        target: Dict with keys canonical_name, aliases, definition, doc_title for target concept.

    Returns:
        User message string describing both concepts for the LLM.
    """
    src_aliases = ", ".join(source.get("aliases") or []) or "없음"
    tgt_aliases = ", ".join(target.get("aliases") or []) or "없음"
    return (
        f"Source concept (from '{source.get('doc_title', '?')}'):\n"
        f"  canonical: {source.get('canonical_name', '')}\n"
        f"  aliases: {src_aliases}\n"
        f"  definition: {source.get('definition', '')}\n\n"
        f"Target concept (from '{target.get('doc_title', '?')}'):\n"
        f"  canonical: {target.get('canonical_name', '')}\n"
        f"  aliases: {tgt_aliases}\n"
        f"  definition: {target.get('definition', '')}\n\n"
        "What is the relationship between these two concepts?"
    )
