"""Prompt contract tests for VLM and LLM templates."""

from __future__ import annotations

import re

from prompts import note_generation, vlm_classify, vlm_code, vlm_diagram, vlm_text


def test_prompt_common_constants() -> None:
    """All prompt modules should expose name/version constants."""
    modules = [vlm_code, vlm_diagram, vlm_text, vlm_classify, note_generation]
    for module in modules:
        assert module.PROMPT_NAME
        assert re.match(r"v\d+\.\d+\.\d+", module.PROMPT_VERSION)
        assert module.PROMPT


def test_vlm_prompts_require_json_only_and_schema_version() -> None:
    """VLM prompts should enforce JSON-only response and schema_version."""
    for prompt in (vlm_code.PROMPT, vlm_diagram.PROMPT, vlm_text.PROMPT):
        assert "schema_version" in prompt
        assert "Output ONLY valid JSON" in prompt
        assert "errors" in prompt
        assert "has_truncation" in prompt


def test_note_generation_prompt_requires_escaped_markdown() -> None:
    """Note generation prompt should pin escaped markdown contract."""
    prompt = note_generation.PROMPT
    assert "note_markdown" in prompt
    assert "\\n" in prompt or "escaped newlines" in prompt
    assert "Output ONLY valid JSON" in prompt


def test_vlm_classify_prompt_includes_chart_output() -> None:
    """The classifier prompt should expose chart as a first-class output type."""
    assert '"chart"' in vlm_classify.PROMPT
    assert "quantitative visualization" in vlm_classify.PROMPT
