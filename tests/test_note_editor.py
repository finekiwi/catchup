"""Tests for llm/note_editor.py — section splitting, merging, edit intent, and edit_section()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm.note_editor import (
    NoteEditResult,
    _find_section_by_query,
    _merge_sections,
    _split_sections,
    _strip_markdown_fence,
    detect_edit_intent,
    edit_section,
)

# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
## 개요

첫 번째 섹션 내용입니다.

## 핵심 개념

핵심 개념 내용.

### 세부 항목

세부 항목 내용.

## 코드 분석

코드 분석 내용.
"""


def test_split_sections_basic():
    sections = _split_sections(SAMPLE_MD)
    headings = [h for h, _ in sections]
    assert headings == ["## 개요", "## 핵심 개념", "## 코드 분석"]


def test_split_sections_body_content():
    sections = _split_sections(SAMPLE_MD)
    heading_map = {h: b for h, b in sections}
    assert "첫 번째 섹션 내용입니다." in heading_map["## 개요"]
    assert "핵심 개념 내용." in heading_map["## 핵심 개념"]
    assert "세부 항목" in heading_map["## 핵심 개념"]  # ### is part of parent section


def test_split_sections_nested_headings_not_split_point():
    """### headings must not create a new section split."""
    sections = _split_sections(SAMPLE_MD)
    # Only 3 sections; ### is part of ## 핵심 개념
    assert len(sections) == 3


def test_split_sections_preamble():
    """Content before the first ## heading becomes ("", preamble)."""
    md = "Intro paragraph.\n\n## Section A\n\nBody."
    sections = _split_sections(md)
    assert sections[0] == ("", "Intro paragraph.")
    assert sections[1][0] == "## Section A"


def test_split_sections_empty():
    sections = _split_sections("")
    assert sections == [("", "")]


def test_split_sections_single_section():
    md = "## Only\n\nSome content."
    sections = _split_sections(md)
    assert len(sections) == 1
    assert sections[0][0] == "## Only"
    assert "Some content." in sections[0][1]


def test_split_sections_code_fence_with_heading():
    """## inside code fence must NOT create a new section."""
    md = "## 코드 분석\n\n```python\n## utility function\ndef foo(): pass\n```\n"
    sections = _split_sections(md)
    assert len(sections) == 1
    assert sections[0][0] == "## 코드 분석"
    assert "## utility function" in sections[0][1]


def test_split_sections_unclosed_fence():
    """Unclosed code fence: remaining content stays in same section."""
    md = "## 섹션A\n\n```python\n## 헤딩처럼생긴주석\nsome code\n"
    sections = _split_sections(md)
    assert len(sections) == 1


# ---------------------------------------------------------------------------
# _merge_sections
# ---------------------------------------------------------------------------


def test_merge_sections_replaces_target():
    sections = _split_sections(SAMPLE_MD)
    new_body = "완전히 새로운 내용으로 교체됩니다."
    result = _merge_sections(sections, 1, new_body)
    assert "완전히 새로운 내용으로 교체됩니다." in result
    assert "핵심 개념 내용." not in result  # original body replaced
    assert "## 개요" in result
    assert "## 코드 분석" in result


def test_merge_sections_preserves_other_sections():
    sections = _split_sections(SAMPLE_MD)
    result = _merge_sections(sections, 0, "새 개요 내용.")
    assert "새 개요 내용." in result
    assert "핵심 개념 내용." in result
    assert "코드 분석 내용." in result


# ---------------------------------------------------------------------------
# _find_section_by_query
# ---------------------------------------------------------------------------


def test_find_section_by_query_exact_substring():
    sections = _split_sections(SAMPLE_MD)
    idx = _find_section_by_query(sections, "핵심 개념")
    assert idx == 1


def test_find_section_by_query_partial():
    sections = _split_sections(SAMPLE_MD)
    idx = _find_section_by_query(sections, "코드")
    assert idx == 2


def test_find_section_by_query_no_match():
    sections = _split_sections(SAMPLE_MD)
    idx = _find_section_by_query(sections, "존재하지않는섹션")
    assert idx is None


def test_find_section_by_query_multiple_matches_returns_first(caplog):
    """When multiple sections match, return first and log a warning."""
    md = "## 개념 A\n\nA.\n\n## 개념 B\n\nB."
    sections = _split_sections(md)
    import logging

    with caplog.at_level(logging.WARNING, logger="llm.note_editor"):
        idx = _find_section_by_query(sections, "개념")
    assert idx == 0
    assert any("Multiple sections" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _strip_markdown_fence
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_removes_fence():
    text = "```markdown\nSome content.\n```"
    assert _strip_markdown_fence(text) == "Some content."


def test_strip_markdown_fence_no_fence():
    text = "Plain text."
    assert _strip_markdown_fence(text) == "Plain text."


# ---------------------------------------------------------------------------
# detect_edit_intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "핵심 개념 섹션에 코드 예제 추가해줘",
        "요약 부분 좀 수정해줘",
        "이 섹션 삭제해줘",
        "내용을 좀 더 자세하게 바꿔줘",
        "add a code example here",
        "remove this section",
    ],
)
def test_detect_edit_intent_positive(message: str):
    assert detect_edit_intent(message, has_note=True) is True


@pytest.mark.parametrize(
    "message",
    [
        "핵심 개념이 뭔지 설명해줘",
        "이 부분 요약해줘",
        "두 개념을 비교해줘",
        "어떻게 작동하는지 알려줘",
        "explain how it works",
        "what is the difference",
        "describe how this works",
    ],
)
def test_detect_edit_intent_negative_qa(message: str):
    """Q&A override keywords should prevent edit intent detection."""
    assert detect_edit_intent(message, has_note=True) is False


def test_detect_edit_intent_no_note():
    """Without a note loaded, never detect edit intent."""
    assert detect_edit_intent("이 섹션 수정해줘", has_note=False) is False


# ---------------------------------------------------------------------------
# edit_section
# ---------------------------------------------------------------------------


def _make_mock_call_fn(return_text: str):
    """Return a mock provider call function that returns (text, 100, 50)."""
    mock = MagicMock(return_value=(return_text, 100, 50))
    return mock


@patch("llm.note_editor._PROVIDER_DISPATCH")
@patch("llm.note_editor.log_api_call")
def test_edit_section_success(mock_log, mock_dispatch):
    mock_dispatch.__getitem__ = MagicMock(
        return_value=_make_mock_call_fn("업데이트된 내용입니다.")
    )

    result = edit_section(
        full_markdown=SAMPLE_MD,
        section_heading="## 개요",
        instruction="더 자세하게 써줘",
        model="gpt-4o-mini",
    )

    assert result.success is True
    assert "업데이트된 내용입니다." in result.edited_markdown
    assert result.edited_section_body == "업데이트된 내용입니다."
    assert result.edited_section == "## 개요"
    assert "## 핵심 개념" in result.edited_markdown
    assert "## 코드 분석" in result.edited_markdown
    mock_log.assert_called_once()


@patch("llm.note_editor._PROVIDER_DISPATCH")
@patch("llm.note_editor.log_api_call")
def test_edit_section_fuzzy_heading(mock_log, mock_dispatch):
    """Fuzzy heading match should work when exact heading not given."""
    mock_dispatch.__getitem__ = MagicMock(
        return_value=_make_mock_call_fn("새 핵심 개념 내용.")
    )

    result = edit_section(
        full_markdown=SAMPLE_MD,
        section_heading="핵심",  # partial, not "## 핵심 개념"
        instruction="예제 추가해줘",
        model="gpt-4o-mini",
    )

    assert result.success is True
    assert "새 핵심 개념 내용." in result.edited_markdown


def test_edit_section_invalid_heading():
    """Unknown heading returns failure result without modifying markdown."""
    result = edit_section(
        full_markdown=SAMPLE_MD,
        section_heading="## 없는섹션",
        instruction="뭔가 해줘",
        model="gpt-4o-mini",
    )

    assert result.success is False
    assert result.error is not None
    assert result.edited_markdown == SAMPLE_MD  # unchanged


def test_edit_section_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        edit_section(
            full_markdown=SAMPLE_MD,
            section_heading="## 개요",
            instruction="테스트",
            model="nonexistent-model",
        )


@patch("llm.note_editor._PROVIDER_DISPATCH")
@patch("llm.note_editor.log_api_call")
def test_edit_section_api_failure(mock_log, mock_dispatch):
    """API errors should return failure result without raising."""
    mock_fn = MagicMock(side_effect=RuntimeError("API timeout"))
    mock_dispatch.__getitem__ = MagicMock(return_value=mock_fn)

    result = edit_section(
        full_markdown=SAMPLE_MD,
        section_heading="## 개요",
        instruction="업데이트",
        model="gpt-4o-mini",
    )

    assert result.success is False
    assert "API timeout" in result.error
    assert result.edited_markdown == SAMPLE_MD  # unchanged


@patch("llm.note_editor._PROVIDER_DISPATCH")
@patch("llm.note_editor.log_api_call")
def test_edit_section_multiturn(mock_log, mock_dispatch):
    """Multi-turn history is passed to the provider call function."""
    call_fn = _make_mock_call_fn("멀티턴 수정 결과.")
    mock_dispatch.__getitem__ = MagicMock(return_value=call_fn)

    history = [
        {"role": "user", "content": "코드 예제 추가해줘"},
        {"role": "assistant", "content": "코드 예제를 추가했습니다."},
    ]

    result = edit_section(
        full_markdown=SAMPLE_MD,
        section_heading="## 개요",
        instruction="예시를 파이썬으로 바꿔줘",
        model="gpt-4o-mini",
        history=history,
    )

    assert result.success is True
    # The call should have received history + current instruction
    called_messages = call_fn.call_args[0][2]  # third positional arg: messages
    assert len(called_messages) == 3  # 2 history + 1 current
    assert called_messages[-1]["content"] == "예시를 파이썬으로 바꿔줘"
