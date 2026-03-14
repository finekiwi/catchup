"""Unit tests for rag/query_rewriter.py (CU-13)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.query_rewriter import rewrite_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(content: str, input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_rewrite_query_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """rewrite_query returns the expanded query string from the LLM response."""
    expanded = "MLP (Multilayer Perceptron, 다층 퍼셉트론) 구조"

    mock_response = _make_openai_response(expanded, input_tokens=20, output_tokens=12)

    mock_create = MagicMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("rag.query_rewriter.openai.OpenAI", return_value=mock_client):
        with patch("rag.query_rewriter.log_api_call") as mock_log:
            result, latency_ms, input_tok, output_tok = rewrite_query("MLP 구조")

    assert result == expanded
    assert input_tok == 20
    assert output_tok == 12
    assert latency_ms >= 0.0
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args.kwargs
    assert call_kwargs["success"] is True
    assert call_kwargs["stage"] == "query_rewrite"


# ---------------------------------------------------------------------------
# Failure / fallback path
# ---------------------------------------------------------------------------


def test_rewrite_query_api_exception_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """rewrite_query falls back to the original question on API exception."""
    question = "커밋 로그 보는 법"

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")

    with patch("rag.query_rewriter.openai.OpenAI", return_value=mock_client):
        with patch("rag.query_rewriter.log_api_call") as mock_log:
            result, latency_ms, input_tok, output_tok = rewrite_query(question)

    assert result == question
    assert input_tok == 0
    assert output_tok == 0
    assert latency_ms >= 0.0
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args.kwargs
    assert call_kwargs["success"] is False
    assert "API timeout" in call_kwargs["error"]


# ---------------------------------------------------------------------------
# Empty response fallback
# ---------------------------------------------------------------------------


def test_rewrite_query_empty_response_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """rewrite_query returns the original question when the LLM emits an empty string."""
    question = "RAG 파이프라인 설명"

    mock_response = _make_openai_response("", input_tokens=15, output_tokens=0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("rag.query_rewriter.openai.OpenAI", return_value=mock_client):
        with patch("rag.query_rewriter.log_api_call"):
            result, latency_ms, input_tok, output_tok = rewrite_query(question)

    assert result == question


# ---------------------------------------------------------------------------
# Whitespace-only response treated as empty
# ---------------------------------------------------------------------------


def test_rewrite_query_whitespace_response_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """rewrite_query returns the original question when the LLM response is whitespace only."""
    question = "노드 조회 방법"

    mock_response = _make_openai_response("   \n  ", input_tokens=10, output_tokens=2)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("rag.query_rewriter.openai.OpenAI", return_value=mock_client):
        with patch("rag.query_rewriter.log_api_call"):
            result, _latency, _in_tok, _out_tok = rewrite_query(question)

    assert result == question


# ---------------------------------------------------------------------------
# Custom model parameter is forwarded
# ---------------------------------------------------------------------------


def test_rewrite_query_uses_specified_model() -> None:
    """rewrite_query passes the caller-supplied model name to the OpenAI client."""
    custom_model = "gpt-4o-mini"
    mock_response = _make_openai_response("expanded query")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("rag.query_rewriter.openai.OpenAI", return_value=mock_client):
        with patch("rag.query_rewriter.log_api_call"):
            rewrite_query("some question", model=custom_model)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == custom_model
