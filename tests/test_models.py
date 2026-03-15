"""Unit tests for utils/models.py shared registry and dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utils.models import (
    MODEL_REGISTRY,
    LLMResponse,
    VLMDispatchResult,
    call_llm,
    call_vlm,
    compute_cost,
    get_model_config,
)


def _make_stub_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


if "anthropic" not in sys.modules:
    _make_stub_module("anthropic")
if "google" not in sys.modules:
    _make_stub_module("google")
if "google.generativeai" not in sys.modules:
    _make_stub_module("google.generativeai")


@pytest.fixture()
def fake_image(tmp_path: Path) -> Path:
    """Minimal valid 1×1 PNG for vision dispatch tests."""
    import struct
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = (
        struct.pack(">I", len(compressed))
        + b"IDAT"
        + compressed
        + struct.pack(">I", idat_crc)
    )
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    image = tmp_path / "test.png"
    image.write_bytes(sig + ihdr + idat + iend)
    return image


def _make_openai_response(content: str, input_tokens: int, output_tokens: int):
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_anthropic_response(content: str, input_tokens: int, output_tokens: int):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    block = SimpleNamespace(text=content)
    return SimpleNamespace(content=[block], usage=usage)


def _make_google_response(content: str, input_tokens: int, output_tokens: int):
    meta = SimpleNamespace(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
    )
    return SimpleNamespace(text=content, usage_metadata=meta)


def test_model_registry_contains_all_providers():
    providers = {config["provider"] for config in MODEL_REGISTRY.values()}
    assert providers == {"openai", "anthropic", "google"}


def test_get_model_config_returns_registry_entry():
    config = get_model_config("gpt-4o-mini")
    assert config["provider"] == "openai"
    assert config["input"] == 0.15


def test_get_model_config_unknown_model_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_config("gpt-99-missing")


def test_compute_cost_unknown_model_returns_zero():
    assert compute_cost("missing", 100, 50) == 0.0


def test_call_llm_openai_success():
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "hello", 11, 7
        )

        response = call_llm(
            "gpt-4o-mini",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hello"},
            ],
            response_format={"type": "json_object"},
        )

    assert response == LLMResponse(content="hello", input_tokens=11, output_tokens=7)
    assert mock_client.chat.completions.create.call_args.kwargs["response_format"] == {
        "type": "json_object"
    }


def test_call_llm_anthropic_success():
    mock_anthropic = MagicMock()

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response(
            "claude", 13, 9
        )

        response = call_llm(
            "claude-sonnet-4-6",
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "prompt"},
            ],
        )

    assert response == LLMResponse(content="claude", input_tokens=13, output_tokens=9)


def test_call_llm_google_success():
    mock_genai = MagicMock()

    with (
        patch.dict(sys.modules, {"google.generativeai": mock_genai}),
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
    ):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = _make_google_response("gemini", 5, 3)

        response = call_llm(
            "gemini-3-flash-preview",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    assert response == LLMResponse(content="gemini", input_tokens=5, output_tokens=3)


def test_call_llm_unknown_model_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        call_llm("not-registered", [{"role": "user", "content": "hi"}])


def test_call_vlm_openai_success(fake_image: Path):
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "vision", 17, 8
        )

        result = call_vlm("gpt-4o-mini", fake_image, "describe")

    assert isinstance(result, VLMDispatchResult)
    assert result.success is True
    assert result.content == "vision"
    assert result.cost_usd > 0


def test_call_vlm_google_success(fake_image: Path):
    mock_genai = MagicMock()

    with (
        patch.dict(sys.modules, {"google.generativeai": mock_genai}),
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch("PIL.Image.open"),
    ):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = _make_google_response(
            "gemini-vision", 9, 4
        )

        result = call_vlm("gemini-3-flash-preview", fake_image, "describe")

    assert result.success is True
    assert result.content == "gemini-vision"


def test_call_vlm_anthropic_success(fake_image: Path):
    mock_anthropic = MagicMock()

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response(
            "claude-vision", 12, 6
        )

        result = call_vlm("claude-sonnet-4-6", fake_image, "describe")

    assert result.success is True
    assert result.content == "claude-vision"


def test_call_vlm_failure_returns_error(fake_image: Path):
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("rate limit")

        result = call_vlm("gpt-4o-mini", fake_image, "describe")

    assert result.success is False
    assert result.content == ""
    assert "rate limit" in (result.error or "")
