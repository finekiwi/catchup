"""Tests for vlm/client.py — all API calls are mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inject stub modules for packages that may not be installed in test env
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# anthropic stub
if "anthropic" not in sys.modules:
    _make_stub_module("anthropic")

# google / google.generativeai stub
if "google" not in sys.modules:
    _make_stub_module("google")
if "google.generativeai" not in sys.modules:
    _make_stub_module("google.generativeai")

# PIL is installed (Pillow) — no stub needed; tests patch PIL.Image.open directly


from vlm.client import call_vlm, _compute_cost, _MODEL_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_image(tmp_path: Path) -> Path:
    """Minimal valid 1×1 PNG."""
    import struct
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    img = tmp_path / "test.png"
    img.write_bytes(sig + ihdr + idat + iend)
    return img


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------

def test_compute_cost_openai():
    cost = _compute_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert abs(cost - expected) < 1e-10


def test_compute_cost_unknown_model_returns_zero():
    cost = _compute_cost("unknown-model", 1000, 500)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# Unsupported model
# ---------------------------------------------------------------------------

def test_call_vlm_unsupported_model_raises(fake_image: Path):
    with pytest.raises(ValueError, match="Unsupported model"):
        call_vlm("gpt-99-turbo", fake_image, "describe this")


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

def _make_openai_response(content: str, input_tokens: int, output_tokens: int):
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_call_vlm_openai_success(fake_image: Path):
    mock_resp = _make_openai_response("hello from gpt", 100, 50)

    with patch("openai.OpenAI") as mock_cls, \
         patch("vlm.client.log_api_call") as mock_log:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp

        result = call_vlm("gpt-4o-mini", fake_image, "describe this")

    assert result.success is True
    assert result.content == "hello from gpt"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.latency_ms >= 0
    assert result.cost_usd > 0
    mock_log.assert_called_once()


def test_call_vlm_openai_failure(fake_image: Path):
    with patch("openai.OpenAI") as mock_cls, \
         patch("vlm.client.log_api_call"):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("rate limit")

        result = call_vlm("gpt-4o-mini", fake_image, "describe this")

    assert result.success is False
    assert result.content == ""
    assert "rate limit" in result.error


# ---------------------------------------------------------------------------
# Google provider
# ---------------------------------------------------------------------------

def _make_google_response(text: str, prompt_tokens: int, candidates_tokens: int):
    meta = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=candidates_tokens)
    return SimpleNamespace(text=text, usage_metadata=meta)


def test_call_vlm_google_success(fake_image: Path):
    mock_resp = _make_google_response("hello from gemini", 80, 40)
    mock_genai = MagicMock()

    with patch.dict(sys.modules, {"google.generativeai": mock_genai}), \
         patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}), \
         patch("PIL.Image.open"), \
         patch("vlm.client.log_api_call"):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = mock_resp

        result = call_vlm("gemini-3-flash-preview", fake_image, "describe this")

    assert result.success is True
    assert result.content == "hello from gemini"
    assert result.input_tokens == 80
    assert result.output_tokens == 40


def test_call_vlm_google_failure(fake_image: Path):
    mock_genai = MagicMock()

    with patch.dict(sys.modules, {"google.generativeai": mock_genai}), \
         patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}), \
         patch("PIL.Image.open"), \
         patch("vlm.client.log_api_call"):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = ConnectionError("timeout")

        result = call_vlm("gemini-3-flash-preview", fake_image, "describe this")

    assert result.success is False
    assert result.content == ""


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _make_anthropic_response(text: str, input_tokens: int, output_tokens: int):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    content_block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[content_block], usage=usage)


def test_call_vlm_anthropic_success(fake_image: Path):
    mock_resp = _make_anthropic_response("hello from claude", 120, 60)
    mock_anthropic = MagicMock()

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}), \
         patch("vlm.client.log_api_call"):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_resp

        result = call_vlm("claude-sonnet-4-6", fake_image, "describe this")

    assert result.success is True
    assert result.content == "hello from claude"
    assert result.input_tokens == 120
    assert result.output_tokens == 60
    assert result.cost_usd > 0


def test_call_vlm_anthropic_failure(fake_image: Path):
    mock_anthropic = MagicMock()

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}), \
         patch("vlm.client.log_api_call"):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("api error")

        result = call_vlm("claude-haiku-4-5-20251001", fake_image, "describe this")

    assert result.success is False
    assert result.content == ""


# ---------------------------------------------------------------------------
# log_api_call is always invoked
# ---------------------------------------------------------------------------

def test_log_api_call_always_called_on_success(fake_image: Path):
    mock_resp = _make_openai_response("ok", 10, 5)

    with patch("openai.OpenAI") as mock_cls, \
         patch("vlm.client.log_api_call") as mock_log:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_resp

        call_vlm("gpt-4o", fake_image, "test", stage="test_stage")

    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["stage"] == "test_stage"
    assert kwargs["success"] is True


def test_log_api_call_always_called_on_failure(fake_image: Path):
    with patch("openai.OpenAI") as mock_cls, \
         patch("vlm.client.log_api_call") as mock_log:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("fail")

        call_vlm("gpt-4o-mini", fake_image, "test")

    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["success"] is False


# ---------------------------------------------------------------------------
# SUPPORTED_MODELS completeness
# ---------------------------------------------------------------------------

def test_supported_models_contains_all_providers():
    actual_providers = {v["provider"] for v in _MODEL_REGISTRY.values()}
    assert actual_providers == {"openai", "google", "anthropic"}
