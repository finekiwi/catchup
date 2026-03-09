"""VLM API client — pure call layer.

Accepts (model, image_path, prompt) → VLMResult.
Prompt selection / routing lives in image_parser.py (CU-05).

Supported providers:
- OpenAI   : gpt-4o-mini, gpt-4o
- Google   : gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview
- Anthropic: claude-haiku-4-5-20251001, claude-sonnet-4-6
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

from utils.logging import log_api_call

load_dotenv()

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry: model_id → {provider, input_cost_per_1m, output_cost_per_1m}
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini":               {"provider": "openai",     "input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"provider": "openai",     "input": 2.50,  "output": 10.00},
    "gpt-4.1-mini":              {"provider": "openai",     "input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":              {"provider": "openai",     "input": 0.10,  "output": 0.40},
    "gpt-5-nano":                {"provider": "openai",     "input": 0.20,  "output": 0.80},
    "gemini-3-flash-preview":         {"provider": "google",     "input": 0.10,  "output": 0.40},
    "gemini-3.1-pro-preview":         {"provider": "google",     "input": 1.25,  "output": 10.00},
    "gemini-3.1-flash-lite-preview":  {"provider": "google",     "input": 0.04,  "output": 0.15},
    "claude-haiku-4-5-20251001": {"provider": "anthropic",  "input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"provider": "anthropic",  "input": 3.00,  "output": 15.00},
}

SUPPORTED_MODELS: list[str] = list(_MODEL_REGISTRY.keys())

_MEDIA_TYPES: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """Result of a single VLM API call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute estimated cost in USD from token counts."""
    info = _MODEL_REGISTRY.get(model, {})
    return (
        input_tokens  * info.get("input",  0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


def _encode_image_b64(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for the given image file."""
    path = Path(image_path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return b64, media_type


# ---------------------------------------------------------------------------
# Provider-level call functions
# ---------------------------------------------------------------------------

def _call_openai(model: str, image_path: str | Path, prompt: str) -> VLMResult:
    """Call OpenAI vision API."""
    import openai  # lazy import — only required if using OpenAI models

    client = openai.OpenAI()
    b64, media_type = _encode_image_b64(image_path)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=2048,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        content = resp.choices[0].message.content or ""
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        cost = _compute_cost(model, input_tokens, output_tokens)
        return VLMResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=True,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return VLMResult(content="", model=model, latency_ms=latency_ms, success=False, error=str(exc))


def _call_google(model: str, image_path: str | Path, prompt: str) -> VLMResult:
    """Call Google Gemini vision API."""
    import google.generativeai as genai  # lazy import
    import PIL.Image  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    img = PIL.Image.open(image_path)
    gemini = genai.GenerativeModel(model)

    t0 = time.perf_counter()
    try:
        resp = gemini.generate_content([prompt, img])
        latency_ms = (time.perf_counter() - t0) * 1000
        content = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        cost = _compute_cost(model, input_tokens, output_tokens)
        return VLMResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=True,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return VLMResult(content="", model=model, latency_ms=latency_ms, success=False, error=str(exc))


def _call_anthropic(model: str, image_path: str | Path, prompt: str) -> VLMResult:
    """Call Anthropic Claude vision API."""
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    b64, media_type = _encode_image_b64(image_path)

    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        content = resp.content[0].text if resp.content else ""
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        cost = _compute_cost(model, input_tokens, output_tokens)
        return VLMResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=True,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return VLMResult(content="", model=model, latency_ms=latency_ms, success=False, error=str(exc))


_PROVIDER_DISPATCH = {
    "openai":    _call_openai,
    "google":    _call_google,
    "anthropic": _call_anthropic,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_vlm(
    model: str,
    image_path: str | Path,
    prompt: str,
    stage: str = "vlm_parse",
) -> VLMResult:
    """Call a VLM model with an image and a prompt string.

    Prompt selection is the caller's responsibility (see image_parser.py).

    Args:
        model: Model identifier. Must be one of SUPPORTED_MODELS.
        image_path: Path to the image file (JPEG / PNG / GIF / WebP).
        prompt: Prompt string to send alongside the image.
        stage: Pipeline stage name used for observability logging.

    Returns:
        VLMResult with content, token counts, latency, cost, and success flag.
        On API failure, content is "" and success is False — never raises.
    """
    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model!r}. Choose from: {SUPPORTED_MODELS}")

    provider = _MODEL_REGISTRY[model]["provider"]
    result = _PROVIDER_DISPATCH[provider](model, image_path, prompt)

    log_api_call(
        model=model,
        stage=stage,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        success=result.success,
        error=result.error,
    )

    if not result.success:
        LOGGER.warning("VLM call failed: model=%s stage=%s error=%s", model, stage, result.error)

    return result
