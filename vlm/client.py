"""VLM API client — pure call layer.

Accepts (model, image_path, prompt) → VLMResult.
Prompt selection / routing lives in image_parser.py (CU-05).

Supported providers:
- OpenAI   : gpt-4o-mini, gpt-4o
- Google   : gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview
- Anthropic: claude-haiku-4-5-20251001, claude-sonnet-4-6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from utils.logging import log_api_call
from utils.models import MODEL_REGISTRY, call_vlm as _dispatch_vlm

LOGGER = logging.getLogger(__name__)

SUPPORTED_MODELS: list[str] = list(MODEL_REGISTRY.keys())


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
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {SUPPORTED_MODELS}"
        )

    result = _dispatch_vlm(model, image_path, prompt)
    wrapped = VLMResult(
        content=result.content,
        model=model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        success=result.success,
        error=result.error,
    )

    log_api_call(
        model=model,
        stage=stage,
        input_tokens=wrapped.input_tokens,
        output_tokens=wrapped.output_tokens,
        latency_ms=wrapped.latency_ms,
        cost_usd=wrapped.cost_usd,
        success=wrapped.success,
        error=wrapped.error,
    )

    if not wrapped.success:
        LOGGER.warning(
            "VLM call failed: model=%s stage=%s error=%s", model, stage, wrapped.error
        )

    return wrapped
