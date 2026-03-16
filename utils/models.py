"""Shared model registry and provider dispatch helpers."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "anthropic", "google"]


class ModelConfig(TypedDict):
    """Static model metadata shared across modules."""

    provider: Provider
    input: float
    output: float
    context_window: NotRequired[int]


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4.1-nano": {"provider": "openai", "input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"provider": "openai", "input": 0.15, "output": 0.60},
    "gpt-4o": {"provider": "openai", "input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"provider": "openai", "input": 0.40, "output": 1.60},
    "gpt-5-nano": {"provider": "openai", "input": 0.20, "output": 0.80},
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "input": 0.80,
        "output": 4.00,
    },
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    "gemini-3-flash-preview": {"provider": "google", "input": 0.10, "output": 0.40},
    "gemini-3.1-pro-preview": {"provider": "google", "input": 1.25, "output": 10.00},
    "gemini-3.1-flash-lite-preview": {
        "provider": "google",
        "input": 0.04,
        "output": 0.15,
    },
}

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class LLMResponse:
    """Normalized response from a text LLM call."""

    content: str
    input_tokens: int
    output_tokens: int


@dataclass
class VLMDispatchResult:
    """Normalized response from a vision LLM call."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: str | None = None


def get_model_config(model: str) -> ModelConfig:
    """Return registry metadata for a model or raise for unknown names."""
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model: {model!r}. Choose from: {list(MODEL_REGISTRY)}"
        ) from exc


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model call."""
    info = MODEL_REGISTRY.get(model, {})
    return (
        input_tokens * info.get("input", 0) / 1_000_000
        + output_tokens * info.get("output", 0) / 1_000_000
    )


def _split_system_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Separate system messages from regular chat messages."""
    system_parts: list[str] = []
    non_system: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            system_parts.append(content)
        else:
            non_system.append({"role": role, "content": content})
    if not non_system:
        non_system = [{"role": "user", "content": ""}]
    return "\n\n".join(system_parts), non_system


def _flatten_google_messages(messages: list[dict[str, str]]) -> str:
    """Serialize multi-turn history into a single Gemini prompt."""
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def _call_openai_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> LLMResponse:
    import openai  # lazy import

    client = openai.OpenAI()
    openai_messages: list[dict[str, str]] = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    return LLMResponse(
        content=resp.choices[0].message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )


def _call_anthropic_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> LLMResponse:
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return LLMResponse(
        content=resp.content[0].text if resp.content else "",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def _call_google_llm(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> LLMResponse:
    del max_tokens

    import google.generativeai as genai  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    kwargs: dict[str, Any] = {}
    if system:
        kwargs["system_instruction"] = system
    gemini = genai.GenerativeModel(model, **kwargs)
    resp = gemini.generate_content(_flatten_google_messages(messages))
    meta = getattr(resp, "usage_metadata", None)
    return LLMResponse(
        content=resp.text or "",
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
    )


def call_llm(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    response_format: dict[str, Any] | None = None,
) -> LLMResponse:
    """Dispatch a text LLM call to the provider configured for model."""
    config = get_model_config(model)
    system, non_system = _split_system_messages(messages)
    provider = config["provider"]
    if provider == "openai":
        return _call_openai_llm(
            model,
            system,
            non_system,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    if provider == "anthropic":
        return _call_anthropic_llm(
            model,
            system,
            non_system,
            max_tokens=max_tokens,
        )
    return _call_google_llm(
        model,
        system,
        non_system,
        max_tokens=max_tokens,
    )


def _encode_image_b64(image_path: str | Path) -> tuple[str, str]:
    """Return the base64 payload and media type for an image path."""
    path = Path(image_path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return payload, media_type


def _call_openai_vlm(
    model: str,
    image_path: str | Path,
    prompt: str,
    *,
    max_tokens: int,
) -> VLMDispatchResult:
    import openai  # lazy import

    client = openai.OpenAI()
    payload, media_type = _encode_image_b64(image_path)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{payload}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        return VLMDispatchResult(
            content=resp.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
            success=True,
        )
    except Exception as exc:
        return VLMDispatchResult(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )


def _call_google_vlm(
    model: str,
    image_path: str | Path,
    prompt: str,
    *,
    max_tokens: int,
) -> VLMDispatchResult:
    del max_tokens

    import google.generativeai as genai  # lazy import
    import PIL.Image  # lazy import

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    image = PIL.Image.open(image_path)
    gemini = genai.GenerativeModel(model)

    t0 = time.perf_counter()
    try:
        resp = gemini.generate_content([prompt, image])
        meta = getattr(resp, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        return VLMDispatchResult(
            content=resp.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
            success=True,
        )
    except Exception as exc:
        return VLMDispatchResult(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )


def _call_anthropic_vlm(
    model: str,
    image_path: str | Path,
    prompt: str,
    *,
    max_tokens: int,
) -> VLMDispatchResult:
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    payload, media_type = _encode_image_b64(image_path)

    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": payload,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        return VLMDispatchResult(
            content=resp.content[0].text if resp.content else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
            success=True,
        )
    except Exception as exc:
        return VLMDispatchResult(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )


def call_vlm(
    model: str,
    image_path: str | Path,
    prompt: str,
    *,
    max_tokens: int = 2048,
) -> VLMDispatchResult:
    """Dispatch a vision LLM call to the provider configured for model."""
    provider = get_model_config(model)["provider"]
    if provider == "openai":
        return _call_openai_vlm(model, image_path, prompt, max_tokens=max_tokens)
    if provider == "anthropic":
        return _call_anthropic_vlm(model, image_path, prompt, max_tokens=max_tokens)
    return _call_google_vlm(model, image_path, prompt, max_tokens=max_tokens)


__all__ = [
    "LLMResponse",
    "MODEL_REGISTRY",
    "ModelConfig",
    "Provider",
    "VLMDispatchResult",
    "call_llm",
    "call_vlm",
    "compute_cost",
    "get_model_config",
]
