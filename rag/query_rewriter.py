"""Query rewriter for retrieval-friendly expansion.

Uses an LLM to expand abbreviations, translate Korean technical terms to English,
and decompose compound words before embedding. Gracefully falls back to the
original question on any failure.
"""

from __future__ import annotations

import logging
import os
import time

import openai

from prompts.query_rewrite import PROMPT
from utils.logging import log_api_call

LOGGER = logging.getLogger(__name__)

_REWRITE_MODEL = "gpt-4.1-nano"
_MAX_TOKENS = 128
_REWRITE_COST_INPUT_PER_1M = 0.10   # USD per 1M input tokens for gpt-4.1-nano
_REWRITE_COST_OUTPUT_PER_1M = 0.40  # USD per 1M output tokens for gpt-4.1-nano


def rewrite_query(
    question: str,
    model: str = _REWRITE_MODEL,
) -> tuple[str, float, int, int]:
    """Rewrite a user query for better embedding retrieval.

    Expands abbreviations, adds Korean↔English synonyms, and decomposes compound
    words so the rewritten query has higher embedding similarity to document text.

    Args:
        question: Original user question.
        model: OpenAI model to use for rewriting. Defaults to gpt-4.1-nano.

    Returns:
        Tuple of (rewritten_query, latency_ms, input_tokens, output_tokens).
        On any failure, returns (original question, 0.0, 0, 0).
    """
    t_start = time.perf_counter()
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000
        rewritten = (response.choices[0].message.content or "").strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if not rewritten:
            LOGGER.warning("rewrite_query: empty response, returning original question")
            return question, latency_ms, input_tokens, output_tokens

        cost_usd = (
            input_tokens * _REWRITE_COST_INPUT_PER_1M / 1_000_000
            + output_tokens * _REWRITE_COST_OUTPUT_PER_1M / 1_000_000
        )
        log_api_call(
            model=model,
            stage="query_rewrite",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=True,
        )
        return rewritten, latency_ms, input_tokens, output_tokens

    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        LOGGER.warning("rewrite_query failed (%s), returning original question", exc)
        log_api_call(
            model=model,
            stage="query_rewrite",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            success=False,
            error=str(exc),
        )
        return question, latency_ms, 0, 0
