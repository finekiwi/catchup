"""Utility functions for structured API call logging.

Logs each API call to a local JSON Lines file (always) and optionally to Langfuse
when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment.

Langfuse integration is optional — if the package is not installed or keys are
missing, logging falls back to JSONL only without raising any errors.

Session grouping:
    Pass ``session_id`` to ``log_api_call`` to group related observations under
    a single Langfuse session (e.g. one Streamlit session or one eval run).
    Uses ``langfuse.propagate_attributes(session_id=...)`` from the v4 SDK.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_API_LOG_PATH = "data/logs/api_calls.jsonl"

# Thread-local storage for the active session ID set by langfuse_session()
_local = threading.local()

# ---------------------------------------------------------------------------
# Langfuse client — initialized lazily, None if unavailable
# ---------------------------------------------------------------------------

_langfuse_client = None
_langfuse_init_attempted = False


def _get_langfuse_client():
    """Return a Langfuse v4 client, or None if unavailable.

    Initialization is attempted once. Subsequent calls return the cached result.
    Fails silently (returns None) if langfuse is not installed or keys are missing.
    """
    global _langfuse_client, _langfuse_init_attempted
    if _langfuse_init_attempted:
        return _langfuse_client

    _langfuse_init_attempted = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        LOGGER.debug("Langfuse keys not set — using JSONL-only logging")
        return None

    try:
        from langfuse import get_client as _lf_get_client  # lazy import

        _langfuse_client = _lf_get_client()
        _langfuse_client.auth_check()
        LOGGER.info("Langfuse client initialized (host=%s)", host)
    except Exception as exc:
        LOGGER.warning("Langfuse init failed — falling back to JSONL-only: %s", exc)
        _langfuse_client = None

    return _langfuse_client


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _api_log_path() -> Path:
    """Return API call log path, optionally overridden by environment variable."""
    return Path(os.getenv("CATCHUP_API_LOG_PATH", DEFAULT_API_LOG_PATH))


def _write_jsonl(payload: dict) -> None:
    """Append one record to the JSONL log file."""
    log_path = _api_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        LOGGER.exception("Failed to write API call log to path=%s", log_path)


# ---------------------------------------------------------------------------
# Langfuse helpers
# ---------------------------------------------------------------------------


def _send_to_langfuse(client, payload: dict, session_id: Optional[str] = None) -> None:
    """Send one API call record to Langfuse as a generation observation (v4 SDK).

    Uses ``start_as_current_observation(as_type="generation")`` to record the call.
    If ``session_id`` is provided, wraps the observation with
    ``propagate_attributes(session_id=...)`` to group it under a Langfuse session.

    Args:
        client: Initialized Langfuse v4 client (from ``get_client()``).
        payload: The same dict written to JSONL.
        session_id: Optional session identifier for grouping related observations.
    """
    try:
        from langfuse import propagate_attributes  # lazy import (v4)

        # session_id: explicit arg takes priority, then thread-local from langfuse_session()
        effective_sid = session_id or getattr(_local, "session_id", None)

        # propagate_attributes goes INSIDE the outer span: it sets session_id on the
        # currently active span (root_span) AND propagates to all child observations.
        with client.start_as_current_observation(
            as_type="span",
            name=payload["stage"],
        ) as root_span:
            ctx = propagate_attributes(session_id=effective_sid) if effective_sid else nullcontext()
            with ctx:
                with root_span.start_as_current_observation(
                    as_type="generation",
                    name=payload["stage"],
                    model=payload["model"],
                    input=payload.get("input_text"),
                    output=payload.get("output_text"),
                ) as gen:
                    gen.update(
                        usage={
                            "input": payload["input_tokens"],
                            "output": payload["output_tokens"],
                            "total": payload["input_tokens"] + payload["output_tokens"],
                        },
                        metadata={
                            "latency_ms": payload["latency_ms"],
                            "cost_usd": payload["cost_usd"],
                            "success": payload["success"],
                            "error": payload["error"],
                        },
                    )

    except Exception as exc:
        LOGGER.warning("Langfuse generation send failed: %s", exc)


# ---------------------------------------------------------------------------
# Session context manager (optional helper for callers)
# ---------------------------------------------------------------------------


@contextmanager
def langfuse_session(session_id: str) -> Iterator[None]:
    """Context manager to group all ``log_api_call`` invocations under a session.

    Usage (e.g. Streamlit session or eval run)::

        with langfuse_session("streamlit-session-abc123"):
            result = query("What is X?")         # log_api_call called internally
            note = generate_note(doc)            # log_api_call called internally

    All calls within the block share the same ``session_id`` in Langfuse,
    enabling session replay. Falls back to a no-op if Langfuse is unavailable.

    Args:
        session_id: Identifier for the session (max 200 chars, US-ASCII).
    """
    # Store session_id in thread-local so nested log_api_call() calls pick it up
    # without requiring explicit session_id= on every call.
    previous = getattr(_local, "session_id", None)
    _local.session_id = session_id
    try:
        yield
    finally:
        _local.session_id = previous
        # Flush buffered Langfuse observations once at session exit
        # instead of per-call in _send_to_langfuse (avoids SDK batching defeat)
        client = _get_langfuse_client()
        if client is not None:
            try:
                client.flush()
            except Exception:
                LOGGER.debug("Langfuse flush failed at session exit")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_api_call(
    model: str,
    stage: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_usd: float,
    success: bool,
    error: Optional[str] = None,
    session_id: Optional[str] = None,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
) -> dict:
    """Append one API call record to JSON Lines log file and optionally to Langfuse.

    The JSONL log is always written. Langfuse is used additionally when
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini", "text-embedding-3-small").
        stage: Pipeline stage name (e.g. "rag_embed", "rag_generate", "eval_run").
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        latency_ms: End-to-end latency in milliseconds.
        cost_usd: Estimated cost in USD.
        success: Whether the API call succeeded.
        error: Optional error message if the call failed.
        session_id: Optional Langfuse session ID for grouping (max 200 chars).
                    Prefer using the ``langfuse_session()`` context manager instead
                    of passing this per-call.

    Returns:
        The logged payload dict.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "success": success,
        "error": error,
        "input_text": input_text,
        "output_text": output_text,
    }

    # Always write to JSONL
    _write_jsonl(payload)

    # Optionally send to Langfuse
    client = _get_langfuse_client()
    if client is not None:
        _send_to_langfuse(client, payload, session_id=session_id)

    return payload
