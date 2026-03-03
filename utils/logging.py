"""Utility functions for structured API call logging."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_API_LOG_PATH = "data/logs/api_calls.jsonl"


def _api_log_path() -> Path:
    """Return API call log path, optionally overridden by environment variable."""
    return Path(os.getenv("CATCHUP_API_LOG_PATH", DEFAULT_API_LOG_PATH))


def log_api_call(
    model: str,
    stage: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_usd: float,
    success: bool,
    error: Optional[str] = None,
) -> dict:
    """Append one API call record to JSON Lines log file and return the record."""
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
    }

    log_path = _api_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        LOGGER.exception("Failed to write API call log to path=%s", log_path)

    return payload

