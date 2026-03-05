"""Tests for API logging utility."""

from __future__ import annotations

import json

from utils.logging import log_api_call


def test_log_api_call_writes_jsonl_record(tmp_path, monkeypatch) -> None:
    """log_api_call should append one JSON object per line."""
    log_path = tmp_path / "logs" / "api_calls.jsonl"
    monkeypatch.setenv("CATCHUP_API_LOG_PATH", str(log_path))

    record = log_api_call(
        model="gpt-4o-mini",
        stage="note_generation",
        input_tokens=120,
        output_tokens=320,
        latency_ms=450.5,
        cost_usd=0.0042,
        success=True,
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    persisted = json.loads(lines[0])
    assert persisted["model"] == "gpt-4o-mini"
    assert persisted["stage"] == "note_generation"
    assert persisted["input_tokens"] == 120
    assert persisted["output_tokens"] == 320
    assert persisted["latency_ms"] == 450.5
    assert persisted["cost_usd"] == 0.0042
    assert persisted["success"] is True
    assert persisted["error"] is None
    assert "timestamp" in persisted
    assert persisted == record

