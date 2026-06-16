"""Tests for grammar production truth smoke gate helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from grammar_truth_gate import (
    format_degraded_reason_groups,
    format_mode_summary,
    validate_truth_payload,
)


def test_validate_sql_writer_payload_requires_core_fields() -> None:
    errors = validate_truth_payload("sql-writer", {"degraded": False, "degraded_reasons": []})
    assert errors
    assert any("missing fields" in e for e in errors)


def test_validate_flags_degraded_without_reasons() -> None:
    payload = {k: None for k in [
        "ok", "degraded", "degraded_reasons", "grammar_channel_enabled", "subscribed_channels",
        "grammar_worker_count", "grammar_queue", "grammar_fallbacks", "latest_by_source_service",
        "grammar_index", "grammar_retention",
    ]}
    payload["degraded"] = True
    payload["degraded_reasons"] = []
    errors = validate_truth_payload("sql-writer", payload)
    assert any("degraded=true but degraded_reasons empty" in e for e in errors)


def test_format_mode_summary_includes_retention_and_channels() -> None:
    text = format_mode_summary(
        {
            "grammar_channel_enabled": True,
            "grammar_worker_count": 4,
            "subscribed_channels": ["orion:grammar:event"],
            "grammar_retention": {"enabled": True, "remaining_debt": 0, "failure_reason": None},
            "latest_by_source_service": [{"source_service": "orion-bus", "latest_created_at": "2026-06-13"}],
        },
        {
            "grammar_poll_interval_sec": 5.0,
            "enabled_reducers": {"transport_bus": True},
            "cursor_lag_by_reducer": {"biometrics_grammar_consumer": 12.0},
            "stream_lag_by_reducer": {"transport_grammar_reducer": 40000.0},
            "pending_backlog_by_reducer": {"transport_grammar_reducer": 1000},
            "degraded_reasons": ["cursor_lag:transport_grammar_reducer"],
            "reducer_health_by_name": {
                "transport_bus": {
                    "classification": "alive_behind",
                    "cursor_name": "transport_grammar_reducer",
                    "stream_lag_sec": 40000.0,
                    "last_tick_at": "2026-06-16T00:00:00+00:00",
                    "blocked_event_id": None,
                }
            },
            "tail_seed": {"count": 0},
            "accepted_pressure_output_channel": "orion:grammar:accepted-pressure",
        },
    )
    assert "grammar_enabled=True" in text
    assert "orion:grammar:accepted-pressure" in text
    assert "reducer health" in text
    assert "backlog=" in text


def test_format_degraded_reason_groups_classifies_cursor_lag() -> None:
    grouped = format_degraded_reason_groups(
        ["cursor_lag:transport_grammar_reducer", "reducer_heartbeat_stale:execution_grammar_reducer"]
    )
    assert "stale_reducer_cursor" in grouped
    assert "missing_reducer_heartbeat" in grouped


def test_validate_substrate_requires_reducer_health_fields() -> None:
    payload = {
        "ok": True,
        "degraded": False,
        "degraded_reasons": [],
        "enabled_reducers": {},
        "grammar_poll_interval_sec": 5.0,
        "cursor_settings": {},
        "cursor_positions": [],
        "cursor_lag_by_reducer": {},
        "tail_seed": {},
        "operator_cursor_reset": {},
        "accepted_pressure_output_channel": "x",
        "canonical_grammar_input_channel": "y",
    }
    errors = validate_truth_payload("substrate-runtime", payload)
    assert any("missing fields" in e for e in errors)
    assert "reducer_health_by_name" in errors[0]
