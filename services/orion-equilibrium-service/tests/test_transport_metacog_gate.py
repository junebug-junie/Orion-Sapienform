from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.transport_metacog_gate import (
    build_transport_metacog_trigger_from_grammar_atom,
    build_transport_metacog_trigger_from_snapshot,
)


def _snapshot_payload(**overrides) -> dict:
    base = {
        "service": "cortex-exec",
        "node": "athena",
        "instance": None,
        "window_start": "2026-07-24T00:00:00+00:00",
        "window_end": "2026-07-24T00:00:30+00:00",
        "success_count": 18,
        "timeout_count": 0,
        "success_latency_ms_p50": 12.0,
        "success_latency_ms_p95": 40.0,
        "success_latency_ms_max": 55.0,
        "timeout_elapsed_ms_max": None,
        "channel_counts": {"orion:cortex:exec:request": 18},
        "truncated": False,
    }
    base.update(overrides)
    return base


def _grammar_atom(**overrides) -> dict:
    base = {
        "semantic_role": "rpc_transport_timeout",
        "text_value": "orion:cortex:exec:request:background",
        "summary": "RPC timeout: orion:cortex:exec:request:background -> reply after 60.0s",
    }
    base.update(overrides)
    return base


# --- Option A: RpcHealthSnapshotV1-driven ---------------------------------


def test_snapshot_no_timeout_no_latency_spike_fires_nothing():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is None


def test_snapshot_empty_window_fires_nothing():
    """Absence of traffic is not evidence of transport trouble -- matches the
    rpc_health organ adapter's own healthy-by-absence rule."""
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(success_count=0, timeout_count=0, success_latency_ms_p95=None),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is None


def test_snapshot_real_timeout_fires():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(success_count=15, timeout_count=3),
        zen_state="zen",
        pressure=0.2,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is not None
    assert trigger.trigger_kind == "transport"
    assert "timeout_count=3" in trigger.upstream["fired_conditions"]
    assert trigger.upstream["evidence_source"] == "rpc_health_snapshot"
    assert trigger.upstream["timeout_count"] == 3
    assert trigger.upstream["service"] == "cortex-exec"


def test_snapshot_latency_spike_above_threshold_fires():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(success_latency_ms_p95=9000.0),
        zen_state="zen",
        pressure=0.2,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is not None
    assert any(c.startswith("success_latency_ms_p95=") for c in trigger.upstream["fired_conditions"])


def test_snapshot_latency_below_threshold_does_not_fire_alone():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(success_latency_ms_p95=100.0),
        zen_state="zen",
        pressure=0.2,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is None


def test_snapshot_both_conditions_fire_together():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(timeout_count=2, success_latency_ms_p95=9000.0),
        zen_state="not_zen",
        pressure=0.5,
        recall_enabled=False,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is not None
    assert len(trigger.upstream["fired_conditions"]) == 2


def test_snapshot_upstream_carries_full_evidence():
    trigger = build_transport_metacog_trigger_from_snapshot(
        _snapshot_payload(timeout_count=1),
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
        latency_p95_threshold_ms=5000.0,
    )
    assert trigger is not None
    for key in (
        "success_count",
        "timeout_count",
        "success_latency_ms_p50",
        "success_latency_ms_p95",
        "success_latency_ms_max",
        "channel_counts",
        "window_start",
        "window_end",
    ):
        assert key in trigger.upstream


# --- Option C: grammar-atom-driven -----------------------------------------


def test_grammar_atom_wrong_role_fires_nothing():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(semantic_role="exec_turn_timeout"),
        correlation_id="corr-1",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is None


def test_grammar_atom_non_dict_fires_nothing():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        None,  # type: ignore[arg-type]
        correlation_id="corr-1",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is None


def test_grammar_atom_rpc_timeout_always_fires():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(),
        correlation_id="corr-1",
        zen_state="not_zen",
        pressure=0.3,
        recall_enabled=True,
    )
    assert trigger is not None
    assert trigger.trigger_kind == "transport"
    assert trigger.upstream["evidence_source"] == "rpc_transport_timeout_grammar"
    assert trigger.upstream["fired_conditions"] == ["rpc_timeout"]
    assert trigger.upstream["request_channel"] == "orion:cortex:exec:request:background"
    assert trigger.signal_refs == ["corr-1"]


def test_grammar_atom_no_correlation_id_still_fires():
    trigger = build_transport_metacog_trigger_from_grammar_atom(
        _grammar_atom(),
        correlation_id="",
        zen_state="zen",
        pressure=0.1,
        recall_enabled=True,
    )
    assert trigger is not None
    assert trigger.signal_refs == []
