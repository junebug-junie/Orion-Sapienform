from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.transport_metacog_gate import (
    build_transport_metacog_trigger_from_bus_synaptic,
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


class TestBusSynapticEvidence:
    def test_below_threshold_does_not_fire(self):
        trigger = build_transport_metacog_trigger_from_bus_synaptic(
            0.5,
            zen_state="zen",
            pressure=0.1,
            recall_enabled=True,
            error_threshold=1.0,
        )
        assert trigger is None

    def test_at_threshold_fires(self):
        trigger = build_transport_metacog_trigger_from_bus_synaptic(
            1.0,
            zen_state="not_zen",
            pressure=0.6,
            recall_enabled=True,
            error_threshold=1.0,
            edge_count=204,
        )
        assert trigger is not None
        assert trigger.trigger_kind == "transport"
        assert trigger.upstream["evidence_source"] == "bus_synaptic_prediction_error"
        assert trigger.upstream["error"] == 1.0
        assert trigger.upstream["edge_count"] == 204
        assert trigger.signal_refs == ["node:substrate.bus_synaptic"]
        assert "bus_synaptic" in trigger.reason

    def test_above_threshold_fires(self):
        trigger = build_transport_metacog_trigger_from_bus_synaptic(
            1.0,
            zen_state="zen",
            pressure=0.1,
            recall_enabled=True,
            error_threshold=0.8,
        )
        assert trigger is not None

    def test_custom_threshold_respected(self):
        trigger = build_transport_metacog_trigger_from_bus_synaptic(
            0.6,
            zen_state="zen",
            pressure=0.1,
            recall_enabled=True,
            error_threshold=0.5,
        )
        assert trigger is not None


# ---------------------------------------------------------------------------
# bus_synaptic branch: edge-triggering, staleness, hysteresis (2026-07-30)
# ---------------------------------------------------------------------------


def _bs(error, **kw):
    kw.setdefault("zen_state", "not_zen")
    kw.setdefault("pressure", 0.4)
    kw.setdefault("recall_enabled", False)
    kw.setdefault("error_threshold", 1.0)
    return build_transport_metacog_trigger_from_bus_synaptic(error, **kw)


def test_bus_synaptic_fires_on_the_rising_edge_only() -> None:
    """The core fix. This branch was a pure level check evaluated every 30s, so
    one sustained condition re-drafted an LLM reflection on every tick -- live,
    1,812 transport rows in 24h, ~48% from this branch alone. Metacognition is
    "something notable HAPPENED", not "something is STILL the case"."""
    first = _bs(1.2, previously_above=False)
    assert first is not None
    assert first.upstream["transition"] == "below_to_above"
    assert "rising_edge" in first.upstream["fired_conditions"]
    assert "episode_start" in first.reason

    # Same condition, still true on the next poll -> silent.
    assert _bs(1.2, previously_above=True) is None
    assert _bs(9.9, previously_above=True) is None


def test_bus_synaptic_below_threshold_never_fires_regardless_of_edge_state() -> None:
    assert _bs(0.5, previously_above=False) is None
    assert _bs(0.5, previously_above=True) is None


def test_bus_synaptic_refuses_a_stale_node() -> None:
    """Confirmed live: node:substrate.bus_synaptic sat frozen at a stale 1.0 for
    HOURS while this loop fired off it every 30s. A frozen value is not a
    present-tense reading."""
    assert _bs(1.2, previously_above=False, node_age_sec=3600.0, max_node_age_sec=300.0) is None
    # Fresh node with the same value still fires.
    fresh = _bs(1.2, previously_above=False, node_age_sec=12.0, max_node_age_sec=300.0)
    assert fresh is not None
    assert fresh.upstream["node_age_sec"] == 12.0


def test_bus_synaptic_unknown_node_age_does_not_suppress() -> None:
    """Deliberate asymmetry: a frozen node is detectable and is guarded, but an
    unparseable/absent timestamp must NOT silently switch this evidence source
    off -- that would be the same "detector quietly stops detecting" failure
    this whole arc has been chasing."""
    assert _bs(1.2, previously_above=False, node_age_sec=None, max_node_age_sec=300.0) is not None
    assert _bs(1.2, previously_above=False, node_age_sec=99999.0, max_node_age_sec=None) is not None


def test_edge_and_hysteresis_collapse_a_real_firing_pattern() -> None:
    """End-to-end over a synthetic series shaped like the real bimodal metric,
    replicating the service's own edge/hysteresis bookkeeping.

    Asserts the property that actually matters: the number of entries equals the
    number of EPISODES, not the number of polls.
    """
    threshold, clear_ratio = 1.0, 0.8
    # Two genuine anomaly episodes, each sustained across several polls, with a
    # stretch of threshold-adjacent flapping in between.
    series = (
        [0.1] * 5
        + [1.0] * 10          # episode 1 (sustained)
        + [0.1] * 5
        + [1.0, 0.85, 1.0, 0.9, 1.0]   # flapping inside the hysteresis band
        + [0.1] * 5
        + [2.0] * 8           # episode 2 (sustained)
    )

    above = False
    fires = 0
    for value in series:
        if _bs(value, previously_above=above) is not None:
            fires += 1
        if value >= threshold:
            above = True
        elif value < threshold * clear_ratio:
            above = False

    # 3 episodes: the two sustained ones plus the flap's first crossing. The
    # flap's four subsequent re-crossings are absorbed by the hysteresis band.
    assert fires == 3
    # The pre-fix level check fired on every poll at-or-above threshold: 21 of
    # 38 polls here, versus 3 episodes. That ~7x collapse on a 38-sample series
    # is the same shape as the live 1,812-rows-in-24h number.
    assert sum(1 for v in series if v >= threshold) == 21
    assert fires < sum(1 for v in series if v >= threshold)
