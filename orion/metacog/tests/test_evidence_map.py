"""Unit tests for orion/metacog/evidence_map.py.

Upstream shapes are copied from live `metacog_trigger` rows (2026-09-24);
the transport_baseline shape is the contract the section-A gate will emit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.metacog.evidence_map import (
    CRITICAL_FLOOR,
    DEGRADED_FLOOR,
    KNOWN_KINDS,
    banded,
    map_trigger,
    severity_rank,
)

FIXTURE = Path(__file__).parent / "fixtures" / "metacog_trigger_sample.jsonl"


def _band_ok(m) -> bool:
    lo, hi = {
        "nominal": (0.0, DEGRADED_FLOOR),
        "degraded": (DEGRADED_FLOOR, CRITICAL_FLOOR),
        "critical": (CRITICAL_FLOOR, 1.0000001),
    }[m.severity]
    return lo <= m.magnitude < hi


# --- banded -----------------------------------------------------------------


def test_banded_edges_and_monotone():
    assert banded(0.0, 1, 2, 4) == ("nominal", 0.0)
    assert banded(-3.0, 1, 2, 4) == ("nominal", 0.0)
    assert banded(1.0, 1, 2, 4) == ("degraded", DEGRADED_FLOOR)
    assert banded(2.0, 1, 2, 4) == ("critical", CRITICAL_FLOOR)
    assert banded(99.0, 1, 2, 4) == ("critical", 1.0)
    xs = [i / 10 for i in range(0, 60)]
    mags = [banded(x, 1, 2, 4)[1] for x in xs]
    assert mags == sorted(mags)


def test_banded_rejects_misordered_cuts():
    with pytest.raises(ValueError):
        banded(1.0, 2, 1, 4)


# --- transport: today's rpc_health_snapshot shape ---------------------------


def _rpc(**over):
    up = {
        "service": "cortex-exec",
        "success_count": 9,
        "timeout_count": 0,
        "channel_counts": {"orion:state:request": 1, "orion:exec:request:LLMGatewayService": 8},
        "evidence_source": "rpc_health_snapshot",
        "fired_conditions": ["success_latency_ms_p95=12916.6"],
        "success_latency_ms_p50": 6000.0,
        "success_latency_ms_p95": 12916.6,
        "timeout_elapsed_ms_max": None,
        "latency_p95_threshold_ms": 5000.0,
    }
    up.update(over)
    return up


def test_transport_latency_only_under_3x_is_nominal():
    m = map_trigger("transport", "r", _rpc())
    assert m.severity == "nominal"  # 2.6x the flat 5s limit
    assert 0 < m.magnitude < DEGRADED_FLOOR
    assert any("p95 12.9s vs limit 5.0s (2.6x)" in e for e in m.evidence)
    assert m.touches[0] == "cortex-exec"
    assert "orion:exec:request:LLMGatewayService" in m.touches


def test_transport_one_timeout_degraded_two_critical():
    one = map_trigger("transport", "r", _rpc(timeout_count=1, timeout_elapsed_ms_max=90002.7))
    two = map_trigger("transport", "r", _rpc(timeout_count=2))
    assert one.severity == "degraded"
    assert two.severity == "critical"
    assert two.magnitude > one.magnitude
    assert any("timeouts 1/10 calls (longest wait 90.0s)" in e for e in one.evidence)


def test_transport_thin_sample_latency_capped_at_degraded():
    m = map_trigger(
        "transport",
        "r",
        _rpc(service="cortex-orch", success_count=2, success_latency_ms_p95=60000.0),
    )
    assert m.severity == "degraded"  # 12x, but only 2 calls: "the slowest call"
    assert any("thin sample" in e for e in m.evidence)
    full = map_trigger("transport", "r", _rpc(success_count=9, success_latency_ms_p95=60000.0))
    assert full.severity == "critical"


def test_transport_rpc_timeout_grammar():
    up = {
        "summary": "RPC timeout: orion:cortex:request -> orion:cortex:result:abc after 420.0s (elapsed 420001.5ms)",
        "correlation_id": "c",
        "evidence_source": "rpc_transport_timeout_grammar",
        "request_channel": "orion:cortex:request",
        "fired_conditions": ["rpc_timeout"],
    }
    m = map_trigger("transport", "transport:rpc_timeout:orion:cortex:request", up)
    assert m.severity == "degraded"
    assert m.touches == ["orion:cortex:request"]
    assert "waited 420.0s before giving up" in m.evidence


def test_transport_bus_synaptic():
    base = {
        "error": 0.196,
        "edge_count": None,
        "transition": "below_to_above",
        "error_threshold": 0.15,
        "evidence_source": "bus_synaptic_prediction_error",
    }
    assert map_trigger("transport", "r", base).severity == "nominal"
    assert map_trigger("transport", "r", {**base, "error": 0.35}).severity == "degraded"
    assert map_trigger("transport", "r", {**base, "error": 0.73}).severity == "critical"


# --- transport: new condition-shaped upstream (section A gate contract) ----


def _cond(**over):
    up = {
        "evidence_source": "transport_baseline",
        "condition": "spike",
        "phase": "open",
        "key": "cortex-exec:orion:exec:request:LLMGatewayService",
        "z": 3.4,
        "saturation_ratio": 1.1,
        "baseline_ms": 9100.0,
        "floor_ms": 8800.0,
        "window_mean_ms": 17400.0,
        "calls_per_min": 40.0,
        "calls_per_min_usual": 6.0,
        "duration_s": None,
        "peak_ms": None,
        "timeout_count": 0,
    }
    up.update(over)
    return up


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"condition": "spike", "z": 2.0}, "nominal"),
        ({"condition": "spike", "z": 3.4}, "degraded"),
        ({"condition": "spike", "z": 5.0}, "critical"),
        ({"condition": "saturation", "saturation_ratio": 1.5}, "nominal"),
        ({"condition": "saturation", "saturation_ratio": 2.5}, "degraded"),
        ({"condition": "saturation", "saturation_ratio": 3.2}, "critical"),
        ({"condition": "timeout", "timeout_count": 1}, "degraded"),
        ({"condition": "timeout", "timeout_count": 2}, "critical"),
        ({"condition": "zero_success", "timeout_count": 0}, "critical"),
        ({"condition": "regime_shift", "saturation_ratio": 5.0}, "degraded"),
        ({"condition": "regime_shift", "saturation_ratio": None}, "degraded"),
        ({"condition": "spike", "z": 9.0, "phase": "close", "duration_s": 600, "peak_ms": 31000}, "nominal"),
        ({"condition": "spike", "z": 6.0, "phase": "escalate"}, "critical"),
    ],
)
def test_transport_baseline_conditions(over, expected):
    m = map_trigger("transport", "r", _cond(**over))
    assert m.severity == expected
    assert _band_ok(m)
    assert m.touches == ["cortex-exec:orion:exec:request:LLMGatewayService"]


def test_transport_baseline_evidence_reads_like_the_spec():
    m = map_trigger("transport", "r", _cond())
    assert "mean 17.4s vs normal 9.1s (z=3.4)" in m.evidence
    assert "load 40.0/min vs usual 6.0/min (slow while busy)" in m.evidence
    idle = map_trigger("transport", "r", _cond(calls_per_min=1.0))
    assert any("slow while idle" in e for e in idle.evidence)


def test_transport_close_of_big_episode_outranks_close_of_small():
    big = map_trigger("transport", "r", _cond(z=9.0, phase="close"))
    small = map_trigger("transport", "r", _cond(z=3.1, phase="close"))
    assert big.severity == small.severity == "nominal"
    assert big.magnitude > small.magnitude > 0


@pytest.mark.parametrize(
    "up",
    [
        _cond(condition="bogus"),
        _cond(key=""),
        _cond(condition="spike", z=None),
        _cond(condition="saturation", saturation_ratio="not-a-number"),
        {"evidence_source": "something_new"},
        {"evidence_source": "rpc_health_snapshot"},
        {"evidence_source": "bus_synaptic_prediction_error", "error": 0.3, "error_threshold": 0},
    ],
)
def test_transport_malformed_is_nominal_no_evidence(up):
    m = map_trigger("transport", "r", up)
    assert m.severity == "nominal"
    assert m.magnitude == 0.0
    assert m.density_rationale.startswith("no_evidence")


# --- telemetry_anomaly -----------------------------------------------------


def _tel(loss):
    return {
        "threshold": 0.014353016380838766,
        "encoder_id": "mood-arc-encoder:v4",
        "recon_loss": loss,
        "top_channels": ["failure_pressure=0.35126", "reliability_pressure=0.13724"],
        "deviation_direction": "elevated",
        "mean_signed_deviation": 0.0125,
    }


@pytest.mark.parametrize(
    "ratio,expected", [(1.1, "nominal"), (1.49, "nominal"), (1.5, "degraded"), (2.4, "degraded"), (2.5, "critical")]
)
def test_telemetry_ratio_bands(ratio, expected):
    m = map_trigger("telemetry_anomaly", "r", _tel(0.014353016380838766 * ratio))
    assert m.severity == expected
    assert _band_ok(m)


def test_telemetry_touches_and_evidence():
    m = map_trigger("telemetry_anomaly", "r", _tel(0.01572))
    assert m.touches == ["mood-arc-encoder:v4", "channel:failure_pressure", "channel:reliability_pressure"]
    assert m.evidence[0].startswith("recon_loss 0.0157 vs threshold 0.0144 (1.10x), elevated")
    assert "top channel failure_pressure=0.35126" in m.evidence


def test_telemetry_malformed():
    assert map_trigger("telemetry_anomaly", "r", {"recon_loss": "x"}).density_rationale.startswith("no_evidence")
    assert map_trigger("telemetry_anomaly", "r", {"recon_loss": 1.0, "threshold": 0}).magnitude == 0.0


# --- chat_turn ------------------------------------------------------------


def test_chat_turn_compliance_failed_is_critical():
    m = map_trigger(
        "chat_turn",
        "r",
        {
            "fired_conditions": ["alignment_verdict=misaligned", "compliance_verdict=failed"],
            "compliance_verdict": "failed",
            "alignment_verdict": "misaligned",
            "timed_out": False,
            "exit_code": None,
            "grounding_status": "fcc_stream_stalled",
        },
    )
    assert m.severity == "critical"
    assert "compliance failed" in m.evidence
    assert "grounding: fcc_stream_stalled" in m.evidence
    assert set(m.touches) >= {"chat_turn", "harness_run", "reflection"}


@pytest.mark.parametrize(
    "up",
    [
        {"timed_out": True, "timeout_reason": "exec_turn_timeout", "fired_conditions": ["timeout=exec_turn_timeout"]},
        {"exit_code": 2, "compliance_verdict": "completed", "fired_conditions": ["exit_code=2"]},
    ],
)
def test_chat_turn_timeout_or_nonzero_exit_is_critical(up):
    assert map_trigger("chat_turn", "r", up).severity == "critical"


def test_chat_turn_misalignment_or_strain_alone_degraded():
    a = map_trigger("chat_turn", "r", {"alignment_verdict": "misaligned", "compliance_verdict": "completed", "exit_code": 0})
    s = map_trigger("chat_turn", "r", {"strain_unresolved": True, "compliance_verdict": "completed", "exit_code": 0})
    both = map_trigger("chat_turn", "r", {"strain_unresolved": True, "alignment_verdict": "misaligned"})
    assert a.severity == s.severity == both.severity == "degraded"
    assert both.magnitude > a.magnitude


def test_chat_turn_surprise_only_is_nominal_nonzero():
    m = map_trigger(
        "chat_turn",
        "r",
        {"surprise_level": 0.8, "fired_conditions": ["surprise_level=0.800"], "compliance_verdict": "completed"},
    )
    assert m.severity == "nominal"
    assert m.magnitude > 0


def test_chat_turn_empty_is_no_evidence():
    assert map_trigger("chat_turn", "r", {}).density_rationale.startswith("no_evidence")


# --- relational / repair_pressure_trend -------------------------------------


def test_relational_level_bands():
    base = {"level_label": "MEDIUM", "confidence": 0.65, "evidence": [], "behavior_applied": "concrete_bias"}
    assert map_trigger("relational", "r", {**base, "level": 0.55}).severity == "nominal"
    assert map_trigger("relational", "r", {**base, "level": 0.70}).severity == "degraded"
    assert map_trigger("relational", "r", {**base, "level": 0.85}).severity == "critical"
    m = map_trigger(
        "relational",
        "r",
        {**base, "level": 0.7, "evidence": [{"evidence_kind": "trust_rupture", "score": 0.91, "confidence": 0.65}]},
    )
    assert "repair:trust_rupture" in m.touches


def test_repair_trend_z_bands():
    up = {"latest_level": 0.44, "baseline_ewma": 0.26, "latest_zscore": 1.19, "consecutive_elevated": 3}
    assert map_trigger("repair_pressure_trend", "r", up).severity == "nominal"
    assert map_trigger("repair_pressure_trend", "r", {**up, "latest_zscore": 2.5}).severity == "degraded"
    assert map_trigger("repair_pressure_trend", "r", {**up, "latest_zscore": 3.5}).severity == "critical"
    assert map_trigger("repair_pressure_trend", "r", {}).density_rationale.startswith("no_evidence")


# --- insight / flow / llm_surface_instability / baseline / manual -----------


def test_insight_and_flow_are_nominal_but_graded():
    small = map_trigger("insight", "r", {"low_value": 0.70, "high_value": 0.75, "ticks_to_cross": 1})
    big = map_trigger("insight", "r", {"low_value": 0.60, "high_value": 0.95, "ticks_to_cross": 1})
    assert small.severity == big.severity == "nominal"
    assert big.magnitude > small.magnitude > 0
    f = map_trigger("flow", "r", {"floor": 0.9, "mean_value": 0.967, "stdev_value": 0.02, "span_sec": 829.5, "tick_count": 20})
    assert f.severity == "nominal" and 0 < f.magnitude < DEGRADED_FLOOR
    assert map_trigger("flow", "r", {"floor": 1.0, "mean_value": 1.0}).magnitude == 0.0


def test_llm_surface_instability():
    up = {
        "phase": "semantic_synthesis",
        "llm_uncertainty": {"unstable_span_count": 1, "low_logprob_token_count": 5, "token_count_observed": 97},
    }
    m = map_trigger("llm_surface_instability", "language_surface_unstable", up)
    assert m.severity == "nominal"
    assert "low-logprob tokens 5/97 (5%)" in m.evidence
    assert "orion-mind" in m.touches
    up["llm_uncertainty"]["unstable_span_count"] = 3
    assert map_trigger("llm_surface_instability", "r", up).severity == "critical"
    # margin far below the gate's 0.75 line is critical on its own
    low_margin = {"llm_uncertainty": {"unstable_span_count": 0, "mean_top1_margin": 0.1, "token_count_observed": 50}}
    assert map_trigger("llm_surface_instability", "r", low_margin).severity == "critical"
    # nothing past any firing line -> no_evidence, never a fake zero event
    calm = {"llm_uncertainty": {"unstable_span_count": 0, "mean_top1_margin": 9.0, "low_logprob_token_count": 0,
                                "token_count_observed": 50}}
    assert map_trigger("llm_surface_instability", "r", calm).density_rationale.startswith("no_evidence")


def test_transport_routes_on_evidence_source_not_condition_key():
    m = map_trigger("transport", "r", _rpc(condition="something_new"))
    assert not m.density_rationale.startswith("no_evidence")


def test_transport_latency_row_without_threshold_is_no_evidence_not_fake_nominal():
    m = map_trigger("transport", "r", _rpc(latency_p95_threshold_ms=None))
    assert m.density_rationale.startswith("no_evidence")


def test_baseline_is_nominal_zero():
    m = map_trigger("baseline", "scheduled_check", {})
    assert (m.severity, m.magnitude) == ("nominal", 0.0)
    m2 = map_trigger("baseline", "scheduled_check", None)
    assert (m2.severity, m2.magnitude) == ("nominal", 0.0)


def test_manual():
    m = map_trigger("manual", "user_collapse_event", {"event_id": "collapse_x"})
    assert m.severity == "nominal"
    assert "source event collapse_x" in m.evidence


# --- never crash ------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(KNOWN_KINDS) + ["unknown_kind", "", None])
@pytest.mark.parametrize("up", [None, [], "str", 3, {}, {"garbage": object()}])
def test_every_kind_survives_malformed_upstream(kind, up):
    m = map_trigger(kind, None, up)
    assert m.severity in ("nominal", "degraded", "critical")
    assert 0.0 <= m.magnitude <= 1.0
    assert isinstance(m.summary_fallback, str) and m.summary_fallback
    assert _band_ok(m)


# --- real fixture rows --------------------------------------------------------


def _fixture_rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def test_every_real_fixture_row_maps_consistently():
    rows = _fixture_rows()
    assert len(rows) > 300
    for row in rows:
        m = map_trigger(row["trigger_kind"], row["reason"], row["upstream"])
        assert _band_ok(m), row
        assert m.causal_density["score"] == m.magnitude
        # Real rows with real upstream should map, not fall to no_evidence
        # (baseline is legitimately evidence-free).
        if row["trigger_kind"] != "baseline":
            assert not m.density_rationale.startswith("no_evidence"), row
            assert m.evidence, row
        assert "zen" not in m.summary_fallback.lower()


def test_severity_rank_order():
    assert severity_rank("nominal") < severity_rank("degraded") < severity_rank("critical")
