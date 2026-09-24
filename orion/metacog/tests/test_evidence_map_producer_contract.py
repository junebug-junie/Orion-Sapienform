"""Producer -> evidence_map contract.

Builds triggers with the REAL equilibrium gate builders and asserts the
mapper reads them (not no_evidence). If a gate renames an upstream key, rows
would otherwise silently degrade to nominal/no_evidence; this fails instead.
The gate files are loaded by path (they import only orion.*), so no
equilibrium-service settings/bus are needed.
"""
from __future__ import annotations

import importlib.util
import inspect
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.metacog.evidence_map import map_trigger
from orion.substrate.metacog_trigger_signals import ConfidenceRecovery, FlowRegime

ROOT = Path(__file__).resolve().parents[3]
GATES = ROOT / "services" / "orion-equilibrium-service" / "app"
COMMON = {"zen_state": "zen", "pressure": 0.1, "recall_enabled": False}


def _gate(name: str):
    spec = importlib.util.spec_from_file_location(f"_contract_{name}", GATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _assert_mapped(trigger, *, severity=None):
    assert trigger is not None
    m = map_trigger(trigger.trigger_kind, trigger.reason, trigger.upstream)
    assert not m.density_rationale.startswith("no_evidence"), (trigger.upstream, m)
    assert m.evidence and m.touches
    if severity:
        assert m.severity == severity
    return m


def test_transport_snapshot_gate():
    g = _gate("transport_metacog_gate")
    build = g.build_transport_metacog_trigger_from_snapshot
    # PR #2310 removes the legacy p95 latency branch and its threshold kwarg;
    # pass it only while the builder still accepts it so either merge order works.
    extra = (
        {"latency_p95_threshold_ms": 5000.0}
        if "latency_p95_threshold_ms" in inspect.signature(build).parameters
        else {}
    )
    t = build(
        {
            "service": "cortex-exec",
            "window_start": "2026-09-24T00:00:00Z",
            "window_end": "2026-09-24T00:00:30Z",
            "success_count": 3,
            "timeout_count": 2,
            "success_latency_ms_p50": 900.0,
            "success_latency_ms_p95": 1200.0,
            "success_latency_ms_max": 1300.0,
            "channel_counts": {"orion:state:request": 5},
        },
        **extra,
        **COMMON,
    )
    _assert_mapped(t, severity="critical")


def test_transport_grammar_gate():
    g = _gate("transport_metacog_gate")
    t = g.build_transport_metacog_trigger_from_grammar_atom(
        {
            "semantic_role": "rpc_transport_timeout",
            "text_value": "orion:cortex:request",
            "summary": "RPC timeout: orion:cortex:request -> x after 420.0s (elapsed 420001.5ms)",
        },
        correlation_id="c",
        **COMMON,
    )
    _assert_mapped(t, severity="degraded")


def test_transport_bus_synaptic_gate():
    g = _gate("transport_metacog_gate")
    t = g.build_transport_metacog_trigger_from_bus_synaptic(0.5, error_threshold=0.15, **COMMON)
    _assert_mapped(t, severity="critical")


def test_telemetry_anomaly_gate():
    g = _gate("telemetry_anomaly_metacog_gate")
    t = g.build_telemetry_anomaly_metacog_trigger(
        correlation_id="c",
        score={
            "recon_loss": 0.03,
            "recon_error_p95": 0.004,
            "top_channels": ["failure_pressure=0.35"],
            "deviation_direction": "elevated",
            "encoder_id": "mood-arc-encoder:v4",
        },
        threshold_multiplier=3.0,
        **COMMON,
    )
    _assert_mapped(t, severity="critical")  # 0.03 / 0.012 = 2.5x


def test_chat_turn_gate():
    g = _gate("chat_turn_metacog_gate")
    t = g.build_chat_turn_metacog_trigger(
        correlation_id="c",
        thought_event={"disposition": "proceed"},
        run_artifact={
            "reflection": {"alignment_verdict": "misaligned", "strain_unresolved": True},
            "compliance_verdict": "failed",
            "exit_code": 0,
        },
        timed_out=False,
        surprise_threshold=0.7,
        **COMMON,
    )
    _assert_mapped(t, severity="critical")


def test_relational_gate():
    g = _gate("repair_pressure_metacog_gate")
    t = g.build_repair_pressure_metacog_trigger(
        correlation_id="c",
        appraisal={
            "level": 0.85,
            "level_label": "HIGH",
            "confidence": 0.8,
            "evidence": [{"evidence_kind": "trust_rupture", "score": 0.9, "confidence": 0.8}],
        },
        level_floor=0.5,
        confidence_floor=0.7,
        **COMMON,
    )
    _assert_mapped(t, severity="critical")


def test_flow_and_insight_gates():
    now = datetime.now(timezone.utc)
    flow = _gate("flow_metacog_gate").build_flow_metacog_trigger(
        FlowRegime(
            started_at=now, ended_at=now, tick_count=20, span_sec=800.0,
            min_value=0.92, mean_value=0.97, stdev_value=0.015,
        ),
        floor=0.9,
        max_stdev=0.02,
        **COMMON,
    )
    _assert_mapped(flow, severity="nominal")
    insight = _gate("insight_metacog_gate").build_insight_metacog_trigger(
        ConfidenceRecovery(
            low_at=now, high_at=now, low_value=0.69, high_value=0.93,
            ticks_to_cross=1, cross_span_sec=37.0, confirm_ticks=2, window_ticks=20,
        ),
        low_threshold=0.7,
        high_threshold=0.9,
        **COMMON,
    )
    _assert_mapped(insight, severity="nominal")


# orion-mind builds its trigger inline (services/orion-mind/app/uncertainty_metacog.py);
# its gate fires on any of three conditions. Each must map to a real event.
@pytest.mark.parametrize(
    "unc,detail",
    [
        ({"unstable_span_count": 1, "mean_top1_margin": 14.0, "low_logprob_token_count": 0, "token_count_observed": 97},
         "unstable_span"),
        ({"unstable_span_count": 0, "mean_top1_margin": 0.3, "low_logprob_token_count": 0, "token_count_observed": 97},
         "low_mean_margin"),
        ({"unstable_span_count": 0, "mean_top1_margin": 5.0, "low_logprob_token_count": 30, "token_count_observed": 100},
         "high_low_logprob_ratio"),
    ],
)
def test_llm_surface_instability_every_mind_gate_condition(unc, detail):
    up = {"llm_uncertainty": {"available": True, **unc}, "phase": "semantic_synthesis", "instability_detail": detail}
    m = map_trigger("llm_surface_instability", "language_surface_unstable", up)
    assert not m.density_rationale.startswith("no_evidence")
    assert m.magnitude > 0
    assert f"fired as {detail}" in m.evidence
