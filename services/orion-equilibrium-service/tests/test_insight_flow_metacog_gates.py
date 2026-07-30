"""Trigger-builder tests for the two generative metacog gates.

Mirrors test_repair_pressure_metacog_gate.py's shape. The condition *math* is
tested separately in tests/test_metacog_generative_trigger_signals.py; these
cover the MetacogTriggerV1 construction, and specifically that `upstream`
carries real, non-empty, distinguishable evidence (Acceptance Check 3 of
docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md
-- an empty/placeholder upstream dict is an explicit non-goal).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.flow_metacog_gate import build_flow_metacog_trigger
from app.insight_metacog_gate import build_insight_metacog_trigger
from orion.substrate.metacog_trigger_signals import ConfidenceRecovery, FlowRegime

T0 = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
EXPECTED_EVIDENCE_SOURCE = "attention_self_model_prediction_error_confidence"


def _recovery() -> ConfidenceRecovery:
    return ConfidenceRecovery(
        low_at=T0,
        high_at=T0 + timedelta(seconds=120),
        low_value=0.66,
        high_value=0.92,
        ticks_to_cross=4,
        cross_span_sec=120.0,
        confirm_ticks=2,
        window_ticks=20,
    )


def _regime() -> FlowRegime:
    return FlowRegime(
        started_at=T0,
        ended_at=T0 + timedelta(seconds=570),
        tick_count=20,
        span_sec=570.0,
        min_value=0.91,
        mean_value=0.935,
        stdev_value=0.008,
    )


def _build_insight(recovery=None):
    return build_insight_metacog_trigger(
        _recovery() if recovery is None else recovery,
        zen_state="zen",
        pressure=0.2,
        recall_enabled=False,
        low_threshold=0.70,
        high_threshold=0.90,
    )


def _build_flow(regime=None):
    return build_flow_metacog_trigger(
        _regime() if regime is None else regime,
        zen_state="zen",
        pressure=0.1,
        recall_enabled=False,
        floor=0.90,
        max_stdev=0.02,
    )


# ===========================================================================
# insight
# ===========================================================================


def test_insight_trigger_kind_and_reason() -> None:
    trigger = _build_insight()
    assert trigger is not None
    assert trigger.trigger_kind == "insight"
    assert "confidence_recovery" in trigger.reason
    assert "ticks_to_cross=4" in trigger.reason
    assert trigger.zen_state == "zen"
    assert trigger.pressure == 0.2


def test_insight_upstream_carries_real_evidence() -> None:
    trigger = _build_insight()
    assert trigger is not None
    up = trigger.upstream
    assert up  # never an empty placeholder dict
    assert up["evidence_source"] == EXPECTED_EVIDENCE_SOURCE
    assert up["detector"] == "sustained_low_to_high_transition"
    assert up["low_value"] == 0.66
    assert up["high_value"] == 0.92
    assert up["ticks_to_cross"] == 4
    # Real seconds recorded next to the tick count, so a stored row can be
    # audited for whether those ticks really were consecutive.
    assert up["cross_span_sec"] == 120.0
    assert up["confirm_ticks"] == 2
    assert up["window_ticks"] == 20
    # The thresholds in force are recorded alongside the values, so a stored row
    # stays interpretable after the provisional defaults get retuned.
    assert up["low_threshold"] == 0.70
    assert up["high_threshold"] == 0.90
    assert up["low_at"] == T0.isoformat()
    assert up["high_at"] == (T0 + timedelta(seconds=120)).isoformat()


def test_insight_returns_none_when_no_recovery_detected() -> None:
    """The builder takes the detector's result straight through, so callers
    don't have to branch on None twice."""
    assert build_insight_metacog_trigger(
        None,
        zen_state="zen",
        pressure=0.0,
        recall_enabled=False,
        low_threshold=0.70,
        high_threshold=0.90,
    ) is None


# ===========================================================================
# flow
# ===========================================================================


def test_flow_trigger_kind_and_reason() -> None:
    trigger = _build_flow()
    assert trigger is not None
    assert trigger.trigger_kind == "flow"
    assert "flow_regime" in trigger.reason
    assert "ticks=20" in trigger.reason


def test_flow_upstream_carries_real_evidence() -> None:
    trigger = _build_flow()
    assert trigger is not None
    up = trigger.upstream
    assert up
    assert up["evidence_source"] == EXPECTED_EVIDENCE_SOURCE
    assert up["detector"] == "sustained_high_low_variance_regime"
    assert up["tick_count"] == 20
    assert up["span_sec"] == 570.0
    assert up["min_value"] == 0.91
    assert up["mean_value"] == 0.935
    assert up["stdev_value"] == 0.008
    assert up["floor"] == 0.90
    assert up["max_stdev"] == 0.02


def test_flow_returns_none_when_no_regime_detected() -> None:
    assert build_flow_metacog_trigger(
        None,
        zen_state="zen",
        pressure=0.0,
        recall_enabled=False,
        floor=0.90,
        max_stdev=0.02,
    ) is None


# ===========================================================================
# The two must be distinguishable downstream (Acceptance Check 3)
# ===========================================================================


def test_insight_and_flow_are_distinguishable_despite_sharing_a_field() -> None:
    """Both read the same `prediction_error_confidence` field, so a reader of an
    orion_metacog row must be able to tell which windowing function fired from
    the payload alone -- not have to infer it."""
    insight = _build_insight()
    flow = _build_flow()
    assert insight is not None and flow is not None

    assert insight.trigger_kind != flow.trigger_kind
    # Same evidence source (by design -- one field) ...
    assert insight.upstream["evidence_source"] == flow.upstream["evidence_source"]
    # ... but different detectors, and no overlapping evidence keys that would
    # make the two rows look alike.
    assert insight.upstream["detector"] != flow.upstream["detector"]
    assert insight.reason.split(":")[0] != flow.reason.split(":")[0]
