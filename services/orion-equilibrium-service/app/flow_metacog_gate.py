"""`trigger_kind="flow"` gate -- sustained low-variance / low-distress regime.

Second of the two generative triggers from
docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md.
Reads the same live field as the "insight" gate
(`AttentionSelfModelV1.prediction_error_confidence`, persisted to
`substrate_attention_self_model`) through a different windowing function:
insight looks for a *transition*, flow looks for a *plateau*.

Condition math lives in orion/substrate/metacog_trigger_signals.py
(`detect_flow_regime`) so it stays pure and unit-testable; this module owns only
the trigger construction.
"""

from __future__ import annotations

import logging

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.substrate.metacog_trigger_signals import FlowRegime

logger = logging.getLogger("orion.equilibrium.flow_metacog_gate")

EVIDENCE_SOURCE = "attention_self_model_prediction_error_confidence"


def build_flow_metacog_trigger(
    regime: FlowRegime | None,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    floor: float,
    max_stdev: float,
) -> MetacogTriggerV1 | None:
    """Turn a detected `FlowRegime` into a "flow" metacog trigger.

    Returns None when `regime` is None, so the caller can hand the detector's
    result straight through without branching twice.
    """
    if regime is None:
        return None

    reason = (
        f"flow_regime:min={regime.min_value:.3f}:mean={regime.mean_value:.3f}"
        f":stdev={regime.stdev_value:.4f}:ticks={regime.tick_count}"
    )

    return MetacogTriggerV1(
        trigger_kind="flow",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        upstream={
            "evidence_source": EVIDENCE_SOURCE,
            "detector": "sustained_high_low_variance_regime",
            "started_at": regime.started_at.isoformat(),
            "ended_at": regime.ended_at.isoformat(),
            "tick_count": regime.tick_count,
            # Real seconds covered, so a reader can confirm tick_count ticks
            # really were consecutive rather than 20 rows spanning hours.
            "span_sec": regime.span_sec,
            "min_value": regime.min_value,
            "mean_value": regime.mean_value,
            "stdev_value": regime.stdev_value,
            "floor": floor,
            "max_stdev": max_stdev,
        },
    )
