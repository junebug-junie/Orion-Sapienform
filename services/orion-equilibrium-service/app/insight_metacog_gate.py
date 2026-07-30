"""`trigger_kind="insight"` gate -- surprise resolution / confidence recovery.

The first of two non-error-shaped generative triggers (see
docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md).
Every other trigger_kind built this arc fires on a rupture: a timeout, an
anomaly, a repair need. This one fires when a real drop in
`AttentionSelfModelV1.prediction_error_confidence` *resolves* -- Orion's
predictions started working again.

Condition math lives in orion/substrate/metacog_trigger_signals.py
(`detect_confidence_recovery`) so it stays pure and unit-testable; this module
owns only the trigger construction.
"""

from __future__ import annotations

import logging

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.substrate.metacog_trigger_signals import ConfidenceRecovery

logger = logging.getLogger("orion.equilibrium.insight_metacog_gate")

EVIDENCE_SOURCE = "attention_self_model_prediction_error_confidence"


def build_insight_metacog_trigger(
    recovery: ConfidenceRecovery | None,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    low_threshold: float,
    high_threshold: float,
) -> MetacogTriggerV1 | None:
    """Turn a detected `ConfidenceRecovery` into an "insight" metacog trigger.

    Returns None when `recovery` is None, so the caller can hand the detector's
    result straight through without branching twice.
    """
    if recovery is None:
        return None

    reason = (
        f"confidence_recovery:{recovery.low_value:.3f}->{recovery.high_value:.3f}"
        f":ticks_to_cross={recovery.ticks_to_cross}"
    )

    return MetacogTriggerV1(
        trigger_kind="insight",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        upstream={
            # Named so a reader of the orion_metacog row can tell which of the
            # two prediction_error_confidence windowing functions fired, without
            # having to infer it from trigger_kind alone.
            "evidence_source": EVIDENCE_SOURCE,
            "detector": "sustained_low_to_high_transition",
            "low_at": recovery.low_at.isoformat(),
            "high_at": recovery.high_at.isoformat(),
            "low_value": recovery.low_value,
            "high_value": recovery.high_value,
            "ticks_to_cross": recovery.ticks_to_cross,
            "confirm_ticks": recovery.confirm_ticks,
            "window_ticks": recovery.window_ticks,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
    )
