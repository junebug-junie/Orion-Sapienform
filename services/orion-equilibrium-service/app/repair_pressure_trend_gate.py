from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.metacog.trend_reducer import (
    MetacogTrendReading,
    MetacogTrendStateV1,
    apply_reading,
)
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.repair_pressure_trend_gate")


def state_from_dict(data: dict[str, Any] | None) -> MetacogTrendStateV1:
    """Reconstruct a MetacogTrendStateV1 from its persisted JSON form. Never
    raises -- missing/malformed fields fall back to the cold-start default
    (ewma=0.0, variance=0.0, count=0, consecutive_elevated=0), same as a
    fresh state, so a corrupt or absent checkpoint degrades to "start over"
    rather than crashing the live message loop."""
    if not isinstance(data, dict):
        return MetacogTrendStateV1()
    try:
        return MetacogTrendStateV1(
            ewma=float(data.get("ewma", 0.0)),
            variance=float(data.get("variance", 0.0)),
            count=int(data.get("count", 0)),
            consecutive_elevated=int(data.get("consecutive_elevated", 0)),
        )
    except (TypeError, ValueError):
        return MetacogTrendStateV1()


def state_to_dict(state: MetacogTrendStateV1) -> dict[str, Any]:
    return {
        "ewma": state.ewma,
        "variance": state.variance,
        "count": state.count,
        "consecutive_elevated": state.consecutive_elevated,
    }


def evaluate_repair_pressure_trend(
    state: MetacogTrendStateV1,
    appraisal: dict[str, Any] | None,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    confidence_floor: float,
    min_samples: int,
    elevated_zscore: float,
    sustained_hits: int,
) -> tuple[MetacogTrendStateV1, MetacogTriggerV1 | None]:
    """Fold one live `orion:repair_pressure:appraisal` message into `state`
    and return (new_state, trigger_or_None).

    `confidence_floor` is this gate's own confidence-gating decision (per
    `orion/metacog/trend_reducer.py`'s own docstring: "Confidence-gating is
    the CALLER's responsibility, not this module's"). Post the 2026-07-30
    `reduce_repair_level()` fix, `confidence` genuinely reflects evidence
    quality (including a real ~0.65 for the common "confidently calm"
    text-fallback case) rather than being pinned to 0.0 for real-but-weak
    evidence -- so a low, inclusive floor (folding in most real appraisals,
    excluding only genuine zero-evidence reads) is the correct default here,
    not the exclusionary `> 0.5`-style threshold the old bug produced.

    Malformed/missing `level`/`confidence` (same defensive contract as
    `repair_pressure_metacog_gate.py::build_repair_pressure_metacog_trigger`)
    leaves `state` unchanged and returns no trigger -- never crashes the live
    message loop, never silently folds a garbage reading into the EWMA.
    """
    if not isinstance(appraisal, dict):
        return state, None

    level = appraisal.get("level")
    confidence = appraisal.get("confidence")
    if not isinstance(level, (int, float)) or not isinstance(confidence, (int, float)):
        return state, None

    confidence = float(confidence)
    if confidence < confidence_floor:
        return state, None

    reading = MetacogTrendReading(
        at=datetime.now(timezone.utc), level=float(level), confidence=confidence
    )
    result = apply_reading(
        state,
        reading,
        min_samples=min_samples,
        elevated_zscore=elevated_zscore,
        sustained_hits=sustained_hits,
    )

    if not result.is_sustained_trend:
        return result.state, None

    trigger = MetacogTriggerV1(
        trigger_kind="repair_pressure_trend",
        reason=(
            f"repair_pressure:sustained_elevated:zscore={result.latest_zscore:.2f}"
            f":consecutive={result.state.consecutive_elevated}"
        ),
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        upstream={
            "latest_level": float(level),
            "latest_confidence": confidence,
            "latest_zscore": result.latest_zscore,
            "consecutive_elevated": result.state.consecutive_elevated,
            "baseline_ewma": result.state.ewma,
        },
    )
    return result.state, trigger
