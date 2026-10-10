"""Substrate-derived eventfulness for metacog trigger gating."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from orion.schemas.context_provenance import classify
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1


@dataclass(frozen=True)
class SubstrateEventfulness:
    score: float
    trigger_kind: str | None
    reasons: tuple[str, ...]


def _coerce_execution_projection(raw: Any) -> ExecutionTrajectoryProjectionV1 | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, ExecutionTrajectoryProjectionV1):
            return raw
        if isinstance(raw, dict):
            return ExecutionTrajectoryProjectionV1.model_validate(raw)
    except Exception:
        return None
    return None


def _execution_has_failures(projection: ExecutionTrajectoryProjectionV1 | None) -> bool:
    if projection is None:
        return False
    for run in projection.runs.values():
        if int(run.failed_step_count or 0) > 0:
            return True
    return False


def compute_substrate_eventfulness(
    *,
    execution_trajectory: Any = None,
    dense_threshold: float = 0.55,
    pulse_threshold: float = 0.30,
) -> SubstrateEventfulness:
    """Score substrate signals in [0,1] and suggest a metacog trigger kind.

    2026-07-22 (SelfStateV1 burn): the self_state-derived scoring terms
    (overall_surprise/overall_condition/trajectory_condition/
    prediction_error_scores) are removed -- SelfStateV1 no longer exists, and
    ctx['self_state'] was never populated by anything else. Only the
    execution_trajectory-derived term (0.25 max) survives.

    Disclosed, not silently degraded: with only that one term left, the max
    achievable score is 0.25 -- below the default dense_threshold=0.55, so
    "dense" can never fire with default thresholds anymore. Callers that
    still want a "dense" tier need to either lower dense_threshold or this
    needs a real replacement scoring term designed later; not attempted here.
    """
    score = 0.0
    reasons: list[str] = []

    ex = _coerce_execution_projection(execution_trajectory)
    if _execution_has_failures(ex):
        score += 0.25
        reasons.append("execution_failures")

    score = max(0.0, min(1.0, score))

    trigger_kind: str | None = None
    if score >= dense_threshold:
        trigger_kind = "dense"
    elif score >= pulse_threshold:
        trigger_kind = "pulse"

    return SubstrateEventfulness(score=score, trigger_kind=trigger_kind, reasons=tuple(reasons))


# ===========================================================================
# Generative (non-rupture) trigger detector over AttentionSelfModelV1's
# `prediction_error_confidence` history.
#
# Reads the live field persisted every ~30s to the
# `substrate_attention_self_model` table by orion-substrate-runtime's
# _attention_self_model_tick() (PR #1459), per docs/superpowers/specs/
# 2026-07-28-collapse-mirror-generative-triggers-design.md (Missing Questions
# 2/3). Pure: no I/O, no settings reads.
#
# The sibling "flow" detector (`detect_flow_regime` / `FlowRegime`, a sustained
# high plateau) was RETIRED 2026-10-10: on live data the plateau was the field's
# idle rest state, not a distinct state. See docs/superpowers/pr-reports/
# 2026-10-10-metacog-flow-trigger-calibration-pr.md.
#
# Thresholds are deliberately *required* keyword args rather than module
# defaults: the live values are calibrated constants owned by
# services/orion-equilibrium-service/app/settings.py, and duplicating them
# here would create a second place for them to drift.
# ===========================================================================


@dataclass(frozen=True)
class ConfidenceSample:
    """One persisted `prediction_error_confidence` tick."""

    generated_at: datetime
    value: float


@dataclass(frozen=True)
class ConfidenceRecovery:
    """A sustained low->high transition in `prediction_error_confidence` --
    the "insight" (surprise-resolution) condition.

    `ticks_to_cross` is how many ticks elapsed between the most recent low-band
    tick and the first tick of the confirmed high-band run.
    """

    low_at: datetime
    high_at: datetime
    low_value: float
    high_value: float
    ticks_to_cross: int
    # Real wall-clock seconds from the low tick to the start of the high run.
    # Recorded alongside `ticks_to_cross` because the two diverge whenever rows
    # are missing from the window -- a stored row stays self-auditing.
    cross_span_sec: float
    confirm_ticks: int
    window_ticks: int


def _window_values(samples: Sequence[ConfidenceSample]) -> list[float] | None:
    """Return the window's values, or None if any is non-finite.

    Fails closed rather than dropping bad samples: a NaN silently compares
    False against every threshold, so a window containing one would otherwise
    be silently mis-evaluated instead of skipped. Dropping it instead would
    also break the "N *consecutive* ticks" semantics the detector relies on.
    Same guard rationale as scripts/analysis/measure_attention_self_model_
    confidence_baseline.py::_finite_float_or_none.
    """
    values: list[float] = []
    for sample in samples:
        value = sample.value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def detect_confidence_recovery(
    samples: Sequence[ConfidenceSample],
    *,
    low_threshold: float,
    high_threshold: float,
    max_ticks_to_cross: int,
    confirm_ticks: int,
    max_cross_span_sec: float,
) -> ConfidenceRecovery | None:
    """Detect a *sustained* low->high recovery ending at the newest sample.

    `samples` must be ordered oldest -> newest.

    Deliberately NOT a single-tick `>= high_threshold` check, unlike every
    other gate in orion-equilibrium-service (chat_turn/transport/relational are
    all single-tick-crossing gates). This is the one documented exception, and
    the reason is measured, not stylistic: PR #1463's baseline pass over real
    `substrate_attention_self_model` history found confidence recoveries unfold
    over a median of 3 ticks (~90s), max 12 -- a gradual climb, not a sharp
    jump. A single-tick crossing gate would fire on noise partway up that climb.
    See the design doc's Missing Question 2 / Acceptance Check 1.

    Conditions, all required:
      1. The most recent `confirm_ticks` samples are all >= `high_threshold`
         (the "sustained" part -- anchors the event to now and rejects a lone
         noisy spike mid-climb).
      2. Somewhere before that high run, within the supplied window, a sample
         sits at/below `low_threshold` (the "surprise" that got resolved).
      3. The gap between that low tick and the first tick of the high run is
         <= `max_ticks_to_cross` *ticks* AND <= `max_cross_span_sec` *seconds*.

    Both halves of (3) are required because row-index distance is not time
    distance. Review finding 2026-07-30, reproduced against the real detector:
    with only the index bound, a low tick 5 hours before the high run reported
    `ticks_to_cross=1` and fired, because the caller's reader silently drops
    rows whose `prediction_error_confidence` is missing/non-finite -- so 20
    "consecutive" rows can span hours. The seconds bound is what actually makes
    this function's "not an hours-old low" claim true; the tick bound alone
    never did.

    Returns None when any condition fails. Firing exactly-once per real
    recovery is the *caller's* job -- see the equilibrium service's `low_at`
    de-dupe, and note that `high_at` is NOT a safe identity key: it re-anchors
    whenever the high run breaks on a single sub-threshold tick and re-forms.
    """
    if confirm_ticks < 1 or max_ticks_to_cross < 0 or max_cross_span_sec < 0:
        return None
    if len(samples) < confirm_ticks:
        return None

    values = _window_values(samples)
    if values is None:
        return None

    # (1) sustained high band through the newest sample.
    if any(v < high_threshold for v in values[-confirm_ticks:]):
        return None

    # First tick of the trailing high-band run.
    high_idx = len(values) - 1
    while high_idx > 0 and values[high_idx - 1] >= high_threshold:
        high_idx -= 1

    # (2) most recent low-band tick strictly before that run.
    low_idx: int | None = None
    for i in range(high_idx - 1, -1, -1):
        if values[i] <= low_threshold:
            low_idx = i
            break
    if low_idx is None:
        return None

    # (3) the climb was recent enough to call it one event -- in ticks AND in
    # wall-clock seconds, since dropped rows make those two different things.
    ticks_to_cross = high_idx - low_idx
    if ticks_to_cross > max_ticks_to_cross:
        return None
    cross_span_sec = (
        samples[high_idx].generated_at - samples[low_idx].generated_at
    ).total_seconds()
    if cross_span_sec > max_cross_span_sec:
        return None

    return ConfidenceRecovery(
        low_at=samples[low_idx].generated_at,
        high_at=samples[high_idx].generated_at,
        low_value=values[low_idx],
        high_value=values[high_idx],
        ticks_to_cross=ticks_to_cross,
        cross_span_sec=cross_span_sec,
        confirm_ticks=confirm_ticks,
        window_ticks=len(values),
    )


def build_metacog_substrate_cue(
    ctx: Mapping[str, Any],
    *,
    max_chars: int = 400,
    eventfulness: SubstrateEventfulness | None = None,
) -> str:
    """Compact substrate cue for metacog prompts (not raw JSON).

    2026-07-22 (SelfStateV1 burn): the self_state clause removed --
    ctx['self_state'] is never populated. execution_trajectory_projection is
    the only remaining source for this cue, still registered
    "live_runtime_projection" (orion/schemas/context_provenance.py) -- tag
    switched from classify("self_state") to classify("execution_trajectory_projection")
    since that's the real surviving source, not a cosmetic no-op.
    """
    parts: list[str] = []
    ex = ctx.get("execution_trajectory_projection")
    if isinstance(ex, dict):
        runs = ex.get("runs") if isinstance(ex.get("runs"), dict) else {}
        failed = sum(1 for r in runs.values() if int((r or {}).get("failed_step_count") or 0) > 0)
        if failed:
            parts.append(f"execution: failed_runs={failed}")
    ev = eventfulness or compute_substrate_eventfulness(
        execution_trajectory=ctx.get("execution_trajectory_projection"),
    )
    if ev.reasons:
        parts.append(f"eventfulness={ev.score:.2f} ({'; '.join(ev.reasons[:3])})")
    if not parts:
        return ""
    # Budget is reserved for the tag and it's appended after truncation, not
    # before: on an eventful turn the joined clauses can already approach
    # max_chars, and a tag appended before truncation is the first thing a
    # tail-truncate cuts, silently dropping the provenance signal on exactly
    # the turns most likely to need it.
    tag = f" (source={classify('execution_trajectory_projection')})"
    body = " | ".join(parts)
    budget = max_chars - len(tag)
    if len(body) > budget:
        body = body[:budget]
    return body + tag
