"""Substrate-derived eventfulness for metacog trigger gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
