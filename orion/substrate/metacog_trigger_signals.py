"""Substrate-derived eventfulness for metacog trigger gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.self_state import SelfStateV1


@dataclass(frozen=True)
class SubstrateEventfulness:
    score: float
    trigger_kind: str | None
    reasons: tuple[str, ...]


def _coerce_self_state(raw: Any) -> SelfStateV1 | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, SelfStateV1):
            return raw
        if isinstance(raw, dict):
            return SelfStateV1.model_validate(raw)
    except Exception:
        return None
    return None


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
    self_state: Any = None,
    execution_trajectory: Any = None,
    dense_threshold: float = 0.55,
    pulse_threshold: float = 0.30,
) -> SubstrateEventfulness:
    """Score substrate signals in [0,1] and suggest a metacog trigger kind."""
    score = 0.0
    reasons: list[str] = []

    ss = _coerce_self_state(self_state)
    if ss is not None:
        surprise = float(ss.overall_surprise or 0.0)
        if surprise >= 0.55:
            score += 0.35
            reasons.append(f"overall_surprise={surprise:.2f}")
        if ss.overall_condition in ("strained", "unstable"):
            score += 0.25
            reasons.append(f"overall_condition={ss.overall_condition}")
        if ss.trajectory_condition == "degrading":
            score += 0.20
            reasons.append("trajectory_degrading")
        pe = 0.0
        if ss.prediction_error_scores:
            pe = max(float(v or 0.0) for v in ss.prediction_error_scores.values())
        if pe >= 0.5:
            score += 0.20
            reasons.append(f"prediction_error_max={pe:.2f}")

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
    """Compact substrate cue for metacog prompts (not raw JSON)."""
    parts: list[str] = []
    ss = ctx.get("self_state")
    if isinstance(ss, dict):
        parts.append(
            "self_state:"
            f" condition={ss.get('overall_condition', '?')}"
            f" surprise={ss.get('overall_surprise', '?')}"
            f" trajectory={ss.get('trajectory_condition', '?')}"
        )
    ex = ctx.get("execution_trajectory_projection")
    if isinstance(ex, dict):
        runs = ex.get("runs") if isinstance(ex.get("runs"), dict) else {}
        failed = sum(1 for r in runs.values() if int((r or {}).get("failed_step_count") or 0) > 0)
        if failed:
            parts.append(f"execution: failed_runs={failed}")
    ev = eventfulness or compute_substrate_eventfulness(
        self_state=ctx.get("self_state"),
        execution_trajectory=ctx.get("execution_trajectory_projection"),
    )
    if ev.reasons:
        parts.append(f"eventfulness={ev.score:.2f} ({'; '.join(ev.reasons[:3])})")
    cue = " | ".join(parts)
    return cue[:max_chars] if len(cue) > max_chars else cue
