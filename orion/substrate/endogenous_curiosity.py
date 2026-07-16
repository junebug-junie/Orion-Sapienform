"""Endogenous agency — rung 5 of the self-modeling loop. FLAG OFF BY DEFAULT.

``FrontierCuriosityEvaluator`` only crosses from observation into invocation
when an operator asks (``explicit_operator_request``). This module lets
intrinsic signals seed ``curiosity_candidate`` signals with no operator
trigger:

- sustained prediction error on substrate nodes (rung 1's surprise) --
  staleness-decayed by node age so "sustained" means currently still
  surprising, not merely surprising once (see
  ``_prediction_error_staleness_decay`` below);
- repair-pressure appraisals (``appraisal/repair_pressure.py``);
- unresolved open-loops from the rung-3 attention broadcast.

Guardrails (all load-bearing, do not relax casually):
- master flag ``ORION_ENDOGENOUS_CURIOSITY_ENABLED`` defaults false, and the
  kill switch ``ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH`` beats it;
- hard per-cycle budget, clamped to ``HARD_BUDGET_CEILING`` regardless of env;
- candidates target ``concept_graph`` only — never the strict
  self/relationship zone, never the autonomy zone directly;
- output is *signals only*: they ride the existing frontier decision/plan
  path and anything that proposes change goes through rung-6 governance
  (trials + rollback). No external action, no auto-apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.pressure import PressureConfig

_TRUTHY = {"1", "true", "yes", "on"}

ENDOGENOUS_FLAG = "ORION_ENDOGENOUS_CURIOSITY_ENABLED"
KILL_SWITCH_FLAG = "ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH"
BUDGET_ENV = "ORION_ENDOGENOUS_CURIOSITY_BUDGET"
MIN_REPAIR_LEVEL_ENV = "ORION_ENDOGENOUS_CURIOSITY_MIN_REPAIR_LEVEL"
HARD_BUDGET_CEILING = 8

# Prediction-error staleness ceiling. `metadata["prediction_error"]` is a raw
# snapshot (never decays on its own; see `pressure.py::prediction_error_pressure()`'s
# docstring) written by the transport reducer whenever `transport_prediction_error()`
# fires. Left unguarded, a node that was surprising once is "sustained prediction
# error" forever and, since candidates sort strongest-first, wins the bounded
# per-cycle budget on every tick. Live-confirmed 2026-07-16: `node:substrate.transport`
# held `signal_strength=1.0` identically across all 1,428 persisted candidate sets
# over the prior 24h (path live since 2026-07-02, not dormant).
#
# Deliberately NOT switched to reading `dynamic_pressure` directly, unlike the
# sibling fix in `attention_broadcast.py::_node_salience()` (PR #1061):
# `dynamic_pressure` is a composite of drive/prediction-error/contradiction pressure
# propagated across edges (`dynamics.py::_compute_pressures()`), so it can be driven
# entirely by an unrelated neighbor's pressure with zero prediction error of its
# own -- reading it here would sometimes make
# `evidence_summary="sustained prediction error on {node_id}"` literally false.
# Instead the raw value is decayed by the node's own age, reusing the same horizon
# `prediction_error_pressure()` already applies to this same field -- keeps the
# signal specifically about prediction error while fixing the staleness.
#
# Captured once at import time from `PressureConfig()`'s bare default (matching
# `pressure.py`'s own current usage -- neither module reads this from env today).
# If `PressureConfig` ever becomes env-configurable, this constant would silently
# pin to whatever was true at process start; re-derive per-call if that happens.
_PREDICTION_ERROR_DECAY_HORIZON_SECONDS = PressureConfig().prediction_error_decay_horizon_seconds


def _prediction_error_staleness_decay(node: Any, *, now: datetime) -> float:
    """Linear decay-to-zero by node age, mirroring `prediction_error_pressure()`.

    A node with no ``temporal.observed_at`` (duck-typed test doubles, mainly --
    real ``BaseSubstrateNodeV1`` instances always carry ``temporal``) is treated as
    unaged: decay factor 1.0, i.e. no staleness penalty, rather than raising or
    silently zeroing it out.
    """
    observed = getattr(getattr(node, "temporal", None), "observed_at", None)
    if observed is None:
        return 1.0
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    horizon = _PREDICTION_ERROR_DECAY_HORIZON_SECONDS
    if horizon <= 0:
        return 1.0
    age_seconds = max(0.0, (now - observed).total_seconds())
    return max(0.0, 1.0 - (age_seconds / horizon))


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "false")).strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EndogenousCuriosityConfig:
    enabled: bool = False
    kill_switch: bool = False
    budget: int = 3
    min_prediction_error: float = 0.55
    min_repair_level: float = 0.6
    min_loop_salience: float = 0.5

    @classmethod
    def from_env(cls) -> EndogenousCuriosityConfig:
        try:
            budget = int(os.getenv(BUDGET_ENV) or 3)
        except (TypeError, ValueError):
            budget = 3
        return cls(
            enabled=_env_flag(ENDOGENOUS_FLAG),
            kill_switch=_env_flag(KILL_SWITCH_FLAG),
            budget=budget,
            min_repair_level=_env_float(MIN_REPAIR_LEVEL_ENV, 0.6),
        )


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _prediction_error_candidates(
    nodes: Sequence[Any],
    *,
    anchor_scope: str,
    subject_ref: str | None,
    min_error: float,
    now: datetime,
) -> list[FrontierInvocationSignalV1]:
    candidates: list[FrontierInvocationSignalV1] = []
    for node in nodes:
        try:
            metadata = dict(getattr(node, "metadata", None) or {})
            raw_error = _clamp01(metadata.get("prediction_error"))
            if raw_error <= 0.0:
                continue
            error = raw_error * _prediction_error_staleness_decay(node, now=now)
            if error < min_error:
                continue
            node_id = str(getattr(node, "node_id", "") or "")
            if not node_id:
                continue
            candidates.append(
                FrontierInvocationSignalV1(
                    signal_type="curiosity_candidate",
                    anchor_scope=anchor_scope,
                    subject_ref=subject_ref,
                    target_zone="concept_graph",
                    task_type_candidate="evidence_gap_scan",
                    focal_node_refs=[node_id],
                    signal_strength=error,
                    evidence_summary=f"sustained prediction error on {node_id}",
                    confidence=0.7,
                    notes=["endogenous_seed", "source:prediction_error"],
                )
            )
        except Exception:
            continue
    return candidates


def _repair_pressure_candidate(
    appraisal: Any,
    *,
    anchor_scope: str,
    subject_ref: str | None,
    min_level: float,
) -> FrontierInvocationSignalV1 | None:
    if appraisal is None:
        return None
    try:
        dimensions = dict(getattr(appraisal, "dimensions", None) or {})
        level = _clamp01(dimensions.get("level", getattr(appraisal, "level", 0.0)))
        if level < min_level:
            return None
        refs = [str(r) for r in (getattr(appraisal, "causal_molecule_ids", None) or [])][:8]
        summary = str(getattr(appraisal, "summary", None) or "repair pressure elevated")
        return FrontierInvocationSignalV1(
            signal_type="curiosity_candidate",
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            target_zone="concept_graph",
            task_type_candidate="relation_discovery",
            focal_node_refs=refs,
            signal_strength=level,
            evidence_summary=summary,
            confidence=_clamp01(getattr(appraisal, "confidence", 0.6)),
            notes=["endogenous_seed", "source:repair_pressure"],
        )
    except Exception:
        return None


def _attention_loop_candidates(
    attention_frame: Any,
    *,
    anchor_scope: str,
    subject_ref: str | None,
    min_salience: float,
) -> list[FrontierInvocationSignalV1]:
    if attention_frame is None:
        return []
    candidates: list[FrontierInvocationSignalV1] = []
    try:
        deferred = set(getattr(attention_frame, "deferred_items", None) or [])
        for loop in getattr(attention_frame, "open_loops", None) or []:
            try:
                if getattr(loop, "already_known", False):
                    continue
                if deferred and loop.id not in deferred:
                    continue
                strength = _clamp01(getattr(loop, "novelty", 0.0))
                if strength < min_salience:
                    continue
                refs = [str(r) for r in (getattr(loop, "source_refs", None) or [])][:8]
                candidates.append(
                    FrontierInvocationSignalV1(
                        signal_type="curiosity_candidate",
                        anchor_scope=anchor_scope,
                        subject_ref=subject_ref,
                        target_zone="concept_graph",
                        task_type_candidate="concept_expand",
                        focal_node_refs=refs,
                        signal_strength=strength,
                        evidence_summary=f"unresolved attention open-loop: {loop.description}",
                        confidence=_clamp01(getattr(loop, "confidence", 0.6)),
                        notes=["endogenous_seed", "source:attention_open_loop"],
                    )
                )
            except Exception:
                continue
    except Exception:
        return candidates
    return candidates


def _coverage_gap_candidates(
    signals: Sequence[FrontierInvocationSignalV1],
    *,
    anchor_scope: str,
    subject_ref: str | None,
) -> list[FrontierInvocationSignalV1]:
    out: list[FrontierInvocationSignalV1] = []
    for sig in signals:
        if str(sig.signal_type) != "world_coverage_gap":
            continue
        out.append(
            FrontierInvocationSignalV1(
                signal_type="curiosity_candidate",
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                target_zone="concept_graph",
                task_type_candidate=sig.task_type_candidate,
                focal_node_refs=list(sig.focal_node_refs[:8]),
                signal_strength=_clamp01(sig.signal_strength),
                evidence_summary=sig.evidence_summary,
                confidence=_clamp01(sig.confidence),
                notes=[
                    "endogenous_seed",
                    "world_coverage_gap",
                    "source:world_coverage_gap",
                    *list(sig.notes[:6]),
                ],
            )
        )
    return out


def endogenous_curiosity_candidates(
    *,
    anchor_scope: str = "orion",
    subject_ref: str | None = "entity:orion",
    nodes: Sequence[Any] = (),
    repair_appraisal: Any = None,
    attention_frame: Any = None,
    coverage_gap_signals: Sequence[FrontierInvocationSignalV1] = (),
    config: EndogenousCuriosityConfig | None = None,
    now: datetime | None = None,
) -> list[FrontierInvocationSignalV1]:
    """Bounded set of self-seeded curiosity candidates; [] unless enabled.

    The kill switch always wins. The returned list is sorted strongest-first
    and hard-capped, so a runaway signal source cannot flood the frontier
    decision path.
    """
    cfg = config if config is not None else EndogenousCuriosityConfig.from_env()
    if not cfg.enabled or cfg.kill_switch:
        return []
    resolved_now = now or datetime.now(timezone.utc)

    candidates = _prediction_error_candidates(
        nodes,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        min_error=cfg.min_prediction_error,
        now=resolved_now,
    )
    repair = _repair_pressure_candidate(
        repair_appraisal,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        min_level=cfg.min_repair_level,
    )
    if repair is not None:
        candidates.append(repair)
    candidates.extend(
        _coverage_gap_candidates(
            coverage_gap_signals,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
        )
    )
    candidates.extend(
        _attention_loop_candidates(
            attention_frame,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            min_salience=cfg.min_loop_salience,
        )
    )

    budget = max(1, min(cfg.budget, HARD_BUDGET_CEILING))
    candidates.sort(key=lambda s: (s.signal_strength, s.confidence), reverse=True)
    return candidates[:budget]
