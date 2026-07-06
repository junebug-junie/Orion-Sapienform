"""Endogenous agency — rung 5 of the self-modeling loop. FLAG OFF BY DEFAULT.

``FrontierCuriosityEvaluator`` only crosses from observation into invocation
when an operator asks (``explicit_operator_request``). This module lets
intrinsic signals seed ``curiosity_candidate`` signals with no operator
trigger:

- sustained prediction error on substrate nodes (rung 1's surprise);
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
from typing import Any, Sequence

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1

_TRUTHY = {"1", "true", "yes", "on"}

ENDOGENOUS_FLAG = "ORION_ENDOGENOUS_CURIOSITY_ENABLED"
KILL_SWITCH_FLAG = "ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH"
BUDGET_ENV = "ORION_ENDOGENOUS_CURIOSITY_BUDGET"
MIN_REPAIR_LEVEL_ENV = "ORION_ENDOGENOUS_CURIOSITY_MIN_REPAIR_LEVEL"
HARD_BUDGET_CEILING = 8


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
) -> list[FrontierInvocationSignalV1]:
    candidates: list[FrontierInvocationSignalV1] = []
    for node in nodes:
        try:
            metadata = dict(getattr(node, "metadata", None) or {})
            error = _clamp01(metadata.get("prediction_error"))
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
) -> list[FrontierInvocationSignalV1]:
    """Bounded set of self-seeded curiosity candidates; [] unless enabled.

    The kill switch always wins. The returned list is sorted strongest-first
    and hard-capped, so a runaway signal source cannot flood the frontier
    decision path.
    """
    cfg = config if config is not None else EndogenousCuriosityConfig.from_env()
    if not cfg.enabled or cfg.kill_switch:
        return []

    candidates = _prediction_error_candidates(
        nodes,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        min_error=cfg.min_prediction_error,
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
