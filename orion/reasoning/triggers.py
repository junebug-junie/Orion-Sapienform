from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.endogenous import (
    EndogenousTriggerDebugV1,
    EndogenousTriggerDecisionV1,
    EndogenousTriggerRequestV1,
    EndogenousTriggerSignalV1,
)
from orion.reasoning.trigger_history import InMemoryTriggerHistoryStore


@dataclass(frozen=True)
class TriggerPolicy:
    contradiction_threshold: float = 0.75
    concept_threshold: float = 0.68
    autonomy_threshold: float = 0.72
    mentor_threshold: float = 0.82
    reflective_threshold: float = 0.45
    workflow_cooldowns_seconds: dict[str, int] | None = None
    subject_cooldown_seconds: int = 120
    contradiction_debounce_seconds: int = 180
    mentor_cooldown_seconds: int = 600

    def cooldown_for(self, workflow_type: str) -> int:
        base = self.workflow_cooldowns_seconds or {
            "contradiction_review": 180,
            "concept_refinement": 240,
            "autonomy_review": 240,
            "mentor_critique": 600,
            "reflective_journal": 120,
            "no_action": 0,
        }
        return base.get(workflow_type, 0)


class EndogenousTriggerEvaluator:
    """Deterministic pressure evaluator that selects bounded workflow triggers."""

    def __init__(self, *, history: InMemoryTriggerHistoryStore | None = None, policy: TriggerPolicy | None = None) -> None:
        self._history = history or InMemoryTriggerHistoryStore()
        self._policy = policy or TriggerPolicy()

    @property
    def history(self) -> InMemoryTriggerHistoryStore:
        return self._history

    def evaluate(self, request: EndogenousTriggerRequestV1) -> EndogenousTriggerDecisionV1:
        signal = self._build_signal(request)
        scored = self._candidate_scores(request, signal)
        if all(score == 0.0 for workflow, score in scored.items() if workflow != "no_action"):
            scored["no_action"] = 1.0
        chosen_workflow, chosen_score = max(scored.items(), key=lambda item: item[1])
        reasons = self._reasons_for(request, signal, chosen_workflow)
        cause_signature = self._cause_signature(request, chosen_workflow)

        alternatives = [
            f"{workflow}:{score:.3f}"
            for workflow, score in sorted(scored.items(), key=lambda item: item[1], reverse=True)
            if workflow != chosen_workflow and score > 0.0
        ]

        debug = EndogenousTriggerDebugV1(
            considered_workflows=sorted(scored.keys()),
            selected_workflow_score=round(chosen_score, 3),
            cause_signature=cause_signature,
            policy_counters={
                "contradiction_threshold": self._policy.contradiction_threshold,
                "concept_threshold": self._policy.concept_threshold,
                "autonomy_threshold": self._policy.autonomy_threshold,
                "mentor_threshold": self._policy.mentor_threshold,
                "reflective_threshold": self._policy.reflective_threshold,
            },
        )

        if request.lifecycle_state in {"dormant", "retired"} and chosen_workflow != "reflective_journal":
            debug.suppression_reasons.append(f"lifecycle_state:{request.lifecycle_state}")
            return EndogenousTriggerDecisionV1(
                request_id=request.request_id,
                outcome="suppress",
                workflow_type="no_action",
                reasons=["entity_not_active"],
                alternatives_not_chosen=alternatives,
                cooldown_applied=False,
                signal=signal,
                debug=debug,
            )

        if chosen_workflow == "no_action":
            return EndogenousTriggerDecisionV1(
                request_id=request.request_id,
                outcome="noop",
                workflow_type="no_action",
                reasons=["insufficient_pressure"],
                alternatives_not_chosen=alternatives,
                signal=signal,
                debug=debug,
            )

        debounce_applied = False
        coalesced = False
        outcome = "trigger"
        if chosen_workflow == "contradiction_review":
            seen = self._history.has_recent_signature(
                cause_signature=cause_signature,
                subject_ref=request.subject_ref,
                now=request.evaluated_at,
                within_seconds=self._policy.contradiction_debounce_seconds,
            )
            if seen:
                debounce_applied = True
                coalesced = True
                outcome = "coalesce"
                debug.suppression_reasons.append("contradiction_debounce")

        cooldown = max(self._policy.cooldown_for(chosen_workflow), self._policy.subject_cooldown_seconds)
        if chosen_workflow == "mentor_critique":
            cooldown = max(cooldown, self._policy.mentor_cooldown_seconds)
        remaining = self._history.cooldown_remaining_seconds(
            workflow_type=chosen_workflow,
            subject_ref=request.subject_ref,
            now=request.evaluated_at,
            cooldown_seconds=cooldown,
        )
        if remaining > 0 and outcome != "coalesce":
            debug.cooldown_seconds_remaining = remaining
            debug.suppression_reasons.append("workflow_cooldown")
            return EndogenousTriggerDecisionV1(
                request_id=request.request_id,
                outcome="suppress",
                workflow_type=chosen_workflow,
                reasons=["cooldown_active"],
                alternatives_not_chosen=alternatives,
                cooldown_applied=True,
                signal=signal,
                debug=debug,
            )

        return EndogenousTriggerDecisionV1(
            request_id=request.request_id,
            outcome=outcome,
            workflow_type=chosen_workflow,
            reasons=reasons,
            alternatives_not_chosen=alternatives,
            cooldown_applied=False,
            debounce_applied=debounce_applied,
            coalesced=coalesced,
            signal=signal,
            debug=debug,
        )

    def _build_signal(self, request: EndogenousTriggerRequestV1) -> EndogenousTriggerSignalV1:
        contradiction_pressure = min(
            1.0,
            (request.unresolved_contradiction_count / 3.0) * 0.7 + request.contradiction_severity_score * 0.3,
        )
        concept_pressure = min(
            1.0,
            request.concept_fragmentation_score * 0.7 + min(1.0, request.low_confidence_artifact_count / 6.0) * 0.3,
        )
        autonomy_pressure = min(1.0, request.autonomy_pressure * 0.7 + request.spark_pressure * 0.2 + request.spark_instability * 0.1)
        fallback_bonus = 0.2 if request.reasoning_summary and request.reasoning_summary.fallback_recommended else 0.0
        mentor_pressure = min(1.0, min(1.0, request.mentor_gap_count / 3.0) * 0.7 + request.spark_instability * 0.2 + fallback_bonus)
        reflective_pressure = min(1.0, max(contradiction_pressure * 0.45, concept_pressure * 0.5, autonomy_pressure * 0.5, mentor_pressure * 0.4))
        total = min(1.0, contradiction_pressure * 0.35 + concept_pressure * 0.25 + autonomy_pressure * 0.2 + mentor_pressure * 0.2)
        return EndogenousTriggerSignalV1(
            contradiction_pressure=round(contradiction_pressure, 3),
            concept_pressure=round(concept_pressure, 3),
            autonomy_pressure=round(autonomy_pressure, 3),
            mentor_pressure=round(mentor_pressure, 3),
            reflective_pressure=round(reflective_pressure, 3),
            total_pressure=round(total, 3),
            triggerable=total >= self._policy.reflective_threshold,
            counters={
                "unresolved_contradiction_count": request.unresolved_contradiction_count,
                "selected_artifact_count": len(request.selected_artifact_ids),
                "low_confidence_artifact_count": request.low_confidence_artifact_count,
                "mentor_gap_count": request.mentor_gap_count,
            },
        )

    def _candidate_scores(self, request: EndogenousTriggerRequestV1, signal: EndogenousTriggerSignalV1) -> dict[str, float]:
        return {
            "contradiction_review": signal.contradiction_pressure if signal.contradiction_pressure >= self._policy.contradiction_threshold else 0.0,
            "concept_refinement": signal.concept_pressure if signal.concept_pressure >= self._policy.concept_threshold else 0.0,
            "autonomy_review": signal.autonomy_pressure if signal.autonomy_pressure >= self._policy.autonomy_threshold else 0.0,
            "mentor_critique": signal.mentor_pressure if signal.mentor_pressure >= self._policy.mentor_threshold else 0.0,
            "reflective_journal": signal.reflective_pressure if signal.total_pressure >= self._policy.reflective_threshold else 0.0,
            "no_action": 0.0,
        }

    def _reasons_for(self, request: EndogenousTriggerRequestV1, signal: EndogenousTriggerSignalV1, workflow: str) -> list[str]:
        if workflow == "contradiction_review":
            return [
                "unresolved_contradictions_high",
                f"contradiction_count:{request.unresolved_contradiction_count}",
                f"contradiction_pressure:{signal.contradiction_pressure}",
            ]
        if workflow == "concept_refinement":
            return [
                "concept_fragmentation_detected",
                f"concept_pressure:{signal.concept_pressure}",
                f"low_confidence_artifacts:{request.low_confidence_artifact_count}",
            ]
        if workflow == "autonomy_review":
            return [
                "autonomy_drive_strain",
                f"autonomy_pressure:{signal.autonomy_pressure}",
            ]
        if workflow == "mentor_critique":
            return [
                "mentor_worthy_gap_cluster",
                f"mentor_gap_count:{request.mentor_gap_count}",
                f"mentor_pressure:{signal.mentor_pressure}",
            ]
        if workflow == "reflective_journal":
            return ["moderate_unresolved_tension", f"total_pressure:{signal.total_pressure}"]
        return ["insufficient_pressure"]

    @staticmethod
    def _cause_signature(request: EndogenousTriggerRequestV1, workflow: str) -> str:
        contradiction_slice = ",".join(sorted(request.contradiction_refs)[:3])
        return (
            f"{workflow}|{request.subject_ref or '-'}|"
            f"c{request.unresolved_contradiction_count}|m{request.mentor_gap_count}|"
            f"l{request.low_confidence_artifact_count}|r{contradiction_slice}"
        )
