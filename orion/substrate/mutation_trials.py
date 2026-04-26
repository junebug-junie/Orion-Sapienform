from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from orion.core.schemas.substrate_mutation import MutationProposalV1, MutationTrialV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_control_surface import get_chat_reflective_lane_threshold
from orion.substrate.mutation_scoring import ClassSpecificScorer


@dataclass
class ReplayCorpusRegistry:
    corpus_by_class: dict[str, str]
    baseline_metric_ref_by_class: dict[str, str]

    def ready_for_class(self, mutation_class: str) -> bool:
        return mutation_class in self.corpus_by_class and mutation_class in self.baseline_metric_ref_by_class


@dataclass(frozen=True)
class RoutingReplayCase:
    telemetry_id: str
    execution_outcome: str
    runtime_duration_ms: int
    priority_score: float
    desired_escalate: bool
    selection_reason: str
    decision_confidence: float | None = None
    route_quality_signal: float | None = None
    task_completion_signal: float | None = None
    false_escalation: bool = False
    false_downgrade: bool = False
    operator_correction: str | None = None
    used_rich_artifact: bool = False


@dataclass(frozen=True)
class RoutingReplayEvaluation:
    case_count: int
    escalation_count: int
    borderline_count: int
    route_appropriateness: float
    escalation_appropriateness: float
    downgrade_appropriateness: float
    helpfulness_proxy: float
    avg_runtime_ms: float
    corpus_coverage: float
    evaluator_confidence: float
    rich_signal_case_count: int


@dataclass(frozen=True)
class RoutingReplayEvaluator:
    borderline_band: float = 0.05

    def build_cases(self, records: list[GraphReviewTelemetryRecordV1]) -> list[RoutingReplayCase]:
        cases: list[RoutingReplayCase] = []
        for record in records:
            parsed = self._parse_rich_artifacts(record)
            priority = self._priority_score(record, parsed_confidence=parsed.get("decision_confidence"))
            desired = self._desired_escalation(record, parsed)
            cases.append(
                RoutingReplayCase(
                    telemetry_id=record.telemetry_id,
                    execution_outcome=str(record.execution_outcome),
                    runtime_duration_ms=max(0, int(record.runtime_duration_ms)),
                    priority_score=priority,
                    desired_escalate=desired,
                    selection_reason=record.selection_reason,
                    decision_confidence=parsed.get("decision_confidence"),
                    route_quality_signal=parsed.get("route_quality_signal"),
                    task_completion_signal=parsed.get("task_completion_signal"),
                    false_escalation=bool(parsed.get("false_escalation")),
                    false_downgrade=bool(parsed.get("false_downgrade")),
                    operator_correction=parsed.get("operator_correction"),
                    used_rich_artifact=bool(parsed.get("used_rich_artifact")),
                )
            )
        return cases

    def evaluate_threshold(self, *, cases: list[RoutingReplayCase], threshold: float) -> RoutingReplayEvaluation:
        if not cases:
            return RoutingReplayEvaluation(
                case_count=0,
                escalation_count=0,
                borderline_count=0,
                route_appropriateness=0.0,
                escalation_appropriateness=0.0,
                downgrade_appropriateness=0.0,
                helpfulness_proxy=0.0,
                avg_runtime_ms=0.0,
                corpus_coverage=0.0,
                evaluator_confidence=0.0,
                rich_signal_case_count=0,
            )
        threshold_value = max(0.0, min(1.0, float(threshold)))
        total = float(len(cases))
        simulated_escalations = 0
        borderline = 0
        matched = 0
        escalate_matches = 0
        escalate_total = 0
        downgrade_matches = 0
        downgrade_total = 0
        runtime_total = 0.0
        rich_signal_cases = 0
        route_quality_sum = 0.0
        route_quality_count = 0
        completion_sum = 0.0
        completion_count = 0
        for case in cases:
            escalated = case.priority_score >= threshold_value
            runtime_total += float(case.runtime_duration_ms)
            if case.used_rich_artifact:
                rich_signal_cases += 1
            if escalated:
                simulated_escalations += 1
            if abs(case.priority_score - threshold_value) <= self.borderline_band:
                borderline += 1
            if case.desired_escalate:
                escalate_total += 1
                if escalated:
                    escalate_matches += 1
            else:
                downgrade_total += 1
                if not escalated:
                    downgrade_matches += 1
            if escalated == case.desired_escalate:
                matched += 1
            if isinstance(case.route_quality_signal, float):
                route_quality_sum += max(0.0, min(1.0, float(case.route_quality_signal)))
                route_quality_count += 1
            if isinstance(case.task_completion_signal, float):
                completion_sum += max(0.0, min(1.0, float(case.task_completion_signal)))
                completion_count += 1
        escalation_appropriateness = (float(escalate_matches) / float(escalate_total)) if escalate_total else 1.0
        downgrade_appropriateness = (float(downgrade_matches) / float(downgrade_total)) if downgrade_total else 1.0
        route_appropriateness = float(matched) / total
        route_quality_proxy = (route_quality_sum / float(route_quality_count)) if route_quality_count else route_appropriateness
        completion_proxy = (completion_sum / float(completion_count)) if completion_count else route_appropriateness
        helpfulness_proxy = (route_quality_proxy * 0.5) + (completion_proxy * 0.3) + (route_appropriateness * 0.2)
        avg_runtime = runtime_total / total
        coverage = float(rich_signal_cases) / total
        evaluator_confidence = min(1.0, (coverage * 0.7) + ((1.0 if total >= 8 else (total / 8.0)) * 0.3))
        return RoutingReplayEvaluation(
            case_count=int(total),
            escalation_count=simulated_escalations,
            borderline_count=borderline,
            route_appropriateness=route_appropriateness,
            escalation_appropriateness=escalation_appropriateness,
            downgrade_appropriateness=downgrade_appropriateness,
            helpfulness_proxy=helpfulness_proxy,
            avg_runtime_ms=avg_runtime,
            corpus_coverage=coverage,
            evaluator_confidence=evaluator_confidence,
            rich_signal_case_count=rich_signal_cases,
        )

    @staticmethod
    def _priority_score(record: GraphReviewTelemetryRecordV1, *, parsed_confidence: float | None = None) -> float:
        if isinstance(parsed_confidence, float):
            return max(0.0, min(1.0, float(parsed_confidence)))
        if isinstance(record.selected_priority, int):
            return max(0.0, min(1.0, float(record.selected_priority) / 100.0))
        default_map = {
            "failed": 0.9,
            "suppressed": 0.75,
            "terminated": 0.7,
            "noop": 0.45,
            "executed": 0.35,
        }
        return max(0.0, min(1.0, float(default_map.get(str(record.execution_outcome), 0.5))))

    @staticmethod
    def _desired_escalation(record: GraphReviewTelemetryRecordV1, parsed: dict[str, Any]) -> bool:
        if parsed.get("false_escalation"):
            return False
        if parsed.get("false_downgrade"):
            return True
        correction = parsed.get("operator_correction")
        if correction == "escalate":
            return True
        if correction == "downgrade":
            return False
        if "requeue_review" in set(record.consolidation_outcomes or []):
            return True
        return bool(record.execution_outcome in {"failed", "suppressed", "terminated"} or record.degraded)

    def _parse_rich_artifacts(self, record: GraphReviewTelemetryRecordV1) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        evidence_items = [str(record.selection_reason or "")] + list(record.notes or []) + list(record.consolidation_outcomes or [])
        for item in evidence_items:
            token = str(item or "").strip()
            if not token:
                continue
            self._maybe_parse_signal(token=token, output=parsed)
        parsed["used_rich_artifact"] = any(
            key in parsed
            for key in (
                "decision_confidence",
                "route_quality_signal",
                "task_completion_signal",
                "false_escalation",
                "false_downgrade",
                "operator_correction",
            )
        )
        return parsed

    @staticmethod
    def _maybe_parse_signal(*, token: str, output: dict[str, Any]) -> None:
        normalized = token.lower()
        if normalized in {"false_escalation", "routing:false_escalation", "quality:false_escalation"}:
            output["false_escalation"] = True
            return
        if normalized in {"false_downgrade", "routing:false_downgrade", "quality:false_downgrade"}:
            output["false_downgrade"] = True
            return
        if normalized.startswith("operator_correction:"):
            value = normalized.split(":", 1)[1].strip()
            if value in {"escalate", "downgrade"}:
                output["operator_correction"] = value
            return

        def parse_float(prefixes: tuple[str, ...], field: str) -> None:
            if field in output:
                return
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    maybe = normalized.split(":", 1)[1].strip()
                    try:
                        output[field] = float(maybe)
                    except Exception:
                        pass
                    return
                match = re.search(rf"{re.escape(prefix)}\s*([-+]?\d*\.?\d+)", normalized)
                if match:
                    try:
                        output[field] = float(match.group(1))
                    except Exception:
                        pass
                    return

        parse_float(("decision_confidence:", "routing_confidence:", "confidence:"), "decision_confidence")
        parse_float(("route_quality:", "route_quality_signal:", "task_quality:"), "route_quality_signal")
        parse_float(("task_completion:", "completion_proxy:", "task_success:"), "task_completion_signal")

        if "decision_confidence" in output and isinstance(output["decision_confidence"], float):
            output["decision_confidence"] = max(0.0, min(1.0, float(output["decision_confidence"])))
        for field in ("route_quality_signal", "task_completion_signal"):
            if field in output and isinstance(output[field], float):
                output[field] = max(0.0, min(1.0, float(output[field])))


@dataclass
class SubstrateTrialRunner:
    scorer: ClassSpecificScorer
    corpus_registry: ReplayCorpusRegistry
    routing_replay_evaluator: RoutingReplayEvaluator = RoutingReplayEvaluator()

    def run_trial(
        self,
        *,
        proposal: MutationProposalV1,
        measured_metrics: dict[str, float],
        replay_records: list[GraphReviewTelemetryRecordV1] | None = None,
    ) -> MutationTrialV1:
        if not self.corpus_registry.ready_for_class(proposal.mutation_class):
            return MutationTrialV1(
                proposal_id=proposal.proposal_id,
                mutation_class=proposal.mutation_class,
                replay_corpus_id=self.corpus_registry.corpus_by_class.get(proposal.mutation_class, "missing"),
                baseline_metric_ref=self.corpus_registry.baseline_metric_ref_by_class.get(proposal.mutation_class, "missing"),
                status="inconclusive",
                metrics=measured_metrics,
                notes=["missing_replay_corpus_or_baseline_metrics"],
            )
        derived_metrics = self._derive_replay_metrics(
            proposal=proposal,
            measured_metrics=measured_metrics,
            replay_records=replay_records or [],
        )
        status, notes = self.scorer.evaluate(mutation_class=proposal.mutation_class, metrics=measured_metrics)
        if derived_metrics:
            measured_metrics = dict(derived_metrics)
            status, notes = self.scorer.evaluate(mutation_class=proposal.mutation_class, metrics=measured_metrics)
        return MutationTrialV1(
            proposal_id=proposal.proposal_id,
            mutation_class=proposal.mutation_class,
            replay_corpus_id=self.corpus_registry.corpus_by_class[proposal.mutation_class],
            baseline_metric_ref=self.corpus_registry.baseline_metric_ref_by_class[proposal.mutation_class],
            status=status,
            metrics=measured_metrics,
            notes=notes,
        )

    def inspect_routing_replay(
        self,
        *,
        proposal: MutationProposalV1,
        replay_records: list[GraphReviewTelemetryRecordV1],
        baseline_threshold: float | None = None,
    ) -> dict[str, Any]:
        cases = self.routing_replay_evaluator.build_cases(replay_records)
        patch_threshold = self._routing_threshold_from_patch(proposal)
        baseline = baseline_threshold if baseline_threshold is not None else self._routing_baseline_threshold(proposal)
        baseline_eval = self.routing_replay_evaluator.evaluate_threshold(cases=cases, threshold=baseline)
        candidate_eval = self.routing_replay_evaluator.evaluate_threshold(cases=cases, threshold=patch_threshold)
        rich_count = sum(1 for case in cases if case.used_rich_artifact)
        correction_escalate = sum(1 for case in cases if case.operator_correction == "escalate")
        correction_downgrade = sum(1 for case in cases if case.operator_correction == "downgrade")
        false_escalations = sum(1 for case in cases if case.false_escalation)
        false_downgrades = sum(1 for case in cases if case.false_downgrade)
        return {
            "mutation_class": proposal.mutation_class,
            "proposal_id": proposal.proposal_id,
            "baseline_threshold": baseline,
            "candidate_threshold": patch_threshold,
            "case_count": len(cases),
            "corpus_composition": {
                "rich_signal_case_count": rich_count,
                "rich_signal_coverage": (float(rich_count) / float(len(cases))) if cases else 0.0,
                "operator_correction_escalate_count": correction_escalate,
                "operator_correction_downgrade_count": correction_downgrade,
                "false_escalation_count": false_escalations,
                "false_downgrade_count": false_downgrades,
            },
            "sample_cases": [
                {
                    "telemetry_id": case.telemetry_id,
                    "execution_outcome": case.execution_outcome,
                    "priority_score": case.priority_score,
                    "desired_escalate": case.desired_escalate,
                    "decision_confidence": case.decision_confidence,
                    "route_quality_signal": case.route_quality_signal,
                    "task_completion_signal": case.task_completion_signal,
                    "false_escalation": case.false_escalation,
                    "false_downgrade": case.false_downgrade,
                    "operator_correction": case.operator_correction,
                    "used_rich_artifact": case.used_rich_artifact,
                    "runtime_duration_ms": case.runtime_duration_ms,
                    "selection_reason": case.selection_reason,
                }
                for case in cases[:20]
            ],
            "baseline_eval": baseline_eval.__dict__,
            "candidate_eval": candidate_eval.__dict__,
            "derived_metrics": self._routing_metrics_from_evals(
                baseline_eval=baseline_eval,
                candidate_eval=candidate_eval,
            ),
        }

    def _derive_replay_metrics(
        self,
        *,
        proposal: MutationProposalV1,
        measured_metrics: dict[str, float],
        replay_records: list[GraphReviewTelemetryRecordV1],
    ) -> dict[str, float] | None:
        if measured_metrics:
            return None
        if proposal.mutation_class != "routing_threshold_patch":
            return None
        cases = self.routing_replay_evaluator.build_cases(replay_records)
        if not cases:
            return None
        baseline = self._routing_baseline_threshold(proposal)
        candidate = self._routing_threshold_from_patch(proposal)
        baseline_eval = self.routing_replay_evaluator.evaluate_threshold(cases=cases, threshold=baseline)
        candidate_eval = self.routing_replay_evaluator.evaluate_threshold(cases=cases, threshold=candidate)
        return self._routing_metrics_from_evals(baseline_eval=baseline_eval, candidate_eval=candidate_eval)

    @staticmethod
    def _routing_metrics_from_evals(
        *,
        baseline_eval: RoutingReplayEvaluation,
        candidate_eval: RoutingReplayEvaluation,
    ) -> dict[str, float]:
        return {
            "success_rate_delta": candidate_eval.route_appropriateness - baseline_eval.route_appropriateness,
            "latency_ms_delta": baseline_eval.avg_runtime_ms - candidate_eval.avg_runtime_ms,
            "route_appropriateness_proxy": candidate_eval.route_appropriateness,
            "escalation_appropriateness": candidate_eval.escalation_appropriateness,
            "downgrade_appropriateness": candidate_eval.downgrade_appropriateness,
            "helpfulness_proxy": candidate_eval.helpfulness_proxy,
            "corpus_coverage": candidate_eval.corpus_coverage,
            "evaluator_confidence": candidate_eval.evaluator_confidence,
            "rich_signal_case_count": float(candidate_eval.rich_signal_case_count),
            "replay_case_count": float(candidate_eval.case_count),
        }

    @staticmethod
    def _routing_threshold_from_patch(proposal: MutationProposalV1) -> float:
        value = proposal.patch.patch.get("chat_reflective_lane_threshold")
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return get_chat_reflective_lane_threshold()

    @staticmethod
    def _routing_baseline_threshold(proposal: MutationProposalV1) -> float:
        value = proposal.patch.rollback_payload.get("chat_reflective_lane_threshold")
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return get_chat_reflective_lane_threshold()
