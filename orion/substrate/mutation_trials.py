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


@dataclass(frozen=True)
class GraphConsolidationReplayEvaluation:
    case_count: int
    resolution_rate: float
    requeue_rate: float


@dataclass(frozen=True)
class GraphConsolidationReplayEvaluator:
    """Derives graph_consolidation_param_patch's real trial metrics.

    Routing's evaluator can literally recompute "would this case have
    escalated under threshold X", because a case's priority_score is a
    static, already-stored value -- re-thresholding it is a pure function.
    graph_consolidation's params (query_limit_nodes/edges, revisit cadence)
    don't have that property: what a wider node query would have returned
    depends on live graph state at review time, which telemetry doesn't
    snapshot. There is no honest counterfactual replay here.

    What IS honestly measurable from real telemetry: whether the queue's
    own resolution behavior is trending better or worse right now. Split
    the real replay window in half chronologically and compare the two
    halves' outcome rates. This is a trend signal, not a controlled
    estimate of this specific proposal's effect -- see queue_resolution_delta
    and requeue_rate_delta's docstrings in derive_metrics() for the exact
    caveat, since the two are easy to conflate.

    ``resolved_outcomes`` is ``{"retire", "noop"}``, matching exactly the
    two outcomes GraphReviewScheduler._schedule_for_decision() treats as
    terminating (review_schedule.py: only these return `queue_item=None`,
    so the item is never re-upserted). "reinforce" is NOT resolution:
    _schedule_for_decision() groups it with "keep_provisional" under the
    same "bounded monitoring" cadence -- the item stays in the queue
    either way. Getting this wrong (an earlier version of this evaluator
    counted reinforce as resolved) would make the metric read "the queue
    is resolving faster" for exactly the outcome that keeps every one of
    those items in the queue unchanged.

    Independence caveat, named rather than hidden: queue_resolution_delta
    and requeue_rate_delta both come from the same outcome list and the
    same before/after split -- a shift from requeue_review to retire moves
    both at once. They are not two independent pieces of evidence, even
    though the contract (mutation_contracts.py) requires both to pass. Not
    fixed here -- changing what the contract measures is a bigger,
    separate decision than giving this class its first real evaluator.
    """

    resolved_outcomes: frozenset[str] = frozenset({"retire", "noop"})
    requeue_outcome: str = "requeue_review"
    min_window_size: int = 3

    def evaluate_window(self, records: list[GraphReviewTelemetryRecordV1]) -> GraphConsolidationReplayEvaluation | None:
        outcomes = [outcome for record in records for outcome in record.consolidation_outcomes]
        if not outcomes:
            return None
        total = 0
        resolved = 0
        requeued = 0
        for outcome in outcomes:
            total += 1
            if outcome in self.resolved_outcomes:
                resolved += 1
            if outcome == self.requeue_outcome:
                requeued += 1
        return GraphConsolidationReplayEvaluation(
            case_count=total,
            resolution_rate=float(resolved) / float(total),
            requeue_rate=float(requeued) / float(total),
        )

    def derive_metrics(self, *, replay_records: list[GraphReviewTelemetryRecordV1]) -> dict[str, float] | None:
        """queue_resolution_delta / requeue_rate_delta, or None if there is
        nothing real to compare yet.

        Two distinct "nothing to report" cases, both correctly None here
        (ClassSpecificScorer.evaluate() reads a missing metric as
        `inconclusive` -- the right read for "not enough evidence", never
        silently `passed`):

        1. Too few real rows to split into two meaningful halves. Returning
           a delta from a handful of records would be noise dressed as
           signal.
        2. Enough rows exist, but the two halves show the EXACT SAME
           outcome distribution -- zero evidence of any real change, not
           "at least it didn't get worse". metric_passed() (mutation_
           contracts.py) treats a delta of exactly 0.0 as passing (its
           floor is `>= 0.0`), which would auto-promote a proposal based on
           a total absence of signal -- backwards. This was not a rare
           edge case: 100% keep_provisional, both halves, every single
           real telemetry row, was the *entire* recorded history of this
           system until 2026-09-08 (PR #2162 first made any other outcome
           reachable at all).

        Only rows with a real consolidation_outcomes list count -- everything
        else (noop/suppressed/terminated/failed cycles) never ran
        consolidate() and has nothing to measure.
        """
        consolidated = sorted(
            (record for record in replay_records if record.consolidation_outcomes),
            key=lambda record: record.selected_at,
        )
        if len(consolidated) < self.min_window_size * 2:
            return None
        midpoint = len(consolidated) // 2
        before = self.evaluate_window(consolidated[:midpoint])
        after = self.evaluate_window(consolidated[midpoint:])
        # Both are guaranteed non-None here: `consolidated` was already
        # filtered to non-empty consolidation_outcomes above, and the
        # min_window_size*2 floor guarantees both slices are non-empty too.
        assert before is not None and after is not None
        if before.resolution_rate == after.resolution_rate and before.requeue_rate == after.requeue_rate:
            return None
        return {
            # Positive = the queue is terminating (retire/noop) more of its
            # reviews in the recent half of this window than the older half.
            "queue_resolution_delta": after.resolution_rate - before.resolution_rate,
            # Reversed on purpose: a LOWER requeue rate is the improvement,
            # but metric_passed() (mutation_contracts.py) treats every
            # metric the same way -- "passes if >= 0.0". A drop in requeue
            # rate has to read as a positive delta here, or this metric
            # would silently mean the opposite of what its name says.
            "requeue_rate_delta": before.requeue_rate - after.requeue_rate,
            # Not a contract metric (ClassSpecificScorer only checks the
            # two above) -- carried through as auditable evidence of how
            # much real data this trial's verdict rests on, mirroring
            # RoutingReplayEvaluation.case_count's role for routing.
            "graph_consolidation_replay_case_count": float(before.case_count + after.case_count),
        }


@dataclass
class SubstrateTrialRunner:
    scorer: ClassSpecificScorer
    corpus_registry: ReplayCorpusRegistry
    routing_replay_evaluator: RoutingReplayEvaluator = RoutingReplayEvaluator()
    graph_consolidation_replay_evaluator: GraphConsolidationReplayEvaluator = GraphConsolidationReplayEvaluator()

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
        if proposal.mutation_class == "graph_consolidation_param_patch":
            return self.graph_consolidation_replay_evaluator.derive_metrics(replay_records=replay_records)
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
