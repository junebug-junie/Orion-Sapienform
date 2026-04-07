from __future__ import annotations

from collections import Counter

from orion.core.schemas.endogenous_eval import (
    EndogenousEvaluationRequestV1,
    EndogenousEvaluationResultV1,
    EndogenousMetricSummaryV1,
    PromotionCalibrationSummaryV1,
    ReasoningSummaryCalibrationSummaryV1,
)
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeExecutionRecordV1


class EndogenousOfflineEvaluator:
    """Deterministic offline metric evaluator over durable endogenous runtime records."""

    def evaluate(
        self,
        request: EndogenousEvaluationRequestV1,
        records: list[EndogenousRuntimeExecutionRecordV1],
    ) -> EndogenousEvaluationResultV1:
        filtered = self._filter_records(request, records)
        metrics = self._metrics(filtered)
        promotion = self._promotion_summary(filtered)
        summary = self._summary_calibration(filtered)

        warnings: list[str] = []
        if metrics.sample_size < request.min_sample_size:
            warnings.append(f"insufficient_sample_size:{metrics.sample_size}<{request.min_sample_size}")
        if not filtered:
            warnings.append("no_records_after_filter")

        return EndogenousEvaluationResultV1(
            request=request,
            metrics=metrics,
            promotion=promotion,
            reasoning_summary=summary,
            warnings=warnings,
        )

    def _filter_records(
        self,
        request: EndogenousEvaluationRequestV1,
        records: list[EndogenousRuntimeExecutionRecordV1],
    ) -> list[EndogenousRuntimeExecutionRecordV1]:
        out: list[EndogenousRuntimeExecutionRecordV1] = []
        for record in records:
            if request.invocation_surfaces and record.invocation_surface not in request.invocation_surfaces:
                continue
            if request.workflow_types and record.decision.workflow_type not in request.workflow_types:
                continue
            if request.outcomes and record.decision.outcome not in request.outcomes:
                continue
            if request.subject_ref and record.subject_ref != request.subject_ref:
                continue
            if request.mentor_invoked is not None and record.mentor_invoked is not request.mentor_invoked:
                continue
            if request.created_after and record.created_at < request.created_after:
                continue
            out.append(record)
            if len(out) >= request.limit:
                break
        return out

    def _metrics(self, records: list[EndogenousRuntimeExecutionRecordV1]) -> EndogenousMetricSummaryV1:
        total = len(records)
        by_surface = Counter(record.invocation_surface for record in records)
        by_workflow = Counter(record.decision.workflow_type for record in records)
        by_outcome = Counter(record.decision.outcome for record in records)

        if total == 0:
            return EndogenousMetricSummaryV1(sample_size=0)

        triggers = by_outcome.get("trigger", 0)
        noops = by_outcome.get("noop", 0)
        suppress = by_outcome.get("suppress", 0)
        failures = sum(1 for record in records if not record.execution_success)
        cooldown = sum(1 for record in records if record.decision.cooldown_applied)
        coalesce = by_outcome.get("coalesce", 0)
        debounce = sum(1 for record in records if record.decision.debounce_applied)
        mentor_selected = sum(1 for record in records if record.decision.workflow_type == "mentor_critique")
        mentor_invoked = sum(1 for record in records if record.mentor_invoked)
        mentor_disabled = sum(1 for record in records if "mentor_runtime_disabled" in record.decision.reasons)
        materialized_total = sum(len(record.materialized_artifact_ids) for record in records)

        subject_counts = Counter(record.subject_ref or "-" for record in records)
        repeated_density = max(subject_counts.values()) / total if subject_counts else 0.0

        return EndogenousMetricSummaryV1(
            sample_size=total,
            by_surface=dict(sorted(by_surface.items())),
            by_workflow=dict(sorted(by_workflow.items())),
            by_outcome=dict(sorted(by_outcome.items())),
            trigger_rate=round(triggers / total, 4),
            noop_rate=round(noops / total, 4),
            suppress_rate=round(suppress / total, 4),
            failure_rate=round(failures / total, 4),
            cooldown_hit_rate=round(cooldown / total, 4),
            coalesce_rate=round(coalesce / total, 4),
            debounce_rate=round(debounce / total, 4),
            mentor_selected_rate=round(mentor_selected / total, 4),
            mentor_invoked_rate=round(mentor_invoked / total, 4),
            mentor_disabled_suppression_rate=round(mentor_disabled / total, 4),
            contradiction_review_rate=round(by_workflow.get("contradiction_review", 0) / total, 4),
            concept_refinement_rate=round(by_workflow.get("concept_refinement", 0) / total, 4),
            autonomy_review_rate=round(by_workflow.get("autonomy_review", 0) / total, 4),
            reflective_journal_rate=round(by_workflow.get("reflective_journal", 0) / total, 4),
            materialized_artifact_avg=round(materialized_total / total, 4),
            repeated_subject_density=round(repeated_density, 4),
        )

    def _promotion_summary(self, records: list[EndogenousRuntimeExecutionRecordV1]) -> PromotionCalibrationSummaryV1:
        total = len(records)
        if total == 0:
            return PromotionCalibrationSummaryV1(sample_size=0)

        blocked_like = sum(1 for record in records if any("runtime_workflow_not_allowed" in reason for reason in record.decision.reasons))
        materialized = sum(1 for record in records if len(record.materialized_artifact_ids) > 0)
        blocked_rate = blocked_like / total
        materialized_rate = materialized / total

        if total < 30:
            return PromotionCalibrationSummaryV1(
                sample_size=total,
                blocked_like_rate=round(blocked_rate, 4),
                materialization_success_rate=round(materialized_rate, 4),
                recommendation="hold",
                rationale="insufficient_data",
            )

        recommendation = "hold"
        rationale = "steady_state"
        if blocked_rate > 0.4 and materialized_rate < 0.05:
            recommendation = "review_promotion_evidence_threshold"
            rationale = "high_block_like_with_low_materialization"

        return PromotionCalibrationSummaryV1(
            sample_size=total,
            blocked_like_rate=round(blocked_rate, 4),
            materialization_success_rate=round(materialized_rate, 4),
            recommendation=recommendation,
            rationale=rationale,
        )

    def _summary_calibration(self, records: list[EndogenousRuntimeExecutionRecordV1]) -> ReasoningSummaryCalibrationSummaryV1:
        total = len(records)
        if total == 0:
            return ReasoningSummaryCalibrationSummaryV1(sample_size=0)

        fallback_proxy = sum(1 for record in records if (record.signal_digest.mentor_gap_count or 0) > 0)
        fallback_rate = fallback_proxy / total
        if total < 30:
            return ReasoningSummaryCalibrationSummaryV1(
                sample_size=total,
                fallback_proxy_rate=round(fallback_rate, 4),
                recommendation="hold",
                rationale="insufficient_data",
            )

        recommendation = "hold"
        rationale = "steady_state"
        if fallback_rate > 0.45:
            recommendation = "loosen_summary_inclusion"
            rationale = "high_fallback_proxy_rate"
        elif fallback_rate < 0.05:
            recommendation = "tighten_summary_inclusion"
            rationale = "very_low_fallback_proxy_rate"

        return ReasoningSummaryCalibrationSummaryV1(
            sample_size=total,
            fallback_proxy_rate=round(fallback_rate, 4),
            recommendation=recommendation,
            rationale=rationale,
        )
