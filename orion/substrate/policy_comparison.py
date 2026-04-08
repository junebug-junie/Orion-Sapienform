from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_policy_comparison import (
    SubstratePolicyComparisonRequestV1,
    SubstratePolicyEffectivenessReportV1,
    SubstratePolicyMetricDeltaV1,
)
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetrySummaryV1


@dataclass(frozen=True)
class SubstratePolicyComparisonAnalyzer:
    """Deterministic bounded comparison analyzer for operator policy effectiveness reads."""

    min_sample_size: int = 5

    def compare(
        self,
        *,
        request: SubstratePolicyComparisonRequestV1,
        baseline: GraphReviewTelemetrySummaryV1,
        candidate: GraphReviewTelemetrySummaryV1,
    ) -> SubstratePolicyEffectivenessReportV1:
        baseline_total = sum(baseline.outcome_counts.values())
        candidate_total = sum(candidate.outcome_counts.values())
        if baseline_total < self.min_sample_size or candidate_total < self.min_sample_size:
            return SubstratePolicyEffectivenessReportV1(
                request_id=request.request_id,
                candidate_profile_id=request.candidate_profile_id,
                baseline_profile_id=request.baseline_profile_id,
                baseline_window_label=request.baseline_window_label,
                candidate_window_label=request.candidate_window_label,
                verdict="insufficient_data",
                confidence=0.2,
                notes=[f"baseline_total:{baseline_total}", f"candidate_total:{candidate_total}", "insufficient_sample"],
            )

        metrics = [
            ("avg_runtime_duration_ms", float(baseline.avg_runtime_duration_ms), float(candidate.avg_runtime_duration_ms), -1.0),
            ("total_failed", float(baseline.total_failed), float(candidate.total_failed), -1.0),
            ("total_suppressed", float(baseline.total_suppressed), float(candidate.total_suppressed), -0.5),
            ("total_executions", float(baseline.total_executions), float(candidate.total_executions), 0.5),
        ]

        deltas: list[SubstratePolicyMetricDeltaV1] = []
        weighted_score = 0.0
        for name, base, cand, weight in metrics:
            delta = cand - base
            pct = (delta / base) if abs(base) > 1e-9 else (1.0 if delta > 0 else 0.0)
            deltas.append(
                SubstratePolicyMetricDeltaV1(
                    metric=name,
                    baseline_value=base,
                    candidate_value=cand,
                    delta=delta,
                    pct_delta=pct,
                )
            )
            weighted_score += delta * weight

        if weighted_score > 0.5:
            verdict = "improved"
            confidence = 0.75
        elif weighted_score < -0.5:
            verdict = "degraded"
            confidence = 0.75
        else:
            verdict = "neutral"
            confidence = 0.6

        return SubstratePolicyEffectivenessReportV1(
            request_id=request.request_id,
            candidate_profile_id=request.candidate_profile_id,
            baseline_profile_id=request.baseline_profile_id,
            baseline_window_label=request.baseline_window_label,
            candidate_window_label=request.candidate_window_label,
            verdict=verdict,
            confidence=confidence,
            metric_deltas=deltas,
            notes=[f"weighted_score:{weighted_score:.3f}", f"baseline_total:{baseline_total}", f"candidate_total:{candidate_total}"],
        )
