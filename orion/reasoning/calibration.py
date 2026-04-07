from __future__ import annotations

from orion.core.schemas.endogenous_eval import (
    EndogenousCalibrationProfileV1,
    EndogenousCalibrationRecommendationV1,
    EndogenousEvaluationResultV1,
)


class EndogenousCalibrationEngine:
    """Deterministic advisory-only calibration recommendation engine."""

    def recommend(self, evaluation: EndogenousEvaluationResultV1) -> EndogenousEvaluationResultV1:
        metrics = evaluation.metrics
        recommendations: list[EndogenousCalibrationRecommendationV1] = []
        warnings = list(evaluation.warnings)

        if metrics.sample_size < evaluation.request.min_sample_size:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="trigger_threshold",
                    parameter="global_threshold_posture",
                    current_value="current",
                    recommended_value="hold",
                    direction="hold",
                    confidence="low",
                    rationale="insufficient_sample_size",
                )
            )
            profile = EndogenousCalibrationProfileV1(
                profile_name="endogenous-offline-calibration-v1",
                generated_from_request_id=evaluation.request.request_id,
                overrides={},
                advisory_only=True,
            )
            out = evaluation.model_copy(update={"recommendations": recommendations, "warnings": warnings, "generated_profile": profile})
            return out

        if metrics.trigger_rate > 0.75:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="trigger_threshold",
                    parameter="reflective_threshold",
                    current_value="0.45",
                    recommended_value="0.5",
                    direction="increase",
                    confidence="medium",
                    rationale="high_trigger_rate_overfiring",
                )
            )
        elif metrics.trigger_rate < 0.1 and metrics.noop_rate + metrics.suppress_rate > 0.85:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="trigger_threshold",
                    parameter="reflective_threshold",
                    current_value="0.45",
                    recommended_value="0.4",
                    direction="decrease",
                    confidence="medium",
                    rationale="high_noop_suppress_underfiring",
                )
            )

        if metrics.cooldown_hit_rate > 0.35:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="workflow_cooldown",
                    parameter="dominant_workflow_cooldown_seconds",
                    current_value="policy_default",
                    recommended_value="policy_default+60",
                    direction="increase",
                    confidence="medium",
                    rationale="high_cooldown_hit_rate",
                )
            )

        if metrics.mentor_disabled_suppression_rate > 0.15:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="mentor_gating",
                    parameter="allow_mentor_branch_operator_only",
                    current_value="false",
                    recommended_value="true_for_operator_surface_only",
                    direction="increase",
                    confidence="low",
                    rationale="mentor_disabled_suppression_present",
                )
            )

        if evaluation.reasoning_summary.recommendation == "loosen_summary_inclusion":
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="summary_inclusion_threshold",
                    parameter="summary_confidence_floor",
                    current_value="0.55",
                    recommended_value="0.5",
                    direction="decrease",
                    confidence="medium",
                    rationale="high_fallback_proxy_rate",
                )
            )
        elif evaluation.reasoning_summary.recommendation == "tighten_summary_inclusion":
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="summary_inclusion_threshold",
                    parameter="summary_confidence_floor",
                    current_value="0.55",
                    recommended_value="0.6",
                    direction="increase",
                    confidence="low",
                    rationale="very_low_fallback_proxy_rate",
                )
            )

        if evaluation.promotion.recommendation == "review_promotion_evidence_threshold":
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="promotion_threshold",
                    parameter="promotion_evidence_floor",
                    current_value="current",
                    recommended_value="current-0.05",
                    direction="decrease",
                    confidence="low",
                    rationale="high_block_like_with_low_materialization",
                )
            )

        if not recommendations:
            recommendations.append(
                EndogenousCalibrationRecommendationV1(
                    target="trigger_threshold",
                    parameter="global_threshold_posture",
                    current_value="current",
                    recommended_value="hold",
                    direction="hold",
                    confidence="medium",
                    rationale="no_strong_adjustment_signal",
                )
            )

        profile = EndogenousCalibrationProfileV1(
            profile_name="endogenous-offline-calibration-v1",
            generated_from_request_id=evaluation.request.request_id,
            overrides={rec.parameter: rec.recommended_value for rec in recommendations if rec.direction != "hold"},
            advisory_only=True,
        )

        return evaluation.model_copy(update={"recommendations": recommendations, "generated_profile": profile, "warnings": warnings})


def render_evaluation_report(evaluation: EndogenousEvaluationResultV1) -> str:
    """Compact operator-readable markdown report for offline calibration review."""

    lines = [
        "# Endogenous Offline Evaluation",
        f"request_id: {evaluation.request.request_id}",
        f"sample_size: {evaluation.metrics.sample_size}",
        f"trigger_rate: {evaluation.metrics.trigger_rate}",
        f"noop_rate: {evaluation.metrics.noop_rate}",
        f"suppress_rate: {evaluation.metrics.suppress_rate}",
        f"failure_rate: {evaluation.metrics.failure_rate}",
        f"cooldown_hit_rate: {evaluation.metrics.cooldown_hit_rate}",
        f"mentor_selected_rate: {evaluation.metrics.mentor_selected_rate}",
        f"mentor_invoked_rate: {evaluation.metrics.mentor_invoked_rate}",
        f"summary_fallback_proxy_rate: {evaluation.reasoning_summary.fallback_proxy_rate}",
        f"promotion_block_like_rate: {evaluation.promotion.blocked_like_rate}",
        "## Recommendations",
    ]
    for rec in evaluation.recommendations:
        lines.append(f"- [{rec.confidence}] {rec.target}.{rec.parameter}: {rec.current_value} -> {rec.recommended_value} ({rec.rationale})")
    if evaluation.warnings:
        lines.append("## Warnings")
        for warning in evaluation.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)
