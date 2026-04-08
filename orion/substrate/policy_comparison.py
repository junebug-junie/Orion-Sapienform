from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from orion.core.schemas.substrate_policy_comparison import (
    SubstratePolicyComparisonRequestV1,
    SubstratePolicyEffectivenessReportV1,
    SubstratePolicyMetricDeltaV1,
)
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryQueryV1
from orion.substrate.policy_profiles import SubstratePolicyProfileStore
from orion.substrate.review_telemetry import GraphReviewTelemetryRecorder

PairMode = Literal["baseline_vs_active", "previous_vs_current", "selected_pair"]


@dataclass(frozen=True)
class ResolvedProfilePair:
    mode: PairMode
    baseline_profile_id: str
    candidate_profile_id: str
    baseline_label: str
    candidate_label: str
    notes: list[str]


@dataclass(frozen=True)
class SubstratePolicyComparisonAnalyzer:
    """Deterministic bounded comparison analyzer for operator policy effectiveness reads."""

    min_sample_size: int = 5

    def compare_metrics(
        self,
        *,
        request: SubstratePolicyComparisonRequestV1,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        baseline_total: int,
        candidate_total: int,
    ) -> SubstratePolicyEffectivenessReportV1:
        if baseline_total < self.min_sample_size or candidate_total < self.min_sample_size:
            return SubstratePolicyEffectivenessReportV1(
                request_id=request.request_id,
                candidate_profile_id=request.candidate_profile_id,
                baseline_profile_id=request.baseline_profile_id,
                baseline_window_label=request.baseline_window_label,
                candidate_window_label=request.candidate_window_label,
                verdict="insufficient_data",
                confidence=0.2,
                notes=[
                    f"baseline_total:{baseline_total}",
                    f"candidate_total:{candidate_total}",
                    "insufficient_sample",
                ],
            )

        weights: dict[str, float] = {
            "execution_rate": 0.15,
            "noop_rate": -0.1,
            "suppressed_rate": -0.7,
            "terminated_rate": -0.5,
            "failed_rate": -1.0,
            "avg_cycles_to_resolution": -0.3,
            "frontier_followup_rate": 0.2,
            "operator_only_rate": 0.2,
            "strict_zone_surface_rate": 0.2,
            "queue_revisit_rate": -0.3,
        }

        deltas: list[SubstratePolicyMetricDeltaV1] = []
        weighted_score = 0.0
        all_metrics = sorted(set(baseline_metrics).union(candidate_metrics))
        for metric in all_metrics:
            base = float(baseline_metrics.get(metric, 0.0))
            cand = float(candidate_metrics.get(metric, 0.0))
            delta = cand - base
            pct = (delta / base) if abs(base) > 1e-9 else (1.0 if delta > 0 else 0.0)
            deltas.append(
                SubstratePolicyMetricDeltaV1(
                    metric=metric,
                    baseline_value=base,
                    candidate_value=cand,
                    delta=delta,
                    pct_delta=pct,
                )
            )
            weighted_score += delta * float(weights.get(metric, 0.0))

        if weighted_score > 0.05:
            verdict = "improved"
            confidence = 0.75
        elif weighted_score < -0.05:
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
            notes=[
                f"weighted_score:{weighted_score:.3f}",
                f"baseline_total:{baseline_total}",
                f"candidate_total:{candidate_total}",
            ],
        )


@dataclass(frozen=True)
class SubstratePolicyComparisonService:
    policy_store: SubstratePolicyProfileStore
    telemetry_recorder: GraphReviewTelemetryRecorder
    analyzer: SubstratePolicyComparisonAnalyzer = SubstratePolicyComparisonAnalyzer()

    def compare(self, *, request: SubstratePolicyComparisonRequestV1) -> dict[str, object]:
        pair = self.resolve_pair(request=request)
        records = self.telemetry_recorder.query(
            GraphReviewTelemetryQueryV1(
                limit=request.sample_limit,
                since=datetime.now(timezone.utc) - timedelta(seconds=request.window_seconds),
                invocation_surface=request.invocation_surface,  # type: ignore[arg-type]
                target_zone=request.target_zone,  # type: ignore[arg-type]
            )
        )
        if request.operator_only is not None:
            records = [
                r for r in records if (r.invocation_surface == "operator_review") is bool(request.operator_only)
            ]
        baseline_records = [r for r in records if (r.policy_profile_id or "baseline") == pair.baseline_profile_id]
        candidate_records = [r for r in records if (r.policy_profile_id or "baseline") == pair.candidate_profile_id]

        baseline_metrics = self._aggregate_metrics(baseline_records)
        candidate_metrics = self._aggregate_metrics(candidate_records)

        report_req = request.model_copy(
            update={
                "baseline_profile_id": pair.baseline_profile_id,
                "candidate_profile_id": pair.candidate_profile_id,
                "baseline_window_label": pair.baseline_label,
                "candidate_window_label": pair.candidate_label,
            }
        )
        report = self.analyzer.compare_metrics(
            request=report_req,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_total=len(baseline_records),
            candidate_total=len(candidate_records),
        )

        return {
            "pair": {
                "mode": pair.mode,
                "baseline_profile_id": pair.baseline_profile_id,
                "candidate_profile_id": pair.candidate_profile_id,
                "baseline_label": pair.baseline_label,
                "candidate_label": pair.candidate_label,
                "notes": pair.notes,
            },
            "baseline": self._profile_meta(pair.baseline_profile_id),
            "candidate": self._profile_meta(pair.candidate_profile_id),
            "samples": {
                "baseline": len(baseline_records),
                "candidate": len(candidate_records),
                "window_seconds": request.window_seconds,
                "limit": request.sample_limit,
            },
            "report": report.model_dump(mode="json"),
            "advisory": {
                "mutating": False,
                "message": "comparison is advisory-only; no activation/rollback performed",
            },
        }

    def resolve_pair(self, *, request: SubstratePolicyComparisonRequestV1) -> ResolvedProfilePair:
        mode = request.pair_mode
        if mode == "selected_pair":
            if not request.baseline_profile_id or not request.candidate_profile_id:
                raise ValueError("selected_pair requires both baseline_profile_id and candidate_profile_id")
            self._ensure_profile_or_baseline(request.baseline_profile_id)
            self._ensure_profile_or_baseline(request.candidate_profile_id)
            return ResolvedProfilePair(
                mode=mode,
                baseline_profile_id=request.baseline_profile_id,
                candidate_profile_id=request.candidate_profile_id,
                baseline_label="selected_baseline",
                candidate_label="selected_candidate",
                notes=["explicit_profile_pair"],
            )

        inspection = self.policy_store.inspect(audit_limit=200)
        active = sorted(inspection.active_profiles, key=lambda p: p.activated_at or p.created_at, reverse=True)
        current = active[0] if active else None

        if mode == "baseline_vs_active":
            if current is None:
                raise ValueError("no_active_profile_available")
            return ResolvedProfilePair(
                mode=mode,
                baseline_profile_id="baseline",
                candidate_profile_id=current.profile_id,
                baseline_label="baseline_window",
                candidate_label="active_window",
                notes=["active_resolved_from_sql_state"],
            )

        if mode == "previous_vs_current":
            if current is None:
                raise ValueError("no_active_profile_available")
            if not current.previous_profile_id:
                raise ValueError("no_previous_profile_available")
            self._ensure_profile_or_baseline(current.previous_profile_id)
            return ResolvedProfilePair(
                mode=mode,
                baseline_profile_id=current.previous_profile_id,
                candidate_profile_id=current.profile_id,
                baseline_label="previous_window",
                candidate_label="current_window",
                notes=["previous_from_active_linkage"],
            )

        raise ValueError(f"unsupported_pair_mode:{mode}")

    def _ensure_profile_or_baseline(self, profile_id: str) -> None:
        if profile_id == "baseline":
            return
        if self.policy_store.get_profile(profile_id) is None:
            raise ValueError(f"profile_not_found:{profile_id}")

    def _profile_meta(self, profile_id: str) -> dict[str, object]:
        if profile_id == "baseline":
            return {
                "profile_id": "baseline",
                "activation_state": "baseline",
                "scope": {},
                "operator_id": None,
            }
        profile = self.policy_store.get_profile(profile_id)
        if profile is None:
            return {
                "profile_id": profile_id,
                "activation_state": "missing",
                "scope": {},
                "operator_id": None,
            }
        return {
            "profile_id": profile.profile_id,
            "activation_state": profile.activation_state,
            "scope": profile.rollout_scope.model_dump(mode="json"),
            "operator_id": profile.operator_id,
            "previous_profile_id": profile.previous_profile_id,
        }

    @staticmethod
    def _aggregate_metrics(records: list) -> dict[str, float]:
        total = max(1, len(records))
        outcomes = Counter(r.execution_outcome for r in records)
        followups = sum(1 for r in records if r.frontier_followup_invoked)
        operator_only = sum(1 for r in records if r.invocation_surface == "operator_review")
        strict_zone = sum(1 for r in records if r.target_zone == "self_relationship_graph")
        revisit = sum(1 for r in records if (r.cycle_count_before or 0) > 1)
        avg_cycles = sum((r.cycle_count_before or 0) for r in records) / total

        return {
            "execution_count": float(len(records)),
            "execution_rate": outcomes.get("executed", 0) / total,
            "noop_rate": outcomes.get("noop", 0) / total,
            "suppressed_rate": outcomes.get("suppressed", 0) / total,
            "terminated_rate": outcomes.get("terminated", 0) / total,
            "failed_rate": outcomes.get("failed", 0) / total,
            "avg_cycles_to_resolution": avg_cycles,
            "frontier_followup_rate": followups / total,
            "operator_only_rate": operator_only / total,
            "strict_zone_surface_rate": strict_zone / total,
            "queue_revisit_rate": revisit / total,
        }
