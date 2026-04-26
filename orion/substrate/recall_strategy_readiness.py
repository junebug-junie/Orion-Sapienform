"""Deterministic recall strategy / Recall V2 shadow promotion readiness (advisory)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from orion.core.schemas.recall_strategy_readiness import (
    RecallStrategyReadinessRecommendationV1,
    RecallStrategyReadinessV1,
)


def default_eval_corpus_total_cases() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    corpus_path = repo_root / "services" / "orion-recall" / "app" / "recall_eval_corpus.json"
    try:
        if corpus_path.is_file():
            data = json.loads(corpus_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return max(1, len(data))
    except Exception:
        pass
    return 12


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _collect_compare_rows_from_pressure(pressure: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Pull v1_v2_compare dicts from snapshot + bounded evidence history."""
    compares: list[dict[str, Any]] = []
    categories: list[str] = []
    snap = getattr(pressure, "recall_evidence_snapshot", None) or {}
    if isinstance(snap, dict):
        rc = snap.get("recall_compare")
        if isinstance(rc, dict) and (rc.get("v1_latency_ms") is not None or rc.get("v2_latency_ms") is not None):
            compares.append(dict(rc))
            fc = str(snap.get("failure_category") or "").strip()
            categories.append(fc or "unknown")
    for entry in list(getattr(pressure, "recall_evidence_history", None) or []):
        if not isinstance(entry, dict):
            continue
        rc = entry.get("recall_compare")
        if isinstance(rc, dict) and (rc.get("v1_latency_ms") is not None or rc.get("v2_latency_ms") is not None):
            compares.append(dict(rc))
            categories.append(str(entry.get("failure_category") or "").strip() or "unknown")
    return compares, categories


def compare_rows_from_telemetry_records(records: Iterable[Any]) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    compares: list[dict[str, Any]] = []
    categories: list[str] = []
    eval_case_ids: set[str] = set()
    for row in records:
        events = getattr(row, "pressure_events", None) or []
        for ev in events:
            meta = ev.metadata if isinstance(ev.metadata, dict) else {}
            cmp = meta.get("v1_v2_compare")
            if not isinstance(cmp, dict):
                continue
            if cmp.get("v1_latency_ms") is None and cmp.get("v2_latency_ms") is None:
                continue
            compares.append(dict(cmp))
            categories.append(str(getattr(ev, "pressure_category", "") or "unknown"))
            if str(cmp.get("source") or "") == "recall_eval_suite":
                cid = cmp.get("case_id")
                if cid is not None:
                    eval_case_ids.add(str(cid))
                rec_case = meta.get("recall_eval_case")
                if isinstance(rec_case, dict) and rec_case.get("case_id") is not None:
                    eval_case_ids.add(str(rec_case["case_id"]))
    return compares, categories, eval_case_ids


def compute_recall_strategy_readiness(
    *,
    compare_rows: list[dict[str, Any]],
    failure_categories: list[str],
    corpus_total_cases: int,
    minimum_evidence_cases_required: int = 3,
    max_irrelevant_cousin_rate: float = 0.35,
    max_latency_regression_ms: float = 200.0,
    min_precision_delta_for_promotion: float = 0.08,
    corpus_coverage_promotion_threshold: float = 0.75,
    corpus_coverage_shadow_threshold: float = 0.5,
) -> RecallStrategyReadinessV1:
    """
    Deterministic readiness from aggregated V1/V2 compare dicts (telemetry and/or pressure history).
    Does not mutate production recall or substrate apply surfaces.
    """
    notes: list[str] = []
    gates: list[str] = []
    n = len(compare_rows)
    corpus_total = max(1, int(corpus_total_cases))

    if not compare_rows:
        return RecallStrategyReadinessV1(
            corpus_coverage=0.0,
            precision_proxy=0.0,
            answer_support_coverage=0.0,
            irrelevant_cousin_rate=0.0,
            entity_time_match_rate=0.0,
            latency_delta_ms_mean=0.0,
            explainability_completeness=0.0,
            failure_categories_improved={},
            failure_categories_regressed={},
            minimum_evidence_cases_required=minimum_evidence_cases_required,
            evidence_observation_count=0,
            minimum_evidence_met=False,
            recommendation="not_ready",
            readiness_notes=["no_compare_observations"],
            gates_blocked=["insufficient_evidence"],
        )

    eval_case_ids = {
        str(r.get("case_id"))
        for r in compare_rows
        if str(r.get("source") or "") == "recall_eval_suite" and r.get("case_id") is not None
    }
    corpus_coverage = min(1.0, len(eval_case_ids) / float(corpus_total)) if eval_case_ids else min(1.0, n / float(corpus_total))

    v2_prec = [_f(r.get("v2_precision_proxy")) for r in compare_rows]
    v1_prec = [_f(r.get("v1_precision_proxy")) for r in compare_rows]
    precision_proxy = sum(v2_prec) / float(n) if n else 0.0

    v2_ans = [_f(r.get("v2_answer_support_coverage", r.get("v2_precision_proxy"))) for r in compare_rows]
    answer_support_coverage = sum(v2_ans) / float(n) if n else 0.0

    v2_cousin = [_f(r.get("v2_irrelevant_cousin_rate")) for r in compare_rows]
    irrelevant_cousin_rate = sum(v2_cousin) / float(n) if n and any(x > 0 for x in v2_cousin) else 0.0

    v2_ent = [_f(r.get("v2_entity_time_match_rate")) for r in compare_rows]
    entity_time_match_rate = sum(v2_ent) / float(n) if n and any(x > 0 for x in v2_ent) else 0.0

    lat_deltas: list[float] = []
    for r in compare_rows:
        v1l = r.get("v1_latency_ms")
        v2l = r.get("v2_latency_ms")
        if v1l is not None and v2l is not None:
            lat_deltas.append(_f(v2l) - _f(v1l))
    latency_delta_ms_mean = sum(lat_deltas) / float(len(lat_deltas)) if lat_deltas else 0.0

    v2_expl = [_f(r.get("v2_explainability_completeness")) for r in compare_rows]
    explainability_completeness = sum(v2_expl) / float(n) if n and any(x > 0 for x in v2_expl) else 0.0

    improved: dict[str, int] = {}
    regressed: dict[str, int] = {}
    cats = failure_categories + ["unknown"] * max(0, n - len(failure_categories))
    for idx, r in enumerate(compare_rows):
        cat = str(cats[idx] if idx < len(cats) else "unknown")
        p1 = _f(r.get("v1_precision_proxy"))
        p2 = _f(r.get("v2_precision_proxy"))
        if p2 > p1 + 0.02:
            improved[cat] = improved.get(cat, 0) + 1
        elif p2 < p1 - 0.02:
            regressed[cat] = regressed.get(cat, 0) + 1

    evidence_met = n >= minimum_evidence_cases_required
    if not evidence_met:
        gates.append("insufficient_evidence")
        notes.append(f"observations={n}_required={minimum_evidence_cases_required}")

    if irrelevant_cousin_rate > max_irrelevant_cousin_rate:
        gates.append("high_irrelevant_cousin_rate")
        notes.append(f"mean_v2_cousin_rate={irrelevant_cousin_rate:.3f}")

    if lat_deltas and latency_delta_ms_mean > max_latency_regression_ms:
        gates.append("latency_regression")
        notes.append(f"mean_latency_delta_ms={latency_delta_ms_mean:.1f}")

    mean_delta = sum(v2_prec[i] - v1_prec[i] for i in range(n)) / float(n) if n else 0.0
    if mean_delta < -0.05:
        gates.append("precision_regression")
        notes.append(f"mean_precision_delta={mean_delta:.3f}")

    recommendation: RecallStrategyReadinessRecommendationV1 = "not_ready"
    if gates:
        if any(g in ("insufficient_evidence", "high_irrelevant_cousin_rate", "latency_regression") for g in gates):
            recommendation = "not_ready"
        elif "precision_regression" in gates and not any(
            g in ("insufficient_evidence", "high_irrelevant_cousin_rate", "latency_regression") for g in gates
        ):
            recommendation = "review_candidate"
    elif evidence_met and corpus_coverage >= corpus_coverage_promotion_threshold:
        if mean_delta >= min_precision_delta_for_promotion and irrelevant_cousin_rate <= max_irrelevant_cousin_rate * 0.72:
            if latency_delta_ms_mean <= max_latency_regression_ms * 0.5:
                recommendation = "ready_for_operator_promotion"
                notes.append("promotion_gate_passed_advisory_only")
            else:
                recommendation = "review_candidate"
        else:
            recommendation = "review_candidate"
    elif evidence_met and corpus_coverage < corpus_coverage_shadow_threshold:
        recommendation = "ready_for_shadow_expansion"
        notes.append("collect_more_eval_or_shadow_coverage")
    elif evidence_met:
        recommendation = "review_candidate"

    return RecallStrategyReadinessV1(
        corpus_coverage=round(corpus_coverage, 4),
        precision_proxy=round(precision_proxy, 4),
        answer_support_coverage=round(answer_support_coverage, 4),
        irrelevant_cousin_rate=round(irrelevant_cousin_rate, 4),
        entity_time_match_rate=round(entity_time_match_rate, 4),
        latency_delta_ms_mean=round(latency_delta_ms_mean, 2),
        explainability_completeness=round(explainability_completeness, 4),
        failure_categories_improved=improved,
        failure_categories_regressed=regressed,
        minimum_evidence_cases_required=minimum_evidence_cases_required,
        evidence_observation_count=n,
        minimum_evidence_met=evidence_met,
        recommendation=recommendation,
        readiness_notes=notes[:24],
        gates_blocked=gates[:16],
    )


def readiness_for_pressure(pressure: Any, *, corpus_total_cases: int | None = None) -> RecallStrategyReadinessV1:
    compares, cats = _collect_compare_rows_from_pressure(pressure)
    total = corpus_total_cases if corpus_total_cases is not None else default_eval_corpus_total_cases()
    return compute_recall_strategy_readiness(
        compare_rows=compares,
        failure_categories=cats,
        corpus_total_cases=total,
    )


def readiness_from_telemetry_records(records: Iterable[Any], *, corpus_total_cases: int | None = None) -> RecallStrategyReadinessV1:
    compares, cats, eval_ids = compare_rows_from_telemetry_records(records)
    total = corpus_total_cases if corpus_total_cases is not None else default_eval_corpus_total_cases()
    base = compute_recall_strategy_readiness(
        compare_rows=compares,
        failure_categories=cats,
        corpus_total_cases=total,
    )
    if not eval_ids:
        return base
    coverage = min(1.0, len(eval_ids) / float(max(1, total)))
    return base.model_copy(
        update={
            "corpus_coverage": round(coverage, 4),
            "readiness_notes": list(base.readiness_notes) + [f"eval_suite_cases_observed={len(eval_ids)}"],
        }
    )
