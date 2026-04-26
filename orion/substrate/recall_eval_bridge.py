"""Helpers to map recall_eval suite rows into mutation pressure metadata (proposal-only path)."""

from __future__ import annotations

from typing import Any, Dict, List

from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1

_RECALL_PRESSURE_CATEGORIES = frozenset(
    {
        "recall_miss_or_dissatisfaction",
        "unsupported_memory_claim",
        "irrelevant_semantic_neighbor",
        "missing_exact_anchor",
        "stale_memory_selected",
    }
)


def eval_row_to_v1_v2_compare(case_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact V1 vs V2 summary for MutationPressureEvidenceV1.metadata["v1_v2_compare"] when
    ingesting recall_eval output into review telemetry.
    """
    v1 = case_result.get("v1") if isinstance(case_result.get("v1"), dict) else {}
    v2 = case_result.get("v2") if isinstance(case_result.get("v2"), dict) else {}
    v1c = int(v1.get("selected_count") or 0)
    v2c = int(v2.get("selected_count") or 0)
    return {
        "source": "recall_eval_suite",
        "case_id": case_result.get("case_id"),
        "v1_latency_ms": v1.get("latency_ms"),
        "v2_latency_ms": v2.get("latency_ms"),
        "v1_selected_count": v1c,
        "v2_selected_count": v2c,
        "selected_count_delta": v2c - v1c,
        "v1_precision_proxy": v1.get("precision_proxy"),
        "v2_precision_proxy": v2.get("precision_proxy"),
        "v1_answer_support_coverage": v1.get("answer_support_coverage", v1.get("precision_proxy")),
        "v2_answer_support_coverage": v2.get("answer_support_coverage", v2.get("precision_proxy")),
        "v1_irrelevant_cousin_rate": v1.get("irrelevant_cousin_rate"),
        "v2_irrelevant_cousin_rate": v2.get("irrelevant_cousin_rate"),
        "v2_entity_time_match_rate": v2.get("entity_time_match_rate"),
        "v2_explainability_completeness": v2.get("explainability_completeness"),
    }


def infer_pressure_category_for_eval_row(row: Dict[str, Any]) -> str:
    explicit = str(row.get("pressure_category") or "").strip()
    if explicit in _RECALL_PRESSURE_CATEGORIES:
        return explicit
    v1 = row.get("v1") if isinstance(row.get("v1"), dict) else {}
    v2 = row.get("v2") if isinstance(row.get("v2"), dict) else {}
    v1p = float(v1.get("precision_proxy") or 0.0)
    v2p = float(v2.get("precision_proxy") or 0.0)
    v1c = float(v1.get("irrelevant_cousin_rate") or 0.0)
    v2c = float(v2.get("irrelevant_cousin_rate") or 0.0)
    if v2c > v1c + 0.15:
        return "irrelevant_semantic_neighbor"
    if v2p > v1p + 0.08:
        return "recall_miss_or_dissatisfaction"
    return "unsupported_memory_claim"


def pressure_evidence_from_eval_suite_rows(
    rows: List[Dict[str, Any]],
    *,
    suite_run_id: str | None = None,
) -> List[MutationPressureEvidenceV1]:
    """Build first-class pressure evidence rows from recall_eval-style dicts (manual ingest only)."""
    rid = str(suite_run_id or "").strip() or "recall-eval-manual"
    out: list[MutationPressureEvidenceV1] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        compare = eval_row_to_v1_v2_compare(row)
        cat = infer_pressure_category_for_eval_row(row)
        cid = str(row.get("case_id") if row.get("case_id") is not None else idx)
        meta: Dict[str, Any] = {
            "v1_v2_compare": compare,
            "recall_eval_case": {
                "case_id": row.get("case_id"),
                "query": row.get("query"),
                "type": row.get("type"),
            },
            "recall_evidence_kind": "eval_suite",
            "suite_run_id": rid,
        }
        out.append(
            MutationPressureEvidenceV1(
                source_service="orion-recall-eval",
                source_event_id=f"{rid}:{cid}",
                correlation_id=str(row.get("correlation_id") or rid),
                pressure_category=cat,  # type: ignore[arg-type]
                confidence=float(row.get("confidence") or 0.75),
                evidence_refs=[f"recall_eval_case:{cid}", f"recall_eval_suite:{rid}"],
                metadata=meta,
            )
        )
    return out[:32]
