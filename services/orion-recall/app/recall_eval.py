from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from orion.core.contracts.recall import RecallQueryV1

from orion.substrate.recall_eval_bridge import eval_row_to_v1_v2_compare

from .recall_v2 import run_recall_v2_shadow
from .worker import process_recall


def load_eval_corpus() -> List[Dict[str, Any]]:
    path = Path(__file__).resolve().parent / "recall_eval_corpus.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def _coverage_score(snippets: List[str], expected_terms: List[str]) -> float:
    if not expected_terms:
        return 1.0
    joined = " ".join(snippets).lower()
    matched = sum(1 for term in expected_terms if term.lower() in joined)
    return matched / float(len(expected_terms))


def _cousin_rate(snippets: List[str], forbidden_terms: List[str]) -> float:
    if not snippets:
        return 0.0
    bad = 0
    for snippet in snippets:
        lowered = snippet.lower()
        if any(term.lower() in lowered for term in forbidden_terms):
            bad += 1
    return bad / float(len(snippets))


def _entity_time_match(cards: List[Dict[str, Any]]) -> float:
    if not cards:
        return 0.0
    matched = 0
    for card in cards:
        why = card.get("why_selected") if isinstance(card.get("why_selected"), dict) else {}
        if bool(why.get("entity_match")) or bool(why.get("project_match")) or bool(why.get("exact_anchor")):
            matched += 1
    return matched / float(len(cards))


def _explainability_completeness(cards: List[Dict[str, Any]]) -> float:
    if not cards:
        return 0.0
    complete = 0
    for card in cards:
        why = card.get("why_selected")
        if isinstance(why, dict) and why:
            complete += 1
    return complete / float(len(cards))


async def run_recall_eval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    query_text = str(case.get("query") or "")
    expected_terms = [str(item) for item in list(case.get("expected_terms") or [])]
    forbidden_terms = [str(item) for item in list(case.get("forbidden_terms") or [])]
    query = RecallQueryV1(fragment=query_text, profile="reflect.v1")

    bundle_v1, decision_v1 = await process_recall(query, corr_id=f"recall-eval:{case.get('case_id')}", diagnostic=True)
    bundle_v2, debug_v2 = await run_recall_v2_shadow(query)

    v1_snippets = [item.snippet for item in bundle_v1.items]
    v2_cards = list(debug_v2.get("ranked_cards") or [])
    v2_snippets = [str(item.get("snippet") or "") for item in v2_cards]

    v1_precision_proxy = _coverage_score(v1_snippets, expected_terms)
    v2_precision_proxy = _coverage_score(v2_snippets, expected_terms)
    v1_cousin_rate = _cousin_rate(v1_snippets, forbidden_terms)
    v2_cousin_rate = _cousin_rate(v2_snippets, forbidden_terms)
    v2_entity_time = _entity_time_match(v2_cards)
    v2_explainability = _explainability_completeness(v2_cards)

    return {
        "case_id": case.get("case_id"),
        "query": query_text,
        "type": case.get("type"),
        "v1": {
            "selected_count": len(bundle_v1.items),
            "precision_proxy": v1_precision_proxy,
            "answer_support_coverage": v1_precision_proxy,
            "irrelevant_cousin_rate": v1_cousin_rate,
            "entity_time_match_rate": 0.0,
            "latency_ms": decision_v1.latency_ms,
            "explainability_completeness": 0.0,
        },
        "v2": {
            "selected_count": len(bundle_v2.items),
            "precision_proxy": v2_precision_proxy,
            "answer_support_coverage": v2_precision_proxy,
            "irrelevant_cousin_rate": v2_cousin_rate,
            "entity_time_match_rate": v2_entity_time,
            "latency_ms": int(debug_v2.get("latency_ms") or 0),
            "explainability_completeness": v2_explainability,
        },
    }


async def run_recall_eval_suite() -> Dict[str, Any]:
    corpus = load_eval_corpus()
    rows: List[Dict[str, Any]] = []
    for case in corpus:
        rows.append(await run_recall_eval_case(case))
    return {
        "cases_total": len(rows),
        "rows": rows,
    }
