from __future__ import annotations

import asyncio

from app import worker
from app.recall_v2 import build_recall_v2_plan, run_recall_v2_shadow
from orion.core.contracts.recall import RecallQueryV1


def test_recall_v2_plan_extracts_exact_entity_project_and_temporal_hints() -> None:
    plan = build_recall_v2_plan(
        RecallQueryV1(
            fragment="today check services/orion-hub for Athena CPU123 incidents",
            profile="reflect.v1",
        )
    )
    assert "services/orion-hub" in plan.project_anchors
    assert any("Athena" in item for item in plan.entity_anchors)
    assert "CPU123" in plan.exact_anchor_tokens
    assert plan.temporal_anchor == "today"
    assert plan.time_window_days == 1


def test_recall_v1_emits_pressure_events_when_selection_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(worker, "get_profile", lambda _name: {"profile": "reflect.v1", "max_per_source": 4, "max_total_items": 4, "render_budget_tokens": 128})

    async def _empty_anchor(**kwargs):
        return [], {}

    async def _empty_backends(*args, **kwargs):
        return [], {}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _empty_anchor)
    monkeypatch.setattr(worker, "_query_backends", _empty_backends)

    q = RecallQueryV1(fragment="find exact anchor COMMIT123 in memory", profile="reflect.v1")
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-pressure-miss", diagnostic=True))
    assert bundle.items == []
    pressure_events = list((decision.recall_debug or {}).get("pressure_events") or [])
    assert pressure_events
    categories = {str(item.get("pressure_category") or "") for item in pressure_events if isinstance(item, dict)}
    assert "recall_miss_or_dissatisfaction" in categories
    assert "missing_exact_anchor" in categories


def test_recall_v2_shadow_exposes_explainable_ranked_cards(monkeypatch) -> None:
    async def _sql_exact(**kwargs):
        return []

    async def _sql_recent(*args, **kwargs):
        return []

    monkeypatch.setattr("app.recall_v2.fetch_exact_fragments", _sql_exact)
    monkeypatch.setattr("app.recall_v2.fetch_recent_fragments", _sql_recent)
    monkeypatch.setattr("app.recall_v2.fetch_vector_exact_matches", lambda **kwargs: [])
    monkeypatch.setattr("app.recall_v2.fetch_rdf_chatturn_exact_matches", lambda **kwargs: [])
    monkeypatch.setattr("app.recall_v2.fetch_vector_fragments", lambda **kwargs: [])
    monkeypatch.setattr("app.recall_v2.fetch_rdf_fragments", lambda **kwargs: [])
    monkeypatch.setattr(
        "app.recall_v2._pageindex_candidates",
        lambda plan, top_k=8: [
            {
                "id": "page-1",
                "source": "pageindex_lexical",
                "source_ref": "pageindex",
                "text": "Athena routing threshold notes",
                "score": 0.8,
                "tags": ["pageindex"],
                "meta": {
                    "entry_id": "entry-1",
                    "heading": "Reflective heading",
                    "provenance": {
                        "reflective_themes": ["continuity"],
                        "active_tensions": ["speed_vs_depth"],
                    },
                },
            }
        ],
    )

    bundle, debug = asyncio.run(run_recall_v2_shadow(RecallQueryV1(fragment="Athena routing threshold", profile="reflect.v1")))
    assert bundle.items
    cards = list(debug.get("ranked_cards") or [])
    assert cards
    assert isinstance(cards[0].get("why_selected"), dict)
    assert cards[0].get("source") == "pageindex_lexical"
    assert "pageindex" in list(cards[0].get("tags") or [])
