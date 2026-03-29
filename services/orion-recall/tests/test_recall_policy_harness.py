from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.fusion import fuse_candidates
from orion.core.contracts.recall import RecallQueryV1


def _profile() -> dict:
    return {
        "profile": "reflect.v1",
        "max_per_source": 8,
        "max_total_items": 10,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "enable_query_expansion": False,
        "enable_sql_timeline": False,
        "vector_top_k": 8,
        "rdf_top_k": 0,
        "sql_top_k": 0,
        "relevance": {"backend_weights": {"vector": 1.0}},
    }


def test_fusion_diagnostic_stats_capture_dedupe_and_drop_reasons():
    cands = [
        {"id": "vec-a", "source": "vector", "text": "User: hi\nOrion: hello", "score": 0.9, "tags": ["vector-assoc"]},
        {"id": "vec-b", "source": "vector", "text": "User: hi\nOrion: hello", "score": 0.8, "tags": ["vector-assoc"]},
        {"id": "vec-c", "source": "vector", "text": "User: hi there\nOrion: hello", "score": 0.7, "tags": ["vector-assoc"]},
    ]

    bundle, ranking_debug = fuse_candidates(candidates=cands, profile=_profile(), query_text="hi", diagnostic=True)

    assert len(ranking_debug) >= 1
    diag = bundle.stats.diagnostic or {}
    assert diag.get("transcript_dedupe_collapsed", 0) >= 1
    assert isinstance(diag.get("drop_counts"), dict)
    assert isinstance(diag.get("source_candidate_counts"), dict)
    assert isinstance(diag.get("source_selected_counts"), dict)


def test_process_recall_diagnostic_contains_gating_suppression_and_selection(monkeypatch):
    monkeypatch.setattr(worker, "get_profile", lambda _name: _profile())
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", True)

    async def _anchor(**kwargs):
        return [], {}

    async def _query(fragment, profile, *, session_id, node_id, entities, diagnostic=False, exclusion=None):
        return [
            {
                "id": "corr-diagnostic",
                "source": "vector",
                "source_ref": "orion_chat",
                "text": "Teddy loves Addy",
                "ts": 1000.0,
                "score": 0.9,
                "tags": [],
            },
            {
                "id": "stable-memory-2",
                "source": "vector",
                "source_ref": "orion_chat",
                "text": "Older durable memory",
                "ts": 900.0,
                "score": 0.7,
                "tags": [],
            },
        ], {"vector": 2}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", _query)

    q = RecallQueryV1(
        fragment="Teddy loves Addy",
        profile="reflect.v1",
        exclude={"active_turn_text": "Teddy loves Addy", "active_turn_ids": ["corr-diagnostic"], "active_turn_ts": 1000.0},
    )

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-diagnostic", diagnostic=True))
    assert any(item.id == "stable-memory-2" for item in bundle.items)
    assert "corr-diagnostic" not in decision.selected_ids
    assert decision.recall_debug.get("source_gating", {}).get("vector") == "enabled"
    assert decision.recall_debug.get("source_gating", {}).get("sql_timeline") == "disabled_by_profile_or_global"
    assert "self_hit_suppressed" in (decision.recall_debug.get("active_turn", {}) or {})
    assert isinstance(decision.recall_debug.get("selected_summary"), list)
