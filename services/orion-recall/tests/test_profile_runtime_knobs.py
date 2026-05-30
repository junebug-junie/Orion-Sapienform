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

from app.fusion import fuse_candidates
from app import worker
from app.profiles import get_profile, load_profiles


def test_assist_light_profile_vector_top_k_zero() -> None:
    load_profiles.cache_clear()
    p = get_profile("assist.light.v1")
    assert int(p.get("vector_top_k", -1)) == 0


def _profile(**overrides):
    base = {
        "profile": "reflect.v1",
        "max_per_source": 10,
        "max_total_items": 10,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "relevance": {
            "backend_weights": {"vector": 1.0},
            "score_weight": 0.7,
            "text_similarity_weight": 0.15,
            "recency_weight": 0.1,
            "enable_recency": True,
        },
    }
    base.update(overrides)
    return base


def test_fusion_scoring_mix_changes_ranking_order():
    candidates = [
        {"id": "base-heavy", "source": "vector", "text": "unrelated archive note", "score": 0.95, "ts": 1.0, "tags": []},
        {"id": "text-heavy", "source": "vector", "text": "User: Teddy launch plan\nOrion: agreed plan", "score": 0.3, "ts": 1.0, "tags": []},
    ]

    score_profile = _profile(relevance={"backend_weights": {"vector": 1.0}, "score_weight": 1.0, "text_similarity_weight": 0.0, "recency_weight": 0.0, "enable_recency": False})
    text_profile = _profile(relevance={"backend_weights": {"vector": 1.0}, "score_weight": 0.0, "text_similarity_weight": 1.0, "recency_weight": 0.0, "enable_recency": False})

    score_bundle, _ = fuse_candidates(candidates=candidates, profile=score_profile, query_text="Teddy launch plan", diagnostic=True)
    text_bundle, _ = fuse_candidates(candidates=candidates, profile=text_profile, query_text="Teddy launch plan", diagnostic=True)

    assert score_bundle.items[0].id == "base-heavy"
    assert text_bundle.items[0].id == "text-heavy"


def test_fusion_enable_recency_false_disables_recency_effect():
    candidates = [
        {"id": "older", "source": "vector", "text": "Teddy launch plan", "score": 0.7, "ts": 1.0, "tags": []},
        {"id": "newer", "source": "vector", "text": "Teddy launch plan", "score": 0.6, "ts": 9999999999.0, "tags": []},
    ]
    profile = _profile(relevance={"backend_weights": {"vector": 1.0}, "score_weight": 1.0, "text_similarity_weight": 0.0, "recency_weight": 1.0, "enable_recency": False})

    bundle, _ = fuse_candidates(candidates=candidates, profile=profile, query_text="Teddy launch plan", diagnostic=True)
    assert bundle.items[0].id == "older"


def test_query_backends_never_emits_vector(monkeypatch):
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)

    candidates, counts = asyncio.run(
        worker._query_backends(
            "hello",
            {"profile": "self.factual.v1", "vector_top_k": 0, "enable_sql_timeline": False, "rdf_top_k": 0},
            session_id=None,
            node_id=None,
            entities=[],
            diagnostic=True,
            exclusion={},
        )
    )

    assert candidates == []
    assert counts.get("vector", 0) == 0


def test_query_backends_profile_disables_sql_timeline_even_when_global_on(monkeypatch):
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", True)

    async def _fail_recent(*args, **kwargs):
        raise AssertionError("sql timeline should be disabled by profile")

    async def _fail_related(*args, **kwargs):
        raise AssertionError("sql timeline should be disabled by profile")

    monkeypatch.setattr(worker, "fetch_recent_fragments", _fail_recent)
    monkeypatch.setattr(worker, "fetch_related_by_entities", _fail_related)

    candidates, counts = asyncio.run(
        worker._query_backends(
            "hello",
            {"profile": "chat.general.v1", "vector_top_k": 0, "enable_sql_timeline": False, "rdf_top_k": 0},
            session_id=None,
            node_id=None,
            entities=[],
            diagnostic=True,
            exclusion={},
        )
    )

    assert candidates == []
    assert counts.get("sql_timeline", 0) == 0
