from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.fusion import pcr_fuse_belief_candidates
from app.pcr_collectors import collectors_for_intent
from app.profiles import get_profile
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, RecallQueryV1


def test_belief_profiles_load():
    for name in (
        "chat.belief.relational.v1",
        "chat.belief.semantic.v1",
        "chat.belief.procedural.v1",
        "chat.belief.open_loop.v1",
        "chat.belief.contradiction.v1",
    ):
        profile = get_profile(name)
        assert profile["profile"] == name
        assert profile["render_lane"] == "belief"
        assert profile["sql_chat_top_k"] == 0
        # Retired from chat (#3461): no weight left behind for a source that never runs.
        assert "active_packet" not in profile["relevance"]["backend_weights"]


def test_collectors_for_intent_semantic():
    plan = collectors_for_intent("semantic")
    assert plan["cards"] is True
    assert plan["rdf"] is True


def test_no_intent_plans_active_packet_or_graphiti():
    """#3461: the active_packet digest (and the graphiti walk that rode on it)
    is retired from chat recall for every intent, not just weighted down."""
    for intent in ("relational", "semantic", "procedural", "open_loop", "contradiction"):
        plan = collectors_for_intent(intent)
        assert "active_packet" not in plan
        assert "graphiti" not in plan


def test_pcr_fuse_belief_candidates_ranks_concept_region_and_drops_social():
    candidates = [
        {"id": "card-1", "source": "cards", "snippet": "Card belief", "score": 0.95},
        {"id": "cr-1", "source": "concept_region", "snippet": "Move solo logistics", "score": 0.9},
        {"id": "social-1", "source": "cards", "snippet": "User: hey\nAssistant: hi there", "score": 0.99},
    ]
    profile = {
        "profile": "chat.belief.semantic.v1",
        "render_budget_tokens": 128,
        "max_total_items": 8,
        "max_per_source": 4,
    }
    bundle, _ = pcr_fuse_belief_candidates(
        candidates=candidates,
        profile=profile,
        retrieval_intent="semantic",
        query_text="move logistics",
        latency_ms=1,
    )
    assert "purposeful recall" in bundle.rendered
    assert "hey" not in bundle.rendered.lower()
    assert bundle.items[0].source == "concept_region"


def test_worker_uses_purposeful_fusion_when_phase_purposeful(monkeypatch):
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_CONCEPT_REGION_ENABLED", False)
    monkeypatch.setattr(worker.settings, "RECALL_BELIEF_RENDER_BUDGET", 128)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(worker.settings, "RECALL_INTENT_ROUTING_ENABLED", False)

    belief_profile = get_profile("chat.belief.semantic.v1")
    monkeypatch.setattr(worker, "get_profile", lambda _name: dict(belief_profile))

    async def _anchor(**kwargs):
        return [], {}

    fuse_called = []
    belief_fuse_called = []

    def _mock_fuse(**kwargs):
        fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="fuse", stats=MemoryBundleStatsV1()), []

    def _mock_belief_fuse(**kwargs):
        belief_fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="belief ok", stats=MemoryBundleStatsV1()), []

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(worker, "fuse_candidates", _mock_fuse)
    monkeypatch.setattr(worker, "pcr_fuse_belief_candidates", _mock_belief_fuse)

    q = RecallQueryV1(
        fragment="what did we decide about the move",
        profile="chat.belief.semantic.v1",
        recall_phase="purposeful",
        retrieval_intent="semantic",
        task_hints={"rule_id": "topic_shift"},
        session_id="sess-1",
    )

    import asyncio

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-pcr-belief", diagnostic=True))

    assert belief_fuse_called
    assert not fuse_called
    assert bundle.rendered == "belief ok"
    assert decision.recall_debug["pcr"]["phase"] == "purposeful"
    assert decision.recall_debug["pcr"]["retrieval_intent"] == "semantic"
    assert "active_packet" not in decision.recall_debug["pcr"]["backend_plan"]
    assert "active_packet_refs" not in decision.recall_debug["pcr"]


def test_pcr_phase_preserves_profile_against_intent_routing(monkeypatch):
    """PCR continuity/purposeful phases must not lose their profile to intent routing."""
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_CONTINUITY_SQL_MINUTES", 120)
    monkeypatch.setattr(worker.settings, "RECALL_CONTINUITY_RENDER_BUDGET", 96)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(worker.settings, "RECALL_INTENT_ROUTING_ENABLED", True)

    continuity_profile = {
        "profile": "chat.continuity.v1",
        "enable_query_expansion": False,
        "enable_anchor_candidates": False,
        "enable_sql_timeline": False,
        "enable_sql_chat": True,
        "sql_chat_top_k": 6,
        "sql_since_minutes": 120,
        "max_total_items": 6,
        "render_budget_tokens": 96,
        "render_lane": "continuity",
    }
    profile_calls: list[str] = []

    def _track_profile(name):
        profile_calls.append(str(name))
        return dict(continuity_profile)

    monkeypatch.setattr(worker, "get_profile", _track_profile)

    async def _fetch_pairs(**kwargs):
        return []

    async def _fetch_msgs(**kwargs):
        return []

    async def _anchor(**kwargs):
        return [], {}

    def _mock_render(**kwargs):
        return MemoryBundleV1(rendered="continuity ok", stats=MemoryBundleStatsV1()), []

    monkeypatch.setattr(worker, "fetch_chat_history_pairs", _fetch_pairs)
    monkeypatch.setattr(worker, "fetch_chat_messages", _fetch_msgs)
    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "render_continuity_bundle", _mock_render)

    q = RecallQueryV1(
        fragment="who am I",
        profile="chat.continuity.v1",
        recall_phase="continuity",
        session_id="sess-1",
    )

    import asyncio

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-pcr-intent-guard", diagnostic=True))

    assert bundle.rendered == "continuity ok"
    assert decision.profile == "chat.continuity.v1"
    assert profile_calls == ["chat.continuity.v1"]
    intent_debug = decision.recall_debug.get("recall_intent") or {}
    assert intent_debug.get("selected_profile") == "chat.continuity.v1"
    assert intent_debug.get("profile_explicit") is True
