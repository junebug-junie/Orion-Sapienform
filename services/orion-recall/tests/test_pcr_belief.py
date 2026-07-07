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
from app.collectors.active_packet import fetch_active_packet_fragments
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
        assert float(profile["relevance"]["backend_weights"]["active_packet"]) == 1.0


def test_collectors_for_intent_semantic():
    plan = collectors_for_intent("semantic")
    assert plan["active_packet"] is True
    assert plan["cards"] is True
    assert plan["rdf"] is True


@pytest.mark.asyncio
async def test_active_packet_maps_crystallization_to_fragment():
    fake_packet = type(
        "Pkt",
        (),
        {
            "items": [
                {
                    "crystallization_id": "crys_abc",
                    "summary": "Move solo",
                    "salience": 0.9,
                    "kind": "semantic",
                    "bucket": "semantic",
                }
            ]
        },
    )()
    with patch("app.collectors.active_packet.list_crystallizations", new=AsyncMock(return_value=[])), patch(
        "app.collectors.active_packet.retrieve_active_packet", new=AsyncMock(return_value=fake_packet)
    ):
        frags = await fetch_active_packet_fragments(
            query=type(
                "Q",
                (),
                {"session_id": "s1", "node_id": "n1", "retrieval_intent": "semantic", "fragment": "move logistics"},
            )(),
            pool=object(),
            settings=type("S", (), {"RECALL_ACTIVE_PACKET_ENABLED": True})(),
        )
    assert frags[0]["source"] == "active_packet"
    assert "Move solo" in frags[0]["snippet"]


def test_pcr_fuse_belief_candidates_ranks_active_packet_and_drops_social():
    candidates = [
        {"id": "card-1", "source": "cards", "snippet": "Card belief", "score": 0.95},
        {"id": "ap-1", "source": "active_packet", "snippet": "Move solo logistics", "score": 0.9},
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
    assert bundle.items[0].source == "active_packet"


def test_worker_uses_purposeful_fusion_when_phase_purposeful(monkeypatch):
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_ACTIVE_PACKET_ENABLED", True)
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
    active_packet_called = []

    def _mock_fuse(**kwargs):
        fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="fuse", stats=MemoryBundleStatsV1()), []

    def _mock_belief_fuse(**kwargs):
        belief_fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="belief ok", stats=MemoryBundleStatsV1()), []

    async def _mock_active_packet(query, *, pool, settings):
        active_packet_called.append(query)
        return [{"id": "ap-1", "source": "active_packet", "snippet": "Move solo", "score": 0.9}]

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(worker, "fuse_candidates", _mock_fuse)
    monkeypatch.setattr(worker, "pcr_fuse_belief_candidates", _mock_belief_fuse)
    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _mock_active_packet)

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

    assert active_packet_called
    assert belief_fuse_called
    assert not fuse_called
    assert bundle.rendered == "belief ok"
    assert decision.recall_debug["pcr"]["phase"] == "purposeful"
    assert decision.recall_debug["pcr"]["retrieval_intent"] == "semantic"
    assert "active_packet" in decision.recall_debug["pcr"]["backend_plan"]
