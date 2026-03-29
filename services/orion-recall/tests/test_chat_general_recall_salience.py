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
from orion.core.contracts.recall import RecallQueryV1


def _chat_profile() -> dict:
    return {
        "profile": "chat.general.v1",
        "enable_query_expansion": False,
        "enable_sql_timeline": False,
        "vector_top_k": 6,
        "rdf_top_k": 0,
        "sql_top_k": 0,
        "max_total_items": 6,
    }


def test_chat_general_substantive_turn_drops_greeting_pollution(monkeypatch):
    monkeypatch.setattr(worker, "get_profile", lambda _name: _chat_profile())

    async def _anchor(**kwargs):
        return [], {}

    async def _query(fragment, profile, *, session_id, node_id, entities, diagnostic=False, exclusion=None):
        return [
            {"id": "g1", "source": "vector", "text": "hi friend", "score": 0.95, "tags": []},
            {"id": "g2", "source": "vector", "text": "Thanks, all good for now.", "score": 0.93, "tags": []},
            {
                "id": "s1",
                "source": "vector",
                "text": "You said the move stress is heavy because you're carrying most logistics while your wife is incapacitated.",
                "score": 0.72,
                "tags": [],
            },
        ], {"vector": 3}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", _query)

    q = RecallQueryV1(
        fragment="I'm under move stress and carrying the move mostly alone while my wife is incapacitated.",
        profile="chat.general.v1",
        verb="chat_general",
    )

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-salience-1", diagnostic=True))
    ids = [item.id for item in bundle.items]
    assert "s1" in ids
    assert "g1" not in ids
    assert "g2" not in ids
    assert (decision.recall_debug.get("fusion") or {}).get("drop_counts", {}).get("low_info_social", 0) >= 1


def test_chat_general_instruction_tail_does_not_shift_query_anchor(monkeypatch):
    monkeypatch.setattr(worker, "get_profile", lambda _name: _chat_profile())
    seen_fragments: list[str] = []

    async def _anchor(**kwargs):
        return [], {}

    async def _query(fragment, profile, *, session_id, node_id, entities, diagnostic=False, exclusion=None):
        seen_fragments.append(fragment)
        return [
            {
                "id": "s1",
                "source": "vector",
                "text": "Move stress and carrying relocation logistics alone are the core issues.",
                "score": 0.8,
                "tags": [],
            }
        ], {"vector": 1}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", _query)

    base = "I'm under move stress and carrying this move mostly alone while my wife is incapacitated."
    with_tail = f"{base}\nBe sure to remain based on your recall and process injection."

    q1 = RecallQueryV1(fragment=base, profile="chat.general.v1", verb="chat_general")
    q2 = RecallQueryV1(fragment=with_tail, profile="chat.general.v1", verb="chat_general")

    _, d1 = asyncio.run(worker.process_recall(q1, corr_id="corr-salience-2a", diagnostic=True))
    _, d2 = asyncio.run(worker.process_recall(q2, corr_id="corr-salience-2b", diagnostic=True))

    assert len(seen_fragments) >= 2
    assert seen_fragments[0] == seen_fragments[1]
    t1 = d1.recall_debug.get("query_targeting") or {}
    t2 = d2.recall_debug.get("query_targeting") or {}
    assert t1.get("query_fragment") == t2.get("query_fragment")
    assert t2.get("tail_stripped") is True


def test_chat_general_social_turn_keeps_lightweight_behavior(monkeypatch):
    monkeypatch.setattr(worker, "get_profile", lambda _name: _chat_profile())

    async def _anchor(**kwargs):
        return [], {}

    async def _query(fragment, profile, *, session_id, node_id, entities, diagnostic=False, exclusion=None):
        return [
            {"id": "g1", "source": "vector", "text": "Hi Orion!", "score": 0.92, "tags": []},
            {"id": "g2", "source": "vector", "text": "Hey friend", "score": 0.84, "tags": []},
        ], {"vector": 2}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", _query)

    q = RecallQueryV1(fragment="Hi Orion!", profile="chat.general.v1", verb="chat_general")
    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-salience-3", diagnostic=True))

    assert len(bundle.items) >= 1
    targeting = decision.recall_debug.get("query_targeting") or {}
    assert targeting.get("turn_type") == "social"
    assert (decision.recall_debug.get("fusion") or {}).get("drop_counts", {}).get("low_info_social", 0) == 0
