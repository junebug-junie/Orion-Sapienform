"""PCR purposeful (belief) recall must run the same entity injection the
default recall path runs.

Live regression, corr 21c8a610-f852-4af3-a855-c337a30486da (2026-10-14):
"Tell Pip and I an imaginative story about Pip going to space..." routed
to intent=open_loop, and the belief digest held only 3 unrelated
active_packet lines. The older turn "We celebrated my son, Pip's, 7th
birthday" was never fetched, because the purposeful branch skipped
_compute_entity_relatedness_boost_map entirely, and even if it had been
fetched, rank-0 active_packet lines filled the 128-word budget first.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app import worker
from app.fusion import ENTITY_INJECTED_TAG, pcr_fuse_belief_candidates
from app.profiles import get_profile
from orion.core.contracts.recall import RecallQueryV1

FRAGMENT = "Tell Pip and I an imaginative story about Pip going to space and a visit to the moon."
PIP_TURN_ID = "turn-pip-birthday-0816"
PIP_PROMPT = "We celebrated my son, Pip's, 7th birthday today. He loved the rocket cake."


def _unrelated_active_packet():
    filler = " ".join(["the substrate maintenance window keeps drifting and needs a follow up review"] * 4)
    return [
        {"id": f"ap-{i}", "source": "active_packet", "snippet": f"Open loop {i}: {filler}", "score": 0.568}
        for i in range(3)
    ]


def _patch_common(monkeypatch, profile_name: str):
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_ACTIVE_PACKET_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_CONCEPT_REGION_ENABLED", False)
    monkeypatch.setattr(worker.settings, "RECALL_BELIEF_RENDER_BUDGET", 128)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(worker.settings, "RECALL_INTENT_ROUTING_ENABLED", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED", True)

    belief_profile = get_profile(profile_name)
    monkeypatch.setattr(worker, "get_profile", lambda _name: dict(belief_profile))

    async def _anchor(**kwargs):
        return [], {}

    async def _active_packet(query, *, pool, settings):
        return _unrelated_active_packet()

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", AsyncMock(return_value=([], {})))
    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _active_packet)

    # Falkor/Postgres seams of the entity injection: one older turn naming
    # Pip as the user's son. Degree 3 -> K/degree discount clamps to 1.0.
    monkeypatch.setattr(worker, "fetch_related_entities", AsyncMock(return_value=[]))
    monkeypatch.setattr(worker, "fetch_entity_degrees", AsyncMock(return_value={"pip": 3}))
    monkeypatch.setattr(
        worker,
        "fetch_turns_mentioning_entities",
        AsyncMock(return_value=[{"turn_id": PIP_TURN_ID, "ts": "2026-08-16T20:00:00+00:00"}]),
    )
    monkeypatch.setattr(
        worker,
        "fetch_chat_turns_by_id",
        AsyncMock(return_value={PIP_TURN_ID: (PIP_PROMPT, "Happy birthday to Pip! Seven is a big one.", {})}),
    )
    monkeypatch.setattr(
        worker, "fetch_entity_matches_for_turns", AsyncMock(return_value={PIP_TURN_ID: ["pip"]})
    )


def test_purposeful_open_loop_injects_entity_turn_into_belief_digest(monkeypatch):
    _patch_common(monkeypatch, "chat.belief.open_loop.v1")
    q = RecallQueryV1(
        fragment=FRAGMENT,
        profile="chat.belief.open_loop.v1",
        recall_phase="purposeful",
        retrieval_intent="open_loop",
        task_hints={"rule_id": "open_loops_present"},
        session_id="sess-pip",
    )

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-pip", diagnostic=True))

    assert decision.backend_counts.get("falkor_chat_entity_injected", 0) >= 1
    assert "my son, Pip" in bundle.rendered
    assert bundle.items[0].id == PIP_TURN_ID
    # active_packet still renders after it, not replaced wholesale.
    assert "[active_packet" in bundle.rendered


def test_purposeful_procedural_gets_no_entity_injection(monkeypatch):
    _patch_common(monkeypatch, "chat.belief.procedural.v1")
    boost_spy = AsyncMock(return_value=({}, []))
    monkeypatch.setattr(worker, "_compute_entity_relatedness_boost_map", boost_spy)
    q = RecallQueryV1(
        fragment=FRAGMENT,
        profile="chat.belief.procedural.v1",
        recall_phase="purposeful",
        retrieval_intent="procedural",
        session_id="sess-pip",
    )

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-pip-proc", diagnostic=True))

    boost_spy.assert_not_called()
    assert "falkor_chat_entity_injected" not in decision.backend_counts
    assert "Pip" not in bundle.rendered


def test_belief_fusion_reserves_slot_for_entity_turn_ahead_of_active_packet():
    profile = dict(get_profile("chat.belief.open_loop.v1"))
    long_response = " ".join(["Orion talked at length about cake and candles and the party."] * 30)
    pip = {
        "id": PIP_TURN_ID,
        "source": "falkor_chat",
        "uri": PIP_TURN_ID,
        "text": f'ExactUserText: "{PIP_PROMPT}"\nOrionResponse: "{long_response}"',
        "score": 0.5,
        "tags": ["falkor", "chat", "chatturn", ENTITY_INJECTED_TAG],
    }
    bundle, _ = pcr_fuse_belief_candidates(
        candidates=_unrelated_active_packet() + [pip],
        profile=profile,
        retrieval_intent="open_loop",
        query_text=FRAGMENT,
        entity_boost_map={PIP_TURN_ID: 1.0},
    )
    assert bundle.items[0].id == PIP_TURN_ID
    assert "my son, Pip" in bundle.rendered
    assert bundle.rendered.startswith("[Durable beliefs")


def test_belief_fusion_without_injected_turns_is_unchanged():
    profile = dict(get_profile("chat.belief.open_loop.v1"))
    cands = _unrelated_active_packet()
    plain_turn = {"id": "t-plain", "source": "falkor_chat", "text": "Pip said hi", "score": 0.5, "tags": []}
    bundle, _ = pcr_fuse_belief_candidates(
        candidates=cands + [plain_turn], profile=profile, retrieval_intent="open_loop", query_text=FRAGMENT
    )
    assert bundle.items[0].source == "active_packet"


def test_injected_turn_is_cut_around_the_name_not_from_its_start():
    """Live smoke, corr 21c8a610: clamping from the start kept "Here are the
    photos of the sensor array..." and cut the mention the turn was injected
    for. The window must centre on the matched name."""
    from app.fusion import _window_around_terms

    opening = " ".join(["photos of the sensor array in the cabinet"] * 12)
    text = f'ExactUserText: "{opening} and my son Pip loved the blinking lights" OrionResponse: "nice"'
    window = _window_around_terms(text, ["pip"], 16)
    assert "my son Pip" in window
    assert len(window.split()) <= 18
    assert _window_around_terms(text, [], 16).startswith('ExactUserText: "photos')


def test_four_entity_turns_fit_ahead_of_active_packet():
    profile = dict(get_profile("chat.belief.open_loop.v1"))
    turns = [
        {"id": f"t{i}", "source": "falkor_chat", "uri": f"t{i}",
         "text": f'ExactUserText: "{" ".join(["filler words here"] * 20)} my son Pip did thing {i}"',
         "score": 0.5, "tags": ["falkor", "chat", "chatturn", ENTITY_INJECTED_TAG],
         "meta": {"entity_terms": ["pip"]}}
        for i in range(4)
    ]
    bundle, _ = pcr_fuse_belief_candidates(
        candidates=_unrelated_active_packet() + turns,
        profile=profile,
        retrieval_intent="open_loop",
        query_text=FRAGMENT,
        entity_boost_map={f"t{i}": 1.0 for i in range(4)},
    )
    assert {it.id for it in bundle.items[:4]} == {"t0", "t1", "t2", "t3"}
    assert bundle.rendered.count("my son Pip") == 4
