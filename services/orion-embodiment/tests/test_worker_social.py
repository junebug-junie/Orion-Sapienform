from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import EmbodimentWorker
from orion.schemas.embodiment import WorldPerceptionV1


def _worker(initiate_distance=0.0, last_social=None, cooldown=120.0):
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._social_cooldown_sec = cooldown
    w._last_social_attempt_at = last_social
    w._faced_conversations = set()
    w._settings = SimpleNamespace(social_initiate_distance=initiate_distance)
    return w


def _perc(active=None, nearby=None, pathfinding=False, position=None):
    return WorldPerceptionV1(
        player_id="orion", position=position or {"x": 0.0, "y": 0.0},
        pathfinding=pathfinding,
        nearby_players=nearby or [], active_conversation=active,
    )


def test_orion_conversation_id_matches_status():
    convos = [
        {"id": "c:1", "participants": [{"playerId": "zz", "status": {"kind": "participating"}}]},
        {"id": "c:2", "participants": [{"playerId": "orion", "status": {"kind": "invited"}}]},
    ]
    assert EmbodimentWorker._orion_conversation_id(convos, "orion") == "c:2"
    assert EmbodimentWorker._orion_conversation_id(convos, "orion", want_status="participating") is None
    assert EmbodimentWorker._orion_conversation_id(convos, "orion", want_status="invited") == "c:2"


def test_engage_accepts_invite():
    w = _worker()
    perc = _perc(active={"conversation_id": "c:2", "status": "invited", "other": {"player_id": "p9"}})
    with patch("app.worker.aitown_client.accept_invite") as acc, \
         patch.object(w, "_actuate", new=AsyncMock()) as act:
        asyncio.run(w._engage_conversation(perc))
    acc.assert_called_once_with(player_id="orion", conversation_id="c:2", world_id="w1")
    act.assert_not_awaited()


def test_engage_walks_to_partner_when_walking_over():
    w = _worker()
    perc = _perc(active={"conversation_id": "c:2", "status": "walkingOver",
                         "other": {"player_id": "p9", "position": {"x": 5.0, "y": 0.0}}})
    with patch.object(w, "_actuate", new=AsyncMock()) as act:
        asyncio.run(w._engage_conversation(perc))
    act.assert_awaited_once()
    intent = act.await_args.args[0]
    assert intent.kind == "approach_player" and intent.ref == "p9"


def test_engage_initiates_with_nearby_player():
    w = _worker(initiate_distance=4.0)
    perc = _perc(nearby=[{"player_id": "p9", "distance": 2.0}])
    with patch.object(w, "_actuate", new=AsyncMock()) as act:
        asyncio.run(w._engage_conversation(perc))
    act.assert_awaited_once()
    intent = act.await_args.args[0]
    assert intent.kind == "start_conversation" and intent.ref == "p9"
    assert w._last_social_attempt_at is not None


def test_initiate_skipped_when_out_of_range():
    w = _worker(initiate_distance=4.0)
    perc = _perc(nearby=[{"player_id": "p9", "distance": 9.0}])
    with patch.object(w, "_actuate", new=AsyncMock()) as act:
        asyncio.run(w._engage_conversation(perc))
    act.assert_not_awaited()


def test_initiate_respects_cooldown():
    now = datetime(2026, 7, 7, 0, 0, 30, tzinfo=timezone.utc)
    w = _worker(initiate_distance=4.0, last_social=datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc),
                cooldown=120.0)
    perc = _perc(nearby=[{"player_id": "p9", "distance": 1.0}])
    with patch.object(w, "_actuate", new=AsyncMock()) as act, \
         patch("app.worker._utcnow", return_value=now):
        asyncio.run(w._engage_conversation(perc))
    act.assert_not_awaited()


def test_initiate_off_when_distance_zero():
    w = _worker(initiate_distance=0.0)
    perc = _perc(nearby=[{"player_id": "p9", "distance": 1.0}])
    with patch.object(w, "_actuate", new=AsyncMock()) as act:
        asyncio.run(w._engage_conversation(perc))
    act.assert_not_awaited()


def test_engage_stops_once_to_face_partner_when_pathfinding():
    w = _worker()
    active = {"conversation_id": "c:3", "status": "participating",
              "other": {"player_id": "p9", "position": {"x": 5.0, "y": 0.0}}}
    perc = _perc(active=active, pathfinding=True, position={"x": 2.3, "y": 3.8})
    with patch("app.worker.aitown_client.move_to") as mv:
        asyncio.run(w._engage_conversation(perc))
        # Second tick for the same conversation must NOT re-issue the stop.
        asyncio.run(w._engage_conversation(perc))
    mv.assert_called_once_with(player_id="orion", x=2.5, y=3.5, world_id="w1")
    assert "c:3" in w._faced_conversations


def test_engage_does_not_stop_when_not_pathfinding():
    w = _worker()
    active = {"conversation_id": "c:4", "status": "participating",
              "other": {"player_id": "p9", "position": {"x": 5.0, "y": 0.0}}}
    perc = _perc(active=active, pathfinding=False, position={"x": 2.0, "y": 3.0})
    with patch("app.worker.aitown_client.move_to") as mv:
        asyncio.run(w._engage_conversation(perc))
    mv.assert_not_called()
    assert "c:4" not in w._faced_conversations
