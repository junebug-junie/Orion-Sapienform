"""Worker leaves a dead participating conversation via leaveConversation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from orion.schemas.embodiment import WorldPerceptionV1
from app.worker import EmbodimentWorker


def _worker(**kwargs) -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._settings = SimpleNamespace(
        conversation_abandon_sec=kwargs.get("abandon_sec", 180.0),
        social_enabled=True,
        social_initiate_distance=0.0,
        social_cooldown_sec=120.0,
    )
    w._orion_player_id = "p:29"
    w._world_id = "w1"
    w._abandon_participating_since_ms = {}
    w._abandon_own_utterances = {}
    w._faced_conversations = set()
    w._opened_conversations = set()
    w._speaking_conversations = set()
    w._speech_exhausted_partner_lines = set()
    w._last_social_attempt_at = None
    return w


def test_engage_abandons_when_last_line_stale():
    w = _worker(abandon_sec=180.0)
    w._abandon_participating_since_ms["c:1"] = 1_000.0
    w._abandon_own_utterances["c:1"] = 1
    perception = WorldPerceptionV1(
        player_id="p:29",
        position={"x": 1.0, "y": 1.0},
        nearby_players=[],
        active_conversation={
            "conversation_id": "c:1",
            "status": "participating",
            "participants": ["p:29", "p:0"],
            "other": {"player_id": "p:0", "name": "Mara Vale"},
            "messages": [
                {"author_id": "p:29", "text": "old line", "created_ms": 1_000.0},
            ],
            "facing_partner": True,
        },
    )
    with patch("app.worker._utcnow") as now_mock, patch(
        "app.worker.aitown_client.leave_conversation"
    ) as leave, patch("app.worker.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        now_mock.return_value = datetime.fromtimestamp(181.0, tz=timezone.utc)
        to_thread.side_effect = lambda fn, **kwargs: fn(**kwargs)
        asyncio.run(w._engage_conversation(perception))
    leave.assert_called_once_with(
        player_id="p:29", conversation_id="c:1", world_id="w1"
    )


def test_engage_does_not_abandon_fresh_partner_line():
    w = _worker(abandon_sec=180.0)
    perception = WorldPerceptionV1(
        player_id="p:29",
        position={"x": 1.0, "y": 1.0},
        nearby_players=[],
        active_conversation={
            "conversation_id": "c:1",
            "status": "participating",
            "participants": ["p:29", "p:0"],
            "other": {"player_id": "p:0", "name": "Mara Vale"},
            "messages": [
                {"author_id": "p:0", "text": "fresh", "created_ms": 170_000.0},
            ],
            "facing_partner": True,
        },
    )
    with patch("app.worker._utcnow") as now_mock, patch(
        "app.worker.aitown_client.leave_conversation"
    ) as leave, patch("app.worker.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        now_mock.return_value = datetime.fromtimestamp(181.0, tz=timezone.utc)
        to_thread.side_effect = lambda fn, **kwargs: fn(**kwargs)
        asyncio.run(w._engage_conversation(perception))
    leave.assert_not_called()


def test_engage_skips_abandon_while_speaking():
    w = _worker(abandon_sec=180.0)
    w._speaking_conversations.add("c:1")
    w._abandon_participating_since_ms["c:1"] = 1_000.0
    perception = WorldPerceptionV1(
        player_id="p:29",
        position={"x": 1.0, "y": 1.0},
        nearby_players=[],
        active_conversation={
            "conversation_id": "c:1",
            "status": "participating",
            "participants": ["p:29", "p:0"],
            "other": {"player_id": "p:0", "name": "Mara Vale"},
            "messages": [
                {"author_id": "p:29", "text": "old", "created_ms": 1_000.0},
            ],
            "facing_partner": True,
        },
    )
    with patch("app.worker._utcnow") as now_mock, patch(
        "app.worker.aitown_client.leave_conversation"
    ) as leave, patch("app.worker.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        now_mock.return_value = datetime.fromtimestamp(181.0, tz=timezone.utc)
        to_thread.side_effect = lambda fn, **kwargs: fn(**kwargs)
        asyncio.run(w._engage_conversation(perception))
    leave.assert_not_called()


def test_engage_abandons_never_spoke_with_fresh_partner_spam():
    w = _worker(abandon_sec=180.0)
    w._abandon_participating_since_ms["c:1"] = 1_000.0
    perception = WorldPerceptionV1(
        player_id="p:29",
        position={"x": 1.0, "y": 1.0},
        nearby_players=[],
        active_conversation={
            "conversation_id": "c:1",
            "status": "participating",
            "participants": ["p:29", "p:0"],
            "other": {"player_id": "p:0", "name": "Mara Vale"},
            "messages": [
                {"author_id": "p:0", "text": "spam", "created_ms": 170_000.0},
            ],
            "facing_partner": True,
        },
    )
    with patch("app.worker._utcnow") as now_mock, patch(
        "app.worker.aitown_client.leave_conversation"
    ) as leave, patch("app.worker.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
        now_mock.return_value = datetime.fromtimestamp(181.0, tz=timezone.utc)
        to_thread.side_effect = lambda fn, **kwargs: fn(**kwargs)
        asyncio.run(w._engage_conversation(perception))
    leave.assert_called_once_with(
        player_id="p:29", conversation_id="c:1", world_id="w1"
    )