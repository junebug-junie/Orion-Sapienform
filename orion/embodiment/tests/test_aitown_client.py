from __future__ import annotations

from unittest.mock import patch

from orion.embodiment import aitown_client


def test_list_players_calls_query(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "convex_query", return_value=[{"id": "p1"}]) as q:
        players = aitown_client.list_players()
    q.assert_called_once_with("aiTown/world:players", {"worldId": "w1"})
    assert players == [{"id": "p1"}]


def test_move_to_builds_send_input(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "send_input", return_value={"ok": True}) as s:
        aitown_client.move_to(player_id="p1", x=3.0, y=4.0)
    s.assert_called_once_with(
        name="moveTo", args={"playerId": "p1", "destination": {"x": 3.0, "y": 4.0}}
    )


def test_send_input_requires_world_id(monkeypatch):
    monkeypatch.delenv("AITOWN_WORLD_ID", raising=False)
    try:
        aitown_client.send_input(name="moveTo", args={})
        assert False, "expected AitownClientError"
    except aitown_client.AitownClientError:
        pass
