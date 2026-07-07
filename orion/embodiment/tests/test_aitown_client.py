from __future__ import annotations

from unittest.mock import patch

from orion.embodiment import aitown_client


def test_list_players_shapes_world_state(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    ws = {"world": {"players": [{"id": "p:0", "position": {"x": 1.0, "y": 2.0}}]}}
    gd = {"playerDescriptions": [{"playerId": "p:0", "name": "Orion"}]}
    with patch.object(aitown_client, "convex_query", side_effect=[ws, gd]) as q:
        players = aitown_client.list_players()
    assert q.call_args_list[0][0][0] == "world:worldState"
    assert q.call_args_list[1][0][0] == "world:gameDescriptions"
    assert players == [{"id": "p:0", "position": {"x": 1.0, "y": 2.0}, "name": "Orion"}]


def test_join_player_sends_join_input(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "send_input", return_value=None) as s:
        aitown_client.join_player(name="Orion", character="f1", description="d")
    s.assert_called_once_with(
        name="join", args={"name": "Orion", "character": "f1", "description": "d"}, world_id=None
    )


def test_move_to_builds_send_input(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "send_input", return_value={"ok": True}) as s:
        aitown_client.move_to(player_id="p1", x=3.0, y=4.0)
    s.assert_called_once_with(
        name="moveTo", args={"playerId": "p1", "destination": {"x": 3.0, "y": 4.0}}
    )


def test_heartbeat_world_calls_mutation(monkeypatch):
    monkeypatch.setenv("AITOWN_WORLD_ID", "w1")
    with patch.object(aitown_client, "convex_mutation", return_value=None) as m:
        aitown_client.heartbeat_world()
    m.assert_called_once_with("world:heartbeatWorld", {"worldId": "w1"})


def test_send_input_requires_world_id(monkeypatch):
    monkeypatch.delenv("AITOWN_WORLD_ID", raising=False)
    try:
        aitown_client.send_input(name="moveTo", args={})
        assert False, "expected AitownClientError"
    except aitown_client.AitownClientError:
        pass
