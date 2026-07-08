from __future__ import annotations

from orion.embodiment.perception import build_perception


def test_build_perception_computes_distances_and_nearby():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "j", "name": "Juniper", "position": {"x": 3.0, "y": 4.0}},
    ]
    perc = build_perception(players=players, orion_player_id="orion", max_nearby=5)
    assert perc.player_id == "orion"
    assert perc.position == {"x": 0.0, "y": 0.0}
    assert len(perc.nearby_players) == 1
    assert perc.nearby_players[0]["distance"] == 5.0


def test_build_perception_none_when_orion_absent():
    assert build_perception(players=[{"id": "x", "position": {"x": 1, "y": 1}}],
                            orion_player_id="orion") is None


def test_build_perception_shapes_active_conversation():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}},
    ]
    conversations = [
        {"id": "c:1", "participants": [
            {"playerId": "orion", "status": {"kind": "participating"}},
            {"playerId": "p9", "status": {"kind": "participating"}},
        ]},
        {"id": "c:2", "participants": [{"playerId": "zz", "status": {"kind": "invited"}}]},
    ]
    messages = [{"author": "p9", "authorName": "Juniper", "text": "hey Orion"},
                {"author": "p9", "text": "  "}]
    perc = build_perception(players=players, orion_player_id="orion",
                            conversations=conversations, messages=messages)
    convo = perc.active_conversation
    assert convo["conversation_id"] == "c:1"
    assert convo["status"] == "participating"
    assert set(convo["participants"]) == {"orion", "p9"}
    assert convo["other"] == {"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}}
    # whitespace-only message dropped; author_id preserved for turn-taking
    assert convo["messages"] == [{"author_id": "p9", "author": "Juniper", "text": "hey Orion"}]


def test_build_perception_no_conversation_when_orion_not_member():
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    conversations = [{"id": "c:2", "participants": [{"playerId": "zz", "status": {"kind": "invited"}}]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation is None


def test_build_perception_reports_invited_status():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 5.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:9", "participants": [
        {"playerId": "p9", "status": {"kind": "walkingOver"}},
        {"playerId": "orion", "status": {"kind": "invited"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation["status"] == "invited"
    assert perc.active_conversation["other"]["player_id"] == "p9"


def test_build_perception_exposes_own_facing_and_pathfinding():
    players = [
        {"id": "orion", "position": {"x": 2.0, "y": 3.0},
         "facing": {"dx": 1.0, "dy": 0.0},
         "pathfinding": {"destination": {"x": 9, "y": 9}, "started": 1, "state": {"kind": "moving"}}},
    ]
    perc = build_perception(players=players, orion_player_id="orion")
    assert perc.facing == {"dx": 1.0, "dy": 0.0}
    assert perc.pathfinding is True


def test_build_perception_pathfinding_false_when_stopped():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}, "facing": {"dx": 0.0, "dy": -1.0}},
    ]
    perc = build_perception(players=players, orion_player_id="orion")
    assert perc.facing == {"dx": 0.0, "dy": -1.0}
    assert perc.pathfinding is False


def test_facing_partner_true_when_oriented_toward_partner():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}, "facing": {"dx": 1.0, "dy": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 5.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:1", "participants": [
        {"playerId": "orion", "status": {"kind": "participating"}},
        {"playerId": "p9", "status": {"kind": "participating"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation["facing_partner"] is True


def test_facing_partner_false_when_facing_away():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}, "facing": {"dx": -1.0, "dy": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 5.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:1", "participants": [
        {"playerId": "orion", "status": {"kind": "participating"}},
        {"playerId": "p9", "status": {"kind": "participating"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation["facing_partner"] is False


def test_facing_partner_none_when_facing_unknown():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 5.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:1", "participants": [
        {"playerId": "orion", "status": {"kind": "participating"}},
        {"playerId": "p9", "status": {"kind": "participating"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation["facing_partner"] is None


def test_facing_partner_none_when_not_participating():
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}, "facing": {"dx": 1.0, "dy": 0.0}},
        {"id": "p9", "name": "Juniper", "position": {"x": 5.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:1", "participants": [
        {"playerId": "orion", "status": {"kind": "walkingOver"}},
        {"playerId": "p9", "status": {"kind": "participating"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    assert perc.active_conversation["facing_partner"] is None
