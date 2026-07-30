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
    assert convo["other"] == {
        "player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "is_human": False,
    }
    # whitespace-only message dropped; author_id preserved for turn-taking
    assert convo["messages"] == [{"author_id": "p9", "author": "Juniper", "text": "hey Orion"}]


def test_build_perception_forwards_is_human_for_nearby_and_partner():
    # ai-town's raw player record carries `human` (a session-token id) only for
    # human-controlled players; NPCs driven by ai-town's own agent framework never
    # have it. This must be forwarded, not dropped, so callers can tag
    # conversation-memory events with the correct participant_kind.
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "human1", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "human": "tok:abc"},
        {"id": "npc1", "name": "Mira", "position": {"x": 2.0, "y": 0.0}},
    ]
    conversations = [{"id": "c:1", "participants": [
        {"playerId": "orion", "status": {"kind": "participating"}},
        {"playerId": "human1", "status": {"kind": "participating"}},
    ]}]
    perc = build_perception(players=players, orion_player_id="orion", conversations=conversations)
    nearby_by_id = {n["player_id"]: n for n in perc.nearby_players}
    assert nearby_by_id["human1"]["is_human"] is True
    assert nearby_by_id["npc1"]["is_human"] is False
    assert perc.active_conversation["other"]["is_human"] is True


def test_build_perception_computes_nearby_landmarks_sorted_by_distance():
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    locations = {
        "waterfall": {"x": 5.0, "y": 0.0},
        "campfire": {"x": 2.0, "y": 0.0},
        "windmill_east": {"x": 100.0, "y": 100.0},
    }
    perc = build_perception(players=players, orion_player_id="orion", locations=locations, max_landmarks=2)
    names = [lm["name"] for lm in perc.nearby_landmarks]
    assert names == ["campfire", "waterfall"]  # nearest 2, sorted, windmill_east excluded
    assert perc.nearby_landmarks[0]["distance"] == 2.0


def test_build_perception_empty_landmarks_when_locations_unset():
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    perc = build_perception(players=players, orion_player_id="orion")
    assert perc.nearby_landmarks == []


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
