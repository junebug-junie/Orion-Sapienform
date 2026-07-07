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
