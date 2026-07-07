from __future__ import annotations

from orion.substrate.relational.adapters.town_perception_ctx import map_town_perception_to_substrate


def test_maps_nearby_players_to_nodes():
    ctx = {"perception": {
        "player_id": "orion", "position": {"x": 0.0, "y": 0.0},
        "nearby_players": [{"player_id": "j", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
    }}
    record = map_town_perception_to_substrate(ctx)
    assert record is not None
    assert record.anchor_scope == "orion"
    assert any("Juniper" in (n.label or "") or n.metadata.get("player_id") == "j" for n in record.nodes)


def test_empty_perception_returns_none():
    assert map_town_perception_to_substrate({}) is None
