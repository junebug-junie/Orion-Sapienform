from __future__ import annotations

from app.embodiment_background import (
    LatestPerceptionCache,
    build_background_embodiment_intent,
)

from orion.schemas.embodiment import WorldPerceptionV1


def _perception_with_neighbor() -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 0.0, "y": 0.0},
        nearby_players=[
            {"player_id": "p9", "name": "Juniper", "position": {"x": 3.0, "y": 0.0}, "distance": 3.0},
            {"player_id": "p2", "name": "Bram", "position": {"x": 8.0, "y": 0.0}, "distance": 8.0},
        ],
    )


def _perception_alone() -> WorldPerceptionV1:
    return WorldPerceptionV1(player_id="orion", position={"x": 1.0, "y": 1.0}, nearby_players=[])


def test_background_intent_approaches_nearest_player():
    intent = build_background_embodiment_intent(_perception_with_neighbor(), correlation_id="bg-1")
    assert intent.source == "deliberate"
    assert intent.kind == "approach_player"
    assert intent.ref == "Juniper"
    assert intent.correlation_id == "bg-1"
    assert intent.player_id == "orion"
    assert intent.reason.strip()


def test_background_intent_wanders_when_alone():
    intent = build_background_embodiment_intent(_perception_alone(), correlation_id="bg-2")
    assert intent.source == "deliberate"
    assert intent.kind == "wander"
    assert intent.reason.strip()


def test_cache_latest_wins_keyed_by_player_id():
    clock = {"t": 100.0}
    cache = LatestPerceptionCache(ttl_sec=30.0, clock=lambda: clock["t"])
    first = WorldPerceptionV1(player_id="orion", position={"x": 0.0, "y": 0.0})
    second = WorldPerceptionV1(player_id="orion", position={"x": 5.0, "y": 5.0})
    cache.put(first)
    cache.put(second)
    got = cache.get("orion")
    assert got is not None
    assert got.position == {"x": 5.0, "y": 5.0}


def test_cache_ttl_evicts_stale_entries():
    clock = {"t": 0.0}
    cache = LatestPerceptionCache(ttl_sec=10.0, clock=lambda: clock["t"])
    cache.put(WorldPerceptionV1(player_id="orion", position={"x": 0.0, "y": 0.0}))
    clock["t"] = 5.0
    assert cache.get("orion") is not None
    clock["t"] = 20.0
    assert cache.get("orion") is None
