from __future__ import annotations

import importlib


def test_store_social_room_inspection_from_route_debug() -> None:
    cache = importlib.import_module("scripts.social_room_inspection_cache")
    cache._latest.clear()
    route_debug = {
        "verb": "chat_social_room",
        "social_room_mode": "hub_direct",
        "social_inspection": {"platform": "hub", "room_id": "hub-direct", "sections": []},
    }
    cache.store("hub-direct", route_debug)
    entry = cache.get("hub-direct")
    assert entry is not None
    assert entry["routing_debug"]["verb"] == "chat_social_room"
    assert entry["room_id"] == "hub-direct"
    assert entry["stored_at"]


def test_get_latest_returns_most_recent_room() -> None:
    cache = importlib.import_module("scripts.social_room_inspection_cache")
    cache._latest.clear()
    cache.store("hub-direct", {"verb": "chat_social_room", "corr": "a"})
    cache.store("other-room", {"verb": "chat_social_room", "corr": "b"})
    latest = cache.get_latest()
    assert latest is not None
    assert latest["room_id"] in {"hub-direct", "other-room"}
