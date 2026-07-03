from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "social_room_inspection_cache.py"
SPEC = importlib.util.spec_from_file_location("hub_social_room_inspection_cache", CACHE_PATH)
assert SPEC and SPEC.loader
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)


def test_store_social_room_inspection_from_route_debug() -> None:
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
    cache._latest.clear()
    cache.store("hub-direct", {"verb": "chat_social_room", "corr": "a"})
    cache.store("other-room", {"verb": "chat_social_room", "corr": "b"})
    latest = cache.get_latest()
    assert latest is not None
    assert latest["room_id"] in {"hub-direct", "other-room"}
