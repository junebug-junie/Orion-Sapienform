from __future__ import annotations

from app.world_context_capsule_cache import WorldContextCapsuleCache


def test_world_context_capsule_cache_starts_empty():
    cache = WorldContextCapsuleCache()
    assert cache.get() is None


def test_world_context_capsule_cache_set_then_get():
    cache = WorldContextCapsuleCache()
    cache.set({"salient_topics": [{"topic": "x"}]})
    assert cache.get() == {"salient_topics": [{"topic": "x"}]}
