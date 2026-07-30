from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app import world_context
from app.world_context import fetch_latest_world_context_capsule, filter_world_context_capsule


def test_filter_world_context_capsule_fail_open_missing():
    capsule, diag = filter_world_context_capsule(
        None,
        min_confidence=0.65,
        max_topics=5,
        max_age_hours=36,
        politics_default="only_when_requested",
    )
    assert capsule is None
    assert diag["capsule_filtered_reason"] == "missing_capsule"


def test_filter_world_context_capsule_filters_expired_and_low_confidence():
    now = datetime.now(timezone.utc)
    capsule, diag = filter_world_context_capsule(
        {
            "generated_at": now.isoformat(),
            "salient_topics": [
                {"topic": "expired", "confidence": 0.9, "expires_at": (now - timedelta(hours=1)).isoformat()},
                {"topic": "low", "confidence": 0.1},
                {"topic": "good", "confidence": 0.9},
            ],
        },
        min_confidence=0.65,
        max_topics=5,
        max_age_hours=36,
        politics_default="only_when_requested",
    )
    assert capsule is not None
    assert len(capsule["salient_topics"]) == 1
    assert capsule["salient_topics"][0]["topic"] == "good"
    assert diag["stance_world_context_items_used"] == 1


def test_fetch_latest_world_context_capsule_fails_open_on_unreachable_host(monkeypatch):
    world_context._CAPSULE_CACHE.clear()
    capsule = fetch_latest_world_context_capsule(
        base_url="http://127.0.0.1:1",
        timeout_seconds=0.2,
        cache_ttl_seconds=300,
    )
    assert capsule is None


def test_fetch_latest_world_context_capsule_returns_capsule_from_payload(monkeypatch):
    world_context._CAPSULE_CACHE.clear()

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"capsule": {"salient_topics": [{"topic": "x", "confidence": 0.9}]}}'

    monkeypatch.setattr(world_context, "urlopen", lambda *a, **k: _FakeResp())
    capsule = fetch_latest_world_context_capsule(
        base_url="http://fake-world-pulse:8628",
        timeout_seconds=2.0,
        cache_ttl_seconds=300,
    )
    assert capsule == {"salient_topics": [{"topic": "x", "confidence": 0.9}]}


def test_fetch_latest_world_context_capsule_uses_cache(monkeypatch):
    world_context._CAPSULE_CACHE.clear()
    calls = {"count": 0}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            calls["count"] += 1
            return b'{"capsule": {"salient_topics": []}}'

    monkeypatch.setattr(world_context, "urlopen", lambda *a, **k: _FakeResp())
    fetch_latest_world_context_capsule(base_url="http://fake:8628", timeout_seconds=2.0, cache_ttl_seconds=300)
    fetch_latest_world_context_capsule(base_url="http://fake:8628", timeout_seconds=2.0, cache_ttl_seconds=300)
    assert calls["count"] == 1
