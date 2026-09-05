"""Tests for orion/substrate/chat_stance_belief_bus.py (self-model rebuild
arc, 2026-09-05): the durable publish path for chat_stance's real per-turn
belief computation, previously discarded after every turn."""
from __future__ import annotations

import json

from orion.substrate import chat_stance_belief_bus as bus_module


class _FakeAnchorSlice:
    def __init__(self, degraded: bool) -> None:
        self.degraded = degraded


class _FakeRedisClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    def publish(self, channel: str, data: bytes) -> None:
        self.published.append((channel, data))


class TestNormalizeShiftKind:
    """Review finding (2026-09-05): every current shift_kind producer
    uppercases it (consolidation_gate.py, recall_skip_gate.py,
    retrieval_intent.py) -- this schema's Literal does not tolerate any
    other casing, and a mismatch would otherwise silently drop the whole
    log row via the outer best-effort except."""

    def test_none_and_empty_stay_none(self):
        assert bus_module._normalize_shift_kind(None) is None
        assert bus_module._normalize_shift_kind("") is None

    def test_lowercase_is_normalized(self):
        assert bus_module._normalize_shift_kind("repair") == "REPAIR"

    def test_already_uppercase_passes_through(self):
        assert bus_module._normalize_shift_kind("TOPIC") == "TOPIC"

    def test_unknown_value_becomes_none_not_raised(self):
        assert bus_module._normalize_shift_kind("something_unexpected") is None


def test_build_anchor_summary_none_when_no_anchors():
    assert bus_module.build_anchor_summary(None) is None
    assert bus_module.build_anchor_summary({}) is None


def test_build_anchor_summary_real_content_marks_degraded():
    anchors = {"orion": _FakeAnchorSlice(False), "relationship": _FakeAnchorSlice(True)}

    summary = bus_module.build_anchor_summary(anchors)

    assert summary is not None
    assert "orion" in summary
    assert "relationship(degraded)" in summary


def test_publish_skips_when_bus_disabled(monkeypatch):
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(bus_module, "_sync_redis", lambda: fake_client)

    bus_module.publish_chat_stance_belief_log_sync(
        anchors={"orion": _FakeAnchorSlice(False)},
        degraded_producers=[],
        lineage={},
        ctx={"correlation_id": "corr-1", "session_id": "sess-1"},
    )

    assert fake_client.published == []


def test_publish_skips_quietly_when_no_redis(monkeypatch):
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    monkeypatch.setattr(bus_module, "_sync_redis", lambda: None)

    # Must not raise even with no client available.
    bus_module.publish_chat_stance_belief_log_sync(
        anchors={"orion": _FakeAnchorSlice(False)},
        degraded_producers=["producer_x"],
        lineage={"orion": "producer_x"},
        ctx={"correlation_id": "corr-1", "session_id": "sess-1"},
    )


def test_publish_real_content_reaches_the_client(monkeypatch):
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(bus_module, "_sync_redis", lambda: fake_client)

    bus_module.publish_chat_stance_belief_log_sync(
        anchors={"orion": _FakeAnchorSlice(False), "juniper": _FakeAnchorSlice(True)},
        degraded_producers=["producer_x", "producer_x"],
        lineage={"orion": "producer_x"},
        shift_kind="repair",  # lowercase on purpose -- proves end-to-end normalization
        ctx={"correlation_id": "corr-1", "session_id": "sess-1"},
    )

    assert len(fake_client.published) == 1
    channel, data = fake_client.published[0]
    assert channel == bus_module.CHANNEL_CHAT_STANCE_BELIEF_WRITE
    decoded = json.loads(data.decode("utf-8"))
    payload = decoded["payload"]
    assert payload["shift_kind"] == "REPAIR"
    assert payload["degraded_producers"] == ["producer_x"]
    assert "juniper(degraded)" in payload["anchor_summary"]
    assert payload["correlation_id"] == "corr-1"
    assert payload["session_id"] == "sess-1"


def test_sync_redis_does_not_retry_connect_within_cooldown_after_failure(monkeypatch):
    """Review finding (2026-09-05): this module is called on essentially
    every real chat turn now, unlike its precedent (tier_outcomes_bus.py),
    whose caller only publishes on the rare cold-anchor path. Without a
    cooldown, a down Redis would attempt a fresh blocking connect on every
    single turn."""
    monkeypatch.setattr(bus_module, "_redis_client", None)
    monkeypatch.setattr(bus_module, "_last_connect_failure_monotonic", 0.0)
    monkeypatch.setenv("ORION_BUS_URL", "redis://example-unreachable:6379/0")

    connect_attempts = {"count": 0}

    class _FailingRedisModule:
        class Redis:
            @staticmethod
            def from_url(*args, **kwargs):
                connect_attempts["count"] += 1
                raise ConnectionError("simulated unreachable redis")

    monkeypatch.setattr(bus_module, "redis", _FailingRedisModule)
    monkeypatch.setattr(bus_module.time, "monotonic", lambda: 100.0)

    first = bus_module._sync_redis()
    assert first is None
    assert connect_attempts["count"] == 1

    # Still inside the cooldown window -- must fast-fail, not reconnect.
    monkeypatch.setattr(bus_module.time, "monotonic", lambda: 100.0 + bus_module._CONNECT_FAILURE_COOLDOWN_SEC - 1)
    second = bus_module._sync_redis()
    assert second is None
    assert connect_attempts["count"] == 1

    # Past the cooldown window -- retries once more.
    monkeypatch.setattr(bus_module.time, "monotonic", lambda: 100.0 + bus_module._CONNECT_FAILURE_COOLDOWN_SEC + 1)
    third = bus_module._sync_redis()
    assert third is None
    assert connect_attempts["count"] == 2


def test_publish_never_raises_on_client_failure(monkeypatch):
    class _RaisingClient:
        def publish(self, channel, data):
            raise RuntimeError("connection reset")

    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    monkeypatch.setattr(bus_module, "_sync_redis", lambda: _RaisingClient())
    monkeypatch.setattr(bus_module, "_reset_redis", lambda: None)

    # Must not raise.
    bus_module.publish_chat_stance_belief_log_sync(
        anchors=None,
        degraded_producers=None,
        lineage=None,
        ctx={},
    )
