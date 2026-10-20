"""node:substrate.street in the runtime: listener -> gate -> node write, the
every-tick calm write, the camera/listener absent rule, Rung 6 recording, and
the /health counters (spec 2026-10-20 street continuity, section 6)."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.core.bus import resilience
from orion.substrate.street_attention import STREET_HANDOFF_DOMAIN, STREET_NODE_ID

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _settings(**kw):
    base = dict(
        street_attention_enabled=True,
        street_events_channel="orion:vision:street:events",
        street_attention_max_per_hour=4,
        street_attention_hold_sec=90.0,
        street_attention_tick_interval_sec=30.0,
        street_camera_up_max_age_sec=600.0,
        perception_walkway_stream_ids="walkway",
        perception_log_only_stream_ids="doorbell",
        prediction_raw_readings_enabled=True,
        prediction_raw_reading_retention_hours=192.0,
        prediction_expectation_live_enabled=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _worker(**kw):
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = _settings(**kw)
    w.raw = []
    w._store = SimpleNamespace(
        save_prediction_raw_reading=lambda **r: w.raw.append(r),
        prune_prediction_raw_readings=lambda **r: 0,
    )
    w.writes = []
    w._write_prediction_error_node = lambda **node: w.writes.append(node)
    return w


def _payload(kind="doorbell_ring", *, p=None, streams=("doorbell",), salience=None, corr="9f1c0e7a-0000-4000-8000-000000000001", eid="ev-1"):
    return {
        "event_id": eid,
        "kind": kind,
        "stream_ids": list(streams),
        "occurred_at": NOW.isoformat(),
        "subject_kind": "person",
        "salience": salience if salience is not None else (1.0 if kind == "doorbell_ring" else 0.4),
        "p_value": p,
        "evidence_refs": [],
        "correlation_id": corr,
    }


@pytest.fixture
def listener_up(monkeypatch):
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    resilience._SUBSCRIPTION_STATES["substrate_street_events_listener"] = resilience.SubscriptionState(
        name="substrate_street_events_listener", channel="orion:vision:street:events", subscribed=True
    )


def test_a_ring_passes_writes_the_node_and_logs_the_verdict(caplog) -> None:
    w = _worker()
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        w._process_street_event(_payload(), now=NOW)
    (write,) = w.writes
    assert write["node_id"] == STREET_NODE_ID and write["error"] == 1.0
    assert write["reducer_key"] == "street_attention"
    assert write["contributing_id"] == "9f1c0e7a-0000-4000-8000-000000000001"  # retina -> attention trace
    assert write["evidence_event_ids"] == ["street_event:ev-1"]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("substrate_street_event "))
    assert "kind=doorbell_ring" in line and "verdict=pass" in line
    assert "correlation_id=9f1c0e7a-0000-4000-8000-000000000001" in line and "salience=1.000" in line
    assert w.raw == []  # a ring is not a graded handoff


def test_a_high_p_handoff_is_blocked_but_still_records_its_rung6_reading(caplog) -> None:
    w = _worker()
    w._street_state().frame_seen("doorbell", NOW.timestamp())
    w._street_state().frame_seen("walkway", NOW.timestamp())
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        w._process_street_event(_payload("handoff_met", p=0.5, streams=("doorbell", "walkway")), now=NOW)
    assert w.writes == []
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("substrate_street_event "))
    assert "verdict=block" in line and "p_value 0.5 >= 0.05" in line
    (row,) = w.raw
    assert row["domain"] == STREET_HANDOFF_DOMAIN
    assert row["raw_value"] == pytest.approx(0.693147, abs=1e-5)
    assert row["observed_at"] == NOW  # occurred_at: a re-delivered event hits the PK, not a 2nd row


def test_camera_down_blocks_a_significant_handoff() -> None:
    w = _worker()
    w._street_state().frame_seen("walkway", NOW.timestamp())
    w._process_street_event(_payload("handoff_missed", p=0.001, streams=("doorbell", "walkway")), now=NOW)
    assert w.writes == []
    assert w._street_state().blocked_by_reason == {"camera_unseen": 1}


def test_the_cap_holds_in_the_runtime() -> None:
    w = _worker(street_attention_max_per_hour=2)
    w._street_state().frame_seen("walkway", NOW.timestamp())
    for i in range(4):
        w._process_street_event(
            _payload("handoff_missed", p=0.001, streams=("walkway",), eid=f"e{i}"), now=NOW + timedelta(seconds=i)
        )
    assert len(w.writes) == 2
    assert w._street_state().blocked_by_reason == {"hourly_cap": 2}


def test_an_invalid_payload_is_counted_and_never_raises() -> None:
    w = _worker()
    w._process_street_event({"kind": "doorbell_ring", "extra": 1}, now=NOW)
    assert w.writes == [] and w._street_state().invalid_total == 1


def test_tick_writes_a_real_zero_when_calm_and_the_street_is_visible(listener_up) -> None:
    w = _worker()
    w._street_state().frame_seen("walkway", NOW.timestamp() - 10)
    w._street_attention_tick(now=NOW)
    (write,) = w.writes
    assert write["error"] == 0.0 and write["evidence_event_ids"] == []


def test_the_node_returns_to_calm_after_the_hold(listener_up) -> None:
    w = _worker()
    w._street_state().frame_seen("walkway", NOW.timestamp())
    w._process_street_event(_payload(), now=NOW)
    w._street_attention_tick(now=NOW + timedelta(seconds=60))
    w._street_attention_tick(now=NOW + timedelta(seconds=91))
    assert [x["error"] for x in w.writes] == [1.0, 1.0, 0.0]


def test_tick_writes_nothing_while_the_listener_is_down(monkeypatch) -> None:
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    resilience._SUBSCRIPTION_STATES["substrate_street_events_listener"] = resilience.SubscriptionState(
        name="substrate_street_events_listener", channel="c", subscribed=False
    )
    w = _worker()
    w._street_state().frame_seen("walkway", NOW.timestamp())
    w._street_attention_tick(now=NOW)
    assert w.writes == [] and w._street_state().ticks_unmeasured_total == 1


def test_tick_writes_nothing_while_every_street_camera_is_dark(listener_up) -> None:
    w = _worker()
    w._street_state().frame_seen("walkway", NOW.timestamp() - 601)
    w._street_attention_tick(now=NOW)
    assert w.writes == []


def test_a_held_ring_is_still_written_with_the_cameras_dark(listener_up) -> None:
    w = _worker()
    w._process_street_event(_payload(), now=NOW)
    w._street_attention_tick(now=NOW + timedelta(seconds=30))
    assert [x["error"] for x in w.writes] == [1.0, 1.0]


def test_perception_artifacts_mark_the_camera_up() -> None:
    w = _worker()
    w._bus = SimpleNamespace(
        codec=SimpleNamespace(
            decode=lambda data: SimpleNamespace(
                ok=True, envelope=SimpleNamespace(payload={"inputs": {"stream_id": "walkway"}, "outputs": {}})
            )
        )
    )
    w._handle_perception_prediction_error_message({"data": b"x"})
    assert "walkway" in w._street_state().last_frame_at


def test_health_reports_street_counters(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("NODE_CATALOG_PATH", str(repo / "config" / "biometrics" / "node_catalog.yaml"))
    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main

    monkeypatch.setattr(main, "build_substrate_grammar_truth", lambda store: {"ok": True, "degraded": False})
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    main.worker._street_state().judge(
        SimpleNamespace(kind="doorbell_ring", p_value=None, stream_ids=["doorbell"], salience=1.0, event_id="e", correlation_id="c", person_guess=None),
        now_epoch=time.time(),
        max_per_hour=4,
        camera_up_max_age_sec=600.0,
    )
    out = asyncio.run(main.health())
    street = out["street_attention"]
    assert street["enabled"] is True and street["max_per_hour"] == 4
    assert street["passed_total"] >= 1 and street["node_id"] == STREET_NODE_ID
    assert street["held_value"] == 1.0


def test_listener_resubscribes_after_a_drop(monkeypatch) -> None:
    """The street listener rides run_subscription (the hardened pattern from
    PRs #2779/#2784): one Redis error must not end it for good."""
    w = _worker()
    w._stop = asyncio.Event()
    handled: list = []

    class _PubSub:
        def __init__(self, script):
            self.script = list(script)

        async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
            if not self.script:
                w._stop.set()
                return None
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    sessions = [_PubSub([RuntimeError("Timeout reading from socket")]), _PubSub([{"type": "message", "data": b"x"}])]
    calls = {"n": 0}

    class _Bus:
        @contextlib.asynccontextmanager
        async def subscribe(self, channel):
            calls["n"] += 1
            yield sessions.pop(0)

    w._bus = _Bus()

    async def fake_handle(msg):
        handled.append(msg)

    w._handle_street_event_message = fake_handle
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    asyncio.run(asyncio.wait_for(w._street_events_listener_loop(), timeout=10))
    assert calls["n"] == 2 and handled == [{"type": "message", "data": b"x"}]
