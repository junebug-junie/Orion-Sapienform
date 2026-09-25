"""/api/runtime-activity serves the reducer; the feeds fail open."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.hub.runtime_activity import RuntimeActivity, reset_runtime_activity
from scripts import runtime_activity_routes as mod
from scripts.runtime_activity_routes import RuntimeActivityFeeds, pool_lanes, router


@pytest.fixture
def activity() -> RuntimeActivity:
    return reset_runtime_activity(RuntimeActivity(now=lambda: 1_000.0))


@pytest.fixture
def client(activity: RuntimeActivity, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(mod.settings, "HUB_RUNTIME_ACTIVITY_ENABLED", True, raising=False)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_snapshot_route_returns_the_reducer_state(client: TestClient, activity: RuntimeActivity):
    activity.run_dispatched(run_id="r1", correlation_id="c1", line="investigate")
    body = client.get("/api/runtime-activity").json()
    assert body["busy"] is True
    assert body["curiosity_runs"][0]["run_id"] == "r1"
    assert set(body["lanes"]) == {"chat", "agent"}


def test_routes_answer_503_when_disabled(activity: RuntimeActivity, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod.settings, "HUB_RUNTIME_ACTIVITY_ENABLED", False, raising=False)
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    assert c.get("/api/runtime-activity").status_code == 503
    assert c.get("/api/runtime-activity/stream").status_code == 503


@pytest.mark.asyncio
async def test_stream_sends_a_snapshot_on_connect_then_one_per_change(
    activity: RuntimeActivity, monkeypatch: pytest.MonkeyPatch
):
    """Drives the SSE generator directly: TestClient runs the app on another
    thread, and asyncio.Queue wake-ups do not cross threads, so a client-side
    test would only ever see the heartbeat, not the change."""
    monkeypatch.setattr(mod, "_COALESCE_SEC", 0.0)
    monkeypatch.setattr(mod, "_HEARTBEAT_SEC", 0.05)

    class _Req:
        disconnected = False

        async def is_disconnected(self):
            return self.disconnected

    req = _Req()
    activity.turn_requested(correlation_id="a", mode="orion", model_label=None, source="chat")
    gen = mod._stream(activity, req)  # type: ignore[arg-type]
    first = _parse_event(await gen.__anext__())
    assert first["event"] == "snapshot"
    assert first["data"]["lanes"]["chat"]["queued"][0]["correlation_id"] == "a"

    # Quiet -> a keepalive comment, never a duplicate snapshot.
    assert (await gen.__anext__()).startswith(":")

    activity.harness_step({"correlation_id": "a", "step_index": 0, "step": {}})
    activity.harness_step({"correlation_id": "a", "step_index": 1, "step": {}})
    second = _parse_event(await gen.__anext__())
    assert second["data"]["version"] == first["data"]["version"] + 2, "two folds coalesced into one frame"
    assert second["data"]["lanes"]["chat"]["running"][0]["step_count"] == 2

    req.disconnected = True
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    assert activity._subscribers == set(), "disconnect unsubscribes"


def _parse_event(frame: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in frame.splitlines():
        if line.startswith("event:"):
            out["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            out["data"] = json.loads(line.split(":", 1)[1].strip())
    assert "data" in out, frame
    return out


POOL_STATE = {
    "roles": [
        {"role": "fast", "url": "http://w:8013", "status": "confirmed", "slots": 4, "model_file": "q4.gguf"},
        {"role": "chat", "url": "http://w:8011", "status": "confirmed", "slots": 1, "model_file": "35b.gguf"},
        {"role": "agent-gpu2", "url": "http://w:8016", "status": "unloaded", "slots": 0},
    ],
    "leases": [
        {"status": "granted", "role": "fast", "work_class": "fast"},
        {"status": "queued", "role": None, "work_class": "fast"},
        {"status": "queued", "role": None, "work_class": "experiment"},
    ],
}


def test_pool_lanes_show_slots_in_use_waiting_and_a_5min_ledger():
    now = 1_000_000.0
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc).isoformat()
    events = [
        {"event": "admitted", "generated_at": iso(now - 10)},
        {"event": "granted", "waited_ms": 2400.0, "generated_at": iso(now - 9)},
        {"event": "granted", "waited_ms": 20.0, "generated_at": iso(now - 8)},
        {"event": "admitted", "generated_at": iso(now - 900)},              # outside the 5-min window
    ]
    out = pool_lanes(POOL_STATE, events, now_ts=now)
    by = {l["upstream"]: l for l in out["lanes"]}
    assert by["http://w:8013"]["inflight"] == 1 and by["http://w:8013"]["waiting"] == 1 and by["http://w:8013"]["max_inflight"] == 4
    assert by["http://w:8013"]["routes"][0]["served_by"] == "circe-worker-fast"
    assert "http://w:8016" not in by                                   # unloaded seat: nothing to show
    assert by["queue:experiment"]["waiting"] == 1                       # waiting with no home role still shown
    assert out["ledger"] == {"checked": 1, "queued": 2, "deferrals": 1, "longest_wait_s": 2.4}
    assert out["source"] == "gpu_pool"


@pytest.mark.asyncio
async def test_poll_once_reads_the_pool_feed(activity: RuntimeActivity):
    feed = type("Feed", (), {"state": POOL_STATE, "events": []})()
    feeds = RuntimeActivityFeeds(activity=activity, poll_sec=0, pool_feed=feed)
    await feeds.poll_once()
    g = activity.snapshot()["gateway"]
    assert g["error"] is None and g["snapshot"]["source"] == "gpu_pool"


@pytest.mark.asyncio
async def test_no_pool_state_is_reported_as_unavailable_not_idle(activity: RuntimeActivity):
    feeds = RuntimeActivityFeeds(activity=activity, poll_sec=0, pool_feed=type("Feed", (), {"state": None, "events": []})())
    await feeds.poll_once()
    assert activity.snapshot()["gateway"]["error"] == "gpu_pool_state_unavailable"


@pytest.mark.asyncio
async def test_backfill_adopts_active_rows_and_fails_open(activity: RuntimeActivity):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *_a, **_k):
            class _R:
                def mappings(self):
                    return self

                def all(self):
                    return [
                        {"run_id": "live", "status": "running", "node": "harness_turn", "correlation_id": "c",
                         "generated_at": "2026-09-09T01:00:00Z", "first_seen_at": "2026-09-09T00:00:00Z"},
                        {"run_id": "done", "status": "completed", "node": "finish", "correlation_id": "d"},
                    ]

            return _R()

    class _Engine:
        def connect(self):
            return _Conn()

    feeds = RuntimeActivityFeeds(
        activity=activity, poll_sec=0, engine_factory=lambda: _Engine()
    )
    assert await feeds.backfill() == 1
    assert [r["run_id"] for r in activity.snapshot()["curiosity_runs"]] == ["live"]

    def _broken():
        raise RuntimeError("postgres down")

    feeds2 = RuntimeActivityFeeds(
        activity=activity, poll_sec=0, engine_factory=_broken
    )
    assert await feeds2.backfill() == 0


@pytest.mark.asyncio
async def test_start_with_poll_disabled_only_backfills(activity: RuntimeActivity):
    feeds = RuntimeActivityFeeds(activity=activity, poll_sec=0)
    await feeds.start()
    assert feeds._task is None
    await feeds.stop()
