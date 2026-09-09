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
from scripts.runtime_activity_routes import RuntimeActivityFeeds, merge_gateway, router


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


def test_merge_gateway_joins_routes_to_gauges_by_upstream():
    admission = {
        "checked": 3,
        "upstreams": {
            "http://w:8013": {"inflight": 1, "waiting": 2, "max_inflight": 8},
            "http://w:8099": {"inflight": 0, "waiting": 0, "max_inflight": 8},
        },
    }
    routes = {
        "default_route": "quick",
        "routes": [
            {"id": "quick", "served_by": "fast-1", "upstream": "http://w:8013", "status": "up"},
            {"id": "quick_background", "served_by": "fast-1", "upstream": "http://w:8013", "status": "up", "priority": "background"},
            {"id": "spark", "served_by": None, "upstream": None, "status": "not_configured"},
        ],
    }
    merged = merge_gateway(admission, routes)
    by = {l["upstream"]: l for l in merged["lanes"]}
    assert [r["id"] for r in by["http://w:8013"]["routes"]] == ["quick", "quick_background"]
    assert by["http://w:8013"]["waiting"] == 2
    # A gauge the catalog does not name is still shown -- traffic is evidence.
    assert by["http://w:8099"]["routes"] == []
    assert merged["ledger"] == {"checked": 3}
    assert merged["default_route"] == "quick"
    # No catalog at all (first poll before /routes answered) still yields gauges.
    assert merge_gateway(admission, None)["lanes"][0]["routes"] == []


class _FakeResp:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"http {self.status}")

    async def json(self):
        return self._body


class _FakeSession:
    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url: str, params=None):
        self.calls.append(url)
        for suffix, resp in self._responses.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, {})


@pytest.mark.asyncio
async def test_poll_once_folds_admission_and_caches_routes(activity: RuntimeActivity):
    session = _FakeSession(
        {
            "/admission": _FakeResp(200, {"checked": 1, "upstreams": {"http://w:1": {"inflight": 1}}}),
            "/routes": _FakeResp(200, {"routes": [{"id": "chat", "upstream": "http://w:1"}]}),
        }
    )
    feeds = RuntimeActivityFeeds(
        activity=activity, gateway_url="http://gw/", poll_sec=0, timeout_sec=1, session_factory=lambda: session
    )
    await feeds.poll_once()
    await feeds.poll_once()
    g = activity.snapshot()["gateway"]
    assert g["error"] is None
    assert g["snapshot"]["lanes"][0]["routes"][0]["id"] == "chat"
    # /routes fetched once (cached ~60s), /admission every poll.
    assert session.calls.count("http://gw/admission") == 2
    assert session.calls.count("http://gw/routes") == 1


@pytest.mark.asyncio
async def test_poll_once_reports_a_down_gateway_instead_of_raising(activity: RuntimeActivity):
    class _Boom:
        async def __aenter__(self):
            raise ConnectionError("refused")

        async def __aexit__(self, *exc):
            return False

    feeds = RuntimeActivityFeeds(
        activity=activity, gateway_url="http://gw", poll_sec=0, timeout_sec=1, session_factory=lambda: _Boom()
    )
    await feeds.poll_once()
    assert activity.snapshot()["gateway"]["error"].startswith("ConnectionError")


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
        activity=activity, gateway_url="http://gw", poll_sec=0, timeout_sec=1, engine_factory=lambda: _Engine()
    )
    assert await feeds.backfill() == 1
    assert [r["run_id"] for r in activity.snapshot()["curiosity_runs"]] == ["live"]

    def _broken():
        raise RuntimeError("postgres down")

    feeds2 = RuntimeActivityFeeds(
        activity=activity, gateway_url="http://gw", poll_sec=0, timeout_sec=1, engine_factory=_broken
    )
    assert await feeds2.backfill() == 0


@pytest.mark.asyncio
async def test_start_with_poll_disabled_only_backfills(activity: RuntimeActivity):
    feeds = RuntimeActivityFeeds(activity=activity, gateway_url="http://gw", poll_sec=0, timeout_sec=1)
    await feeds.start()
    assert feeds._task is None
    await feeds.stop()
