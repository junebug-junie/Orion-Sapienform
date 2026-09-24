"""GPU pool operator panel: page + nav wiring, bus RPC shape, live feed/SSE, operator guard, history SQL."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.core.bus.codec import OrionCodec
from orion.schemas.gpu_pool import (
    GPU_POOL_CONTROL_REQUEST_CHANNEL, GPU_POOL_EVENT_CHANNEL, GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_REQUEST_CHANNEL,
)
from scripts import gpu_pool_routes as mod

HUB = Path(__file__).resolve().parents[1]


class RpcBus:
    """Behaves like the real transport: the responder answers only to envelope.reply_to."""

    enabled = True

    def __init__(self, reply_payload):
        self.codec = OrionCodec()
        self.reply_payload = reply_payload
        self.calls = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec=60.0, health_label=None):
        assert envelope.reply_to == reply_channel, "no reply_to -> the pool never answers"
        self.calls.append((channel, envelope.payload, health_label))
        reply = envelope.model_copy(update={"payload": self.reply_payload})
        return {"type": "message", "channel": reply_channel, "data": self.codec.encode(reply)}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mod.settings, "HUB_GPU_POOL_ENABLED", True, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    app.include_router(mod.page_router)
    return TestClient(app)


def _use_bus(monkeypatch, bus):
    monkeypatch.setattr(mod, "_rpc_bus", lambda: bus)


def test_nav_tab_panel_and_script_are_wired():
    index = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="gpuPoolTabButton"' in index and 'data-hash-target="#gpu-pool"' in index
    assert '<section id="gpu-pool" data-panel="gpu-pool"' in index
    assert 'id="gpuPoolPanelFrame" src="/gpu-pool"' in index
    assert '/static/js/gpu_pool_tab.js?v={{HUB_UI_ASSET_VERSION}}' in index
    page = (HUB / "templates" / "gpu_pool.html").read_text(encoding="utf-8")
    assert '/static/js/gpu_pool.js?v={{HUB_UI_ASSET_VERSION}}' in page
    for element_id in ("cards", "yaml", "lendControls", "byRole", "byClass", "series", "events", "walker", "walkSteps"):
        assert f'id="{element_id}"' in page, element_id


def test_page_renders_with_cache_bust(client, monkeypatch):
    # The route borrows TEMPLATES_DIR + the asset-version helper from scripts.main (like curiosity's);
    # stand those two in rather than importing the whole Hub app.
    import sys
    fake_main = SimpleNamespace(TEMPLATES_DIR=HUB / "templates", build_hub_ui_asset_version=lambda: "t123")
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    res = client.get("/gpu-pool")
    assert res.status_code == 200 and "{{HUB_UI_ASSET_VERSION}}" not in res.text and "gpu_pool.js?v=" in res.text


def test_state_route_asks_the_pool_over_the_bus(client, monkeypatch):
    bus = RpcBus({"config_digest": "abc", "cards": [], "roles": [], "history": [{"event": "admit"}]})
    _use_bus(monkeypatch, bus)
    body = client.get("/api/gpu-pool/state?config=true&history_for=L1").json()
    assert body["config_digest"] == "abc"
    [(channel, payload, label)] = bus.calls
    assert channel == GPU_POOL_STATE_REQUEST_CHANNEL and label == "gpu_pool_panel"
    assert payload["include_config"] is True and payload["history_for"] == "L1"


HUB_PAGE = {"x-requested-with": "orion-hub"}


def test_control_requires_the_operator_cookie_then_the_pool_token(client, monkeypatch):
    bus = RpcBus({"ok": True, "reason": None, "detail": {"card": "gpu0", "lent": True}})
    _use_bus(monkeypatch, bus)
    monkeypatch.delenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", raising=False)
    assert client.post("/api/gpu-pool/control", json={"verb": "lend", "card": "gpu0"}, headers=HUB_PAGE).status_code == 503

    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "hub-op")
    assert client.post("/api/gpu-pool/control", json={"verb": "lend", "card": "gpu0"},
                       headers={**HUB_PAGE, "x-orion-operator-token": "wrong"}).status_code == 403

    monkeypatch.setattr(mod.settings, "GPU_POOL_OPERATOR_TOKEN", "", raising=False)
    res = client.post("/api/gpu-pool/control", json={"verb": "lend", "card": "gpu0"}, cookies={"orion_operator_token": "hub-op"},
                      headers=HUB_PAGE)
    assert res.status_code == 503 and res.json()["detail"] == "gpu_pool_operator_token_not_configured"

    monkeypatch.setattr(mod.settings, "GPU_POOL_OPERATOR_TOKEN", "pool-secret", raising=False)
    res = client.post("/api/gpu-pool/control", json={"verb": "lend", "card": "gpu0"}, cookies={"orion_operator_token": "hub-op"},
                      headers=HUB_PAGE)
    assert res.status_code == 200 and res.json()["ok"] is True
    [(channel, payload, _)] = bus.calls
    assert channel == GPU_POOL_CONTROL_REQUEST_CHANNEL
    assert payload["verb"] == "lend" and payload["actor"] == "hub-operator"
    assert "pool-secret" not in json.dumps(payload) and "pool-secret" not in res.text   # never on the bus
    from orion.gpu_pool.control_auth import NonceLedger, verify
    from orion.schemas.gpu_pool import GpuPoolControlV1
    ctl = GpuPoolControlV1.model_validate(payload)
    assert verify(ctl, "pool-secret", ctl.issued_at, NonceLedger()) is None           # the pool accepts it


def test_control_refuses_cross_site_shaped_requests(client, monkeypatch):
    """No custom header, or a typeless body: what a cross-site no-cors fetch can send. Refused before auth."""
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "hub-op")
    body = json.dumps({"verb": "hold", "work_class": "experiment"})
    no_header = client.post("/api/gpu-pool/control", content=body, cookies={"orion_operator_token": "hub-op"},
                            headers={"content-type": "application/json"})
    typeless = client.post("/api/gpu-pool/control", content=body, cookies={"orion_operator_token": "hub-op"},
                           headers={**HUB_PAGE, "content-type": ""})
    assert no_header.status_code == 403 and typeless.status_code == 403


def test_control_rejects_unknown_verbs(client, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "hub-op")
    assert client.post("/api/gpu-pool/control", json={"verb": "rm -rf"}, cookies={"orion_operator_token": "hub-op"},
                       headers=HUB_PAGE).status_code == 422


def test_state_route_sends_only_non_default_fields_and_caps_lease_id(client, monkeypatch):
    bus = RpcBus({"config_digest": "abc", "cards": [], "roles": []})
    _use_bus(monkeypatch, bus)
    client.get("/api/gpu-pool/state")
    assert bus.calls[0][1] == {}                              # an older pool accepts this unchanged
    assert client.get("/api/gpu-pool/state?history_for=" + "x" * 200).status_code == 422


def test_disabled_panel_answers_404(monkeypatch):
    monkeypatch.setattr(mod.settings, "HUB_GPU_POOL_ENABLED", False, raising=False)
    app = FastAPI()
    app.include_router(mod.router)
    assert TestClient(app).get("/api/gpu-pool/history").status_code == 404


def test_feed_keeps_latest_state_and_recent_events_and_fans_out():
    feed = mod.GpuPoolFeed(max_events=3)
    q = feed.subscribe()
    feed.absorb(GPU_POOL_STATE_CHANNEL, {"mode": "observe"})
    for i in range(5):
        feed.absorb(GPU_POOL_EVENT_CHANNEL, {"event": "granted", "lease_id": str(i)})
    snap = feed.snapshot()
    assert snap["state"] == {"mode": "observe"} and [e["lease_id"] for e in snap["events"]] == ["2", "3", "4"]
    assert q.get_nowait()["kind"] == "state" and q.get_nowait()["event"]["lease_id"] == "0"
    feed.unsubscribe(q)
    feed.absorb(GPU_POOL_EVENT_CHANNEL, {"event": "x"})
    assert q.qsize() == 4


def test_stream_sends_a_snapshot_then_live_updates():
    async def go():
        feed = mod.GpuPoolFeed()
        feed.absorb(GPU_POOL_STATE_CHANNEL, {"mode": "observe"})

        class Req:
            async def is_disconnected(self):
                return False

        gen = mod._stream(Req(), feed)
        first = await gen.__anext__()
        assert first.startswith("event: snapshot") and '"mode": "observe"' in first
        feed.absorb(GPU_POOL_EVENT_CHANNEL, {"event": "granted", "lease_id": "z"})
        second = await gen.__anext__()
        assert second.startswith("event: event") and '"lease_id": "z"' in second
        await gen.aclose()
    asyncio.run(go())


PG = os.environ.get("GPU_POOL_TEST_POSTGRES_URI")


@pytest.mark.skipif(not PG, reason="GPU_POOL_TEST_POSTGRES_URI not set")
def test_history_sql_on_postgres():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG.replace("postgresql://", "postgresql+psycopg2://", 1) if "+" not in PG else PG)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS gpu_pool_events"))
        conn.execute(text("""CREATE TABLE gpu_pool_events (event_id text primary key, created_at timestamptz default now(),
            generated_at timestamptz not null, event text not null, lease_id text, holder text, work_class text,
            priority text, role text, cards json, turn_correlation_id text, attempt int, waited_ms float,
            held_ms float, reason text, detail json)"""))
        rows = [("e1", now - timedelta(minutes=5), "granted", "L1", "gw", "metacog", "system", "metacog", 10.0, None, None),
                ("e2", now - timedelta(minutes=4), "released", "L1", "gw", "metacog", "system", "metacog", None, 500.0, "ok"),
                ("e3", now - timedelta(minutes=3), "granted", "L2", "gw", "metacog", "system", "fast", 30.0, None, None),
                ("e4", now - timedelta(minutes=2), "expired", "L2", "gw", "metacog", "system", "fast", None, 30000.0, "heartbeat_lost"),
                ("e5", now - timedelta(days=2), "granted", "L0", "gw", "metacog", "system", "metacog", 99.0, None, None)]
        for r in rows:
            conn.execute(text("INSERT INTO gpu_pool_events (event_id, generated_at, event, lease_id, holder, work_class, "
                              "priority, role, waited_ms, held_ms, reason) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k)"),
                         dict(zip("abcdefghijk", r)))
    with engine.connect() as conn:
        out = mod.history_payload(conn, minutes=60, work_class=None, holder=None, lease_id=None, limit=50)
    roles = {r["role"]: r for r in out["by_role"]}
    assert roles["metacog"]["grants"] == 1 and roles["metacog"]["released_ok"] == 1
    assert roles["fast"]["failures"] == 1 and roles["fast"]["wait_p50_ms"] == 30.0
    assert [e["event_id"] for e in out["events"]] == ["e4", "e3", "e2", "e1"]          # 2-day-old row excluded
    assert sum(s["grants"] for s in out["series"]) == 2 and out["by_class"][0]["grants"] == 2
    json.dumps(out, default=str)
