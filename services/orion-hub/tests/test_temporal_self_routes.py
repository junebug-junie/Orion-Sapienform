"""Temporal Self day tab: Hub proxy routes (fake durable-runs upstream), page, nav/tab wiring."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import temporal_self_routes as mod

HUB = Path(__file__).resolve().parents[1]

FRAME = {"day_id": "2026-10-13", "day_phase": "night", "as_of": "2026-10-14T03:19:26Z",
         "active_arc": {"kind": "attention", "arc_id": "a1", "status": "open"},
         "open_threads": [{"arc_id": "c1", "subject_ref": "open-loop-x", "returns_today": 5}],
         "arcs_today_total": 2}
HEALTH = {"ok": True, "temporal_self": {"arousal_level": "engaged", "since": "2026-10-14T03:07:37Z",
                                        "last_error": None, "chronicle": {"day_id": "2026-10-13"},
                                        "immune": {"open": 1, "findings": [{"summary": "x"}]}}}


class Upstream:
    """Records calls; answers from a path -> (status, body) table, or raises a transport error."""

    def __init__(self, table=None, error: Exception | None = None):
        self.table = table or {}
        self.error = error
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        status, body = self.table.get(request.url.path, (404, {"detail": "not found"}))
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)


@pytest.fixture
def wire(monkeypatch):
    def _wire(upstream: Upstream, base: str = "http://durable.test:8124"):
        monkeypatch.setattr(mod.settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", base, raising=False)
        real = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(upstream.handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(mod.httpx, "AsyncClient", factory)
        app = FastAPI()
        app.include_router(mod.router)
        return TestClient(app)

    return _wire


def test_frame_proxies_verbatim_with_no_cache(wire):
    up = Upstream({"/temporal-self/frame": (200, FRAME)})
    resp = wire(up).get("/api/temporal-self/frame")
    assert resp.status_code == 200
    assert resp.json() == FRAME
    assert "no-store" in resp.headers["cache-control"]
    assert str(up.calls[0].url) == "http://durable.test:8124/temporal-self/frame"


def test_arcs_forwards_day_kind_limit_and_drops_absent_params(wire):
    up = Upstream({"/temporal-self/arcs": (200, {"day_id": "2026-10-12", "count": 0, "arcs": []})})
    client = wire(up)
    assert client.get("/api/temporal-self/arcs?day_id=2026-10-12&kind=attention&limit=50").status_code == 200
    assert dict(up.calls[-1].url.params) == {"day_id": "2026-10-12", "kind": "attention", "limit": "50"}
    assert client.get("/api/temporal-self/arcs").status_code == 200
    assert dict(up.calls[-1].url.params) == {"limit": "2000"}


@pytest.mark.parametrize("url", ["/api/temporal-self/day/2026-1-2", "/api/temporal-self/day/..%2Fhealth",
                                 "/api/temporal-self/arcs?day_id=yesterday", "/api/temporal-self/arcs?kind=A;B"])
def test_bad_input_is_400_and_never_reaches_upstream(wire, url):
    up = Upstream()
    resp = wire(up).get(url)
    assert resp.status_code in (400, 404)
    assert up.calls == []


def test_closed_day_proxies_and_open_day_404_passes_through(wire):
    day = {"day_id": "2026-10-12", "closed_at": "2026-10-13T06:00:00Z", "frame": FRAME, "arcs": []}
    up = Upstream({"/temporal-self/day/2026-10-12": (200, day),
                   "/temporal-self/day/2026-10-13": (404, {"detail": "day 2026-10-13 is not closed (or not chronicled)"})})
    client = wire(up)
    assert client.get("/api/temporal-self/day/2026-10-12").json()["day_id"] == "2026-10-12"
    resp = client.get("/api/temporal-self/day/2026-10-13")
    assert resp.status_code == 404
    assert "not closed" in resp.json()["detail"]


def test_upstream_503_passes_through_with_its_detail(wire):
    up = Upstream({"/temporal-self/threads": (503, {"detail": "temporal self tables missing: apply x.sql"})})
    resp = wire(up).get("/api/temporal-self/threads")
    assert resp.status_code == 503
    assert "tables missing" in resp.json()["detail"]


def test_upstream_down_is_502_not_500(wire):
    up = Upstream(error=httpx.ConnectError("refused"))
    resp = wire(up).get("/api/temporal-self/regulation")
    assert resp.status_code == 502
    assert resp.json()["detail"].startswith("durable_runs_unavailable: ConnectError")


def test_upstream_timeout_is_502(wire):
    up = Upstream(error=httpx.ReadTimeout("slow"))
    resp = wire(up).get("/api/temporal-self/cursors")
    assert resp.status_code == 502
    assert "ReadTimeout" in resp.json()["detail"]


def test_upstream_500_and_bad_json_become_502(wire):
    up = Upstream({"/temporal-self/cursors": (500, {"detail": "boom"}), "/regulation/state": (200, "<html>")})
    client = wire(up)
    r1 = client.get("/api/temporal-self/cursors")
    assert r1.status_code == 502 and "durable_runs_error 500" in r1.json()["detail"]
    r2 = client.get("/api/temporal-self/regulation")
    assert r2.status_code == 502 and "bad_json" in r2.json()["detail"]


def test_unconfigured_base_url_is_503(wire):
    resp = wire(Upstream(), base="").get("/api/temporal-self/frame")
    assert resp.status_code == 503
    assert "HUB_CURIOSITY_DURABLE_RUNS_URL" in resp.json()["detail"]


def test_regulation_proxies_regulation_state(wire):
    up = Upstream({"/regulation/state": (200, {"arousal": {"arousal_level": "calm"}, "drives": []})})
    assert wire(up).get("/api/temporal-self/regulation").json()["arousal"]["arousal_level"] == "calm"
    assert up.calls[0].url.path == "/regulation/state"


def test_immune_extracts_temporal_self_block_from_health(wire):
    up = Upstream({"/health": (200, HEALTH)})
    body = wire(up).get("/api/temporal-self/immune").json()
    assert body["immune"]["open"] == 1
    assert body["arousal_level"] == "engaged"
    assert body["chronicle"] == {"day_id": "2026-10-13"}


def test_immune_without_temporal_self_block_is_503(wire):
    up = Upstream({"/health": (200, {"ok": True})})
    assert wire(up).get("/api/temporal-self/immune").status_code == 503


def test_page_renders_template_with_cache_busted_script(monkeypatch):
    from scripts import main as hub_main

    monkeypatch.setattr(hub_main, "build_hub_ui_asset_version", lambda: "testver")
    app = FastAPI()
    app.include_router(mod.page_router)
    resp = TestClient(app).get("/temporal-self")
    assert resp.status_code == 200
    assert '<script src="/static/js/temporal_self.js?v=testver"></script>' in resp.text
    assert "{{HUB_UI_ASSET_VERSION}}" not in resp.text


def test_main_registers_routers():
    src = (HUB / "scripts" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(temporal_self_router)" in src
    assert "app.include_router(temporal_self_page_router)" in src


def test_nav_tab_panel_and_scripts_are_wired():
    index = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="temporalSelfTabButton"' in index and 'data-hash-target="#temporal-self"' in index
    assert '<section id="temporal-self" data-panel="temporal-self"' in index
    assert 'id="temporalSelfPanelFrame" src="/temporal-self"' in index
    assert '<script src="/static/js/temporal_self_tab.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in index
    tab_js = (HUB / "static" / "js" / "temporal_self_tab.js").read_text(encoding="utf-8")
    for needle in ('"#temporal-self"', "temporalSelfTabButton", "temporalSelfPanelFrame",
                   "temporalSelfPanelRefresh", "function activatePanel()", "function deactivatePanel()"):
        assert needle in tab_js


def test_page_js_fetches_every_proxy_route_and_page_has_its_ids():
    page = (HUB / "templates" / "temporal_self.html").read_text(encoding="utf-8")
    js = (HUB / "static" / "js" / "temporal_self.js").read_text(encoding="utf-8")
    for route in ("/api/temporal-self/frame", "/api/temporal-self/arcs?day_id=", "/api/temporal-self/day/",
                  "/api/temporal-self/regulation", "/api/temporal-self/immune"):
        assert route in js
    for el_id in ("tsDayPicker", "tsPrevDay", "tsNextDay", "tsToday", "tsRefresh", "tsTimeline", "tsThreads",
                  "tsActiveArc", "tsArousal", "tsRest", "tsImmuneCount", "tsImmuneFindings", "tsDayPhase", "tsStatus"):
        assert f'id="{el_id}"' in page, el_id
        assert f'"{el_id}"' in js, el_id
    # Render-only: the page never writes.
    assert "method:" not in js and "POST" not in js
