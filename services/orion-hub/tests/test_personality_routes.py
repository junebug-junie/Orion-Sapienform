"""Personality tab (rev 4 move 4): Hub proxy routes (fake durable-runs upstream), Yes/No write, page, nav/tab wiring."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import personality_routes as mod

HUB = Path(__file__).resolve().parents[1]


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


NOW = datetime.now(timezone.utc)
NODE = {"disposition_id": "d1", "label": "the gap between what I predicted and what arrived",
        "origin": "recurrence", "status": "active", "weight": 2.4, "evidence_count": 7,
        "lane_counts": {"curiosity": 4, "prime": 3}, "chosen_count": 1,
        "first_seen_at": _iso(NOW - timedelta(hours=6)), "last_reinforced_at": _iso(NOW - timedelta(minutes=5))}
FRAME = {"schema_version": "personality.frame.v1", "generated_at": _iso(NOW - timedelta(minutes=2)),
         "dispositions": [NODE], "choice_style": {"window_days": 7, "choices": 5, "took_top": 2, "went_own_way": 3},
         "active_count": 1, "candidate_count": 2, "retired_count": 0, "evidence_rows": 9, "warnings": []}


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
    up = Upstream({"/personality/frame": (200, FRAME)})
    resp = wire(up).get("/api/personality/frame")
    assert resp.status_code == 200
    assert resp.json() == FRAME
    assert "no-store" in resp.headers["cache-control"]
    assert str(up.calls[0].url) == "http://durable.test:8124/personality/frame"


CHOICES = {"count": 1, "choices": [{"choice_id": "curiosity:r1:x", "lane": "curiosity",
                                     "chosen": {"key": "x", "text": "<b>x</b>", "market_rank": 4},
                                     "passed_over": [{"key": "a", "text": "a", "market_rank": 1}],
                                     "declined_all": False}]}
TENDENCIES = {"count": 1, "warnings": [], "tendencies": [{"disposition_id": "disp-1", "label": "x over a, 3 of 3",
                                                          "status": "candidate", "weight": 2.0, "n": 3, "m": 3,
                                                          "examples": [], "juniper": None}]}


def test_choices_tendencies_and_arms_forward_limits(wire):
    up = Upstream({"/personality/choices": (200, CHOICES), "/personality/tendencies": (200, TENDENCIES),
                   "/personality/arms": (200, {"days": 7, "total": 0, "arms": {}})})
    client = wire(up)
    assert client.get("/api/personality/choices").json() == CHOICES
    assert dict(up.calls[-1].url.params) == {"limit": "30"}
    assert client.get("/api/personality/tendencies?limit=5").json() == TENDENCIES
    assert dict(up.calls[-1].url.params) == {"limit": "5"}
    assert client.get("/api/personality/arms").json()["total"] == 0
    assert dict(up.calls[-1].url.params) == {"days": "7"}


def test_verdict_posts_yes_or_no_to_durable_runs(wire):
    ok = {"verdict_id": "jv-1", "disposition_id": "disp-1", "verdict": "no", "status": "retired", "retired": True}
    up = Upstream({"/personality/tendencies/disp-1/verdict": (200, ok)})
    resp = wire(up).post("/api/personality/tendencies/disp-1/verdict", json={"verdict": "no"})
    assert resp.status_code == 200 and resp.json() == ok
    call = up.calls[-1]
    assert call.method == "POST" and json.loads(call.content) == {"verdict": "no"}
    assert "no-store" in resp.headers["cache-control"]


@pytest.mark.parametrize("status", [404, 409, 503])
def test_verdict_upstream_404_409_503_pass_through(wire, status):
    up = Upstream({"/personality/tendencies/disp-1/verdict": (status, {"detail": "named reason"})})
    resp = wire(up).post("/api/personality/tendencies/disp-1/verdict", json={"verdict": "yes"})
    assert resp.status_code == status and resp.json()["detail"] == "named reason"


@pytest.mark.parametrize("url,body,code", [
    ("/api/personality/tendencies/disp-1/verdict", {"verdict": "maybe"}, 422),
    ("/api/personality/tendencies/disp-1/verdict", {}, 422),
    ("/api/personality/tendencies/disp-1/verdict", {"verdict": "no", "x": 1}, 422),
    ("/api/personality/tendencies/disp-1%0A/verdict", {"verdict": "no"}, 400),
    ("/api/personality/tendencies/a%20b/verdict", {"verdict": "no"}, 400),
])
def test_bad_verdict_never_reaches_upstream(wire, url, body, code):
    up = Upstream()
    assert wire(up).post(url, json=body).status_code == code
    assert up.calls == []


@pytest.mark.parametrize("url,code", [
    ("/api/personality/choices?limit=0", 422),
    ("/api/personality/choices?limit=201", 422),
    ("/api/personality/tendencies?limit=101", 422),
    ("/api/personality/arms?days=0", 422),
    ("/api/personality/arms?days=31", 422),
])
def test_bad_input_is_rejected_and_never_reaches_upstream(wire, url, code):
    up = Upstream()
    resp = wire(up).get(url)
    assert resp.status_code == code
    assert up.calls == []


def test_old_prose_panel_routes_are_gone(wire):
    """Rev 4 move 4: the excerpt / recurrence panels are dropped, and so are their proxies."""
    client = wire(Upstream())
    for path in ("/api/personality/dispositions", "/api/personality/evidence", "/api/personality/naming?disposition_id=d1",
                 "/api/personality/cue"):
        assert client.get(path).status_code == 404, path


def test_unconfigured_base_url_is_503(wire):
    resp = wire(Upstream(), base="").get("/api/personality/choices")
    assert resp.status_code == 503
    assert "HUB_CURIOSITY_DURABLE_RUNS_URL" in resp.json()["detail"]


def test_upstream_503_passes_through(wire):
    up = Upstream({"/personality/frame": (503, {"detail": "personality tables missing"})})
    resp = wire(up).get("/api/personality/frame")
    assert resp.status_code == 503 and "tables missing" in resp.json()["detail"]


def test_upstream_down_is_502_not_500(wire):
    down = wire(Upstream(error=httpx.ConnectError("refused"))).get("/api/personality/tendencies")
    assert down.status_code == 502 and down.json()["detail"].startswith("durable_runs_unavailable: ConnectError")


def test_upstream_500_and_bad_json_become_502(wire):
    up = Upstream({"/personality/frame": (500, {"detail": "boom"}), "/personality/choices": (200, "<html>")})
    client = wire(up)
    r1 = client.get("/api/personality/frame")
    assert r1.status_code == 502 and "durable_runs_error 500" in r1.json()["detail"]
    r2 = client.get("/api/personality/choices")
    assert r2.status_code == 502 and "bad_json" in r2.json()["detail"]


def test_malformed_base_url_is_502_not_500(wire):
    resp = wire(Upstream(), base="http://[::1").get("/api/personality/frame")
    assert resp.status_code == 502


def test_page_renders_template_with_cache_busted_script(monkeypatch):
    # The page route imports the full Hub app (scripts.main) for the template dir and asset
    # version, like the Temporal Self page. The slim reading-tests install lacks the app's deps
    # (aiohttp, ...), so this one test needs the full Hub environment; the rest run anywhere.
    try:
        from scripts import main as hub_main
    except ImportError as exc:
        pytest.skip(f"full Hub app not importable here: {exc}")

    monkeypatch.setattr(hub_main, "build_hub_ui_asset_version", lambda: "testver")
    app = FastAPI()
    app.include_router(mod.page_router)
    resp = TestClient(app).get("/personality")
    assert resp.status_code == 200
    assert '<script src="/static/js/personality.js?v=testver"></script>' in resp.text
    assert "{{HUB_UI_ASSET_VERSION}}" not in resp.text
    assert "no-store" in resp.headers["cache-control"]


def test_main_registers_routers():
    src = (HUB / "scripts" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(personality_router)" in src
    assert "app.include_router(personality_page_router)" in src


def test_nav_tab_panel_and_scripts_are_wired():
    index = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="personalityTabButton"' in index and 'data-hash-target="#personality"' in index
    assert '<section id="personality" data-panel="personality"' in index
    assert 'id="personalityPanelFrame" src="/personality"' in index
    assert '<script src="/static/js/personality_tab.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in index
    assert index.count('id="personality"') == 1
    tab_js = (HUB / "static" / "js" / "personality_tab.js").read_text(encoding="utf-8")
    for needle in ('"#personality"', "personalityTabButton", "personalityPanelFrame",
                   "personalityPanelRefresh", "function activatePanel()", "function deactivatePanel()"):
        assert needle in tab_js


def test_page_js_fetches_every_route_and_page_has_its_ids():
    page = (HUB / "templates" / "personality.html").read_text(encoding="utf-8")
    js = (HUB / "static" / "js" / "personality.js").read_text(encoding="utf-8")
    for route in ("/api/personality/choices?limit=", "/api/personality/tendencies?limit=",
                  "/api/personality/arms?days=", "/api/personality/tendencies/${encodeURIComponent(id)}/verdict"):
        assert route in js, route
    for el_id in ("pRefresh", "pStatus", "pChoices", "pTendencies", "pTendencyWarn", "pTendencyFlash", "pSaidNo",
                  "pSaidNoCount", "pArms", "pArmsSummary"):
        assert f'id="{el_id}"' in page, el_id
        assert f'"{el_id}"' in js, el_id
    # The only write is Juniper's answer.
    assert js.count('method: "POST"') == 1
    # The old prose panels are gone from the page.
    for gone in ("pCueText", "pActive", "pCandidates", "excerpt", "evidence"):
        assert gone not in page, gone


def test_page_js_never_writes_html():
    """Option texts and labels come from Orion's lanes: they must only ever reach the DOM as text."""
    js = (HUB / "static" / "js" / "personality.js").read_text(encoding="utf-8")
    assert re.findall(r"\.(innerHTML|outerHTML|insertAdjacentHTML)\b", js) == []
