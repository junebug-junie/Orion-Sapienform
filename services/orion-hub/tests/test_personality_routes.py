"""Personality panel: Hub proxy routes (fake durable-runs upstream), cue render, page, nav/tab wiring."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.personality.cue import project_personality_cue
from scripts import personality_routes as mod

HUB = Path(__file__).resolve().parents[1]


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


NOW = datetime.now(timezone.utc)
NODE = {"disposition_id": "d1", "label": "the gap between what I predicted and what arrived",
        "origin": "recurrence", "status": "active", "weight": 2.4, "evidence_count": 7,
        "lane_counts": {"dream": 4, "journal": 3}, "chosen_count": 1,
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


def test_dispositions_forwards_status_and_limit_with_defaults(wire):
    body = {"dispositions": [{**NODE, "history": [{"at": NODE["first_seen_at"], "weight": 1.0,
                                                   "status": "candidate", "evidence_count": 2}]}]}
    up = Upstream({"/personality/dispositions": (200, body)})
    client = wire(up)
    assert client.get("/api/personality/dispositions").json() == body
    assert dict(up.calls[-1].url.params) == {"status": "active", "limit": "50"}
    assert client.get("/api/personality/dispositions?status=retired&limit=200").status_code == 200
    assert dict(up.calls[-1].url.params) == {"status": "retired", "limit": "200"}


def test_evidence_forwards_disposition_id_and_drops_absent(wire):
    rows = {"evidence": [{"evidence_id": "e1", "lane": "dream", "excerpt": "<b>raw</b>", "disposition_id": "d1"}]}
    up = Upstream({"/personality/evidence": (200, rows)})
    client = wire(up)
    assert client.get("/api/personality/evidence?disposition_id=d1&limit=10").json() == rows
    assert dict(up.calls[-1].url.params) == {"disposition_id": "d1", "limit": "10"}
    assert client.get("/api/personality/evidence").status_code == 200
    assert dict(up.calls[-1].url.params) == {"limit": "50"}


@pytest.mark.parametrize("url,code", [
    ("/api/personality/dispositions?status=weird", 400),
    ("/api/personality/dispositions?status=active%0A", 400),
    ("/api/personality/dispositions?limit=0", 422),
    ("/api/personality/dispositions?limit=201", 422),
    ("/api/personality/evidence?disposition_id=a/../b", 400),
    ("/api/personality/evidence?disposition_id=d1%0A", 400),
    ("/api/personality/evidence?disposition_id=", 400),
    ("/api/personality/evidence?limit=201", 422),
])
def test_bad_input_is_rejected_and_never_reaches_upstream(wire, url, code):
    up = Upstream()
    resp = wire(up).get(url)
    assert resp.status_code == code
    assert up.calls == []


def test_unconfigured_base_url_is_503(wire):
    resp = wire(Upstream(), base="").get("/api/personality/dispositions")
    assert resp.status_code == 503
    assert "HUB_CURIOSITY_DURABLE_RUNS_URL" in resp.json()["detail"]


def test_upstream_503_passes_through(wire):
    up = Upstream({"/personality/frame": (503, {"detail": "personality tables missing"})})
    resp = wire(up).get("/api/personality/frame")
    assert resp.status_code == 503 and "tables missing" in resp.json()["detail"]


def test_upstream_down_is_502_not_500(wire):
    down = wire(Upstream(error=httpx.ConnectError("refused"))).get("/api/personality/evidence")
    assert down.status_code == 502 and down.json()["detail"].startswith("durable_runs_unavailable: ConnectError")


def test_upstream_500_and_bad_json_become_502(wire):
    up = Upstream({"/personality/frame": (500, {"detail": "boom"}), "/personality/dispositions": (200, "<html>")})
    client = wire(up)
    r1 = client.get("/api/personality/frame")
    assert r1.status_code == 502 and "durable_runs_error 500" in r1.json()["detail"]
    r2 = client.get("/api/personality/dispositions")
    assert r2.status_code == 502 and "bad_json" in r2.json()["detail"]


def test_malformed_base_url_is_502_not_500(wire):
    resp = wire(Upstream(), base="http://[::1").get("/api/personality/frame")
    assert resp.status_code == 502


def test_cue_renders_what_the_lanes_get_at_their_default_settings(wire):
    """Same renderer, same defaults as the lanes. The defaults are reported in render_params so a lane
    running non-default knobs is a visible divergence, not a silent one."""
    up = Upstream({"/personality/frame": (200, FRAME)})
    body = wire(up).get("/api/personality/cue").json()
    expected = project_personality_cue(FRAME)
    assert body["cue"]["text"] == expected.text
    assert body["cue"]["text"].startswith("Things you keep coming back to")
    assert NODE["label"] in body["cue"]["text"]
    assert body["cue"]["disposition_ids"] == ["d1"]
    assert body["cue"]["skip_reason"] is None
    assert body["frame"] == FRAME and body["error"] is None and body["upstream_status"] is None
    from orion.personality import cue as cue_mod

    assert body["render_params"] == {"max_age_sec": cue_mod.DEFAULT_MAX_AGE_SEC, "top_k": cue_mod.DEFAULT_TOP_K}


def test_cue_renderer_fault_is_render_failed_not_500(wire, monkeypatch):
    def boom(_frame):
        raise RuntimeError("renderer broke")

    monkeypatch.setattr(mod, "project_personality_cue", boom)
    resp = wire(Upstream({"/personality/frame": (200, FRAME)})).get("/api/personality/cue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cue"]["skip_reason"] == "render_failed" and body["cue"]["text"] == ""
    assert body["error"] == "render_failed: RuntimeError"


def test_cue_empty_graph_reports_skip_reason(wire):
    empty = {**FRAME, "dispositions": [], "choice_style": None, "active_count": 0, "candidate_count": 0,
             "evidence_rows": 0}
    body = wire(Upstream({"/personality/frame": (200, empty)})).get("/api/personality/cue").json()
    assert body["cue"]["text"] == "" and body["cue"]["skip_reason"] == "empty"
    assert body["frame"] == empty


def test_cue_stale_frame_skips_like_the_lanes(wire):
    stale = {**FRAME, "generated_at": _iso(NOW - timedelta(hours=3))}
    body = wire(Upstream({"/personality/frame": (200, stale)})).get("/api/personality/cue").json()
    assert body["cue"]["skip_reason"] == "stale" and body["cue"]["text"] == ""


@pytest.mark.parametrize("upstream,base,status", [
    (Upstream(error=httpx.ConnectError("refused")), "http://durable.test:8124", 502),
    (Upstream({"/personality/frame": (503, {"detail": "tables missing"})}), "http://durable.test:8124", 503),
    (Upstream(), "", 503),
])
def test_cue_fails_open_with_the_upstream_fault(wire, upstream, base, status):
    resp = wire(upstream, base=base).get("/api/personality/cue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cue"]["text"] == "" and body["cue"]["skip_reason"] == "fetch_failed"
    assert body["frame"] is None and body["upstream_status"] == status and body["error"]


def test_cue_invalid_frame_is_named(wire):
    body = wire(Upstream({"/personality/frame": (200, {"nope": 1})})).get("/api/personality/cue").json()
    assert body["cue"]["skip_reason"] == "invalid_frame"


def test_page_renders_template_with_cache_busted_script(monkeypatch):
    from scripts import main as hub_main

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
    for route in ("/api/personality/cue", "/api/personality/dispositions?status=",
                  "/api/personality/evidence?disposition_id="):
        assert route in js
    for el_id in ("pRefresh", "pStatus", "pWarnings", "pCueText", "pCueSkip", "pCueMeta", "pCueParams", "pCounts",
                  "pChoiceStyle", "pGeneratedAt", "pActive", "pActiveCount", "pCandidates", "pCandidateCount",
                  "pRetired", "pRetiredCount"):
        assert f'id="{el_id}"' in page, el_id
        assert f'"{el_id}"' in js, el_id
    # Render-only: the page never writes.
    assert "method:" not in js and "POST" not in js


def test_page_js_only_innerhtml_is_the_escaped_sparkline():
    """Labels and excerpts are Orion's raw words: they must only ever reach the DOM as text."""
    js = (HUB / "static" / "js" / "personality.js").read_text(encoding="utf-8")
    writes = re.findall(r"\.(innerHTML|outerHTML|insertAdjacentHTML)\b", js)
    assert writes == ["innerHTML"], writes
    assert "spark.innerHTML = sparklineSvg(" in js


@pytest.mark.parametrize("scenario", ["graph", "empty", "unavailable", "invalid"])
def test_personality_page_browser_smoke(scenario):
    """The REAL templates/personality.html + static/js/personality.js in Chromium, Hub API intercepted.
    Covers the changed interactions: cue text, node table with sparkline, click-for-evidence (with a
    hostile excerpt that must stay text), candidates/retired, and the empty and unavailable states."""
    pytest.importorskip("playwright.sync_api")
    import json

    from playwright.sync_api import sync_playwright

    template = (HUB / "templates" / "personality.html").read_text().replace("{{HUB_UI_ASSET_VERSION}}", "t")
    script = (HUB / "static" / "js" / "personality.js").read_text()
    hostile = '<img src=x onerror="window.__pwned=1">I keep circling the gap'
    history = [{"at": _iso(NOW - timedelta(hours=h)), "weight": w, "status": "active", "evidence_count": 7 - h}
               for h, w in ((5, 0.4), (3, 1.1), (1, 2.4))]
    nodes = {"active": [{**NODE, "history": history}],
             "candidate": [{**NODE, "disposition_id": "c1", "status": "candidate", "label": "a candidate theme",
                            "history": []}],
             "retired": []}
    empty_frame = {**FRAME, "dispositions": [], "choice_style": None, "active_count": 0, "candidate_count": 0,
                   "retired_count": 0, "evidence_rows": 0}
    if scenario == "graph":
        cue_body = {"cue": project_personality_cue(FRAME).model_dump(mode="json"), "frame": FRAME,
                    "error": None, "upstream_status": None}
    elif scenario == "empty":
        nodes = {"active": [], "candidate": [], "retired": []}
        cue_body = {"cue": project_personality_cue(empty_frame).model_dump(mode="json"), "frame": empty_frame,
                    "error": None, "upstream_status": None}
    elif scenario == "invalid":
        bad = {"nope": 1}
        cue_body = {"cue": project_personality_cue(bad).model_dump(mode="json"), "frame": bad,
                    "error": None, "upstream_status": None}
    else:
        cue_body = {"cue": project_personality_cue(None).model_dump(mode="json"), "frame": None,
                    "error": "durable_runs_url_not_configured (HUB_CURIOSITY_DURABLE_RUNS_URL)",
                    "upstream_status": 503}
    evidence_calls: list[str] = []

    def handle(route):
        url = route.request.url
        if url.endswith("/personality"):
            return route.fulfill(body=template, content_type="text/html")
        if "/static/js/personality.js" in url:
            return route.fulfill(body=script, content_type="application/javascript")
        if "/api/personality/cue" in url:
            return route.fulfill(body=json.dumps(cue_body), content_type="application/json")
        if "/api/personality/dispositions" in url:
            status = url.split("status=")[1].split("&")[0]
            return route.fulfill(body=json.dumps({"dispositions": nodes[status]}), content_type="application/json")
        if "/api/personality/evidence" in url:
            evidence_calls.append(url.split("disposition_id=")[1].split("&")[0])
            rows = [{"evidence_id": "e1", "lane": "dream", "source_ref": "dream:1",
                     "occurred_at": _iso(NOW - timedelta(hours=2)), "excerpt": hostile, "disposition_id": "d1",
                     "similarity": 0.83, "weight_delta": 0.5, "meta": {}}]
            return route.fulfill(body=json.dumps({"evidence": rows}), content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/personality")
        page.wait_for_function("document.getElementById('pStatus').textContent !== 'loading…'")

        if scenario == "invalid":                                  # never read as an empty, calm graph
            assert "does not validate" in page.inner_text("#pStatus")
            assert "starts from today" not in page.inner_text("#pStatus")
            assert page.inner_text("#pCueSkip").startswith("invalid_frame:")
            assert page.inner_text("#pActive") == ""
        elif scenario == "unavailable":
            assert page.inner_text("#pStatus").startswith("durable-runs unavailable: durable_runs_url_not_configured")
            assert page.evaluate("document.getElementById('pCueText').hidden") is True
            assert page.inner_text("#pCueSkip").startswith("fetch_failed:")
            assert page.inner_text("#pActive") == ""
        elif scenario == "empty":
            assert "starts from today and grows as Orion writes" in page.inner_text("#pStatus")
            assert page.inner_text("#pCueSkip").startswith("empty:")
            page.wait_for_selector("#pActive .muted")
            assert page.inner_text("#pActive") == "No active nodes yet."
            assert page.inner_text("#pCounts") == "0 active · 0 candidate · 0 retired · 0 evidence rows"
        else:
            assert page.text_content("#pCueText") == cue_body["cue"]["text"]      # byte-for-byte
            assert page.inner_text("#pStatus") == ""
            assert "went their own way 3 of 5" in page.inner_text("#pChoiceStyle")
            page.wait_for_selector("#pActive tr.node")
            row = page.inner_text("#pActive tr.node")
            assert NODE["label"] in row and "2.400" in row and "dream: 4, journal: 3" in row
            assert page.locator("#pActive tr.node svg.spark polyline").count() == 1
            page.wait_for_selector("#pCandidates tr.node", state="attached")       # collapsed by default
            assert page.inner_text("#pCandidateCount") == "(1 of 2)"                    # frame says 2
            page.click("details.group summary >> nth=0")
            assert "a candidate theme" in page.inner_text("#pCandidates tr.node")
            button = page.locator("#pActive tr.node button.label-btn")
            assert button.get_attribute("aria-expanded") == "false"
            button.focus()
            page.keyboard.press("Enter")                                              # keyboard opens it
            page.wait_for_selector("#pActive tr.evidence-row td.excerpt")
            assert button.get_attribute("aria-expanded") == "true"
            assert page.locator("#" + button.get_attribute("aria-controls")).count() == 1
            assert page.text_content("#pActive tr.evidence-row td.excerpt") == hostile   # shown as text
            assert page.locator("#pActive tr.evidence-row img").count() == 0
            assert page.evaluate("window.__pwned") is None
            assert "+0.500" in page.inner_text("#pActive tr.evidence-row")
            assert evidence_calls == ["d1"]
            page.click("#pActive tr.node td.time >> nth=0")                         # row click closes it
            assert page.locator("#pActive tr.evidence-row").count() == 0
            assert button.get_attribute("aria-expanded") == "false"
        assert errors == [], errors
        browser.close()
