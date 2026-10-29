"""Body economy tab: Hub proxy route (fake durable-runs upstream) and static tab wiring."""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import body_economy_routes as mod

HUB = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((HUB / "tests/fixtures/body_economy_report.json").read_text(encoding="utf-8"))
INDEX = (HUB / "templates/index.html").read_text(encoding="utf-8")
APP_JS = (HUB / "static/js/app.js").read_text(encoding="utf-8")
PAGE = (HUB / "static/body-economy.html").read_text(encoding="utf-8")
PAGE_JS = (HUB / "static/js/body-economy.js").read_text(encoding="utf-8")


class Upstream:
    def __init__(self, status: int = 200, body=None, error: Exception | None = None):
        self.status, self.body, self.error = status, body, error
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if isinstance(self.body, str):
            return httpx.Response(self.status, text=self.body)
        return httpx.Response(self.status, json=self.body)


@pytest.fixture
def wire(monkeypatch):
    monkeypatch.setattr(mod, "_reach", {"last_ok_at": None, "unreachable_since": None})
    real = httpx.AsyncClient  # captured once, so a second _wire() in one test swaps the upstream cleanly

    def _wire(upstream: Upstream, base: str = "http://durable.test:8124"):
        monkeypatch.setattr(mod.settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", base, raising=False)

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(upstream.handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(mod.httpx, "AsyncClient", factory)
        app = FastAPI()
        app.include_router(mod.router)
        return TestClient(app)

    return _wire


def test_report_proxies_verbatim_with_no_cache_and_forwards_params(wire):
    up = Upstream(200, FIXTURE)
    resp = wire(up).get("/api/body-economy/report?window=7d&outcome=deferred&family=hub.outreach&limit=50")
    assert resp.status_code == 200
    assert resp.json() == FIXTURE
    assert "no-store" in resp.headers["cache-control"]
    url = up.calls[0].url
    assert url.path == "/physiology/report"
    assert dict(url.params) == {"window": "7d", "outcome": "deferred", "family": "hub.outreach", "limit": "50"}


def test_default_window_is_24h_and_absent_params_are_dropped(wire):
    up = Upstream(200, FIXTURE)
    wire(up).get("/api/body-economy/report")
    assert dict(up.calls[0].url.params) == {"window": "24h", "limit": "500"}


@pytest.mark.parametrize("query", ["window=1y", "outcome=maybe", "family=bad%20name", "sections=a;b"])
def test_bad_params_are_400_and_never_reach_upstream(wire, query):
    up = Upstream(200, FIXTURE)
    resp = wire(up).get(f"/api/body-economy/report?{query}")
    assert resp.status_code == 400
    assert up.calls == []
    assert "unreachable_since" not in resp.json()


def test_unreachable_is_502_with_since_that_holds_until_a_good_answer(wire):
    down = Upstream(error=httpx.ConnectError("refused"))
    client = wire(down)
    first = client.get("/api/body-economy/report")
    assert first.status_code == 502
    body = first.json()
    assert body["detail"].startswith("durable_runs_unavailable: ConnectError")
    since = body["unreachable_since"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", since)
    assert client.get("/api/body-economy/report").json()["unreachable_since"] == since  # first failure, kept

    ok = wire(Upstream(200, FIXTURE)).get("/api/body-economy/report")
    assert ok.status_code == 200
    assert mod._reach["unreachable_since"] is None and mod._reach["last_ok_at"]


@pytest.mark.parametrize("upstream, status, detail_part", [
    (Upstream(500, {"detail": "boom"}), 502, "durable_runs_error 500"),
    (Upstream(200, "<html>"), 502, "durable_runs_bad_json"),
    (Upstream(200, [1, 2]), 502, "durable_runs_bad_shape"),
    (Upstream(404, {"detail": "Not Found"}), 404, "Not Found"),   # API not deployed yet
    (Upstream(503, {"detail": "want_ledger missing"}), 503, "want_ledger missing"),
])
def test_upstream_faults_are_named_never_500(wire, upstream, status, detail_part):
    resp = wire(upstream).get("/api/body-economy/report")
    assert resp.status_code == status
    assert detail_part in resp.json()["detail"]


def test_unset_base_url_is_503_named(wire):
    resp = wire(Upstream(200, FIXTURE), base="").get("/api/body-economy/report")
    assert resp.status_code == 503
    assert "HUB_CURIOSITY_DURABLE_RUNS_URL" in resp.json()["detail"]


def test_router_is_mounted_in_the_hub_app():
    main = (HUB / "scripts/main.py").read_text(encoding="utf-8")
    assert "from scripts.body_economy_routes import router as body_economy_router" in main
    assert "app.include_router(body_economy_router)" in main


# ---- static wiring (CLAUDE.md section 9: rendered template, linked asset) ----
def test_index_has_button_section_and_versioned_frame():
    assert 'id="bodyEconomyTabButton"' in INDEX
    assert 'data-hash-target="#body-economy"' in INDEX
    assert '<section id="body-economy" data-panel="body-economy"' in INDEX
    assert 'src="/static/body-economy.html?v={{HUB_UI_ASSET_VERSION}}"' in INDEX


def test_app_js_routes_hash_click_and_visibility():
    for needle in ('getElementById("bodyEconomyTabButton")', 'getElementById("body-economy")',
                   'getElementById("bodyEconomyFrame")', 'setActiveTab("body-economy")',
                   'h === "#body-economy"', "styleTabButton(bodyEconomyTabButton, isBodyEconomy)",
                   'bodyEconomyPanelEl.classList.toggle("hidden", !isBodyEconomy)'):
        assert needle in APP_JS, needle


def test_page_loads_versioned_script_and_every_id_the_js_writes_exists():
    assert '"/static/js/body-economy.js" + location.search' in PAGE
    assert "/api/body-economy/report?window=" in PAGE_JS
    written = set(re.findall(r'(?:setBody\(doc, |getElementById\()"(be[A-Z][A-Za-z]+)"', PAGE_JS))
    assert {"beTiles", "beLimitRows", "beWantsChart", "beRhythm", "beMoney", "beUnits", "beSees", "beWaste",
            "beQueue", "beVram", "beClaude", "beValue", "beDecRows", "beDetail", "beTip"} <= written
    for element_id in written:
        assert f'id="{element_id}"' in PAGE, element_id
    for w in ("24h", "7d", "30d"):
        assert f'data-window="{w}"' in PAGE
    assert 'id="beHatch"' in PAGE  # exogenous hours are hatched via this pattern
