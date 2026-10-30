"""Browser smoke for the Body economy tab (spec Phase 3, H2 + H3): the REAL `#body-economy` section of
templates/index.html, its iframe page static/body-economy.html and static/js/body-economy.js, in
Chromium, with the Hub API intercepted by the physiology.report.v1 fixture (no Hub process).

Covers the changed interactions, not page load: a deferred row clicked shows its binding constraint,
its displaced line and its verbatim reason; a drag across the limits chart filters the decision list;
the window buttons re-query; a tooltip appears on hover; and empty sections say why.

Set BODY_ECONOMY_SMOKE_SHOT=<path.png> to keep a full-page screenshot of the tab."""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((HUB / "tests/fixtures/body_economy_report.json").read_text(encoding="utf-8"))


def _host_page() -> str:
    """The real tab section from index.html, in a bare page that shows it for #body-economy (app.js's
    job in the Hub). Sized so the whole iframe page is reachable by the mouse."""
    index = (HUB / "templates/index.html").read_text(encoding="utf-8")
    match = re.search(r'<section id="body-economy".*?</section>', index, re.S)
    assert match, "index.html lost the #body-economy section"
    section = match.group(0).replace("{{HUB_UI_ASSET_VERSION}}", "t")
    return ("<!doctype html><html><head><style>.hidden{display:none}#bodyEconomyFrame{width:1320px;height:3400px;border:0}"
            "body{margin:0;background:#030712}</style></head><body>" + section +
            "<script>if (location.hash === '#body-economy') document.getElementById('body-economy').classList.remove('hidden');"
            "</script></body></html>")


def _run(report_handler, check):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    host = _host_page()
    page_html = (HUB / "static/body-economy.html").read_text(encoding="utf-8")
    script = (HUB / "static/js/body-economy.js").read_text(encoding="utf-8")

    def handle(route):
        url = route.request.url
        if "/api/body-economy/report" in url:
            return report_handler(route, url)
        if "/static/body-economy.html" in url:
            return route.fulfill(body=page_html, content_type="text/html")
        if "/static/js/body-economy.js" in url:
            return route.fulfill(body=script, content_type="application/javascript")
        if url.split("#")[0].rstrip("/") == "http://hub.test":
            return route.fulfill(body=host, content_type="text/html")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1340, "height": 1000}, timezone_id="UTC")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/#body-economy")
        page.wait_for_selector("#bodyEconomyFrame")
        frame = page.frame_locator("#bodyEconomyFrame")
        check(page, frame)
        assert errors == [], errors
        browser.close()


def test_body_economy_tab_deferred_row_brush_window_and_tooltip():
    windows: list[str] = []

    def report(route, url):
        windows.append(url.split("window=")[1].split("&")[0])
        return route.fulfill(body=json.dumps(FIXTURE), content_type="application/json")

    def check(page, frame):
        frame.locator('[data-tile="heat"]').wait_for()
        tiles = frame.locator("#beTiles").inner_text()
        assert "Cabinet heat" in tiles and "32.1 °C" in tiles
        assert "orion-harness-worker" in frame.locator('[data-tile="services"]').inner_text()   # services tile
        assert "2 idle lanes folded away" in frame.locator("#beLimitIdle").inner_text()      # fast + diffusion
        assert frame.locator('#beLimitRows [data-limit="energy"] rect.c-exo').count() == 24    # hatched, not coloured
        assert "other families" in frame.locator("#beWantsLegend").inner_text()
        assert "orion-harness-worker" not in frame.locator("#beWaste").inner_text()
        assert "agent" in frame.locator("#beWaste").inner_text() and "31 min" in frame.locator("#beWaste").inner_text()
        assert "gpu2" in frame.locator("#beVram").inner_text() and "agent-gpu2, diffusion" in frame.locator("#beVram").inner_text()
        assert "$14.62" in frame.locator("#beClaude").inner_text()
        procs = frame.locator("#beProcesses [data-process]")
        assert procs.count() == 4
        assert procs.first.get_attribute("data-process") == "athena:orion-athena-vector-host"   # longest first
        assert "50.6 W" in procs.first.inner_text() and "athena" in procs.first.inner_text()
        assert "CPU below the fitted idle floor" in frame.locator("#beProcesses").inner_text()
        assert "1 section(s) missing or partial" in frame.locator("#beStatus").inner_text()
        assert "Juniper replied" in frame.locator("#beValue").inner_text()
        assert "room without Orion 29.6 C" in frame.locator("#beSees").inner_text()             # verbatim why
        assert frame.locator("#beCount").inner_text() == "10 of 10 decisions"

        # H3: a deferred row shows its binding constraint, its displaced line, and its verbatim reason.
        frame.locator('[data-outcome="deferred"]').click()
        assert frame.locator("#beCount").inner_text() == "5 of 10 decisions"
        frame.locator('tr[data-dec="0"]').click()
        detail = frame.locator("#beDetail")
        assert "hub.outreach · say" in detail.inner_text()
        assert "Blocked by: lane:agent" in detail.locator("[data-binding]").inner_text()
        assert "Displaced:" in detail.locator("[data-displaced]").inner_text()
        assert "lane:agent is full: forecast use" in detail.inner_text()
        assert "lane:agent" in detail.inner_text() and "1.00" in detail.inner_text()             # prices seen
        assert frame.locator('tr[data-dec="0"].sel').count() == 1

        # The "why did Orion wait?" search.
        frame.locator("#beWhyWait").fill("painting")
        assert frame.locator("#beCount").inner_text() == "1 of 10 decisions"
        frame.locator("#beWhyWait").fill("")
        frame.locator('[data-outcome="all"]').click()

        # H2 in the browser: drag across the heat row from 70% to the right edge.
        track = frame.locator('[data-limit-track="heat"]')
        track.scroll_into_view_if_needed()
        box = track.bounding_box()
        y = box["y"] + box["height"] / 2
        page.mouse.move(box["x"] + box["width"] * 0.7, y)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.85, y, steps=4)
        page.mouse.move(box["x"] + box["width"] - 1, y, steps=4)
        page.mouse.up()
        assert frame.locator("#beCount").inner_text() == "7 of 10 decisions"
        assert "from the selection above" in frame.locator("#beBrushNote").inner_text()
        assert frame.locator('[data-limit-track="heat"] .brush:not(.hidden)').count() == 1
        frame.locator("#beBrushClear").click()
        assert frame.locator("#beCount").inner_text() == "10 of 10 decisions"

        # Hover tooltip on the heat strip names the hour, reading and state.
        page.mouse.move(box["x"] + box["width"] - 3, y)
        assert "32.1 °C · binding" in frame.locator("#beTip").inner_text()

        shot = os.environ.get("BODY_ECONOMY_SMOKE_SHOT")
        if shot:
            page.screenshot(path=shot, full_page=True)

        frame.locator('[data-window="7d"]').click()
        frame.locator('[data-window="7d"][aria-pressed="true"]').wait_for()
        assert windows[0] == "24h" and windows[-1] == "7d"

    _run(report, check)


def test_body_economy_tab_empty_sections_say_why():
    missing = copy.deepcopy(FIXTURE)
    missing["wants"] = {"families": [], "hourly": []}
    missing["vram"] = []
    missing["processes"] = []
    missing["services"] = {"checked_at": "2026-10-29T18:55:00Z", "down": []}
    missing["missing"] = [{"section": "wants", "reason": "no wants yet: want_ledger has no rows in this window"},
                          {"section": "vram", "reason": "no VRAM reading: the pool has not reported cards"},
                          {"section": "processes", "reason": "per-container CPU watts are not reported yet"}]
    calls = {"n": 0}

    def report(route, url):
        calls["n"] += 1
        if calls["n"] == 1:   # durable-runs down on first load
            return route.fulfill(status=502, content_type="application/json", body=json.dumps(
                {"detail": "durable_runs_unavailable: ConnectError", "unreachable_since": "2026-10-29T17:00:00Z",
                 "last_ok_at": None}))
        return route.fulfill(body=json.dumps(missing), content_type="application/json")

    def check(page, frame):
        frame.locator('#beWantsChart [data-empty]').wait_for()
        for section in ("#beWantsChart", "#beRhythm", "#beMoney", "#beUnits", "#beDecEmpty", "#beClaude", "#beProcesses"):
            assert "durable-runs unreachable since" in frame.locator(section).inner_text(), section
        assert "unreachable since" in frame.locator("#beStatus").inner_text()

        frame.locator('[data-window="24h"]').click()
        frame.locator('[data-tile="heat"]').wait_for()
        assert "no wants yet: want_ledger has no rows" in frame.locator("#beWantsChart").inner_text()
        assert "the pool has not reported cards" in frame.locator("#beVram").inner_text()
        assert "per-container CPU watts are not reported yet" in frame.locator("#beProcesses [data-empty]").inner_text()
        assert frame.locator("#beProcesses [data-process]").count() == 0
        assert "all running" in frame.locator('[data-tile="services"]').inner_text()
        assert frame.locator("#beDecRows tr").count() == 10

    _run(report, check)


def test_body_economy_tab_malformed_rows_and_non_report_bodies_never_blank_or_crash():
    odd = copy.deepcopy(FIXTURE)
    odd["decisions"][0]["family"] = None                 # producer bug: no family
    odd["decisions"][0]["displaced"] = "not-a-list"
    odd["rhythm"][0]["heavy"] = 2.6                      # fractional count
    odd["money"]["daily_kwh"][0]["d"] = 20261023         # number, not a date string
    calls = {"n": 0}

    def report(route, url):
        calls["n"] += 1
        if calls["n"] == 1:   # an HTML page with 200 (proxy login page, say)
            return route.fulfill(status=200, body="<html>login</html>", content_type="text/html")
        return route.fulfill(body=json.dumps(odd), content_type="application/json")

    def check(page, frame):
        frame.locator('#beWantsChart [data-empty]').wait_for()
        assert "not with a body report" in frame.locator("#beStatus").inner_text()
        assert "not with a body report" in frame.locator("#beLimitRows").inner_text()
        frame.locator('[data-window="24h"]').click()
        frame.locator('[data-tile="heat"]').wait_for()
        assert "could not draw" not in frame.locator("#beStatus").inner_text()
        assert frame.locator("#beDecRows tr").count() == 10
        frame.locator('tr[data-dec="0"]').click()
        assert "Displaced: none" in frame.locator("#beDetail [data-displaced]").inner_text()

    _run(report, check)
