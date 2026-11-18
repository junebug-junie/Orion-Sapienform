"""Browser smoke for the "Walkway history" panel (walkway rollups, 2026-10-18).

The REAL templates/index.html and static/js (app.js + walkway-history.js) in Chromium at
360 px wide, every Hub API intercepted. Covers the changed interaction, not just page load:
the panel is hidden (and fetches nothing) until "Walkway (live)" is chosen; then it shows the
Days tab, each row's line as text (a "<b>" in a line is not markup), a camera-down hour says
"camera down", captions are labelled as one-frame captions, and clicking Hours / Weeks /
Months asks the API for that grain and shows its rows. Nothing overflows 360 px.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]


def _row(grain: str, i: int, **kw) -> dict:
    base = {
        "stream_id": "walkway", "grain": grain, "label": f"{grain}-{i}", "complete": True,
        "period_start": "2026-10-17T06:00:00+00:00", "period_end": "2026-10-18T06:00:00+00:00",
        "line": f"Walkway {grain} {i}: 185 passings (105 people, 38 vehicles) by 93 distinct. <b>x</b>",
        "total": {"passings": 185}, "camera": {"up_minutes": 1212, "expected_minutes": 1440, "status": "ok"},
        "regulars": {"count": 3, "by_kind": {"person": 3}}, "captions": [],
        "stories": {"gate_passed": False, "checked": 8},
    }
    base.update(kw)
    return base


def test_walkway_history_panel_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    asked: list[str] = []

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            f = HUB / path.lstrip("/")
            if f.is_file():
                ctype = "application/javascript" if f.suffix == ".js" else "text/css" if f.suffix == ".css" else "application/octet-stream"
                return route.fulfill(body=f.read_bytes(), content_type=ctype)
            return route.fulfill(status=404, body="")
        if path == "/api/vision/rollups":
            grain = re.search(r"grain=(\w+)", url).group(1)
            asked.append(grain)
            rows = [_row(grain, 1, captions=[{"at": "14:23", "text": "A dog on a leash."}]), _row(grain, 2)]
            if grain == "hour":
                rows.append(_row("hour", 3, line="Walkway 16:00–17:00 (Sat 17 Oct): camera was down (0 of 60 minutes seen), so no count.",
                                 camera={"up_minutes": 0, "expected_minutes": 60, "status": "down"}))
            return route.fulfill(body=json.dumps({"ok": True, "grain": grain, "rows": rows}),
                                 content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [], "tally": {}}),
                                 content_type="application/json")
        if path == "/api/vision/frames/latest/image":
            return route.fulfill(status=404, body="")
        return route.fulfill(body="{}", content_type="application/json")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 360, "height": 800})
        page.route("http://hub.test/**", handle)
        page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
        page.goto("http://hub.test/")
        page.wait_for_function("() => !!window.OrionWalkwayHistory")
        page.add_style_tag(content=".hidden{display:none!important}")

        card = page.locator("#walkwayHistoryCard")
        assert not card.is_visible() and asked == []

        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayHistoryList [data-rollup-label]")
        assert card.is_visible() and asked == ["day"]
        tabs = page.locator("#walkwayHistoryTabs [data-rollup-grain]")
        assert [tabs.nth(i).text_content() for i in range(tabs.count())] == ["Hours", "Days", "Weeks", "Months"]
        assert page.locator('[data-rollup-grain="day"]').get_attribute("aria-selected") == "true"
        first = page.locator("#walkwayHistoryList [data-rollup-label]").first
        assert first.locator("div").first.text_content().endswith("<b>x</b>") and first.locator("b").count() == 0
        assert "caption 14:23 (one frame, may be wrong): A dog on a leash." in first.text_content()
        assert "camera up 20.2/24.0 h" in first.text_content()

        page.click('[data-rollup-grain="hour"]')
        page.wait_for_selector('[data-rollup-label="hour-3"]')
        assert asked[-1] == "hour"
        down = page.locator('[data-rollup-label="hour-3"]').text_content()
        assert "camera was down" in down and "camera down 0/60 min" in down and "nobody" not in down
        assert page.locator('[data-rollup-grain="hour"]').get_attribute("aria-selected") == "true"

        for grain in ("week", "month"):
            page.click(f'[data-rollup-grain="{grain}"]')
            page.wait_for_selector(f'[data-rollup-label="{grain}-1"]')
            assert asked[-1] == grain

        box = card.bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= 360 + 1
        assert page.evaluate("document.getElementById('walkwayHistoryCard').scrollWidth") <= box["width"] + 1

        page.select_option("#visionSource", "none")
        assert not card.is_visible()
        browser.close()
