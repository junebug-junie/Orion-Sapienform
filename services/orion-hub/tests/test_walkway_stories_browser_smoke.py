"""Browser smoke for the "Walkway stories" spot-check card (motion stories slice 1, 2026-10-16).

The REAL templates/index.html and static/js (app.js + walkway-stories.js) in Chromium at
360 px wide, every Hub API intercepted. Covers the changed interaction, not just page load:
the card is hidden until "Walkway (live)" is chosen; then it lists stories with a box map
and the thumbnails that exist, the tally line sits at the top, and clicking "Wrong name"
POSTs the verdict, marks the button pressed and updates the tally. Story text is rendered
as text (a "<b>" in a sentence must not become markup).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
H = "c" * 64
GONE = "d" * 64
PIXEL = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")


def _story(i: int, **kw) -> dict:
    base = {
        "story_id": f"{i:032x}", "stream_id": "walkway", "kind": "person", "started_at": "2026-10-16T22:09:44+00:00",
        "ended_at": "2026-10-16T22:09:49+00:00", "detection_count": 3, "used_detections": 3,
        "speed_class": "walking", "zone_path": ["walkway"], "individual_id": None, "name_label": None,
        "companion_kinds": [], "clean": True, "rules_version": 1, "verdict": None,
        "sentence": f"A person walked left to right on the walkway, over 5 s (3 detections). <b>{i}</b>",
        "evidence": [
            {"crop_id": "c1", "box": [600, 1150, 680, 1350], "clipped": False, "thumb": H},
            {"crop_id": "c2", "box": [760, 1150, 840, 1350], "clipped": False, "thumb": GONE},
            {"crop_id": "c3", "box": [2490, 1150, 2560, 1350], "clipped": True, "thumb": None},
        ],
    }
    base.update(kw)
    return base


def _tally(checked: int, right: int, wrong_name: int = 0) -> dict:
    return {"rules_version": 1, "checked": checked, "right": right, "wrong_name": wrong_name, "wrong_motion": 0,
            "wrong_other": 0, "named_checked": 0, "pct_right": (right / checked) if checked else None,
            "gate": {"min_checked": 50, "min_right": 0.8, "max_wrong_names": 0, "passed": False}}


def test_walkway_stories_card_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    posts: list[tuple[str, dict]] = []
    lists: list[str] = []

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
        if path == "/api/vision/motion-stories":
            lists.append(url)
            body = {"ok": True, "stories": [_story(1), _story(2, name_label="Alex", individual_id="ind-a")],
                    "tally": _tally(3, 3)}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/verdict", path)
        if m:
            posts.append((m.group(1), json.loads(route.request.post_data or "{}")))
            return route.fulfill(body=json.dumps({"ok": True, "story_id": m.group(1), "verdict": "wrong_name",
                                                  "tally": _tally(4, 3, wrong_name=1)}),
                                 content_type="application/json")
        if path == f"/api/vision/crop-thumbs/{H}":
            return route.fulfill(body=PIXEL, content_type="image/gif")
        if path.startswith("/api/vision/crop-thumbs/") or path == "/api/vision/frames/latest/image":
            return route.fulfill(status=404, body="")
        return route.fulfill(body="{}", content_type="application/json")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 360, "height": 800})
        page.route("http://hub.test/**", handle)
        page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
        page.goto("http://hub.test/")
        page.wait_for_function("() => !!window.OrionWalkwayStories")
        # Tailwind comes from a CDN this smoke does not load; its `hidden` is the one class
        # the page's overlays (and this card) need to behave.
        page.add_style_tag(content=".hidden{display:none!important}")

        card = page.locator("#walkwayStoriesCard")
        assert "hidden" in (card.get_attribute("class") or "").split() and not card.is_visible()
        assert lists == []                       # nothing fetched while another view is shown

        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayStoriesList [data-story-id]")
        assert "hidden" not in (card.get_attribute("class") or "").split() and card.is_visible()
        items = page.locator("#walkwayStoriesList [data-story-id]")
        assert items.count() == 2
        first = items.nth(0)
        assert first.locator("div").first.text_content().endswith("<b>1</b>")   # text, not markup
        assert first.locator("b").count() == 0
        assert first.locator("svg[data-box-map] rect").count() == 3
        assert first.locator("svg[data-box-map] rect[stroke-dasharray]").count() == 1
        page.wait_for_function("() => document.querySelector('#walkwayStoriesList [data-story-id]')"
                               ".textContent.includes('(picture gone)')")
        assert first.locator(f"img[data-story-thumb='{H}']").count() == 1     # the one that exists
        assert first.locator(f"img[data-story-thumb='{GONE}']").count() == 0  # pruned: says so instead
        assert "named" in items.nth(1).text_content()
        assert page.locator("#walkwayStoriesTally").text_content().startswith("Checked 3 of 50")
        # The tally sits above the list, at the top of the card.
        tally_box = page.locator("#walkwayStoriesTally").bounding_box()
        list_box = page.locator("#walkwayStoriesList").bounding_box()
        assert tally_box["y"] < list_box["y"]
        # Fits a 360 px phone: no horizontal overflow from the card.
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360

        btn = items.nth(1).locator("[data-story-verdict='wrong_name']")
        btn.click()
        page.wait_for_function(
            "() => document.querySelector(\"#walkwayStoriesList [data-story-id] + [data-story-id] "
            "[data-story-verdict='wrong_name']\").getAttribute('aria-pressed') === 'true'")
        assert posts == [(f"{2:032x}", {"verdict": "wrong_name"})]
        assert "wrong names 1" in page.locator("#walkwayStoriesTally").text_content()
        assert "a wrong name was found" in page.locator("#walkwayStoriesGate").text_content()

        page.select_option("#visionSource", "none")
        assert "hidden" in (card.get_attribute("class") or "").split()
        browser.close()
