"""Browser smoke for the "Walkway stories" spot-check card (motion stories slice 1, 2026-10-16).

The REAL templates/index.html and static/js (app.js + walkway-stories.js) in Chromium at
360 px wide, every Hub API intercepted. Covers the changed interaction, not just page load:
the card is hidden until "Walkway (live)" is chosen; then it lists stories with their own
picture of the moment (2026-10-17: a boxed full frame labelled "Picture from <time>" plus
crops, a crop whose frame was gone says so), "only checkable" is on by default and is sent
to the API, a story with no picture says "No picture saved for this story" and offers
"Can't tell", the tally line sits at the top, and clicking "Wrong name" / "Can't tell"
POSTs the verdict, marks the button pressed and updates the tally. Story text is rendered
as text (a "<b>" in a sentence must not become markup). No picture of today's view is used.
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
        "evidence_strip": {
            "version": 1, "dark": False,
            "frame": {"file": "frame.jpg", "t": "2026-10-16T22:09:46+00:00", "crop_id": "c2"},
            "crops": [{"crop_id": "c1", "t": "2026-10-16T22:09:44+00:00", "file": "crop_0.jpg"},
                      {"crop_id": "c2", "t": "2026-10-16T22:09:46+00:00", "file": "crop_1.jpg"},
                      {"crop_id": "c3", "t": "2026-10-16T22:09:49+00:00", "missing": "frame_gone"}],
        },
        "dark": False,
        "evidence": [
            {"crop_id": "c1", "box": [600, 1150, 680, 1350], "clipped": False, "thumb": H},
            {"crop_id": "c2", "box": [760, 1150, 840, 1350], "clipped": False, "thumb": GONE},
            {"crop_id": "c3", "box": [2490, 1150, 2560, 1350], "clipped": True, "thumb": None},
        ],
    }
    base.update(kw)
    return base


def _tally(checked: int, right: int, wrong_name: int = 0, cant_tell: int = 0) -> dict:
    return {"rules_version": 1, "checked": checked, "right": right, "wrong_name": wrong_name, "wrong_motion": 0,
            "wrong_other": 0, "named_checked": 0, "pct_right": (right / checked) if checked else None,
            "cant_tell": cant_tell,
            "gate": {"min_checked": 50, "min_right": 0.8, "max_wrong_names": 0, "passed": False}}


def test_walkway_stories_card_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    posts: list[tuple[str, dict]] = []
    lists: list[str] = []
    strips: list[str] = []

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
            body = {"ok": True, "stories": [_story(1), _story(2, name_label="Alex", individual_id="ind-a"),
                                            _story(3, evidence_strip=None)],
                    "tally": _tally(3, 3)}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/verdict", path)
        if m:
            sent = json.loads(route.request.post_data or "{}")
            posts.append((m.group(1), sent))
            tally = _tally(4, 3, wrong_name=1, cant_tell=1 if sent.get("verdict") == "cant_tell" else 0)
            return route.fulfill(body=json.dumps({"ok": True, "story_id": m.group(1), "verdict": sent.get("verdict"),
                                                  "tally": tally}),
                                 content_type="application/json")
        if re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/(crop_[0-3]|frame)\.jpg", path):
            strips.append(path)
            return route.fulfill(body=PIXEL, content_type="image/gif")
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
        assert "checkable_only=true" in lists[0]       # "only checkable" is on by default
        assert page.locator("#walkwayStoriesCheckable").is_checked()
        items = page.locator("#walkwayStoriesList [data-story-id]")
        assert items.count() == 3
        first = items.nth(0)
        assert first.locator("div").first.text_content().endswith("<b>1</b>")   # text, not markup
        assert first.locator("b").count() == 0
        # The picture of the moment, never today's view.
        assert first.locator("svg").count() == 0
        assert page.locator("img[src*='/api/vision/frames/latest']").count() == 0
        assert first.locator("img[data-story-frame]").get_attribute("src") == \
            f"/api/vision/motion-stories/{1:032x}/strip/frame.jpg"
        assert "Picture from" in first.text_content()
        assert first.locator("img[data-story-crop]").count() == 2
        assert "frame already gone" in first.text_content()
        assert first.locator("[data-story-verdict='cant_tell']").count() == 1   # an unclear picture is not "wrong"
        page.wait_for_function("() => document.querySelectorAll('img[data-story-frame]').length === 2 && "
                               "[...document.querySelectorAll('img[data-story-frame]')].every(i => i.complete)")
        assert any(p.endswith("/strip/frame.jpg") for p in strips)
        assert "named" in items.nth(1).text_content()
        # No strip: says so, shows the vision-host thumb that exists, and offers "Can't tell".
        third = items.nth(2)
        page.wait_for_function("() => document.querySelectorAll('#walkwayStoriesList [data-story-id]')[2]"
                               ".textContent.includes('(picture gone)')")
        assert "No full picture saved for this story" in third.text_content()
        assert third.locator(f"img[data-story-thumb='{H}']").count() == 1
        assert third.locator(f"img[data-story-thumb='{GONE}']").count() == 0
        assert third.locator("[data-story-verdict]").count() == 5
        assert page.locator("#walkwayStoriesTally").text_content().startswith("Checked 3 of 50")
        # The tally sits above the list, at the top of the card.
        # (Document order, not pixels: other panels keep reflowing while the page loads.)
        assert page.evaluate(
            "() => !!(document.getElementById('walkwayStoriesTally').compareDocumentPosition("
            "document.getElementById('walkwayStoriesList')) & Node.DOCUMENT_POSITION_FOLLOWING)")
        # Fits a 360 px phone: no horizontal overflow from the card.
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
        # The card's pictures must not push the page itself sideways.
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 360

        btn = items.nth(1).locator("[data-story-verdict='wrong_name']")
        btn.click()
        page.wait_for_function(
            "() => document.querySelector(\"#walkwayStoriesList [data-story-id] + [data-story-id] "
            "[data-story-verdict='wrong_name']\").getAttribute('aria-pressed') === 'true'")
        assert posts == [(f"{2:032x}", {"verdict": "wrong_name"})]
        assert "wrong names 1" in page.locator("#walkwayStoriesTally").text_content()
        assert "a wrong name was found" in page.locator("#walkwayStoriesGate").text_content()

        third.locator("[data-story-verdict='cant_tell']").click()
        page.wait_for_function(
            "() => document.querySelectorAll('#walkwayStoriesList [data-story-id]')[2]"
            ".querySelector(\"[data-story-verdict='cant_tell']\").getAttribute('aria-pressed') === 'true'")
        assert posts[-1] == (f"{3:032x}", {"verdict": "cant_tell"})
        assert "can't tell 1 (not counted)" in page.locator("#walkwayStoriesTally").text_content()
        assert "Checked 4 of 50" in page.locator("#walkwayStoriesTally").text_content()

        # Unticking "only checkable" asks for everything.
        before = len(lists)
        page.uncheck("#walkwayStoriesCheckable")
        for _ in range(50):
            if len(lists) > before:
                break
            page.wait_for_timeout(50)
        assert "checkable_only=false" in lists[-1]

        page.select_option("#visionSource", "none")
        assert "hidden" in (card.get_attribute("class") or "").split()
        browser.close()


def _person(iid: str, **kw) -> dict:
    base = {"individual_id": iid, "short_id": iid[:8], "role": "walker", "kind": "person", "crop_ids": ["c1"],
            "thumb": H, "label": None, "label_copied": False, "known": True, "story_action": None}
    base.update(kw)
    return base


def test_walkway_story_people_can_be_named_or_corrected_browser_smoke():
    """Who the story is about (2026-10-18): one box per person, a typed name / "Someone else" /
    "Not sure" POSTs to the person route, and the story re-renders from the server's answer."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    anon = _story(1, sentence="A person walked right to left, over 5 s (3 detections).",
                  display_sentence="A person walked right to left, over 5 s (3 detections).", display_name=None,
                  name_source="rule", people=[_person("ind-anon1234567"),
                                              _person("ind-dog", role="companion", kind="dog", thumb=None)])
    alex = _story(2, name_label="Alex", individual_id="ind-alex", sentence="Alex walked left to right, over 5 s (3 detections).",
                  display_sentence="Alex walked left to right, over 5 s (3 detections).", display_name="Alex",
                  name_source="rule", people=[_person("ind-alex", label="Alex", label_copied=True)])
    posts: list[tuple[str, str, dict]] = []

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
            return route.fulfill(body=json.dumps({"ok": True, "stories": [anon, alex], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/people/([^/]+)", path)
        if m:
            sent = json.loads(route.request.post_data or "{}")
            posts.append((m.group(1), m.group(2), sent))
            if m.group(2) == "ind-anon1234567":
                story = dict(anon, display_sentence="Theo walked right to left, over 5 s (3 detections).",
                             display_name="Theo", name_source="juniper",
                             people=[_person("ind-anon1234567", story_action={"action": "name", "name": "Theo"}),
                                     anon["people"][1]])
                body = {"ok": True, "story": story, "wrong_name_recorded": False, "tally": _tally(1, 1)}
            else:
                story = dict(alex, verdict="wrong_name", display_sentence="A person walked left to right, over 5 s (3 detections).",
                             display_name=None, name_source="juniper",
                             people=[_person("ind-alex", label="Alex", story_action={"action": "someone_else", "name": None})])
                body = {"ok": True, "story": story, "wrong_name_recorded": True, "tally": _tally(2, 1, wrong_name=1)}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        if path == f"/api/vision/crop-thumbs/{H}" or re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/.*", path):
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
        page.add_style_tag(content=".hidden{display:none!important}")
        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayStoriesList [data-story-person]")
        items = page.locator("#walkwayStoriesList [data-story-id]")
        first = items.nth(0)
        boxes = first.locator("[data-story-person]")
        assert boxes.count() == 2
        assert boxes.nth(0).locator("[data-person-title]").text_content() == "unnamed person (individual ind-anon)"
        assert boxes.nth(1).locator("[data-person-title]").text_content() == "With them: unnamed dog (individual ind-dog)"
        assert "Who is this story about?" in first.text_content()
        second = items.nth(1)
        assert second.locator("[data-person-title]").text_content() == "Alex"
        assert "carried over from a same-person answer" in second.text_content()
        assert second.locator("[data-person-name-input]").get_attribute("placeholder") == "Correct the name"
        # Fits a 360 px phone with the name boxes.
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 360

        # "Name them" with nothing typed: nothing is sent.
        boxes.nth(0).locator("[data-person-action='name']").click()
        assert "Type a name first" in page.locator("#walkwayStoriesStatus").text_content()
        assert posts == []
        boxes.nth(0).locator("[data-person-name-input]").fill("  Theo  ")
        boxes.nth(0).locator("[data-person-action='name']").click()
        page.wait_for_function("() => document.querySelector('#walkwayStoriesList [data-story-id] [data-story-sentence]')"
                               ".textContent.startsWith('Theo walked')")
        assert posts[-1] == (f"{1:032x}", "ind-anon1234567", {"action": "name", "name": "Theo"})
        first = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        assert "Orion first wrote: A person walked" in first.locator("[data-story-first-written]").text_content()
        assert first.locator("[data-person-said='name']").text_content() == "You named them Theo on this story."
        assert "named by you" in first.text_content()
        assert 'Saved: Theo' in page.locator("#walkwayStoriesStatus").text_content()

        # "Someone else" on the story's named walker: the story loses the name and gets "Wrong name".
        second = page.locator("#walkwayStoriesList [data-story-id]").nth(1)
        second.locator("[data-person-action='someone_else']").click()
        page.wait_for_function(
            "() => document.querySelectorAll('#walkwayStoriesList [data-story-id]')[1]"
            ".querySelector(\"[data-story-verdict='wrong_name']\").getAttribute('aria-pressed') === 'true'")
        assert posts[-1] == (f"{2:032x}", "ind-alex", {"action": "someone_else", "name": ""})
        second = page.locator("#walkwayStoriesList [data-story-id]").nth(1)
        assert second.locator("[data-story-sentence]").text_content().startswith("A person walked")
        assert "Recorded as a wrong name" in page.locator("#walkwayStoriesStatus").text_content()
        assert "wrong names 1" in page.locator("#walkwayStoriesTally").text_content()
        browser.close()
