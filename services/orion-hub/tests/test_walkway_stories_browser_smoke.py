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
            "thumb": H, "thumb_crop_id": "c1", "label": None, "label_copied": False, "known": True,
            "story_action": None, "pictures": [], "boxes": [], "pictured": True}
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
                                              _person("ind-dog", role="companion", kind="dog"),
                                              _person("ind-ghost", role="companion", thumb=None,
                                                      pictured=False)])
    alex = _story(2, name_label="Alex", individual_id="ind-alex", sentence="Alex walked left to right, over 5 s (3 detections).",
                  display_sentence="Alex walked left to right, over 5 s (3 detections).", display_name="Alex",
                  name_source="rule", people=[_person("ind-alex", label="Alex", label_copied=True)])
    posts: list[tuple[str, str, dict]] = []
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
                                     *anon["people"][1:]])
                body = {"ok": True, "story": story, "wrong_name_recorded": False, "tally": _tally(1, 1)}
            else:
                story = dict(alex, verdict="wrong_name", display_sentence="A person walked left to right, over 5 s (3 detections).",
                             display_name=None, name_source="juniper",
                             people=[_person("ind-alex", label="Alex", story_action={"action": "someone_else", "name": None,
                                                                    "why": sent.get("why") or None})])
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
        assert boxes.count() == 3
        assert boxes.nth(0).locator("[data-person-title]").text_content() == "unnamed person (individual ind-anon)"
        assert boxes.nth(1).locator("[data-person-title]").text_content() == "With them: unnamed dog (individual ind-dog)"
        assert "Who is in this story?" in first.text_content()
        # ind-ghost has no picture of its own: shown greyed, not asked (no name box, no reason box, no buttons).
        assert boxes.nth(2).get_attribute("data-person-asked") == "0"
        assert "No picture of this person - not asked." in boxes.nth(2).text_content()
        assert boxes.nth(2).locator("[data-person-action], [data-person-why-input]").count() == 0
        assert boxes.nth(0).locator(f"img[data-person-picture][src='/api/vision/crop-thumbs/{H}']").count() == 1
        second = items.nth(1)
        assert second.locator("[data-person-title]").text_content() == "Alex"
        assert "carried over from a same-person answer" in second.text_content()
        assert second.locator("[data-person-name-input]").get_attribute("placeholder") == "Or type the right name"
        assert second.locator("[data-person-action='confirm']").text_content() == "Yes, that's Alex"
        assert second.locator("[data-person-action='someone_else']").text_content() == "No, not Alex"
        # Fits a 360 px phone with the name boxes.
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 360

        # "Name them" with nothing typed: nothing is sent.
        boxes.nth(0).locator("[data-person-action='name']").click()
        assert "Type a name first" in page.locator("#walkwayStoriesStatus").text_content()
        assert posts == []
        # Half-typed text in the dog's box survives the walker's save re-rendering the story.
        boxes.nth(1).locator("[data-person-why-input]").fill("on a red leash")
        boxes.nth(0).locator("[data-person-name-input]").fill("  Theo  ")
        boxes.nth(0).locator("[data-person-action='name']").click()
        page.wait_for_function("() => document.querySelector('#walkwayStoriesList [data-story-id] [data-story-sentence]')"
                               ".textContent.startsWith('Theo walked')")
        assert posts[-1] == (f"{1:032x}", "ind-anon1234567", {"action": "name", "name": "Theo", "why": "", "group": False})
        first = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        assert first.locator("[data-story-person='ind-dog'] [data-person-why-input]").input_value() == "on a red leash"
        assert first.locator("[data-story-person='ind-anon1234567'] [data-person-why-input]").input_value() == ""
        assert "Orion first wrote: A person walked" in first.locator("[data-story-first-written]").text_content()
        assert first.locator("[data-person-said='name']").text_content() == "You named them Theo on this story."
        assert "named by you" in first.text_content()
        assert 'Saved: Theo' in page.locator("#walkwayStoriesStatus").text_content()

        # "Someone else" on the story's named walker: the story loses the name and gets "Wrong name".
        # With how she could tell (2026-11-24): the reason goes with the answer and shows back.
        second = page.locator("#walkwayStoriesList [data-story-id]").nth(1)
        second.locator("[data-person-why-input]").fill("  Pushing a stroller; Alex has no young kids.  ")
        second.locator("[data-person-action='someone_else']").click()
        page.wait_for_function(
            "() => document.querySelectorAll('#walkwayStoriesList [data-story-id]')[1]"
            ".querySelector(\"[data-story-verdict='wrong_name']\").getAttribute('aria-pressed') === 'true'")
        assert posts[-1] == (f"{2:032x}", "ind-alex", {"action": "someone_else", "name": "",
                                                       "why": "Pushing a stroller; Alex has no young kids.",
                                                       "group": False})
        second = page.locator("#walkwayStoriesList [data-story-id]").nth(1)
        assert second.locator("[data-person-why]").text_content() == (
            "How you could tell: Pushing a stroller; Alex has no young kids.")
        assert second.locator("[data-story-sentence]").text_content().startswith("A person walked")
        assert "Recorded as a wrong name" in page.locator("#walkwayStoriesStatus").text_content()
        assert "wrong names 1" in page.locator("#walkwayStoriesTally").text_content()

        # A half-typed name does not block a filter change she makes herself.
        page.locator("#walkwayStoriesList [data-person-name-input]").first.fill("Half")
        before = len(lists)
        page.uncheck("#walkwayStoriesCheckable")
        for _ in range(50):
            if len(lists) > before:
                break
            page.wait_for_timeout(50)
        assert "checkable_only=false" in lists[-1]
        browser.close()



def test_walkway_story_each_person_shows_their_own_pictures_and_yes_confirms_browser_smoke():
    """Juniper's story c486f5d8... (2026-11-22, doorbell): Juniper, Sasha, Sasha, an unnamed person and two
    companions, one picture with no link to its row (2026-11-24: "how the f would i know which is which").
    Each row now shows ITS OWN crops, a letter that also tags the time crops, its box numbers, and a
    named row says "Yes, that's <name>"; a person without a picture of their own is not asked."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    sid, partner = f"{7:032x}", f"{8:032x}"

    def pic(story_id, n, crop):
        return {"story_id": story_id, "file": f"crop_{n}.jpg", "t": f"2026-11-22T01:28:3{n}+00:00", "crop_id": crop}

    people = [
        _person("ind-jun", label="Juniper", crop_ids=["k0"], thumb=None, pictures=[pic(sid, 0, "k0")],
                boxes=[{"n": 1, "colour": "green"}], pictured=True),
        _person("ind-sasha1", label="Sasha", crop_ids=["k1", "k2"], thumb=None, pictures=[pic(sid, 1, "k1")],
                boxes=[{"n": 2, "colour": "blue"}, {"n": 3, "colour": "blue"}], pictured=True),
        _person("ind-sasha2", label="Sasha", crop_ids=["k3"], thumb=None, pictures=[pic(sid, 2, "k3")],
                boxes=[{"n": 4, "colour": "blue"}], pictured=True),
        _person("ind-anon38e1", crop_ids=["k4"], thumb=None, pictures=[pic(sid, 3, "k4")],
                boxes=[{"n": 5, "colour": "red"}], pictured=True),
        _person("ind-cab04a49", role="companion", crop_ids=["p0"], thumb=None, pictures=[pic(partner, 0, "p0")],
                boxes=[], pictured=True, story_id=partner),
        _person("ind-c2f73f7e", role="companion", crop_ids=["p1"], thumb=None, pictures=[], boxes=[],
                pictured=False, story_id=partner),
        _person("ind-jun", label="Juniper", crop_ids=["k0"], thumb=None),     # a repeat: shown once
    ]
    story = _story(7, sentence="A person walked right to left and toward the house on the porch, over 6 s "
                               "(5 detections), moving alongside another person.", people=people,
                   evidence_strip={"version": 1, "dark": False,
                                   "frame": {"file": "frame.jpg", "t": "2026-11-22T01:28:33+00:00", "crop_id": "k2"},
                                   "crops": [{"crop_id": c, "t": f"2026-11-22T01:28:3{n}+00:00", "file": f"crop_{n}.jpg"}
                                             for n, c in enumerate(["k0", "k1", "k3", "k4"])]})
    posts: list[tuple[str, dict]] = []

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
            return route.fulfill(body=json.dumps({"ok": True, "stories": [story], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/people/([^/]+)", path)
        if m:
            sent = json.loads(route.request.post_data or "{}")
            posts.append((m.group(2), sent))
            again = dict(story, people=[dict(p, story_action={"action": "confirm", "name": "Sasha"})
                                        if p["individual_id"] == "ind-sasha2" else p for p in people])
            return route.fulfill(body=json.dumps({"ok": True, "story": again, "wrong_name_recorded": False,
                                                  "tally": _tally(1, 1)}), content_type="application/json")
        if re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/(crop_[0-3]|frame)\.jpg", path):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 360, "height": 900})
        page.route("http://hub.test/**", handle)
        page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
        page.goto("http://hub.test/")
        page.wait_for_function("() => !!window.OrionWalkwayStories")
        page.add_style_tag(content=".hidden{display:none!important}")
        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayStoriesList [data-story-person]")
        item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        rows = item.locator("[data-story-person]")
        assert rows.count() == 6                                   # the repeated individual is shown once
        assert [rows.nth(i).get_attribute("data-person-letter") for i in range(6)] == list("ABCDEF")
        # Each row: its own crop, from its own story's strip.
        srcs = [rows.nth(i).locator("img[data-person-picture]").get_attribute("src") for i in range(5)]
        assert srcs == [f"/api/vision/motion-stories/{sid}/strip/crop_0.jpg",
                        f"/api/vision/motion-stories/{sid}/strip/crop_1.jpg",
                        f"/api/vision/motion-stories/{sid}/strip/crop_2.jpg",
                        f"/api/vision/motion-stories/{sid}/strip/crop_3.jpg",
                        f"/api/vision/motion-stories/{partner}/strip/crop_0.jpg"]
        assert rows.nth(0).locator("img[data-person-picture]").get_attribute("alt").startswith("Person A (Juniper)")
        # The two "Sasha" rows are told apart by their box numbers and group ids.
        assert rows.nth(1).get_attribute("data-person-letter") == "B"
        assert rows.nth(1).locator("[data-person-boxes]").get_attribute("data-person-boxes") == "2,3"
        assert rows.nth(2).locator("[data-person-boxes]").get_attribute("data-person-boxes") == "4"
        assert "group ind-sash" in rows.nth(1).text_content() and "group ind-sash" in rows.nth(2).text_content()
        assert "text-red-400" in rows.nth(3).locator("[data-person-box='5']").get_attribute("class")
        assert "they walked alongside" in rows.nth(4).text_content()
        # The time crops above carry the same letters.
        assert [c.get_attribute("data-story-crop-person")
                for c in item.locator("[data-story-crop-person]").all()] == ["A", "B", "C", "D"]
        # The companion with no picture of their own: greyed, not asked.
        assert rows.nth(5).get_attribute("data-person-asked") == "0"
        assert rows.nth(5).locator("[data-person-action]").count() == 0
        assert "1 without a picture of their own is not asked." in item.text_content()
        # Named rows say yes in one click; an unnamed row asks for a name.
        assert rows.nth(3).locator("[data-person-action='confirm']").count() == 0
        assert rows.nth(3).locator("[data-person-action='name']").text_content() == "Name them"
        rows.nth(2).locator("[data-person-action='confirm']").click()
        page.wait_for_function("() => !!document.querySelector(\"[data-person-said='confirm']\")")
        assert posts == [("ind-sasha2", {"action": "confirm", "name": "Sasha", "why": "", "group": False})]
        assert "Saved: yes, that's Sasha." in page.locator("#walkwayStoriesStatus").text_content()
        assert page.locator("[data-person-said='confirm']").text_content() == "You said yes, this is Sasha."
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 360
        browser.close()


def test_walkway_story_linked_groups_are_one_person_asked_once_browser_smoke():
    """Who-card Patch A (2026-11-24): the two "Sasha" camera groups Juniper already linked are ONE row
    (both groups' pictures, "Seen as 2 camera groups"), a yes is sent once with group=true, and the
    row offers "not the same person?" for its direct link (Orion's question 1)."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    html = re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)
    sid = f"{9:032x}"

    def pic(n, crop):
        return {"story_id": sid, "file": f"crop_{n}.jpg", "t": f"2026-11-22T01:28:3{n}+00:00", "crop_id": crop,
                "individual_id": "x"}

    sasha = {"person_key": "ind-sasha1", "individual_ids": ["ind-sasha1", "ind-sasha2"], "crop_ids": ["k1", "k3"],
             "role": "walker", "kind": "person", "label": "Sasha", "label_copied": False, "short_id": "ind-sash",
             "pictures": [pic(2, "k3"), pic(1, "k1")], "boxes": [{"n": 2, "colour": "blue"}, {"n": 4, "colour": "blue"}],
             "pictured": True, "asked_ids": ["ind-sasha1", "ind-sasha2"], "story_action": None,
             "groups": [{"individual_id": "ind-sasha1", "short_id": "ind-sas1", "picture_crop_ids": ["k1"], "asked": True},
                        {"individual_id": "ind-sasha2", "short_id": "ind-sas2", "picture_crop_ids": ["k3"], "asked": True}],
             "links": [{"a": "ind-sasha1", "b": "ind-sasha2"}]}
    anon = {"person_key": "ind-anon", "individual_ids": ["ind-anon"], "crop_ids": ["k4"], "role": "walker",
            "kind": "person", "label": None, "short_id": "ind-anon", "pictures": [pic(3, "k4")],
            "boxes": [{"n": 5, "colour": "red"}], "pictured": True, "asked_ids": ["ind-anon"], "story_action": None,
            "groups": [{"individual_id": "ind-anon", "short_id": "ind-anon", "picture_crop_ids": ["k4"], "asked": True}],
            "links": []}
    story = _story(9, sentence="Sasha walked right to left, over 6 s (4 detections).", people=[], persons=[sasha, anon],
                   evidence_strip={"version": 1, "dark": False,
                                   "frame": {"file": "frame.jpg", "t": "2026-11-22T01:28:33+00:00", "crop_id": "k1"},
                                   "crops": [{"crop_id": c, "t": f"2026-11-22T01:28:3{n}+00:00", "file": f"crop_{n}.jpg"}
                                             for n, c in enumerate(["k0", "k1", "k3", "k4"])]})
    posts: list[tuple[str, dict]] = []

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
            return route.fulfill(body=json.dumps({"ok": True, "stories": [story], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        if route.request.method == "POST":
            sent = json.loads(route.request.post_data or "{}")
            posts.append((path, sent))
            again = dict(story, persons=[dict(sasha, story_action={"action": "confirm", "name": "Sasha"}), anon])
            return route.fulfill(body=json.dumps({"ok": True, "story": again, "wrong_name_recorded": False,
                                                  "tally": _tally(1, 1)}), content_type="application/json")
        if re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/(crop_[0-3]|frame)\.jpg", path):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 360, "height": 900})
        page.route("http://hub.test/**", handle)
        page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
        page.goto("http://hub.test/")
        page.wait_for_function("() => !!window.OrionWalkwayStories")
        page.add_style_tag(content=".hidden{display:none!important}")
        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayStoriesList [data-story-person]")
        item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        rows = item.locator("[data-story-person]")
        assert rows.count() == 2                                   # Sasha once, not twice
        first = rows.nth(0)
        assert first.locator("[data-person-title]").text_content() == "Sasha"
        assert first.locator("[data-person-groups]").text_content().startswith("Seen as 2 camera groups")
        assert [i.get_attribute("src") for i in first.locator("img[data-person-picture]").all()] == [
            f"/api/vision/motion-stories/{sid}/strip/crop_2.jpg", f"/api/vision/motion-stories/{sid}/strip/crop_1.jpg"]
        assert first.locator("[data-person-boxes]").get_attribute("data-person-boxes") == "2,4"
        # Both of Sasha's time crops carry Sasha's letter.
        assert [c.get_attribute("data-story-crop-person") for c in item.locator("[data-story-crop-person]").all()] == [
            "A", "A", "B"]
        assert rows.nth(1).locator("[data-person-not-same]").count() == 0     # one group: nothing to split
        # "Not the same person?" is hidden until asked for, then names the link.
        pair = first.locator("[data-person-not-same-pair]")
        assert not pair.is_visible()
        first.locator("[data-person-not-same-toggle]").click()
        assert pair.is_visible() and pair.text_content() == "group ind-sas1 and group ind-sas2 are different people"
        assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 360
        first.locator("[data-person-action='confirm']").click()
        page.wait_for_function("() => !!document.querySelector(\"[data-person-said='confirm']\")")
        assert posts == [(f"/api/vision/motion-stories/{sid}/people/ind-sasha1",
                          {"action": "confirm", "name": "Sasha", "why": "", "group": True})]
        page.locator("[data-person-not-same-toggle]").first.click()
        page.locator("[data-person-not-same-pair]").first.click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes('not the same person')")
        assert posts[-1] == (f"/api/vision/motion-stories/{sid}/persons/ind-sasha1/not-same",
                             {"a": "ind-sasha1", "b": "ind-sasha2"})
        browser.close()
