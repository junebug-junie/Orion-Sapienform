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


def _pp(key: str, **kw) -> dict:
    """One person as the server lists it (``persons``, who-card Patches A/B): one camera group unless given."""
    ids = kw.pop("individual_ids", [key])
    pics = kw.pop("pictures", None)
    crops = kw.pop("crop_ids", ["c1"])
    if pics is None:
        pics = [{"thumb": H, "crop_id": crops[0], "t": None, "individual_id": key}]
    base = {"person_key": key, "individual_ids": ids, "crop_ids": crops, "short_id": key[:8], "role": "walker",
            "kind": "person", "label": None, "label_copied": False, "known": True, "story_action": None,
            "pictures": pics, "boxes": [], "pictured": bool(pics), "asked_ids": ids if pics else [],
            "groups": [{"individual_id": i, "short_id": i[:8], "picture_crop_ids": crops if pics else [],
                        "asked": bool(pics)} for i in ids],
            "links": []}
    base.update(kw)
    return base


def _page(p, html, handle, height=900):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 360, "height": height})
    page.route("http://hub.test/**", handle)
    page.route(re.compile(r"^(?!http://hub\.test).*"), lambda r: r.fulfill(status=204, body=""))
    page.goto("http://hub.test/")
    page.wait_for_function("() => !!window.OrionWalkwayStories")
    page.add_style_tag(content=".hidden{display:none!important}")
    page.select_option("#visionSource", "walkway-live")
    page.wait_for_selector("#walkwayStoriesList [data-story-person]")
    return browser, page


def _html() -> str:
    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{HUB_CFG}}", "{}")
    return re.sub(r"\{\{[A-Z0-9_]+\}\}", "", html)


def _static(route, path):
    f = HUB / path.lstrip("/")
    if f.is_file():
        ctype = "application/javascript" if f.suffix == ".js" else "text/css" if f.suffix == ".css" else "application/octet-stream"
        return route.fulfill(body=f.read_bytes(), content_type=ctype)
    return route.fulfill(status=404, body="")


NAMES = {"ok": True, "names": [{"name": "Theo", "seen": 40}, {"name": "Alex", "seen": 30}, {"name": "Sasha", "seen": 9}]}


def _fits(page):
    assert page.evaluate("() => document.getElementById('walkwayStoriesCard').scrollWidth") <= 360
    assert page.evaluate("() => document.documentElement.scrollWidth") <= 360


def test_walkway_story_one_person_at_a_time_browser_smoke():
    """Who-card Patch B (2026-11-24): one person at a time; "Is this Alex?" / "Who is this?"; "No, it's ..."
    lists the names Orion knows; a name picked from the list is held as "Changed to X - Undo" until the next
    person (Juniper's answer 1); a typed name and "Can't see them" save at once; Next / progress; 360 px."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    anon = _story(1, sentence="A person walked right to left, over 5 s (3 detections).",
                  display_sentence="A person walked right to left, over 5 s (3 detections).", display_name=None,
                  name_source="rule",
                  persons=[_pp("ind-anon1234567"), _pp("ind-dog", role="companion", kind="dog"),
                           _pp("ind-ghost", role="companion", pictures=[])])
    alex = _story(2, name_label="Alex", individual_id="ind-alex", sentence="Alex walked left to right, over 5 s (3 detections).",
                  display_sentence="Alex walked left to right, over 5 s (3 detections).", display_name="Alex",
                  name_source="rule", persons=[_pp("ind-alex", label="Alex", label_copied=True)], reopened=True)
    posts: list[tuple[str, str, dict]] = []

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [anon, alex], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/people/([^/]+)", path)
        if m:
            sent = json.loads(route.request.post_data or "{}")
            posts.append((m.group(1), m.group(2), sent))
            if m.group(2) == "ind-alex":
                story = dict(alex, verdict="wrong_name", display_sentence="Cora walked left to right, over 5 s (3 detections).",
                             persons=[_pp("ind-alex", label="Alex", story_action={"action": "someone_else", "name": "Cora"})])
                body = {"ok": True, "story": story, "wrong_name_recorded": True, "tally": _tally(2, 1, wrong_name=1)}
            else:
                body = {"ok": True, "story": None, "wrong_name_recorded": False, "tally": _tally(1, 1)}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        if path == f"/api/vision/crop-thumbs/{H}" or re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/.*", path):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser, page = _page(p, html, handle)
        items = page.locator("#walkwayStoriesList [data-story-id]")
        first = items.nth(0)
        # One person shown, of the two with a picture; the third is not asked.
        assert first.locator("[data-story-person]").count() == 1
        assert first.locator("[data-story-progress]").get_attribute("data-story-progress") == "1/2"
        assert "1 without a picture of their own is not asked." in first.text_content()
        assert first.locator("[data-person-question]").text_content() == "Who is this?"
        assert first.locator("[data-person-action='yes']").count() == 0          # nothing to say yes to
        assert first.locator("[data-person-action='cant_see']").text_content() == "Can't see them"
        _fits(page)
        # An unnamed person shows the names at once (no hidden "It's ..."): Orion's known names,
        # most seen first, and a box to type one.
        assert first.locator("[data-person-action='no']").count() == 0
        assert first.locator("[data-person-picker]").is_visible()
        assert "Who is it? Pick a name, or type one:" in first.locator("[data-person-picker]").text_content()
        assert first.locator("[data-person-name-input]").is_visible()
        assert [b.text_content() for b in first.locator("[data-person-pick]").all()] == ["Theo", "Alex", "Sasha"]
        _fits(page)
        # A name picked from the list: no request yet, "Named Theo" with Undo.
        first.locator("[data-person-pick='Theo']").click()
        page.wait_for_selector("[data-person-pending='Theo']")
        assert posts == []
        page.locator("[data-person-undo]").click()
        page.wait_for_function("() => !document.querySelector('[data-person-pending]')")
        assert posts == [] and "Undone" in page.locator("#walkwayStoriesStatus").text_content()
        first = items.nth(0)
        first.locator("[data-person-pick='Theo']").click()
        page.wait_for_selector("[data-person-pending='Theo']")
        # Next person sends it.
        first.locator("[data-story-nav='next']").click()
        page.wait_for_function("() => document.querySelector('[data-story-progress]').getAttribute('data-story-progress') === '2/2'")
        assert posts == [(f"{1:032x}", "ind-anon1234567", {"action": "name", "name": "Theo", "why": "", "group": True,
                                                           "card": "one_at_a_time", "dont_know": False,
                                                           "crop_ids": ["c1"]})]
        first = items.nth(0)
        assert first.locator("[data-person-question]").text_content() == "Walking with them: Who is this?"
        assert first.locator("[data-story-nav-done]").count() == 1
        # "Can't see them": saved at once, says Orion will not ask again.
        first.locator("[data-person-action='cant_see']").click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes(\"won't ask\")")
        assert posts[-1][1:] == ("ind-dog", {"action": "unsure", "name": "", "why": "", "group": True,
                                             "card": "one_at_a_time", "dont_know": False, "crop_ids": ["c1"]})
        # A named person: "Is this Alex?", the list leaves out Alex, a typed name saves at once.
        second = items.nth(1)
        assert second.locator("[data-story-reopened]").count() == 1 and first.locator("[data-story-reopened]").count() == 0
        assert second.locator("[data-person-question]").text_content() == "Is this Alex?"
        assert second.locator("[data-person-action='yes']").text_content() == "Yes, that's Alex"
        assert "carried over from a same-person answer" in second.text_content()
        second.locator("[data-person-action='no']").click()
        assert [b.text_content() for b in second.locator("[data-person-pick]").all()] == ["Theo", "Sasha"]
        second.locator("[data-person-why-input]").fill("  Pushing a stroller; Alex has no young kids.  ")
        second.locator("[data-person-name-input]").fill("  Cora ")
        second.locator("[data-person-action='typed']").click()
        page.wait_for_function(
            "() => document.querySelectorAll('#walkwayStoriesList [data-story-id]')[1]"
            ".querySelector(\"[data-story-verdict='wrong_name']\").getAttribute('aria-pressed') === 'true'")
        assert posts[-1][1:] == ("ind-alex", {"action": "someone_else", "name": "Cora",
                                              "why": "Pushing a stroller; Alex has no young kids.", "group": True,
                                              "card": "one_at_a_time", "dont_know": False, "crop_ids": ["c1"]})
        assert "Recorded as a wrong name" in page.locator("#walkwayStoriesStatus").text_content()
        # "Save name" with nothing typed sends nothing.
        n = len(posts)
        items.nth(1).locator("[data-person-action='no']").click()
        items.nth(1).locator("[data-person-action='typed']").click()
        assert "Type a name first" in page.locator("#walkwayStoriesStatus").text_content() and len(posts) == n
        # A held pick is sent when the page closes (keepalive), never lost.
        items.nth(1).locator("[data-person-pick='Theo']").click()
        page.wait_for_selector("[data-person-pending='Theo']")
        page.evaluate("() => window.dispatchEvent(new Event('pagehide'))")
        for _ in range(50):
            if len(posts) > n:
                break
            page.wait_for_timeout(50)
        assert posts[-1][1:] == ("ind-alex", {"action": "someone_else", "name": "Theo", "why": "", "group": True,
                                              "card": "one_at_a_time", "dont_know": False, "crop_ids": ["c1"]})
        browser.close()


def test_walkway_story_each_person_shows_their_own_pictures_and_yes_confirms_browser_smoke():
    """Juniper's story c486f5d8... (2026-11-22, doorbell): Juniper, Sasha, Sasha, an unnamed person and two
    companions. One at a time, each person shows ITS OWN crops, a letter that also tags the time crops,
    and its box numbers; a person without a picture of their own is never asked; "Yes, that's Sasha"."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    sid, partner = f"{7:032x}", f"{8:032x}"

    def pic(story_id, n, crop):
        return {"story_id": story_id, "file": f"crop_{n}.jpg", "t": f"2026-11-22T01:28:3{n}+00:00", "crop_id": crop}

    persons = [
        _pp("ind-jun", label="Juniper", crop_ids=["k0"], pictures=[pic(sid, 0, "k0")], boxes=[{"n": 1, "colour": "green"}]),
        _pp("ind-sasha1", label="Sasha", crop_ids=["k1", "k2"], pictures=[pic(sid, 1, "k1")],
            boxes=[{"n": 2, "colour": "blue"}, {"n": 3, "colour": "blue"}]),
        _pp("ind-sasha2", label="Sasha", crop_ids=["k3"], pictures=[pic(sid, 2, "k3")], boxes=[{"n": 4, "colour": "blue"}]),
        _pp("ind-anon38e1", crop_ids=["k4"], pictures=[pic(sid, 3, "k4")], boxes=[{"n": 5, "colour": "red"}]),
        _pp("ind-cab04a49", role="companion", crop_ids=["p0"], pictures=[pic(partner, 0, "p0")], story_id=partner),
        _pp("ind-c2f73f7e", role="companion", crop_ids=["p1"], pictures=[], story_id=partner),
    ]
    story = _story(7, sentence="A person walked right to left and toward the house on the porch, over 6 s "
                               "(5 detections), moving alongside another person.", persons=persons,
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
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [story], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/people/([^/]+)", path)
        if m:
            sent = json.loads(route.request.post_data or "{}")
            posts.append((m.group(2), sent))
            again = dict(story, persons=[dict(p, story_action={"action": "confirm", "name": "Sasha"})
                                         if p["person_key"] == "ind-sasha2" else p for p in persons])
            return route.fulfill(body=json.dumps({"ok": True, "story": again, "wrong_name_recorded": False,
                                                  "tally": _tally(1, 1)}), content_type="application/json")
        if re.fullmatch(r"/api/vision/motion-stories/[0-9a-f]{32}/strip/(crop_[0-3]|frame)\.jpg", path):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser, page = _page(p, html, handle)
        item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        assert item.locator("[data-story-progress]").get_attribute("data-story-progress") == "1/5"
        assert "1 without a picture of their own is not asked." in item.text_content()
        # The time crops above carry each person's letter.
        assert [c.get_attribute("data-story-crop-person")
                for c in item.locator("[data-story-crop-person]").all()] == ["A", "B", "C", "D"]
        want = [("A", "Is this Juniper?", f"/api/vision/motion-stories/{sid}/strip/crop_0.jpg", "1"),
                ("B", "Is this Sasha?", f"/api/vision/motion-stories/{sid}/strip/crop_1.jpg", "2,3"),
                ("C", "Is this Sasha?", f"/api/vision/motion-stories/{sid}/strip/crop_2.jpg", "4"),
                ("D", "Who is this?", f"/api/vision/motion-stories/{sid}/strip/crop_3.jpg", "5"),
                ("E", "Walking with them: Who is this?", f"/api/vision/motion-stories/{partner}/strip/crop_0.jpg", "")]
        for k, (letter, question, src, boxes) in enumerate(want):
            item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
            row = item.locator("[data-story-person]")
            assert row.get_attribute("data-person-letter") == letter
            assert row.locator("[data-person-question]").text_content() == question
            assert row.locator("img[data-person-picture]").get_attribute("src") == src
            if boxes:
                assert row.locator("[data-person-boxes]").get_attribute("data-person-boxes") == boxes
            else:
                assert "they walked alongside" in row.text_content()
            if k == 2:
                # The second "Sasha": told apart from the first by its box numbers and group id.
                assert "group ind-sash" in row.text_content()
                row.locator("[data-person-action='yes']").click()
                page.wait_for_function("() => !!document.querySelector(\"[data-person-said='confirm']\")")
                assert posts == [("ind-sasha2", {"action": "confirm", "name": "Sasha", "why": "", "group": True,
                                                 "card": "one_at_a_time", "dont_know": False, "crop_ids": ["k3"]})]
                assert "Saved: yes, that's Sasha." in page.locator("#walkwayStoriesStatus").text_content()
                # Still on the same person after saving: she moves on herself.
                assert page.locator("[data-story-progress]").get_attribute("data-story-progress") == "3/5"
            _fits(page)
            if k < len(want) - 1:
                page.locator("[data-story-nav='next']").click()
                page.wait_for_function(
                    f"() => document.querySelector('[data-story-progress]').getAttribute('data-story-progress') === '{k + 2}/5'")
        assert page.locator("[data-story-nav-done]").count() == 1
        page.locator("[data-story-nav='back']").click()
        page.wait_for_function("() => document.querySelector('[data-story-progress]').getAttribute('data-story-progress') === '4/5'")
        browser.close()


def test_walkway_story_linked_groups_are_one_person_asked_once_browser_smoke():
    """Who-card Patch A (2026-11-24): the two "Sasha" camera groups Juniper already linked are ONE person
    (both groups' pictures, "Seen as 2 camera groups"), a yes is sent once with group=true, and the
    person offers "not the same person?" for its direct link (Orion's question 1)."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    sid = f"{9:032x}"

    def pic(n, crop):
        return {"story_id": sid, "file": f"crop_{n}.jpg", "t": f"2026-11-22T01:28:3{n}+00:00", "crop_id": crop,
                "individual_id": "x"}

    sasha = _pp("ind-sasha1", individual_ids=["ind-sasha1", "ind-sasha2"], crop_ids=["k1", "k3"], label="Sasha",
                short_id="ind-sash", pictures=[pic(2, "k3"), pic(1, "k1")],
                boxes=[{"n": 2, "colour": "blue"}, {"n": 4, "colour": "blue"}],
                groups=[{"individual_id": "ind-sasha1", "short_id": "ind-sas1", "picture_crop_ids": ["k1"], "asked": True},
                        {"individual_id": "ind-sasha2", "short_id": "ind-sas2", "picture_crop_ids": ["k3"], "asked": True}],
                links=[{"a": "ind-sasha1", "b": "ind-sasha2"}])
    anon = _pp("ind-anon", crop_ids=["k4"], pictures=[pic(3, "k4")], boxes=[{"n": 5, "colour": "red"}])
    story = _story(9, sentence="Sasha walked right to left, over 6 s (4 detections).", persons=[sasha, anon],
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
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
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
        browser, page = _page(p, html, handle)
        item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        assert item.locator("[data-story-progress]").get_attribute("data-story-progress") == "1/2"   # Sasha once
        first = item.locator("[data-story-person]")
        assert first.locator("[data-person-question]").text_content() == "Is this Sasha?"
        assert first.locator("[data-person-groups]").text_content().startswith("Seen as 2 camera groups")
        assert [i.get_attribute("src") for i in first.locator("img[data-person-picture]").all()] == [
            f"/api/vision/motion-stories/{sid}/strip/crop_2.jpg", f"/api/vision/motion-stories/{sid}/strip/crop_1.jpg"]
        assert first.locator("[data-person-boxes]").get_attribute("data-person-boxes") == "2,4"
        assert [c.get_attribute("data-story-crop-person") for c in item.locator("[data-story-crop-person]").all()] == [
            "A", "A", "B"]
        pair = first.locator("[data-person-not-same-pair]")
        assert not pair.is_visible()
        first.locator("[data-person-not-same-toggle]").click()
        assert pair.is_visible() and pair.text_content() == "group ind-sas1 and group ind-sas2 are different people"
        _fits(page)
        first.locator("[data-person-action='yes']").click()
        page.wait_for_function("() => !!document.querySelector(\"[data-person-said='confirm']\")")
        assert posts == [(f"/api/vision/motion-stories/{sid}/people/ind-sasha1",
                          {"action": "confirm", "name": "Sasha", "why": "", "group": True, "card": "one_at_a_time",
                           "dont_know": False, "crop_ids": ["k3", "k1"]})]
        page.locator("[data-person-not-same-toggle]").first.click()
        page.locator("[data-person-not-same-pair]").first.click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes('not the same person')")
        assert posts[-1] == (f"/api/vision/motion-stories/{sid}/persons/ind-sasha1/not-same",
                             {"a": "ind-sasha1", "b": "ind-sasha2"})
        # The one unnamed person next: "It's ..." and "I don't know who" (an "unsure" that says why).
        page.locator("[data-story-nav='next']").click()
        page.wait_for_function("() => document.querySelector('[data-person-question]').textContent === 'Who is this?'")
        page.locator("[data-person-action='dont_know']").click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes(\"don't know\")")
        assert posts[-1] == (f"/api/vision/motion-stories/{sid}/people/ind-anon",
                             {"action": "unsure", "name": "", "why": "", "group": True, "card": "one_at_a_time",
                              "dont_know": True, "crop_ids": ["k4"]})
        browser.close()


def test_walkway_story_held_pick_is_never_lost_nor_doubled_browser_smoke():
    """Review fixes (Patch B): a held pick that fails to send comes back (not lost); another answer on the
    same person replaces it (never two rows); the last person's "Done" sends it."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    story = _story(3, name_label="Alex", individual_id="ind-alex", sentence="Alex walked.",
                   persons=[_pp("ind-anon"), _pp("ind-alex", label="Alex")])
    posts: list[tuple[str, dict]] = []
    fail = {"next": True}

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [story], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        m = re.fullmatch(r"/api/vision/motion-stories/([0-9a-f]{32})/people/([^/]+)", path)
        if m:
            if fail["next"]:
                fail["next"] = False
                return route.fulfill(status=503, body=json.dumps({"detail": "motion_story_store_unavailable"}),
                                     content_type="application/json")
            posts.append((m.group(2), json.loads(route.request.post_data or "{}")))
            return route.fulfill(body=json.dumps({"ok": True, "story": None, "tally": _tally(1, 1)}),
                                 content_type="application/json")
        if path == f"/api/vision/crop-thumbs/{H}":
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser, page = _page(p, html, handle)
        item = page.locator("#walkwayStoriesList [data-story-id]").nth(0)
        assert item.locator("[data-person-picker]").is_visible()          # unnamed: the names show at once
        item.locator("[data-person-pick='Theo']").click()
        page.wait_for_selector("[data-person-pending='Theo']")
        # Next fails (503): the held pick comes back, the card stays on this person.
        page.locator("[data-story-nav='next']").click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes('Could not save')")
        page.wait_for_selector("[data-person-pending='Theo']")
        assert page.locator("[data-story-progress]").get_attribute("data-story-progress") == "1/2"
        assert posts == []
        # Another answer on the same person replaces the held pick: one row, not two.
        page.locator("[data-person-action='cant_see']").click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes(\"won't ask\")")
        assert [(k, b["action"]) for k, b in posts] == [("ind-anon", "unsure")]
        assert page.locator("[data-person-pending]").count() == 0
        # The last person: a held pick, then "Done" sends it.
        page.locator("[data-story-nav='next']").click()
        page.wait_for_function("() => document.querySelector('[data-story-progress]').getAttribute('data-story-progress') === '2/2'")
        assert page.locator("[data-story-nav='done']").count() == 0          # nothing held: no Done
        page.locator("[data-person-action='no']").click()
        page.locator("[data-person-pick='Theo']").click()
        page.wait_for_selector("[data-story-nav='done']")
        page.locator("[data-story-nav='done']").click()
        page.wait_for_function("() => !document.querySelector('[data-person-pending]')")
        for _ in range(50):
            if len(posts) == 2:
                break
            page.wait_for_timeout(50)
        assert posts[-1] == ("ind-alex", {"action": "someone_else", "name": "Theo", "why": "", "group": True,
                                          "card": "one_at_a_time", "dont_know": False, "crop_ids": ["c1"]})
        assert len(posts) == 2
        browser.close()


def test_walkway_story_sentence_graded_in_parts_browser_smoke():
    """Who-card Patch C: only the parts the story's record states are asked ("Right people?", "Right
    direction?"); Yes / No / Can't tell POSTs the part; the old four buttons give way to "Can't tell";
    the tally says where verdicts came from. A story with no record keeps the old buttons."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    graded = _story(4, persons=[_pp("ind-a")], parts={"stated": ["people", "direction"], "answers": {}})
    bare = _story(5, persons=[_pp("ind-b")], parts={"stated": None, "answers": {}})
    posts: list[tuple[str, dict]] = []
    tally = dict(_tally(3, 2), by_source={"button": {"right": 2}, "parts": {"wrong_motion": 1}}, half_graded=1)

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [graded, bare], "tally": tally}),
                                 content_type="application/json")
        if path.endswith("/parts"):
            sent = json.loads(route.request.post_data or "{}")
            posts.append((path, sent))
            again = dict(graded, parts={"stated": ["people", "direction"],
                                        "answers": {"people": {"verdict": "yes", "source": "juniper"}}})
            return route.fulfill(body=json.dumps({"ok": True, "story": again, "story_verdict": None, "tally": tally}),
                                 content_type="application/json")
        if path == f"/api/vision/crop-thumbs/{H}":
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser, page = _page(p, html, handle)
        items = page.locator("#walkwayStoriesList [data-story-id]")
        first = items.nth(0)
        assert [r.get_attribute("data-story-part") for r in first.locator("[data-story-part]").all()] == [
            "people", "direction"]
        assert "Right number of people?" in first.text_content() and "Right place?" not in first.text_content()
        # Parts only: no lone whole-story "Can't tell" under them (it read as an answer she never gave).
        assert first.locator("[data-story-verdict]").count() == 0
        assert first.locator("[data-story-parts-sentence]").text_content().startswith('Orion wrote: "A person walked')
        assert "Right number of people?" in first.text_content()
        assert items.nth(1).locator("[data-story-part]").count() == 0
        assert items.nth(1).locator("[data-story-verdict]").count() == 5          # no record: the old buttons
        assert "from parts 1, from the old buttons 2" in page.locator("#walkwayStoriesTally").text_content()
        assert "half-graded 1 (not counted yet)" in page.locator("#walkwayStoriesTally").text_content()
        _fits(page)
        first.locator("[data-story-part='people'] [data-part-verdict='yes']").click()
        page.wait_for_function("() => document.getElementById('walkwayStoriesStatus').textContent.includes('1 more part')")
        assert posts == [(f"/api/vision/motion-stories/{4:032x}/parts", {"part": "people", "verdict": "yes"})]
        assert page.locator("[data-story-part='people'] [data-part-verdict='yes']").get_attribute("aria-pressed") == "true"
        browser.close()


def test_walkway_story_same_name_question_and_link_undo_browser_smoke():
    """Who-card Patch D: "Orion has another Sasha group. Same person?" with both pictures; Yes POSTs her
    answer; the link then shows with Undo, which asks Orion's next pass to remove it."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    html = _html()
    other = {"individual_id": "ind-s2", "short_id": "ind-s2", "label": "Sasha", "thumb": GONE}
    sasha = _pp("ind-s1", label="Sasha", same_name_other=other, card_links=[])
    story = _story(6, persons=[sasha])
    linked = dict(story, persons=[dict(sasha, same_name_other=None, card_links=[
        {"ask_id": "ask-" + "a" * 32, "a": "ind-s1", "b": "ind-s2", "applied": False, "undone_at": None,
         "undo_requested_at": None, "undo_refused": None}])])
    posts: list[tuple[str, dict]] = []

    def handle(route):
        url = route.request.url
        path = url.split("http://hub.test", 1)[1].split("?", 1)[0]
        if path == "/":
            return route.fulfill(body=html, content_type="text/html")
        if path.startswith("/static/"):
            return _static(route, path)
        if path == "/api/vision/known-names":
            return route.fulfill(body=json.dumps(NAMES), content_type="application/json")
        if path == "/api/vision/motion-stories":
            return route.fulfill(body=json.dumps({"ok": True, "stories": [story], "tally": _tally(1, 1)}),
                                 content_type="application/json")
        if route.request.method == "POST":
            posts.append((path, json.loads(route.request.post_data or "{}")))
            body = {"ok": True, "story": linked, "tally": _tally(1, 1)} if path.endswith("/same-name") else {"ok": True}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        if path.startswith("/api/vision/crop-thumbs/"):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser, page = _page(p, html, handle)
        q = page.locator("[data-person-same-name='ind-s2']")
        assert "Orion has another Sasha group. Same person?" in q.text_content()
        assert q.locator("img").count() == 2
        _fits(page)
        q.locator("[data-person-same-name-answer='same']").click()
        page.wait_for_selector("[data-person-link-undo]")
        assert posts == [(f"/api/vision/motion-stories/{6:032x}/persons/ind-s1/same-name",
                          {"other": "ind-s2", "answer": "same"})]
        assert page.locator("[data-person-same-name]").count() == 0
        assert "Orion links them on its next pass" in page.locator("[data-person-card-link]").text_content()
        page.locator("[data-person-link-undo]").click()
        page.wait_for_function("() => document.querySelector('[data-person-card-link]').textContent.includes('Undo asked for')")
        assert posts[-1] == (f"/api/vision/individual-links/ask-{'a' * 32}/undo", {})
        assert page.locator("[data-person-link-undo]").count() == 0
        browser.close()
