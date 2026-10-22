"""Browser smoke for the "People Orion knows" panel (2026-10-22).

The REAL panel markup (cut out of templates/index.html by its id) and the REAL
static/js/people.js in Chromium, with the Hub API intercepted (no Hub process).
Covers adding a fact, removing one, "Not a person", inert database text, and 360 px.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
JS = HUB_ROOT / "static" / "js" / "people.js"
THUMB = "ab" * 32


def _panel_html() -> str:
    import re

    html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'(<details id="peopleCard".*?</details>)', html, re.S)
    assert m, "panel markup not found in index.html"
    return m.group(1)


def test_people_panel_browser_smoke() -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    page_html = f"<!doctype html><html><body>{_panel_html()}<script src=\"/static/js/people.js\"></script></body></html>"
    script = JS.read_text(encoding="utf-8")
    state = {"facts": [{"fact_id": "f1", "category": "relationship", "label": "Relationship",
                        "value": "<img src=x onerror=\"window.__pwned=1\">"}], "status": "active"}
    posted: list = []

    def people():
        return [{"person_id": "p-sam", "name": "Sam", "status": state["status"], "days": 3, "clusters": 2,
                 "belongings": 0, "image_ref": f"thumb:{THUMB}", "facts": state["facts"],
                 "open_question": {"ask_id": "a1", "question": "What pronouns should I use for Sam?"}}]

    def handle(route):
        url = route.request.url
        if url.endswith("/home"):
            return route.fulfill(body=page_html, content_type="text/html")
        if url.endswith("/static/js/people.js"):
            return route.fulfill(body=script, content_type="application/javascript")
        if url.endswith("/api/people") and route.request.method == "GET":
            return route.fulfill(body=json.dumps({"ok": True, "people": people(), "categories": [
                {"key": "relationship", "label": "Relationship"}, {"key": "age", "label": "Age"}]}),
                content_type="application/json")
        if route.request.method == "POST":
            body = json.loads(route.request.post_data or "{}")
            posted.append((url.split("hub.test")[1], body))
            if url.endswith("/facts"):
                state["facts"] = state["facts"] + [{"fact_id": "f2", "category": body["category"], "label": "Age",
                                                    "value": body["value"]}]
            elif url.endswith("/remove"):
                state["facts"] = [f for f in state["facts"] if f["fact_id"] != "f1"]
            elif url.endswith("/status"):
                state["status"] = body["status"]
            return route.fulfill(body=json.dumps({"ok": True}), content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/home")
        page.click("#peopleCard summary")
        page.wait_for_selector('[data-person-id="p-sam"]')
        assert page.inner_text("#peopleStatus") == "Orion knows 1 person by name."
        card = page.inner_text('[data-person-id="p-sam"]')
        assert "Seen on camera on 3 days" in card and "Orion is asking: What pronouns should I use for Sam?" in card
        assert "<img src=x" in card and page.evaluate("window.__pwned") is None   # database text is inert
        page.select_option('[data-person-id="p-sam"] [data-person-field="category"]', "age")
        page.fill('[data-person-id="p-sam"] [data-person-field="value"]', "about ten")
        page.click('[data-person-id="p-sam"] [data-person-action="add-fact"]')
        page.wait_for_function("document.querySelector('[data-person-id=\"p-sam\"]').innerText.includes('Age: about ten')")
        page.click('[data-person-id="p-sam"] [aria-label^="Remove Relationship"]')
        page.wait_for_function("!document.querySelector('[data-person-id=\"p-sam\"]').innerText.includes('Relationship:')")
        page.click('[data-person-id="p-sam"] [data-person-action="status"]')
        page.wait_for_function("document.querySelector('[data-person-id=\"p-sam\"]').innerText.includes('This is a person')")
        assert page.inner_text("#peopleStatus") == "Orion does not know anyone by name yet."
        page.set_viewport_size({"width": 360, "height": 800})
        card_box = page.locator('[data-person-id="p-sam"]').bounding_box()
        assert card_box["x"] + card_box["width"] <= 360 + 1
        browser.close()
    assert posted == [("/api/people/p-sam/facts", {"category": "age", "value": "about ten"}),
                      ("/api/people/p-sam/facts/f1/remove", {}),
                      ("/api/people/p-sam/status", {"status": "not_a_person"})]
    assert errors == []
