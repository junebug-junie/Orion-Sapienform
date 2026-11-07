"""Real-browser smoke for the "Ask Orion" modal (button above Send in Hub's chat).

Loads the real templates/index.html, the real orion-abilities.js/.css, the real catalog
static/catalog/orion-abilities.json and the real app.js in headless Chromium. Every /api/* call is
answered here, so no Hub, bus or database is touched. Checks the changed interactions, not just
page load:

  1. The "Ask Orion" button is visible and sits above Send.
  2. Clicking it opens the modal with every catalog section; the search box has focus.
  3. Tab and Shift+Tab stay inside the dialog; Esc closes it and focus returns to the button.
  4. Typing in the search box filters the cards and updates the count.
  5. Insert puts the message into the chat box after text Juniper already typed (nothing lost),
     closes the modal and focuses the chat box. Nothing is sent: no chat POST, no websocket frame,
     no new bubble in the conversation.
  6. Enter in the search box inserts the first match, again without sending. A control then
     presses Send and confirms the "nothing was sent" checks can see a real send.
  7. At phone width (390 px) the dialog fills the screen and the cards stack in one column.

Run (needs the `playwright` Python package and a Chromium for it):

    python services/orion-hub/evals/orion_abilities_browser.py [--screenshot out.png]

Exit 0 on pass, 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
CATALOG = json.loads((HUB / "static" / "catalog" / "orion-abilities.json").read_text(encoding="utf-8"))
TYPED = "I was thinking about you today."


def render_index() -> str:
    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    subs = {
        "{{HUB_UI_ASSET_VERSION}}": "smoke",
        "{{NOTIFY_TOAST_SECONDS}}": "8",
        "{{HUB_MEMORY_STORE_READY}}": "false",
        "{{HUB_MEMORY_STORE_BANNER_TEXT}}": "",
        "{{HUB_MEMORY_STORE_BANNER_CLASS}}": "hidden",
        "{{HUB_CFG}}": "{}",
        "{{HUB_AUTONOMY_SUBJECT_DISPLAY}}": "Orion",
        "{{HUB_AITOWN_TAB_NAV}}": "",
        "{{HUB_AITOWN_PANEL}}": "",
    }
    for key, value in subs.items():
        html = html.replace(key, value)
    # Offline fallback for the Tailwind rules the hidden modals and screen-reader label rely on.
    return html.replace(
        "</head>",
        "<style>.hidden{display:none!important}.sr-only{position:absolute;width:1px;height:1px;"
        "overflow:hidden;clip:rect(0,0,0,0)}</style></head>",
        1,
    )


def run(screenshot: str | None) -> int:
    from playwright.sync_api import sync_playwright

    failures: list = []
    sent: list = []
    catalog_hits: list = []
    index_html = render_index()

    def check(cond: bool, label: str) -> None:
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    def handle(route, request):
        url = request.url
        if url.startswith("https://cdn.tailwindcss.com") or url.startswith("https://fonts."):
            return route.continue_()
        if not url.startswith("http://hub.smoke"):
            return route.abort()
        rest = url.split("://", 1)[1]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
        path_only = path.split("?", 1)[0]
        if request.method != "GET" and "chat" in path_only:
            sent.append(f"{request.method} {path_only} {request.post_data or ''}"[:300])
        if path_only in ("/", "/index.html"):
            return route.fulfill(status=200, body=index_html, content_type="text/html")
        if path_only.startswith("/static/"):
            if path_only.endswith("orion-abilities.json"):
                catalog_hits.append(path)
            file = HUB / path_only.lstrip("/")
            if file.is_file():
                ctype = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                return route.fulfill(status=200, body=file.read_bytes(), content_type=ctype)
            return route.fulfill(status=404, body="")
        if path_only == "/api/session":
            return route.fulfill(status=200, body=json.dumps({"session_id": "sess-smoke"}), content_type="application/json")
        if path_only.startswith("/api/"):
            return route.fulfill(status=200, body="[]" if request.method == "GET" else "{}", content_type="application/json")
        return route.fulfill(status=404, body="")

    def on_ws(ws):
        ws.on("framesent", lambda payload: sent.append(f"ws {str(payload)[:200]}"))

    def active_id(page) -> str:
        return page.evaluate("() => (document.activeElement && document.activeElement.id) || ''")

    def focus_in_dialog(page) -> bool:
        return page.evaluate("() => document.getElementById('orionAbilitiesDialog').contains(document.activeElement)")

    total = sum(len(s["entries"]) for s in CATALOG["sections"])
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("websocket", on_ws)
        page.route("**/*", handle)
        page.goto("http://hub.smoke/")
        page.wait_for_selector("#orionAbilitiesButton", state="visible", timeout=20000)
        page.wait_for_function("() => !!window.OrionAbilities", timeout=20000)

        # 1. placement
        btn = page.locator("#orionAbilitiesButton").bounding_box()
        send = page.locator("#sendButton").bounding_box()
        check(btn is not None and send is not None and btn["y"] + btn["height"] <= send["y"] + 1,
              "Ask Orion button sits above Send")
        check(btn is not None and send is not None and abs(btn["x"] - send["x"]) < 2, "Ask Orion lines up with Send")

        # Juniper has already typed something; the caret is at its end.
        page.locator("#chatInput").fill(TYPED)
        bubbles_before = page.locator("#conversation > *").count()

        # 2. open
        page.locator("#orionAbilitiesButton").click()
        page.wait_for_selector("#orionAbilitiesList .oa-card", timeout=10000)
        check(page.locator("#orionAbilitiesModalRoot").is_visible(), "modal opens")
        check(page.locator("#orionAbilitiesButton").get_attribute("aria-expanded") == "true", "aria-expanded true")
        check(active_id(page) == "orionAbilitiesSearch", "search box has focus on open")
        check(page.locator("#orionAbilitiesList .oa-section").count() == len(CATALOG["sections"]),
              f"all {len(CATALOG['sections'])} sections shown")
        check(page.locator("#orionAbilitiesList .oa-card").count() == total, f"all {total} cards shown")
        check(page.locator("#orionAbilitiesNav .oa-chip").count() == len(CATALOG["sections"]), "one nav chip per section")
        check(f"{total} things to ask" in page.locator("#orionAbilitiesCount").inner_text(), "count reads the total")
        check(any("?v=" in h for h in catalog_hits), "catalog fetched with the cache-busting version")
        if screenshot:
            page.screenshot(path=screenshot)

        # 3. focus trap + Esc
        page.keyboard.press("Shift+Tab")
        check(focus_in_dialog(page), "Shift+Tab from the first control stays in the dialog")
        for _ in range(3):
            page.keyboard.press("Tab")
        check(focus_in_dialog(page), "Tab stays in the dialog")
        page.keyboard.press("Escape")
        check(not page.locator("#orionAbilitiesModalRoot").is_visible(), "Esc closes the modal")
        check(active_id(page) == "orionAbilitiesButton", "focus returns to the button")
        check(page.locator("#chatInput").input_value() == TYPED, "closing changed nothing in the chat box")

        # 4. filter
        page.locator("#orionAbilitiesButton").click()
        page.wait_for_selector("#orionAbilitiesList .oa-card")
        page.locator("#orionAbilitiesSearch").fill("portrait")
        cards = page.locator("#orionAbilitiesList .oa-card")
        n = cards.count()
        check(0 < n < total, f"search 'portrait' narrows the cards ({n} of {total})")
        check(f"{n} of {total} shown" in page.locator("#orionAbilitiesCount").inner_text(), "count says how many match")
        target = page.locator('#orionAbilitiesList .oa-card[data-entry-id="portrait-write"]')
        check(target.count() == 1, "the portrait-write card is among the matches")
        page.locator("#orionAbilitiesSearch").fill("zzzz no such thing")
        check("Nothing matches" in page.locator("#orionAbilitiesList").inner_text(), "no match says so")
        page.locator("#orionAbilitiesSearch").fill("portrait")

        # 5. insert
        entry = next(e for s in CATALOG["sections"] for e in s["entries"] if e["id"] == "portrait-write")
        target.locator(".oa-insert").click()
        value = page.locator("#chatInput").input_value()
        check(not page.locator("#orionAbilitiesModalRoot").is_visible(), "Insert closes the modal")
        check(value.startswith(TYPED), "text already typed is kept")
        check(entry["message"].strip() in value and entry["hint"] in value, "the message and its hint are inserted")
        check(value == TYPED + "\n\n" + entry["message"].strip() + "\n" + entry["hint"], "inserted after a blank line")
        check(active_id(page) == "chatInput", "the chat box has focus after Insert")
        caret = page.evaluate("() => document.getElementById('chatInput').selectionEnd")
        check(caret == len(value), "caret sits at the end of the inserted text")

        # 6. Enter in search inserts the first match (with a fill-in colon: caret lands after it)
        page.locator("#chatInput").fill("")
        page.locator("#orionAbilitiesButton").click()
        page.wait_for_selector("#orionAbilitiesList .oa-card")
        page.locator("#orionAbilitiesSearch").fill("read this page now")
        first_id = page.locator("#orionAbilitiesList .oa-card").first.get_attribute("data-entry-id")
        page.keyboard.press("Enter")
        value2 = page.locator("#chatInput").input_value()
        first = next(e for s in CATALOG["sections"] for e in s["entries"] if e["id"] == first_id)
        check(value2.startswith(first["message"].strip()), f"Enter in search inserts the first match ({first_id})")
        if first["message"].strip().endswith(":"):
            caret2 = page.evaluate("() => document.getElementById('chatInput').selectionEnd")
            check(caret2 == len(first["message"].strip()) + 1, "caret lands after the colon to fill in")

        page.wait_for_timeout(500)
        check(not sent, f"nothing was sent ({sent[:2]})")
        check(page.locator("#conversation > *").count() == bubbles_before, "no new bubble in the conversation")
        if screenshot:
            page.locator("#composerRow").screenshot(path=screenshot.replace(".png", "-composer.png"))

        # Control: the two "nothing was sent" checks above can see a send. Pressing Send itself must
        # trip at least one of them, or they would pass for the wrong reason.
        page.locator("#sendButton").click()
        page.wait_for_timeout(800)
        check(bool(sent) or page.locator("#conversation > *").count() > bubbles_before,
              "control: pressing Send is detected (so 'nothing was sent' is a real check)")
        sent.clear()

        # 7. phone width
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#orionAbilitiesButton").scroll_into_view_if_needed()
        page.locator("#orionAbilitiesButton").click()
        page.wait_for_selector("#orionAbilitiesList .oa-card")
        box = page.locator("#orionAbilitiesDialog").bounding_box()
        check(box is not None and box["width"] >= 385 and box["height"] >= 830, "dialog fills a phone screen")
        c0 = page.locator("#orionAbilitiesList .oa-card").nth(0).bounding_box()
        c1 = page.locator("#orionAbilitiesList .oa-card").nth(1).bounding_box()
        check(c0 is not None and c1 is not None and c1["y"] > c0["y"] and abs(c1["x"] - c0["x"]) < 2,
              "cards stack in one column on a phone")
        if screenshot:
            page.screenshot(path=screenshot.replace(".png", "-phone.png"))
        page.keyboard.press("Escape")

        oa_errors = [e for e in errors if "OrionAbilities" in e or "orion-abilities" in e or "oa-" in e]
        check(not oa_errors, f"no page errors from the modal ({oa_errors[:2]})")
        browser.close()

    print(json.dumps({"sent": sent, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default=None)
    sys.exit(run(parser.parse_args().screenshot))
