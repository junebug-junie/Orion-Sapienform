"""Browser smoke for the "Orion is asking" panel (memory confirmation loop, 2026-10-06).

The REAL panel markup (cut out of templates/index.html by its id) and the REAL
static/js/vision-asks.js in Chromium, with the Hub API intercepted (no Hub process). Covers the
changed interactions: memory cards render Confirm / Revise / Reject, each button POSTs the right
body to /api/asks/{id}/resolve and the card leaves the list, Revise is prefilled and cannot be
sent empty, a vision card still gets Answer / Dismiss, and database text stays inert.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]


def _panel_html() -> str:
    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'(<div id="visionAsksCard".*?<div id="visionAsksList"[^>]*></div>\s*</div>)', html, re.S)
    assert m, "panel markup not found in index.html"
    return m.group(1)


MEM = {
    "m-confirm": "You told me something on Oct 3, and I wrote it down like this: “Juniper said X.” Want me to remember that?",
    "m-revise": "This came from my own reverie on Oct 4, not from anything you told me. I wrote it down as: “Y.” It's about us, so I don't want to decide it alone. Want me to remember that?",
    "m-reject": "<img src=x onerror=\"window.__pwned=1\"> Want me to remember that?",
}


def _asks(open_ids):
    out = [{"ask_id": i, "question": MEM[i], "source_kind": "memory_confirmation",
            "source_ref": f"memory-confirm-{i}", "memory_statement": f"statement for {i}",
            "created_at": "2026-10-06T12:00:00+00:00", "evidence_refs": []} for i in open_ids if i in MEM]
    if "v1" in open_ids:
        out.append({"ask_id": "v1", "question": "Who is this?", "source_kind": "vision_individual",
                    "source_ref": "ind-1", "created_at": "2026-10-06T11:00:00+00:00", "evidence_refs": []})
    if "p1" in open_ids:
        out.append({"ask_id": "p1", "question": "Are these the same person?", "source_kind": "vision_same_individual",
                    "source_ref": "a|b", "created_at": "2026-10-15T21:40:00+00:00", "evidence_refs": [],
                    "image_ref": f"thumb:{FACE_A}", "image_refs": [f"thumb:{FACE_A}", f"thumb:{FACE_B}"]})
    return out


FACE_A = "a" * 64
FACE_B = "b" * 64
# 1x1 JPEG-ish bytes are not needed: a 1x1 GIF is a valid image for the browser.
PIXEL = bytes.fromhex("47494638396101000100800000ffffff00000021f90401000000002c00000000010001000002024401003b")


def test_orion_is_asking_panel_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    page_html = f"<!doctype html><html><body>{_panel_html()}<script src=\"/static/js/vision-asks.js\"></script></body></html>"
    script = (HUB / "static" / "js" / "vision-asks.js").read_text(encoding="utf-8")
    open_ids = {"m-confirm", "m-revise", "m-reject", "v1", "p1"}
    posted: list[tuple[str, dict]] = []

    def handle(route):
        url = route.request.url
        if url.endswith("/home"):
            return route.fulfill(body=page_html, content_type="text/html")
        if url.endswith("/static/js/vision-asks.js"):
            return route.fulfill(body=script, content_type="application/javascript")
        if url.endswith(f"/api/vision/crop-thumbs/{FACE_A}"):
            return route.fulfill(body=PIXEL, content_type="image/gif")
        if route.request.method == "POST":
            m = re.search(r"/api/asks/([^/]+)/(resolve|answer|dismiss)$", url)
            body = json.loads(route.request.post_data or "{}")
            posted.append((f"{m.group(1)}/{m.group(2)}", body))
            open_ids.discard(m.group(1))
            return route.fulfill(body=json.dumps({"ok": True}), content_type="application/json")
        if "/api/asks?status=open" in url:
            return route.fulfill(body=json.dumps({"ok": True, "asks": _asks(open_ids)}),
                                 content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/home")

        page.wait_for_selector('[data-ask-id="m-confirm"]')
        assert "Orion has 5 questions for you." in page.inner_text("#visionAsksStatus")
        assert "Orion is asking" in page.inner_text("#visionAsksCard")
        # Database text is inert: the markup in a question renders as text and never runs.
        assert "<img src=x" in page.inner_text('[data-ask-id="m-reject"]')
        assert page.evaluate("window.__pwned") is None
        # The vision card keeps its own buttons; memory cards get the three answers.
        assert page.locator('[data-ask-id="v1"] [data-ask-action="answer"]').count() == 1
        assert page.locator('[data-ask-id="m-confirm"] [data-ask-action="answer"]').count() == 0

        # "Same person?" card: both faces side by side, the pruned one says so
        # instead of a broken image, and the answer is a button, not free text.
        page.wait_for_selector('[data-ask-id="p1"] img[alt="Left face Orion is asking about"]')
        assert page.locator('[data-ask-id="p1"] img').count() == 2
        page.wait_for_function(
            "document.querySelector('[data-ask-id=\"p1\"]').innerText.includes('Picture no longer available.')")
        boxes = [page.locator('[data-ask-id="p1"] figure').nth(i).bounding_box() for i in range(2)]
        assert abs(boxes[0]["y"] - boxes[1]["y"]) < 2 and boxes[1]["x"] > boxes[0]["x"], boxes  # side by side
        assert page.locator('[data-ask-id="p1"] [data-ask-input]').count() == 0
        page.set_viewport_size({"width": 360, "height": 800})  # phone width: both still inside the card
        card_box = page.locator('[data-ask-id="p1"]').bounding_box()
        fig_b = page.locator('[data-ask-id="p1"] figure').nth(1).bounding_box()
        assert fig_b["x"] + fig_b["width"] <= card_box["x"] + card_box["width"] + 1, (fig_b, card_box)
        page.click('[data-ask-id="p1"] [data-ask-action="same"]')
        page.wait_for_selector('[data-ask-id="p1"]', state="detached")
        assert posted[-1] == ("p1/answer", {"answer": "yes"})
        posted.clear()

        page.click('[data-ask-id="m-confirm"] [data-ask-action="confirm"]')
        page.wait_for_selector('[data-ask-id="m-confirm"]', state="detached")
        assert posted[-1] == ("m-confirm/resolve", {"resolution": "confirmed", "note": ""})

        # Revise: hidden until clicked, prefilled with the current wording, never sent empty.
        assert not page.is_visible('[data-ask-revise="m-revise"]')
        page.click('[data-ask-id="m-revise"] [data-ask-action="revise"]')
        assert page.is_visible('[data-ask-revise="m-revise"]')
        assert page.input_value('[data-ask-id="m-revise"] textarea') == "statement for m-revise"
        page.fill('[data-ask-id="m-revise"] textarea', "   ")
        page.click('[data-ask-id="m-revise"] [data-ask-action="save-revision"]')
        page.wait_for_function(
            "document.querySelector('[data-ask-id=\"m-revise\"]').textContent.includes('Write how I should remember it first.')")
        assert len(posted) == 1
        page.fill('[data-ask-id="m-revise"] textarea', "We decided Y together, not alone.")
        page.click('[data-ask-id="m-revise"] [data-ask-action="save-revision"]')
        page.wait_for_selector('[data-ask-id="m-revise"]', state="detached")
        assert posted[-1] == ("m-revise/resolve", {"resolution": "revised", "note": "We decided Y together, not alone."})

        page.click('[data-ask-id="m-reject"] [data-ask-action="reject"]')
        page.wait_for_selector('[data-ask-id="m-reject"]', state="detached")
        assert posted[-1] == ("m-reject/resolve", {"resolution": "rejected", "note": ""})
        page.wait_for_function("document.getElementById('visionAsksStatus').textContent.includes('1 question')")
        assert errors == [], errors
        browser.close()
