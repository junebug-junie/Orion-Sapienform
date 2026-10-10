"""Browser smoke for the composer's "GPU leases" modal (static/js/gpu_pool_leases.js).

Renders the real button + modal markup cut from templates/index.html with the real stylesheet and
script, stubs the two pool endpoints, and drives the switches: open, flip on, flip off, a refused
flip that must snap back, and Escape to close. Skips where Playwright is not installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]

STATE = {
    "cards": [
        {"card": "gpu0", "host": "circe", "vram_gb": 32, "lendable": True, "lent": False},
        {"card": "gpu1", "host": "circe", "vram_gb": 32, "lendable": False, "lent": False},
        {"card": "hecate-gpu0", "host": "hecate", "vram_gb": 32, "lendable": True, "lent": False},
    ],
    "roles": [
        {"role": "chat", "cards": ["gpu0"], "status": "confirmed"},
        {"role": "agent", "cards": ["gpu1"], "status": "confirmed"},
        {"role": "agent-deep", "cards": ["hecate-gpu0"], "status": "down"},
    ],
}


def _fragments() -> str:
    html = (HUB / "templates" / "index.html").read_text()
    button = re.search(r'<button\s+id="gpuLeasesOpen".*?</button>', html, re.S).group(0)
    start = html.index('<div id="gpuLeasesModal"')
    modal = html[start:html.index('<div id="mindRunsModal"', start)]
    return button + modal


def test_gpu_leases_modal_switches_flip_and_settle_on_the_pool_reply():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    css = (HUB / "static" / "css" / "style.css").read_text()
    script = (HUB / "static" / "js" / "gpu_pool_leases.js").read_text()
    page_html = f"<!doctype html><html><head><style>{css}</style></head><body>{_fragments()}<script>{script}</script></body></html>"
    posted: list[dict] = []
    refuse = {"card": None}

    def handle(route):
        url = route.request.url
        if url.endswith("/page"):
            return route.fulfill(status=200, content_type="text/html", body=page_html)
        if "/api/gpu-pool/state" in url:
            return route.fulfill(status=200, content_type="application/json", body=json.dumps(STATE))
        if url.endswith("/api/gpu-pool/control"):
            body = json.loads(route.request.post_data or "{}")
            posted.append({**body, "xrw": route.request.headers.get("x-requested-with")})
            if body["card"] == refuse["card"]:
                return route.fulfill(status=200, content_type="application/json",
                                     body=json.dumps({"ok": False, "reason": "unknown_card"}))
            return route.fulfill(status=200, content_type="application/json",
                                 body=json.dumps({"ok": True, "detail": {"card": body["card"], "lent": body["verb"] == "lend"}}))
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.route("**/*", handle)
        page.goto("http://hub.test/page")

        assert page.is_hidden("#gpuLeasesModal")
        page.click("#gpuLeasesOpen")
        page.wait_for_selector('[data-lease-card="hecate-gpu0"]')
        assert page.is_visible("#gpuLeasesModal")
        cards = page.eval_on_selector_all("[data-lease-card]", "els => els.map(e => e.dataset.leaseCard)")
        assert cards == ["gpu0", "hecate-gpu0"]   # gpu1 is not lendable
        assert page.locator('[data-lease-row="hecate-gpu0"] [data-lease-down]').count() == 1

        sw = page.locator('[data-lease-card="hecate-gpu0"]')
        sw.click()
        page.wait_for_function('document.querySelector(\'[data-lease-card="hecate-gpu0"]\').getAttribute("aria-checked") === "true" && !document.querySelector(\'[data-lease-card="hecate-gpu0"]\').disabled')
        knob = page.eval_on_selector('[data-lease-card="hecate-gpu0"] .gpu-lease-knob', "e => getComputedStyle(e).transform")
        assert knob not in ("none", "")   # the knob actually slid
        assert page.text_content("#gpuLeasesBadge") == "1 lent"
        assert page.is_visible("#gpuLeasesBadge")

        sw.click()
        page.wait_for_function('document.querySelector(\'[data-lease-card="hecate-gpu0"]\').getAttribute("aria-checked") === "false" && !document.querySelector(\'[data-lease-card="hecate-gpu0"]\').disabled')
        assert page.is_hidden("#gpuLeasesBadge")

        refuse["card"] = "gpu0"
        page.click('[data-lease-card="gpu0"]')
        page.wait_for_function('document.getElementById("gpuLeasesStatus").textContent.includes("unknown_card")')
        assert page.get_attribute('[data-lease-card="gpu0"]', "aria-checked") == "false"   # snapped back

        page.keyboard.press("Escape")
        assert page.is_hidden("#gpuLeasesModal")
        browser.close()

    assert [(b["verb"], b["card"]) for b in posted] == [("lend", "hecate-gpu0"), ("unlend", "hecate-gpu0"), ("lend", "gpu0")]
    assert all(b["xrw"] == "orion-hub" for b in posted)
