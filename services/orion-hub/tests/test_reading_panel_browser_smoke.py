"""Browser smoke for the Reading tab: the REAL templates/reading.html + static/js/reading.js in
Chromium, with the Hub API intercepted (no Hub process). Covers the changed interactions: list
renders, clicking a read shows its outputs (a rejected handoff is labelled, not shown as learning),
web-derived markup stays inert text, retry POSTs with the CSRF header, and submit queues a URL."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import pytest

HUB = Path(__file__).resolve().parents[1]

STATUS = {
    "available": True, "wallet_a": {"enabled": True, "done_today": 3},
    "wallet_b": {"enabled": True, "done_today": 1},
    "queue": {"pending": 2, "claimed": 1, "done": 9, "failed": 1, "skipped": 4},
    "stage2_queue": {"pending": 1, "claimed": 0, "done": 4, "failed": 1, "skipped": 0},
    "retries": {"max_attempts": 3}, "last_stage1_at": "2026-09-27T05:00:00Z",
}
DONE_ID = "finding:run:abc"
FAILED_ID = "reading:rejected"
ITEMS = [
    {"seed_id": DONE_ID, "kind": "finding", "url": "https://example.org/paper", "title": "A paper",
     "reading_status": "failed", "status": "done", "stage2_status": "failed", "requested_by": "world_pulse",
     "preview": "Orion learned that X.", "updated_at": "2026-09-27T05:30:00Z", "stage2_error": "bad_json"},
    {"seed_id": FAILED_ID, "kind": "reading", "url": "https://example.org/r", "title": "",
     "reading_status": "failed", "status": "failed", "stage2_status": "pending", "requested_by": "juniper",
     "invocation_context": "operator", "preview": "", "updated_at": "2026-09-27T04:00:00Z",
     "last_error": "no_read_evidence"},
    {"seed_id": "reading:dup", "kind": "reading", "url": "https://example.org/paper", "title": "",
     "reading_status": "skipped", "status": "skipped", "stage2_status": "pending", "requested_by": "orion",
     "duplicate_of": DONE_ID, "preview": "", "updated_at": "2026-09-27T03:00:00Z"},
]
DETAIL = {
    DONE_ID: {
        "seed_id": DONE_ID, "kind": "finding", "url": "https://example.org/paper", "title": "A paper",
        "status": "done", "stage2_status": "failed", "attempts": 0, "stage2_attempts": 1,
        "stage2_error": "bad_json", "reading_status": "failed", "requested_by": "world_pulse",
        "handoff_accepted": True, "created_at": "2026-09-27T04:00:00Z", "durable_turns": [], "aliases": [],
        "handoff": {"what_i_learned": "Orion learned that X. <img src=x onerror=\"window.__pwned=1\">",
                    "candidate_priors": [{"claim": "X holds", "confidence": 0.4}],
                    "concept_candidates": [{"label": "X", "definition": "a thing"}], "open_threads": ["why X?"],
                    "read_evidence": [{"tool_name": "WebFetch", "url": "https://example.org/paper", "content_chars": 900}]},
        "stage2_result": None,
        "journal": [{"title": "A paper", "source_ref": "world_pulse_read:t", "body": "journal body", "created_at": None}],
    },
    FAILED_ID: {
        "seed_id": FAILED_ID, "kind": "reading", "url": "https://example.org/r", "title": "",
        "status": "failed", "stage2_status": "pending", "attempts": 1, "stage2_attempts": 0,
        "last_error": "no_read_evidence", "reading_status": "failed", "requested_by": "juniper",
        "request": {"invocation_context": "operator", "why_now": "curious"}, "handoff_accepted": False,
        "handoff": {"what_i_learned": "Unverified summary.", "read_evidence": []}, "stage2_result": None,
        "journal": [], "durable_turns": [], "aliases": [], "created_at": "2026-09-27T04:00:00Z",
    },
}


def test_reading_panel_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    template = (HUB / "templates" / "reading.html").read_text().replace("{{HUB_UI_ASSET_VERSION}}", "t")
    script = (HUB / "static" / "js" / "reading.js").read_text()
    posted: list[tuple[str, dict, str | None]] = []

    def handle(route):
        url = route.request.url
        method = route.request.method
        if url.endswith("/reading"):
            return route.fulfill(body=template, content_type="text/html")
        if "/static/js/reading.js" in url:
            return route.fulfill(body=script, content_type="application/javascript")
        if url.endswith("/api/status"):
            return route.fulfill(body=json.dumps(STATUS), content_type="application/json")
        if method == "POST":
            posted.append((url.split("/api/")[1], json.loads(route.request.post_data or "{}"),
                           route.request.headers.get("x-requested-with")))
            if url.endswith("/api/reads"):
                body = {"seed_id": FAILED_ID, "status": "queued", "queue_position": 3, "queue_depth": 5}
            else:
                body = {"action": "requeued", "stage": 2}
            return route.fulfill(body=json.dumps(body), content_type="application/json")
        if "/api/reads?" in url:
            return route.fulfill(body=json.dumps({"items": ITEMS, "total": len(ITEMS), "phase": "all"}),
                                 content_type="application/json")
        if "/api/reads/" in url:
            seed = unquote(url.split("/api/reads/")[1])
            return route.fulfill(body=json.dumps(DETAIL[seed]), content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/reading")

        page.wait_for_selector(f'#reads tr[data-seed="{DONE_ID}"]')
        assert "3 read today" in page.inner_text("#stats")
        assert "Juniper (Hub)" in page.inner_text("#reads")
        assert "Orion learned that X." in page.inner_text("#reads")
        assert "merged into another read" in page.inner_text('#reads tr[data-seed="reading:dup"]')

        page.click(f'#reads tr[data-seed="{DONE_ID}"]')
        page.wait_for_selector("#retryStage2")
        detail = page.inner_text("#detail")
        assert "X holds" in detail and "40% sure" in detail and "900 characters" in detail
        assert "Rejected, not learned" not in detail
        assert page.evaluate("window.__pwned") is None                                    # markup stayed text
        assert "<img" in detail
        assert page.is_enabled("#retryStage2") and not page.is_enabled("#cancelRead")

        page.once("dialog", lambda d: d.accept())
        page.click("#retryStage2")
        page.wait_for_function("document.getElementById('actionStatus').textContent.includes('requeued')")
        assert posted[-1] == (f"reads/{DONE_ID.replace(':', '%3A')}/retry", {"stage": 2}, "orion-hub")

        page.click(f'#reads tr[data-seed="{FAILED_ID}"]')
        page.wait_for_function("document.getElementById('detail').textContent.includes('Rejected, not learned')")
        assert "curious" in page.inner_text("#detail")
        assert page.is_enabled("#retryStage1") and not page.is_enabled("#retryStage2")

        page.click("#readAgain")                                          # prefills, never POSTs
        assert page.input_value("#submitUrl") == "https://example.org/r"
        assert posted[-1][0].endswith("/retry")

        page.fill("#submitUrl", "https://example.org/new")
        page.fill("#submitWhy", "because")
        page.click("#submitButton")
        page.wait_for_function("document.getElementById('submitStatus').textContent.includes('3 of 5 in line')")
        assert posted[-1] == ("reads", {"url": "https://example.org/new", "why_now": "because", "title": ""}, "orion-hub")
        assert errors == [], errors
        browser.close()
