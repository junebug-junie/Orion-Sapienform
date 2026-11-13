"""Personality tab browser smoke (rev 4 move 4): the REAL templates/personality.html and
static/js/personality.js in Chromium, Hub API intercepted.

Covers the changed interactions: recent choices (picked, passed over with rank, a "said nothing"
decline, a hostile option text that must stay text), tendencies with examples, the Yes button
(saved, card stays), the No button (confirm dialog, POST, card retired, buttons gone), a failed
save (named error, buttons back), the "not live yet" market readout, and an upstream outage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
HOSTILE = '<img src=x onerror="window.__pwned=1">walkway misnaming'

CHOICES = {"count": 2, "choices": [
    {"choice_id": "outreach:o1", "lane": "outreach", "occurred_at": "2026-11-13T08:00:00Z", "chosen": None,
     "passed_over": [{"key": "g", "text": "the gap", "market_rank": 1}], "chosen_market_rank": None,
     "declined_all": True, "care_promoted": False, "picks_in_decision": 1, "tendency": None},
    {"choice_id": "curiosity:r1:x", "lane": "curiosity", "occurred_at": "2026-11-13T07:29:39Z",
     "chosen": {"key": "x", "text": HOSTILE, "market_rank": 4},
     "passed_over": [{"key": "a", "text": "peer briefs refused budget", "market_rank": 1},
                     {"key": "b", "text": "v2 relative switch", "market_rank": 2},
                     {"key": "c", "text": "wp read record grows", "market_rank": 3},
                     {"key": "d", "text": "four wiring points named", "market_rank": None}],
     "chosen_market_rank": 4, "declined_all": False, "care_promoted": False, "picks_in_decision": 1,
     "tendency": {"disposition_id": "disp-1", "label": "walkway misnaming over peer briefs, 3 of 3",
                  "status": "active"}},
]}
TENDENCIES = {"count": 3, "warnings": [], "tendencies": [
    {"disposition_id": "disp-1", "label": "walkway misnaming over peer briefs (+14 more), 21 of 23",
     "status": "active", "weight": 12.0, "n": 21, "m": 23, "name": None,
     "examples": [CHOICES["choices"][1]], "juniper": None},
    {"disposition_id": "disp-2", "label": "say nothing over the gap, 3 of 3", "status": "candidate", "weight": 2.0,
     "n": 3, "m": 3, "name": None, "examples": [CHOICES["choices"][0]], "juniper": None},
    {"disposition_id": "disp-3", "label": "a over b, 4 of 4", "status": "candidate", "weight": 3.0, "n": 4, "m": 4,
     "name": None, "examples": [], "juniper": {"verdict": "yes", "at": "2026-11-12T10:00:00Z"}},
]}
ARMS_OFF = {"days": 7, "since": "2026-11-06T00:00:00Z", "arms": {}, "total": 0}


@pytest.mark.parametrize("scenario", ["ok", "down"])
def test_personality_tab_yes_no_browser_smoke(scenario):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    template = (HUB / "templates" / "personality.html").read_text().replace("{{HUB_UI_ASSET_VERSION}}", "t")
    script = (HUB / "static" / "js" / "personality.js").read_text()
    posts: list[tuple[str, dict]] = []

    def handle(route):
        req = route.request
        url = req.url
        if url.endswith("/personality"):
            return route.fulfill(body=template, content_type="text/html")
        if "/static/js/personality.js" in url:
            return route.fulfill(body=script, content_type="application/javascript")
        if scenario == "down" and "/api/personality/" in url:
            return route.fulfill(status=502, body=json.dumps({"detail": "durable_runs_unavailable: ConnectError"}),
                                 content_type="application/json")
        if "/verdict" in url and req.method == "POST":
            did = url.split("/tendencies/")[1].split("/verdict")[0]
            body = json.loads(req.post_data or "{}")
            posts.append((did, body))
            if did == "disp-3":
                return route.fulfill(status=409, body=json.dumps({"detail": "tendency is already retired"}),
                                     content_type="application/json")
            retired = body.get("verdict") == "no"
            return route.fulfill(body=json.dumps({"verdict_id": "jv-1", "disposition_id": did,
                                                  "verdict": body.get("verdict"), "retired": retired,
                                                  "status": "retired" if retired else "active"}),
                                 content_type="application/json")
        if "/api/personality/choices" in url:
            return route.fulfill(body=json.dumps(CHOICES), content_type="application/json")
        if "/api/personality/tendencies" in url:
            return route.fulfill(body=json.dumps(TENDENCIES), content_type="application/json")
        if "/api/personality/arms" in url:
            return route.fulfill(body=json.dumps(ARMS_OFF), content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        dialogs: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def on_dialog(d):
            dialogs.append(d.message)
            d.accept()

        page.on("dialog", on_dialog)
        page.route("http://hub.test/**", handle)
        page.goto("http://hub.test/personality")
        page.wait_for_function("document.getElementById('pStatus').textContent !== 'loading…'")

        if scenario == "down":
            assert page.inner_text("#pStatus").startswith("Could not load: choices: durable_runs_unavailable")
            assert page.locator("#pChoices .choice").count() == 0
            assert page.locator("#pTendencies .tendency").count() == 0
            assert errors == [], errors
            browser.close()
            return

        assert page.inner_text("#pStatus") == ""
        # 1. Recent choices: a decline, then a pick with its rank and what it passed over.
        cards = page.locator("#pChoices .choice")
        assert cards.count() == 2
        assert "Said nothing" in cards.nth(0).inner_text()
        assert "Reaching out to you" in cards.nth(0).inner_text()
        pick = cards.nth(1).inner_text()
        assert "Picked: " in pick and "(market's #4)" in pick
        assert "#1 peer briefs refused budget · #2 v2 relative switch · #3 wp read record grows" in pick
        assert "and 1 more" in pick
        assert "Counts toward: walkway misnaming over peer briefs, 3 of 3" in pick
        assert HOSTILE in page.text_content("#pChoices .choice >> nth=1")          # shown as text
        assert page.locator("#pChoices img").count() == 0
        # 2. Tendencies: label without the count, n of m, Prime reach, an example.
        first = page.locator('#pTendencies .tendency[data-id="disp-1"]')
        text = first.inner_text()
        assert "walkway misnaming over peer briefs (+14 more)" in text and ", 21 of 23" not in text
        assert "21 of 23 choices went this way" in text and "In Orion's Prime block now" in text
        assert "Example choices:" in text and "You have not answered yet." in text
        assert "You said yes on" in page.locator('#pTendencies .tendency[data-id="disp-3"]').inner_text()
        # 3. Not live yet.
        assert page.inner_text("#pArmsSummary").startswith("Not live yet.")

        # Yes: saved, the card and its buttons stay.
        first.locator("button.verdict.yes").click()
        page.wait_for_function(
            "document.querySelector('.tendency[data-id=\"disp-1\"] .verdict-msg').textContent.startsWith('Saved')")
        assert posts[-1] == ("disp-1", {"verdict": "yes"})
        assert first.locator("button.verdict").count() == 2
        assert first.get_attribute("data-answered") == "yes"
        assert dialogs == []                                                         # yes never asks

        # No: asks first, then retires the card and removes its buttons.
        second = page.locator('#pTendencies .tendency[data-id="disp-2"]')
        second.locator("button.verdict.no").click()
        page.wait_for_selector('.tendency[data-id="disp-2"].retired')
        assert posts[-1] == ("disp-2", {"verdict": "no"})
        assert len(dialogs) == 1 and dialogs[0].startswith("Retire this tendency?")
        assert second.locator("button.verdict").count() == 0
        assert second.locator(".verdict-msg").inner_text() == "Retired. Orion will not lean on this any more."

        # A refused save names the reason and gives the buttons back.
        third = page.locator('#pTendencies .tendency[data-id="disp-3"]')
        third.locator("button.verdict.yes").click()
        page.wait_for_selector('.tendency[data-id="disp-3"] .verdict-msg.err')
        assert third.locator(".verdict-msg").inner_text() == "Not saved: tendency is already retired"
        assert third.locator("button.verdict.yes").is_enabled()

        assert page.evaluate("window.__pwned") is None
        assert errors == [], errors
        browser.close()
