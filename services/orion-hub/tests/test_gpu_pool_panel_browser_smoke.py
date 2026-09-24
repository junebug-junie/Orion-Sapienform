"""Browser smoke for the GPU pool panel: the REAL templates/gpu_pool.html + static/js/gpu_pool.js in
Chromium, with the Hub API intercepted (no Hub process). Covers the changed interactions, not just
page load: cards from discovery, live SSE events into the traffic view, a click on an event row
walking that lease, the lend control POSTing the right verb, and the historical range switch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]

CONFIG = {
    "cards": {"gpu0": {"vram_gb": 32, "lendable": True}, "gpu3": {"vram_gb": 32, "lendable": False}},
    "roles": {
        "chat": {"kind": "llm", "cards": ["gpu0"], "owner": ["chat"], "port": 8011},
        "metacog": {"kind": "llm", "cards": ["gpu3"], "owner": ["metacog", "fast"], "port": 8012},
        "experiment": {"kind": "llm", "cards": ["gpu0", "gpu3"], "owner": ["experiment"], "port": 8099,
                       "operator_only": True, "swap": {"evicts": "all", "load": "x", "unload": "y"}},
    },
    "classes": {"chat": {"roles": ["chat"]}, "metacog": {"roles": ["metacog", "chat"]},
                "fast": {"roles": ["metacog"]}, "experiment": {"roles": ["experiment"]}},
    "routes": {},
}
STATE = {
    "mode": "observe", "config_digest": "d1g3st", "generated_at": "2026-09-24T12:00:00Z",
    "cards": [{"card": "gpu0", "vram_gb": 32, "lendable": True, "lent": False, "swapped_in": [], "swap_state": "idle"},
              {"card": "gpu3", "vram_gb": 32, "lendable": False, "lent": False, "swapped_in": [], "swap_state": "idle"}],
    "roles": [
        {"role": "chat", "kind": "llm", "cards": ["gpu0"], "url": "u", "status": "confirmed", "slots": 1,
         "ctx_per_slot": 65536, "model_file": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", "profile_name": "chat-profile"},
        {"role": "metacog", "kind": "llm", "cards": ["gpu3"], "url": "u", "status": "mismatch", "slots": 4,
         "ctx_per_slot": 4096, "model_file": "Something.gguf", "profile_name": "p", "detail": "profile expects X"},
        {"role": "experiment", "kind": "llm", "cards": ["gpu0", "gpu3"], "url": "u", "status": "unloaded", "slots": 0},
    ],
    "unclaimed_servers": [], "leases": [{"lease_id": "L1", "request_id": "r", "holder": "gw", "work_class": "chat",
                                         "priority": "interactive", "kind": "request", "status": "granted", "role": "chat",
                                         "attempt": 1, "created_at": "2026-09-24T12:00:00Z"}],
    "queue_depth": {}, "backlog_depth": {},
    "config": CONFIG, "config_yaml": "version: 1\ncards:\n  gpu0: {vram_gb: 32}\n",
}
EVENT = {"event": "granted", "lease_id": "L1abcdef", "holder": "gw", "work_class": "chat", "priority": "interactive",
         "role": "chat", "waited_ms": 42.0, "generated_at": "2026-09-24T12:00:01Z"}
HISTORY = [{"event": "admit", "status": "queued", "at": "2026-09-24T12:00:00+00:00"},
           {"event": "grant", "from": "queued", "status": "granted", "role": "chat", "at": "2026-09-24T12:00:01+00:00"}]


def test_gpu_pool_panel_browser_smoke():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    template = (HUB / "templates" / "gpu_pool.html").read_text().replace("{{HUB_UI_ASSET_VERSION}}", "t")
    script = (HUB / "static" / "js" / "gpu_pool.js").read_text()
    posted: list[dict] = []
    history_minutes: list[str] = []
    csrf_headers: list[str | None] = []
    state_calls = {"n": 0}

    def handle(route):
        url = route.request.url
        if url.endswith("/gpu-pool"):
            return route.fulfill(body=template, content_type="text/html")
        if "/static/js/gpu_pool.js" in url:
            return route.fulfill(body=script, content_type="application/javascript")
        if "/api/gpu-pool/stream" in url:
            body = (f"event: snapshot\ndata: {json.dumps({'version': 1, 'state': STATE, 'events': []})}\n\n"
                    f"event: event\ndata: {json.dumps({'version': 2, 'kind': 'event', 'event': EVENT})}\n\n")
            return route.fulfill(body=body, content_type="text/event-stream")
        if "/api/gpu-pool/state" in url:
            state_calls["n"] += 1
            if state_calls["n"] == 1:                     # the pool is down when the page first loads
                return route.fulfill(status=504, body=json.dumps({"detail": "gpu_pool_rpc_timeout"}))
            data = dict(STATE, history=HISTORY, history_lease_id="L1abcdef") if "history_for=" in url else STATE
            return route.fulfill(body=json.dumps(data), content_type="application/json")
        if "/api/gpu-pool/history" in url:
            history_minutes.append(url.split("minutes=")[1].split("&")[0])
            return route.fulfill(body=json.dumps({"by_role": [{"role": "chat", "grants": 7, "failures": 1, "recalls": 0,
                                                              "wait_p50_ms": 12, "wait_p95_ms": 80}],
                                                  "by_class": [], "series": [], "events": [], "minutes": 60}),
                                 content_type="application/json")
        if "/api/gpu-pool/control" in url:
            posted.append(json.loads(route.request.post_data))
            csrf_headers.append(route.request.headers.get("x-requested-with"))
            return route.fulfill(body=json.dumps({"ok": True, "reason": None, "detail": {}}),
                                 content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("http://hub.test/**", handle)
        # The fake stream ends at once (a real one stays open), so the status text flips between
        # "STALE" and "disconnected, retrying"; record every value it takes instead of sampling one.
        page.add_init_script("""document.addEventListener('DOMContentLoaded', () => {
            window.__liveTexts = [];
            const el = document.getElementById('liveText');
            new MutationObserver(() => window.__liveTexts.push(el.textContent))
              .observe(el, {childList: true, characterData: true, subtree: true});
        });""")
        page.goto("http://hub.test/gpu-pool")

        # recovers by itself: the first SSE state frame triggers a config reload after the 504
        page.wait_for_selector('[data-role="chat"]')
        assert state_calls["n"] >= 2
        page.wait_for_function("(window.__liveTexts || []).some(t => t.includes('STALE'))")  # fixture state is old
        chat = page.inner_text('[data-role="chat"]')
        assert "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" in chat and "1/1 slots in use" in chat
        assert "profile expects X" in page.inner_text('[data-role="metacog"]')          # mismatch explained
        assert "may borrow: metacog" in chat                                               # borrow rule shown
        assert "experiment" in page.inner_text("#spanning")                                # multi-card seat
        assert "cards:" in page.text_content("#yaml")

        page.wait_for_selector('#events tr[data-lease="L1abcdef"]')                       # live SSE event
        assert "42 ms" in page.inner_text("#events")
        page.click('#events tr[data-lease="L1abcdef"]')                                    # walk that lease
        page.wait_for_selector("#walkSteps td.ev-grant")
        assert "queued → granted" in page.inner_text("#walkSteps")

        page.once("dialog", lambda d: d.accept())
        page.click('button[data-verb="lend"][data-card="gpu0"]')
        page.wait_for_function("document.getElementById('controlStatus').textContent.includes('lend: ok')")
        assert posted == [{"verb": "lend", "card": "gpu0"}]
        assert csrf_headers == ["orion-hub"]

        page.select_option("#range", "60")
        page.wait_for_function("document.getElementById('byRole').textContent.includes('80 ms')")
        assert history_minutes == ["60"]
        assert errors == [], errors
        browser.close()
