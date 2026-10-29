"""Real-browser smoke for Orion's messages in the notification tray.

Loads the real templates/index.html, the real orion-message-tray.js and the
real app.js in headless Chromium. Every /api/* call is answered by this script,
so no Hub, bus or database is touched. Checks the changed interactions, not
just page load:

  1. Two open Orion messages pin at the top; the header badge says "Orion 2".
  2. A flapping raise/clear alert pair is folded away; "self-cleared" note shows.
  3. Reply switches the chat session, shows the message in the chat, focuses
     the input, and saves `opened` + `dismissed` receipts.
  4. Dismiss saves a `dismissed` receipt; the badge hides at zero.

Run (needs the `playwright` Python package and a Chromium for it):

    python services/orion-hub/evals/orion_message_tray_browser.py [--screenshot out.png]

Exit 0 on pass, 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]

SESSION = "sess-outreach-1"
MESSAGES = [
    {
        "message_id": "11111111-1111-4111-8111-111111111111",
        "session_id": SESSION,
        "correlation_id": "22222222-2222-4222-8222-222222222222",
        "title": "Orion reached out",
        "preview_text": "I noticed the cabinet ran hot again while you were out.",
        "full_text": "I noticed the cabinet ran hot again while you were out. Want to look at it together?",
        "created_at": "2026-10-28T16:42:00Z",
        "tags": ["endogenous_outreach"],
        "status": "seen",
    },
    {
        "message_id": "33333333-3333-4333-8333-333333333333",
        "session_id": SESSION,
        "correlation_id": "44444444-4444-4444-8444-444444444444",
        "title": "Orion reached out",
        "preview_text": "Something from the reading earlier stuck with me.",
        "full_text": "Something from the reading earlier stuck with me.",
        "created_at": "2026-10-28T14:15:00Z",
        "tags": ["endogenous_outreach"],
        "status": "seen",
    },
]
NOTIFICATIONS = [
    {"event_kind": "execution_dispatch_all_refused", "severity": "warning", "title": "Dispatch refused",
     "created_at": "2026-10-28T10:00:00Z", "source_service": "orion-execution-dispatch-runtime"},
    {"event_kind": "execution_dispatch_all_refused_cleared", "severity": "info", "title": "Dispatch admitting",
     "created_at": "2026-10-28T10:05:00Z", "source_service": "orion-execution-dispatch-runtime"},
    {"event_kind": "orion_day.letter", "severity": "info", "title": "Orion's day",
     "created_at": "2026-10-28T09:00:00Z", "source_service": "orion-hub", "body_text": "Letter body"},
]


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
    # Offline fallback for the one Tailwind rule the hidden modals rely on.
    return html.replace("</head>", "<style>.hidden{display:none!important}</style></head>", 1)


def run(screenshot: str | None) -> int:
    from playwright.sync_api import sync_playwright

    receipts: list = []
    failures: list = []
    index_html = render_index()

    def check(cond: bool, label: str) -> None:
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    def handle(route, request):
        url = request.url
        if url.startswith("https://cdn.tailwindcss.com"):
            return route.continue_()
        if not url.startswith("http://hub.smoke"):
            return route.abort()
        path = "/" + url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else "/"
        path_only = path.split("?", 1)[0]
        if path_only in ("/", "/index.html"):
            return route.fulfill(status=200, body=index_html, content_type="text/html")
        if path_only.startswith("/static/"):
            file = HUB / path_only.lstrip("/")
            if file.is_file():
                ctype = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                return route.fulfill(status=200, body=file.read_bytes(), content_type=ctype)
            return route.fulfill(status=404, body="")
        if path_only == "/api/notifications":
            return route.fulfill(status=200, body=json.dumps(NOTIFICATIONS), content_type="application/json")
        if path_only == "/api/chat/messages":
            open_rows = [m for m in MESSAGES if m["message_id"] not in {r["id"] for r in receipts if r["type"] == "dismissed"}]
            return route.fulfill(status=200, body=json.dumps(open_rows), content_type="application/json")
        if path_only.startswith("/api/chat/message/") and path_only.endswith("/receipt"):
            body = json.loads(request.post_data or "{}")
            receipts.append({"id": path_only.split("/")[4], "type": body.get("receipt_type"), "session": body.get("session_id")})
            return route.fulfill(status=200, body="{}", content_type="application/json")
        if path_only == "/api/session":
            return route.fulfill(status=200, body=json.dumps({"session_id": "sess-current"}), content_type="application/json")
        if path_only.startswith("/api/"):
            return route.fulfill(status=200, body="[]" if request.method == "GET" else "{}", content_type="application/json")
        return route.fulfill(status=404, body="")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors: list = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.route("**/*", handle)
        page.goto("http://hub.smoke/")
        page.wait_for_selector("#notificationList .om-orion-msg", timeout=20000)

        rows = page.locator("#notificationList .om-orion-msg")
        check(rows.count() == 2, "two open Orion messages pinned")
        first_id = rows.nth(0).get_attribute("data-message-id")
        check(first_id == MESSAGES[0]["message_id"], "newest Orion message is first")
        badge = page.locator("#orionMessagesBadge")
        check(badge.is_visible() and badge.inner_text().strip() == "Orion 2", "badge reads 'Orion 2'")
        tray_text = page.locator("#notificationList").inner_text()
        check("Dispatch refused" not in tray_text and "Dispatch admitting" not in tray_text,
              "flapping raise/clear pair folded away")
        check("Orion's day" in tray_text, "ordinary notification still shown")
        check("2 self-cleared alerts hidden" in tray_text, "self-cleared note shown")
        if screenshot:
            page.locator("#hubNotificationsPanel").screenshot(path=screenshot)

        rows.nth(0).locator(".om-orion-reply").click()
        page.wait_for_function("() => document.querySelectorAll('#notificationList .om-orion-msg').length === 1")
        sid = page.evaluate("() => localStorage.getItem('orion_sid')")
        check(sid == SESSION, "Reply switched the chat session")
        bubble_text = page.locator("#conversation").inner_text()
        check("ran hot again" in bubble_text, "Reply put the message in the chat window")
        focused = page.evaluate("() => document.activeElement && document.activeElement.id")
        check(focused == "chatInput", "Reply focused the chat input")
        page.wait_for_timeout(300)
        kinds = sorted(r["type"] for r in receipts if r["id"] == MESSAGES[0]["message_id"])
        check("opened" in kinds and "dismissed" in kinds, "Reply saved opened + dismissed receipts")
        check(badge.inner_text().strip() == "Orion 1", "badge dropped to 'Orion 1'")

        page.locator("#notificationList .om-orion-msg .om-orion-dismiss").first.click()
        page.wait_for_function("() => document.querySelectorAll('#notificationList .om-orion-msg').length === 0")
        page.wait_for_timeout(300)
        check(any(r["id"] == MESSAGES[1]["message_id"] and r["type"] == "dismissed" for r in receipts),
              "Dismiss saved a dismissed receipt")
        check(not badge.is_visible(), "badge hidden at zero")

        tray_errors = [e for e in errors if "OrionMessageTray" in e or "renderNotifications" in e or "Orion" in e]
        check(not tray_errors, f"no tray page errors ({tray_errors[:2]})")
        browser.close()

    print(json.dumps({"receipts": receipts, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default=None)
    sys.exit(run(parser.parse_args().screenshot))
