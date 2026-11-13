"""Real-browser smoke for "@claude" in the Hub chat box.

Loads the real templates/index.html, claude-address.js and app.js in headless Chromium. Every
/api/* call is answered here, so no Hub, bus or database is touched. Checks the interaction, not
just page load:

  1. Typing "@claude ..." turns Send amber ("To Claude"); clearing the box turns it back.
  2. Enter sends ONE invite to /api/room/claude/invite with the address stripped, invited_by
     Juniper -- and nothing to Orion (no /api/chat POST, no websocket frame). Her words show as a
     "You" bubble and the box clears.
  3. While Claude is still replying, a second "@claude" is held in the box: no second invite and
     nothing to Orion. Claude's reply frame (sent by the mock socket) then frees it.
  4. "@claude" alone falls back to the last room message, without its "@claude" address.
  5. A refused invite (503) leaves the text in the box and still sends nothing to Orion.
  5b. Pending images block an @claude send (the invite is text-only); text and images stay put.
  5c. The expanded composer: amber cue on its Send, Enter sends one invite, both boxes clear.
  6. "@claudette ..." and a mid-sentence "@claude" are ordinary chat: no amber, no invite, and
     the message goes to Orion (the control that proves the "nothing to Orion" checks can see a
     real Orion send).
  7. Fail closed: with claude-address.js answering 404, "@claude ..." still invites Claude and
     still sends nothing to Orion (app.js's fallback pattern).

The Hub websocket is a mock server (page.route_web_socket), so a real Orion turn shows up as a
chat frame on it -- "nothing to Orion" is measured on the same rail a real turn uses.

Run (needs the `playwright` Python package and a Chromium for it):

    python services/orion-hub/evals/claude_address_browser.py

Exit 0 on pass, 1 on any failed check.
"""

from __future__ import annotations

import json
import mimetypes
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
REFUSE_MARKER = "please refuse this one"


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
    return html.replace(
        "</head>",
        "<style>.hidden{display:none!important}.sr-only{position:absolute;width:1px;height:1px;"
        "overflow:hidden;clip:rect(0,0,0,0)}</style></head>",
        1,
    )


def run() -> int:
    from playwright.sync_api import sync_playwright

    failures: list = []
    invites: list = []
    to_orion: list = []
    static_hits: list = []
    index_html = render_index()
    block_module = [False]

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
        if path_only in ("/", "/index.html"):
            return route.fulfill(status=200, body=index_html, content_type="text/html")
        if path_only.startswith("/static/"):
            static_hits.append(path)
            if block_module[0] and path_only.endswith("/claude-address.js"):
                return route.fulfill(status=404, body="")
            file = HUB / path_only.lstrip("/")
            if file.is_file():
                ctype = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
                return route.fulfill(status=200, body=file.read_bytes(), content_type=ctype)
            return route.fulfill(status=404, body="")
        if path_only == "/api/room/claude/invite" and request.method == "POST":
            body = json.loads(request.post_data or "{}")
            invites.append(body)
            if REFUSE_MARKER in str(body.get("prompt") or ""):
                return route.fulfill(status=503, body="Room companion is disabled")
            return route.fulfill(
                status=200,
                body=json.dumps({"ok": True, "request_id": "r1", "correlation_id": "c1"}),
                content_type="application/json",
            )
        # Image upload stages a file for a later turn; it is not a turn.
        if request.method != "GET" and "chat" in path_only and path_only != "/api/chat/attachments":
            to_orion.append(f"{request.method} {path_only} {request.post_data or ''}"[:300])
        if path_only == "/api/session":
            return route.fulfill(status=200, body=json.dumps({"session_id": "sess-smoke"}), content_type="application/json")
        if path_only.startswith("/api/"):
            return route.fulfill(status=200, body="[]" if request.method == "GET" else "{}", content_type="application/json")
        return route.fulfill(status=404, body="")

    sockets: list = []

    def on_page_ws_message(message) -> None:
        text = message if isinstance(message, str) else message.decode("utf-8", "replace")
        try:
            kind = str(json.loads(text).get("type") or "")
        except Exception:
            kind = ""
        if kind == "session_hello":  # connect handshake, not a turn
            return
        to_orion.append(f"ws {text[:200]}")

    def mock_ws(ws) -> None:
        sockets.append(ws)
        ws.on_message(on_page_ws_message)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.route_web_socket("**/ws", mock_ws)
        page.route("**/*", handle)
        page.goto("http://hub.smoke/")
        page.wait_for_selector("#chatInput", state="visible", timeout=20000)
        page.wait_for_function("() => !!window.OrionClaudeAddress", timeout=20000)
        page.wait_for_timeout(1000)  # let the socket connect, so an Orion turn would use it
        check(bool(sockets), "chat socket connected to the mock (Orion turns would be seen)")
        check(any("claude-address.js?v=" in h for h in static_hits), "template loads claude-address.js (versioned)")

        send = page.locator("#sendButton")
        box = page.locator("#chatInput")

        def bubbles(sender: str) -> list:
            return page.eval_on_selector_all(
                f'#conversation [data-sender="{sender}"] [data-message-body="1"]',
                "els => els.map(e => e.textContent.trim())",
            )

        def wait_settle() -> None:
            page.wait_for_timeout(500)

        # 1. cue
        check(send.get_attribute("data-claude-address") in (None, "0"), "Send starts in its normal state")
        box.fill("@claude what do you make of this?")
        check(send.get_attribute("data-claude-address") == "1", "typing @claude marks Send")
        check("bg-amber-600" in (send.get_attribute("class") or ""), "Send turns amber")
        check("bg-indigo-600" not in (send.get_attribute("class") or ""), "Send loses its indigo")
        check(send.inner_text().strip() == "To Claude", "Send reads 'To Claude'")
        check("border-amber-500" in (box.get_attribute("class") or ""), "chat box border turns amber")
        box.fill("")
        check(send.get_attribute("data-claude-address") == "0" and send.inner_text().strip() == "Send",
              "clearing the box restores Send")

        # 2. Enter routes to Claude, not Orion
        box.fill("@Claude: what do you make of this?")
        box.press("Enter")
        wait_settle()
        check(len(invites) == 1, f"exactly one invite ({len(invites)})")
        if invites:
            check(invites[0].get("prompt") == "what do you make of this?", f"address stripped ({invites[0].get('prompt')!r})")
            check(invites[0].get("invited_by") == "Juniper", "invited_by Juniper")
        check(not to_orion, f"nothing sent to Orion ({to_orion[:2]})")
        check(bubbles("You")[-1:] == ["@Claude: what do you make of this?"], f"her words shown as a You bubble ({bubbles('You')[-1:]})")
        check(box.input_value() == "", "box cleared after the invite")
        check(send.get_attribute("data-claude-address") == "0", "Send back to normal after sending")

        # 3. Claude still replying: held, not doubled, not handed to Orion
        box.fill("@claude and another thing")
        box.press("Enter")
        wait_settle()
        check(len(invites) == 1, "no second invite while Claude is replying")
        check(not to_orion, "held message not sent to Orion")
        check(box.input_value() == "@claude and another thing", "held message stays in the box")
        def claude_replies(text: str) -> None:
            sockets[-1].send(json.dumps({
                "kind": "room_claude_utterance", "speaker": "Claude", "llm_response": text,
                "ok": True, "correlation_id": "c1", "request_id": "r1", "message_id": f"m-{text[:8]}",
            }))
            page.wait_for_timeout(300)

        if sockets:
            claude_replies("Claude's first reply.")
        check(page.locator("#askClaudeButton").is_enabled(), "Claude's reply frees Ask Claude")
        check(bubbles("Claude")[-1:] == ["Claude's first reply."], "Claude's reply shown")

        # 4. empty remainder falls back to the last room message, address stripped
        # Last room message is Claude's reply; make Juniper's @claude bubble the last one instead
        # by sending a fresh @claude message first, then free the button again.
        box.fill("@claude what about now?")
        box.press("Enter")
        wait_settle()
        if sockets:
            claude_replies("")  # a pass: frees the button, adds no bubble
        box.fill("@claude")
        send.click()
        wait_settle()
        check(len(invites) == 3, f"bare @claude invites ({len(invites)})")
        if len(invites) == 3:
            check(invites[2].get("prompt") == "what about now?",
                  f"bare @claude falls back to the last message, address stripped ({invites[2].get('prompt')!r})")
        check(not to_orion, "bare @claude not sent to Orion")
        check(box.input_value() == "", "bare @claude clears the box")
        if sockets:
            claude_replies("")

        # 5. refused invite keeps the text
        refused = f"@claude {REFUSE_MARKER}"
        box.fill(refused)
        box.press("Enter")
        wait_settle()
        check(len(invites) == 4, "refused invite was attempted")
        check(box.input_value() == refused, "refused invite leaves the text in the box")
        check(not to_orion, "refused invite not sent to Orion")
        check(any("Could not invite Claude (503)" in t for t in bubbles("System")), "refusal shown as a System line")
        check(refused not in bubbles("You"), "no You bubble for a refused invite")

        # 5b. pending images block the @claude send
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d4944415478da63f8cfc0f01f0005000201a2dd7d1d0000000049454e44ae426082"
        )
        page.set_input_files("#attachmentFileInput", files=[{"name": "dot.png", "mimeType": "image/png", "buffer": png}])
        page.wait_for_timeout(800)
        pending = page.locator(".om-chip-remove").count()
        check(pending == 1, f"an image is pending ({pending}), so the block below is a real check")
        box.fill("@claude look at this picture")
        box.press("Enter")
        wait_settle()
        check(len(invites) == 4, "pending images block the @claude invite")
        check(not to_orion, "pending images + @claude: nothing to Orion")
        check(box.input_value() == "@claude look at this picture", "pending images + @claude: text stays in the box")
        check("Images cannot go to Claude" in page.locator("#status").inner_text(), "pending images + @claude: says why")
        page.locator(".om-chip-remove").first.click()
        page.wait_for_timeout(200)
        check(page.locator(".om-chip-remove").count() == 0, "image removed again")

        # 5c. expanded composer
        box.fill("")
        page.locator("#chatInputExpandButton").click()
        area = page.locator("#chatInputExpandTextarea")
        area.wait_for(state="visible")
        area.fill("@claude from the big box")
        modal_send = page.locator("#chatInputExpandModalSend")
        check(modal_send.get_attribute("data-claude-address") == "1", "expanded Send is marked for Claude")
        check("bg-amber-600" in (modal_send.get_attribute("class") or ""), "expanded Send turns amber")
        area.press("Enter")
        wait_settle()
        check(not to_orion, "expanded @claude: nothing to Orion")
        check(any(i.get("prompt") == "from the big box" for i in invites), "expanded @claude invites Claude")
        check(box.input_value() == "" and area.input_value() == "", "expanded @claude clears both boxes")
        if sockets:
            claude_replies("")
        invites_before_control = len(invites)

        # 6. not an address: ordinary chat for Orion (also the control for the checks above)
        for text in ("@claudette hello there", "please ask @claude about this"):
            box.fill(text)
            check(send.get_attribute("data-claude-address") == "0", f"no amber for {text!r}")
            before = len(to_orion)
            box.press("Enter")
            page.wait_for_timeout(800)
            check(len(invites) == invites_before_control, f"{text!r} does not invite Claude")
            check(len(to_orion) > before and text in to_orion[-1], f"{text!r} goes to Orion as a chat frame")
            if sockets:  # end the in-flight Orion turn so the next send is not queued behind it
                sockets[-1].send(json.dumps({"llm_response": "ok", "correlation_id": f"o-{before}"}))
                page.wait_for_timeout(300)

        check(not errors, f"no page errors at all ({errors[:2]})")

        # 7. fail closed without claude-address.js
        block_module[0] = True
        page.reload()
        page.wait_for_selector("#chatInput", state="visible", timeout=20000)
        page.wait_for_timeout(1500)
        check(page.evaluate("() => !window.OrionClaudeAddress"), "module really missing on reload")
        orion_before, invites_before = len(to_orion), len(invites)
        page.locator("#chatInput").fill("@claude are you there?")
        page.locator("#chatInput").press("Enter")
        wait_settle()
        check(len(to_orion) == orion_before, f"module missing: @claude still not sent to Orion ({to_orion[orion_before:][:1]})")
        check(len(invites) == invites_before + 1 and invites[-1].get("prompt") == "are you there?",
              "module missing: @claude still invites Claude, address stripped")
        check(page.locator("#sendButton").get_attribute("data-claude-address") in ("0", None), "module missing: cue reset after send")
        browser.close()

    print(json.dumps({"invites": invites, "to_orion": to_orion[:5], "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
