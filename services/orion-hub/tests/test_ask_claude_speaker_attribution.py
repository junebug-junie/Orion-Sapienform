"""Gate: "Ask Claude" with an empty composer must not quote someone else's
message back to Claude under Juniper's name.

Real incident (2026-09-07): clicking "Ask Claude" right after Orion spoke,
with the message box empty, sent Orion's own message to the room companion
labeled `invited_by="Juniper"` (the backend's unconditional default -- see
`RoomClaudeInviteRequest.invited_by` in api_routes.py, which the client never
overrode). `room_prompt.build_turn_prompt` then rendered that straight into
Claude's prompt as `"Juniper: <Orion's words>"`, so Claude replied as if
Juniper had written a paragraph of Orion's own jargon, and asked her if she
was okay.

`app.js` is a single non-modular file (no import surface to unit-test
directly against a DOM), so this follows the same source-assertion pattern as
`test_agent_claude_trace_js.py`: pin the exact code shape that keeps the
fallback honest about who said what, so a future edit can't silently
reintroduce the hardcoded "Juniper" default.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_ask_claude_sends_invited_by_not_hardcoded_juniper() -> None:
    source = _source()
    match = re.search(r"async function askClaude\(\)[\s\S]*?\n  \}\n", source)
    assert match, "askClaude() not found"
    body = match.group(0)
    # The fetch body must carry a variable, not a literal "Juniper" baked
    # into the request payload -- that literal is exactly the regression.
    assert "invited_by: invitedBy" in body
    assert "invited_by: 'Juniper'" not in body
    assert 'invited_by: "Juniper"' not in body
    # The fallback must derive who said the reacted-to message instead of
    # assuming Juniper said it -- and must keep mapping the DOM's internal
    # "You" label back to "Juniper" for the backend, not send "You" verbatim
    # (that exact substring, not just "invitedBy = last.sender", so a future
    # edit can't quietly drop the ternary and still pass this test).
    assert "invitedBy = last.sender === 'You' ? 'Juniper' : last.sender" in body


def test_last_room_message_carries_real_sender_and_skips_system() -> None:
    source = _source()
    assert "function lastRoomMessage()" in source
    assert "function lastUserOrOrionText()" not in source, (
        "old fallback should be replaced, not left dead alongside the new one"
    )
    match = re.search(r"function lastRoomMessage\(\)[\s\S]*?\n  \}\n", source)
    assert match, "lastRoomMessage() not found"
    body = match.group(0)
    # Must read the real per-bubble speaker, not just the coarse role bucket.
    assert "data-sender" in body
    # An error/status banner must never be handed to Claude as a quote.
    assert "sender === 'System'" in body and "continue" in body


def test_append_message_records_real_sender_alongside_role() -> None:
    source = _source()
    assert "div.dataset.sender = sender;" in source
