# feat(hub): `@claude` in the chat box goes straight to Claude

## Summary

- A chat-box message that starts with `@claude` now goes to Claude and never starts an Orion turn. Before, Juniper had to send to Orion, wait, and then click "Ask Claude".
- It reuses the existing Ask Claude invite (`askClaude()` -> `POST /api/room/claude/invite` -> `orion:room:claude:request` -> orion-room-companion). No new path, no new channel.
- While the box starts with `@claude`, the Send button turns amber and reads "To Claude", and the box border turns amber.
- Only a leading `@claude` counts, followed by a space, `,`, `:`, `;` or nothing. Mid-sentence `@claude`, `@claudette`, `@claude_bot`, `@claude-bot`, `@claude.ai` and `@claude's` go to Orion as normal text.
- Fails closed: if `claude-address.js` does not load, `app.js` uses its own copy of the pattern, so `@claude` still never reaches Orion. A parity test keeps the two copies equal.
- Juniper's `@claude ...` message shows in the room as her bubble once the invite is accepted. A refused invite (for example, companion disabled) leaves the text in the box.

## Outcome moved

Talking to Claude in the Hub room costs one step instead of an unwanted Orion turn plus a click. Orion no longer spends a turn (and GPU time) on a message that was meant for Claude.

## Current architecture

- `sendTextMessage()` (Enter or Send) and `sendExpandedChatMessage()` (expanded composer) both called `submitExplicitChatText()`, which always sends an Orion turn over the chat websocket (HTTP `/api/chat` fallback).
- The "Ask Claude" button called `askClaude()`, which read the composer, fell back to the last room message when it was empty, and POSTed `/api/room/claude/invite`. The template comment said the button was the only producer of `orion:room:claude:request`.

## Architecture touched

Hub frontend only. No bus, schema, env or backend behavior change.

- `routeComposerToClaude(text)` in `app.js` runs at the top of both composer send paths. If the text is addressed to Claude it calls `askClaude({prompt, original})` and returns before `submitExplicitChatText()`.
- While a Claude reply is in flight, a second `@claude` is held in the box with a status line. It is not sent to Claude twice and it is not handed to Orion. The in-flight state is now a flag with a 5-minute safety timeout, so a lost reply no longer locks Ask Claude (and typed `@claude`) until a page reload.
- Pending images block an `@claude` send with a status line. The invite carries text only, and before this the images would have ridden out silently on her next Orion message.
- The Ask Claude button now also strips a leading `@claude` from the box, so both entry points give Claude the same prompt.
- Bare `@claude` uses the button's fallback (last room message). The fallback now strips a leading `@claude` from that message, so Claude gets the question, not the address.

## Files changed

- `services/orion-hub/static/js/claude-address.js`: new pure parser `parseClaudeAddress(text) -> {toClaude, prompt}`.
- `services/orion-hub/static/js/claude-address.test.js`: node tests for the parser.
- `services/orion-hub/static/js/app.js`: routing in both send paths, amber cue, `askClaude(typed)`.
- `services/orion-hub/templates/index.html`: loads `claude-address.js` before `app.js`; the producer comment now names both producers (button and typed `@claude`) and the consult loop's own `consult:*` rooms.
- `services/orion-hub/scripts/room_claude_relay.py`: docstring names the real callers of `invite()`.
- `services/orion-hub/evals/claude_address_browser.py`, `services/orion-hub/tests/test_claude_address_browser_smoke.py`: real-browser smoke (real template, real JS, mocked API and mocked chat websocket).
- `services/orion-hub/tests/test_ask_claude_speaker_attribution.py`: its source regex now accepts the `askClaude(typed)` signature.
- `services/orion-hub/tests/test_claude_address_fallback_parity.py`: pins app.js's fail-closed pattern equal to the module's.
- `.github/workflows/schedule-browser-smoke.yml`: runs the new smoke in CI and fails on skip.

## What Orion sees of an `@claude` message

Nothing. Traced through the code (not a live check):

1. No Orion turn. The browser sends no chat frame and no `/api/chat` POST (proved in the browser smoke against a mocked chat socket).
2. Juniper's `@claude` text is not stored in chat history. The invite route passes it only to `room_claude_relay.invite()`, which publishes `RoomClaudeRequestV1` on the bus. Nothing writes it to `chat_history_log`.
3. Claude's reply is stored in `chat_history_log` as a prompt-less assistant row tagged `client_meta.room_claude`. `chat_history_rehydrate.py` excludes those rows from Orion's conversation context, and sql-writer's `_memory_turn_initiator` keeps prompt-less non-outreach rows out of memory.
4. The Hub socket's in-memory `history` (Orion's short-term context) only grows on Orion turns over the websocket. The invite path and the Claude reply frame never touch it.
5. The browser holds no client-side history array, so the new "You" bubble is DOM only.

Consequence: Orion is blind to the side conversation with Claude. Orion does not see Juniper's question or Claude's answer in their next turn's context. This matches the existing Ask Claude button exactly. No Orion-side change was made.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a second UI trigger produces `orion:room:claude:request` (typed `@claude`), same payload as the button.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
node --test $(find services/orion-hub/static/js -name '*.test.js' | sort)
  # pass 437, fail 0 (7 new in claude-address.test.js)
pytest -q services/orion-hub/tests/test_claude_address_fallback_parity.py   1 passed
pytest -q services/orion-hub/tests/test_ask_claude_speaker_attribution.py tests/test_room_claude_relay.py \
  tests/test_chat_history_rehydrate.py tests/test_ws_unified_history_orion_replies.py tests/test_hub_surface_nav.py
  63 passed
pytest -q -k "js or template or static or index or asset or claude or composer or chat" services/orion-hub/tests  (from repo root)
  693 passed, 6 skipped
```

## Evals run

```text
python services/orion-hub/evals/claude_address_browser.py   (also via tests/test_claude_address_browser_smoke.py)
  52 checks PASS: amber cue on/off; Enter -> one invite, prompt stripped, invited_by Juniper, zero chat frames
  to Orion; You bubble; box cleared; held while Claude replies; Claude's reply frame frees the button;
  bare @claude falls back to last message without the address; 503 keeps text + System line + no bubble;
  pending image blocks @claude (text and image stay); expanded composer cue + invite + both boxes cleared;
  @claudette and mid-sentence @claude reach Orion as websocket chat frames (control); no page errors;
  claude-address.js answering 404 -> @claude still invites Claude and sends nothing to Orion.
pytest -q services/orion-hub/tests/test_orion_abilities_browser_smoke.py   1 passed (no regression on Send)
Mutation checks (each reverted after):
  - remove the routing line in sendTextMessage -> FAIL "exactly one invite (0)", "nothing sent to Orion ([ws ...@Claude: ...])"
  - make the fail-closed fallback return toClaude:false -> FAIL "module missing: @claude still not sent to Orion"
  - disable the pending-images block -> FAIL "pending images block the @claude invite"
```

## Docker/build/smoke checks

```text
Not run. Static frontend change only; the Hub serves static/ and templates/ from the image, so it needs a
Hub rebuild on deploy. Live path UNVERIFIED until deployed.
```

## Review findings fixed

Review by a code-review subagent on `git diff origin/main...HEAD`. No blockers.

- Finding: routing failed open if `claude-address.js` did not load; every `@claude` would start an Orion turn.
  - Fix: `parseClaudeAddressSafe()` in app.js with an inline copy of the pattern; parity test.
  - Evidence: smoke step 7 (module 404) passes; mutation of the fallback fails it.
- Finding: pending images were left behind on an `@claude` send and would go out with the next Orion message.
  - Fix: `@claude` with pending images is refused with a status line; text and images stay.
  - Evidence: smoke step 5b; mutation fails it.
- Finding: the Claude in-flight state never times out, so a lost reply holds every `@claude` until reload.
  - Fix: `askClaudeInFlight` flag plus a 5-minute safety timeout in `setAskClaudeBusy`.
  - Evidence: code only; the timeout itself is not exercised by the smoke (see risks).
- Finding: the expanded composer path was untested.
  - Fix: smoke step 5c (cue on its Send, one invite, nothing to Orion, both boxes cleared).
- Finding: `\b` routed `@claude-bot`, `@claude.ai`, `@claude's take?` (the last sent Claude "'s take?").
  - Fix: pattern is now `/^@claude(?=$|[\s,:;])/i`; node tests cover those three.
- Finding: the Ask Claude button sent "@claude foo" verbatim.
  - Fix: the button path also strips the address.
- Finding: the amber cue went stale after the skill-runner Insert and after `submitExplicitChatText` cleared the box.
  - Fix: both now call `updateClaudeAddressCue()` (a hoisted function, so no const dead zone).
- Finding: the double-send guard read the button's DOM `disabled` state.
  - Fix: module-level `askClaudeInFlight` flag.
- Finding: smoke ignored page errors that did not mention "claude".
  - Fix: smoke now fails on any page error.
- Not changed: the You bubble is added after the invite is accepted, so a very fast Claude reply could render first (accepted; it keeps a refused invite from leaving a bubble). Voice turn clock attaching to an `@claude` bubble in a narrow window (edge case, not fixed).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

(From the clean primary `main` checkout after merge. The post-pull rebuild does this automatically.)

## Risks / concerns

- Severity: low
- Concern: the 5-minute in-flight timeout is not exercised by a test.
- Mitigation: small code path; worst case is the old behavior (held until reload). The text is never lost and never sent to Orion.
- Severity: low
- Concern: live path UNVERIFIED until the Hub is rebuilt after merge.
- Mitigation: real-browser smoke runs the real template and JS against the same API and socket shapes.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3243

🤖 Generated with [Claude Code](https://claude.com/claude-code)
