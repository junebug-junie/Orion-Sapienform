# Wire TTS into the HTTP fallback chat path -- it never had one

## Summary

- Fixes a real incident (corr=`11215a1b-d3c8-438b-901b-0d6cadf3d637`): a
  live turn in the regular Hub chat window produced neither text nor voice.
- Root cause: I redeployed hub (verifying PR #1905) minutes before the
  message was sent, killing the live WebSocket connection. `app.js`'s own
  pre-existing fallback (built 2026-08-22, for exactly this scenario)
  correctly detected the dead socket and fell back to `POST /api/chat`,
  with an on-screen warning. That HTTP route has **never had any TTS
  wiring at all, in any mode**, since it existed -- a completely separate
  code path from `websocket_handler.py`'s WS loop, which PR #1905 fixed
  only for the live-socket case.
- Wired synchronous TTS into the HTTP route, and wired the frontend's
  existing audio-playback logic (previously WS-only) to also fire for
  the HTTP fallback response.

## Outcome moved

Before: any turn that fell back to HTTP (dead WS, exactly the scenario the
fallback exists for) got text-only at best, and per this real incident,
possibly nothing visible at all. After: the HTTP fallback speaks too, live
-verified end to end.

## Current architecture (before this patch)

- `services/orion-hub/scripts/api_routes.py`'s `POST /api/chat` route
  (`api_chat` -> `handle_chat_request`) has a `mode == "orion"` branch that
  calls `execute_unified_turn` directly and returns the last frame as the
  HTTP response body. Zero TTS handling anywhere in this file.
- `services/orion-hub/static/js/app.js` has TWO ways a chat message leaves
  the browser: `socket.send(...)` (the live WS path, normal case) and a
  documented fallback to `fetch(.../api/chat, ...)` when the socket is not
  open (dead connection, e.g. a Hub redeploy). The WS path's `onmessage`
  handler already reacted to `audio_response`/`tts_debug`/`tts_error`
  fields and queued playback; the HTTP fallback's own `.then(d => {...})`
  handler never looked at those fields at all.

## Architecture touched

| Seam | Change |
| --- | --- |
| `services/orion-hub/scripts/api_routes.py` | orion-mode HTTP branch now synthesizes TTS synchronously before returning |
| `services/orion-hub/static/js/app.js` | new shared `handleTtsFields(d)`, called from both the WS `onmessage` handler and the HTTP fallback's `.then()` |
| `services/orion-hub/tests/test_handle_chat_request_http_fallback_tts.py` | new, 6 tests |

## Files changed

- `services/orion-hub/scripts/api_routes.py`: imports `TTSRequestPayload`
  and (lazily, matching this file's existing convention for
  `websocket_handler.py` symbols) `extract_unified_turn_final_text`. After
  building `final_frame`, extracts `disable_tts` from the payload, and if
  there's real text, `not disable_tts`, and a configured `tts_client`,
  awaits `tts_client.speak(...)` under `HUB_TTS_TIMEOUT_SEC` and merges
  `audio_response`/`tts_source_text`/`tts_meta` into the response. A
  timeout or exception sets `tts_error` instead -- the text reply, which
  already succeeded, is never lost because of a TTS failure.
- `services/orion-hub/static/js/app.js`: extracted the WS handler's
  existing `audio_response`/`tts_debug`/`tts_error` handling into
  `handleTtsFields(d)` (a hoisted function declaration, same convention as
  the pre-existing `processAudioQueue`), called from both the WS
  `onmessage` handler and the HTTP fallback's `.then()`.
- `services/orion-hub/tests/test_handle_chat_request_http_fallback_tts.py`:
  new, 6 tests, following the exact established pattern
  `test_handle_chat_request_orion_mode_degraded.py` already uses for this
  same HTTP route (fresh `scripts.api_routes`/`scripts.main` imports per
  test, since `conftest.py`'s autouse fixture clears them between tests).

## Schema / bus / API changes

- **Added** (HTTP response only, additive): `audio_response`,
  `tts_source_text`, `tts_meta`, `tts_error` -- same field names the WS
  lane's `run_tts_remote` already uses, so the frontend's shared handler
  works identically regardless of transport.
- No bus channel, schema registry, or payload contract changes -- reuses
  the exact same `TTSClient.speak()` / `orion:tts:intake` RPC the WS lane
  already uses.

## Env/config changes

None.

## Tests run

```text
services/orion-hub/tests/test_handle_chat_request_http_fallback_tts.py  16 passed
services/orion-hub/tests/test_handle_tts_fields_frontend.py              4 passed
services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py
services/orion-hub/tests/test_handle_chat_request_orion_mode_continuity.py
services/orion-hub/tests/test_chat_route_tagging.py
services/orion-hub/tests/test_orion_unified_turn_tts.py
services/orion-hub/tests/test_handle_chat_request_turn_effect.py       (all pre-existing, unaffected
                                                                         except 1 confirmed pre-existing
                                                                         failure on origin/main too)

# whole hub suite, branch (post-review-fix) vs fresh origin/main
branch:        63 failed, 1651 passed, 4 skipped
origin/main:   63 failed, 1638 passed, 4 skipped

FAILED set is byte-identical between the two (comm in both directions:
empty). Zero regressions; the +13 passed are this pass's new/updated
tests.
```

## Evals run

```text
No eval harness exists for this seam. Covered by the live HTTP-request
verification below, which directly replicates the real incident.
```

## Docker/build/smoke checks

```text
docker compose --env-file <primary>/.env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml build && up -d
  => built clean, boot clean ("Application startup complete.")

# Real POST /api/chat call replicating the actual incident (mode=orion,
# no live WebSocket involved at all -- a plain HTTP client):
status: 200, elapsed: 168.8s (matches the original incident's multi-minute
  duration -- the real 60s pre_turn_appraisal timeout is still in there)
llm_response: "I read you, loud and clear. Still here, still tuned in —
  ready whenever you are."
has audio_response: True, len: 819344
tts_meta: {content_type: audio/wav, duration_sec: 12.8, backend: coqui,
  speaker_wav_basename: orion_reference.wav, speaker_wav_used: True,
  synthesis_ms: 6823, gpu_enabled: True}
tts_error: None
```

Redeployed hub live to run this verification -- flagged explicitly before
doing so, since this is the exact same action (a hub redeploy killing any
live WS connection) that caused the original incident.

## Review findings fixed

High-effort `/code-review` run. 9 findings, all real, all fixed.

- **Most severe -- Finding**: `shouldAppendOrionWsPayload(d)` checked
  `if (d.tts_error) return false` as its FIRST condition. Correct for the
  WS lane (a tts_error frame there never carries real text), but the new
  HTTP fallback can merge a real `llm_response` WITH a `tts_error` (text
  succeeded, TTS didn't) into the SAME object -- that merged shape got
  misclassified as "no text", replacing a real, successful reply with
  *"HTTP completed but no assistant text was returned"* -- exactly the
  failure mode this whole PR exists to fix, reintroduced by the fix
  itself.
  - Fix: check specifically for the absence of `llm_response`, not
    `tts_error` alone. Every existing WS-lane case (bare tts_error,
    audio-only follow-up frames) verified unchanged.
  - Evidence: `test_should_append_orion_ws_payload_checks_llm_response_before_tts_error`.

- **Finding**: hand-rolled a THIRD independent implementation of the TTS
  gate + synthesis + timeout/exception handling -- the exact "one lane
  wired, one lane not, nothing keeping them in sync" shape PR #1905's
  `dispatch_tts_reply`/`run_tts_remote` were unified to stop, the same
  day.
  - Fix: extracted `synthesize_tts_reply()` (`websocket_handler.py`) as
    the shared synthesis core; both `run_tts_remote` (WS, queue-based)
    and this HTTP route (synchronous) now call it, emitting the same
    `voice.tts.start`/`done`/`error` log format regardless of transport.
    Only the *gate* stays lane-specific (fire-and-forget task vs
    synchronous await differ enough in shape that forcing them into one
    function added more indirection than it removed).

- **Finding**: a "successful" synthesis with `audio_b64=""` would silently
  drop voice output with zero trace -- no exception, no log, no UI signal.
  - Fix: treated as a real failure inside the shared core, so both lanes
    get it from one place.

- **Finding**: `disable_tts = bool(payload.get("disable_tts", False))` --
  a non-browser caller sending the JSON string `"false"` gets Python
  `True` from a bare `bool()` cast, silently suppressing TTS against their
  actual intent.
  - Fix: uses the file's own established `_normalize_bool()` helper
    (already used for `use_recall` two lines earlier).

- **Finding**: TTS synthesis is awaited synchronously (up to
  `HUB_TTS_TIMEOUT_SEC`) with no check for a client that already gave up
  -- and this fallback exists precisely for Hub-wide WS outages, when many
  clients can hit this route at once.
  - Fix: `request.is_disconnected()` checked before synthesizing.
    Best-effort: a caller with no `Request` object (e.g. direct test
    invocation), or the check itself raising, still gets TTS rather than
    losing it to an unrelated failure.
  - Evidence: `test_http_fallback_skips_synthesis_when_client_already_disconnected`,
    `test_http_fallback_still_synthesizes_when_request_object_is_unavailable`,
    `test_http_fallback_a_broken_disconnect_check_does_not_block_synthesis`.

- **Finding**: no JS test/smoke was added for the `app.js` refactor --
  only a Python backend test -- and as a direct consequence, the severe
  `shouldAppendOrionWsPayload` regression above shipped undetected.
  - Fix: `app.js` has no `module.exports` (a browser IIFE, not
    Node-requireable, unlike this repo's small pure-logic `*.test.js`
    modules) -- added `test_handle_tts_fields_frontend.py` instead,
    following the SAME established convention
    `test_websocket_agent_claude_routing.py` already uses for this exact
    file: assert on the real source's control-flow shape. This test would
    have caught the regression; it did not exist until now.

- **Finding**: a dead/redundant monkeypatch in my own new test file (a
  timeout override set twice, only the second call had any effect).
  - Fix: extended the shared `_wire_common` fixture to accept the
    parameter directly.

- **Finding** (raw exception text surfaced to the user): noted as an
  existing pattern already present in the WS lane's own `run_tts_remote`
  (not a new exposure, just replicated into a second call site). Not
  changed in this pass -- the shared-core refactor above converges it to
  ONE origin instead of two independent copies, which is itself the
  improvement; sanitizing the actual message content is a separate,
  deliberate product decision for both lanes together, not something to
  change unilaterally while fixing this bug.

- **Finding** (`tts_client` global not reset on Hub shutdown): confirmed
  pre-existing and already reachable via the WS lane (which holds
  `tts_client` for a connection's whole lifetime, a wider window than this
  HTTP route's single request). This patch adds a second consumer of an
  existing, unguarded pattern rather than introducing a new one. Noted as
  a known, deferred risk rather than fixed here -- properly reset-guarding
  a module global touched by two independent lanes is a separate change.

Live re-verified post-refactor against the real production Hub: a second
real `POST /api/chat` call succeeded end to end (200, 224.2s, real
`llm_response`, real 781KB `audio_response`, `tts_error: None`) -- confirms
the shared-core refactor didn't change observable behavior.

37 tests total for this seam (13 new/updated in this review-fix pass).
Whole hub suite: 63 failed/1651 passed -- FAILED set is byte-identical to
fresh `origin/main`'s 63 failed/1638 passed (`comm` in both directions:
empty). Zero regressions.

## Restart required

```bash
# Already deployed live during verification. For a future redeploy:
cd <this-worktree>
docker compose --env-file <repo-root>/.env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium -- a hub redeploy still kills every live WebSocket
  connection; this patch does not prevent that, it makes what happens
  *after* it (the fallback) actually deliver both halves of the reply.
  The underlying "redeploying hub disconnects everyone" fact is
  unchanged and not in scope here.
- Severity: low -- the HTTP fallback's total latency now includes TTS
  synthesis time (a real ~7s in the verification run, and TTS itself can
  take up to `HUB_TTS_TIMEOUT_SEC`). Accepted: the alternative is silence,
  and the turn itself was already multi-minute in the case that matters
  (a dead-socket fallback survives a long turn either way).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1911
