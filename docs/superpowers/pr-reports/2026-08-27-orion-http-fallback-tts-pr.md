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
services/orion-hub/tests/test_handle_chat_request_http_fallback_tts.py   6 passed
services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py
services/orion-hub/tests/test_handle_chat_request_orion_mode_continuity.py
services/orion-hub/tests/test_chat_route_tagging.py
services/orion-hub/tests/test_orion_unified_turn_tts.py                38 passed (all pre-existing, unaffected)

# whole hub suite, branch vs fresh origin/main
branch:        64 failed, 1643 passed, 4 skipped
origin/main:   63 failed, 1638 passed, 4 skipped

Diffed the two FAILED sets: zero failures unique to origin/main; one
failure unique to the branch
(test_substrate_mutation_manual_route_routing.py::test_routing_dry_run_produces_trial_and_decision_without_side_effects),
the same file already confirmed order-dependent-flaky multiple times
earlier this session on unrelated PRs. Ran it alone: 1 passed. Not a
regression.
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

Pending -- `/code-review high` launched, findings will be applied and this
section updated before merge is requested.

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

<filled in after push>
