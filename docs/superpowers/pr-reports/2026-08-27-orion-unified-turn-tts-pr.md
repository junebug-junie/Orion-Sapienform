# Orion (unified-turn) chat lane now speaks its replies

## Summary

- Fixes a real, user-reported miss: a spoken turn (corr=`7dc1bab2-97a4-4390-89a2-cdd1fa4f0092`)
  got a real, good reply and no voice came back.
- Root cause was structural, not a regression from anything shipped
  yesterday: the `orion`-mode branch's own `continue` (immediately after
  `run_unified_turn`'s `finally:`) always exited the message loop before
  ever reaching the classic lane's own "4. TTS" trigger block, ~40 lines
  further down in the same function -- the only place `run_tts_remote` was
  ever called from. Voice **input** (STT) has always worked for this lane;
  voice **output** never did, regardless of `disable_tts` or anything else.
- Wired a TTS trigger into the orion branch, mirroring the classic lane's
  own gate (`disable_tts`, `tts_client`) and reusing the same per-connection
  `tts_q`/`drain_task` the classic lane already relies on for audio
  playback -- no new plumbing needed on the playback side, only the trigger.

## Outcome moved

Before: an Orion-mode chat turn (the lane microphone input feeds into) had
**no code path to speech at all**. After: it speaks its final reply exactly
like the classic lane does, gated the same way.

Live-verified end to end against the real production Hub: a real WS turn in
`mode: "orion"` with `disable_tts: false` produced a real
`audio_response` frame -- 230KB WAV, 3.6s, synthesized in 1.4s by the real
Coqui XTTS backend.

## Current architecture (before this patch)

`services/orion-hub/scripts/websocket_handler.py`'s message loop has two
lanes inside the same `while True: raw = await websocket.receive_text()`
body:

- **`client_mode == "orion"`** (checked first): builds and awaits
  `run_unified_turn(...)`, fires the affect bracket's pre/post legs around
  it, then unconditionally `continue`s -- back to the top of the loop.
- **classic** (`mode in {"brain", "agent", ...}`, everything that falls
  through the orion check): builds a chat request, dispatches through
  Cortex, and near the end of that same branch has a `# 4. TTS` block that
  computes `will_tts = orion_response_text and not workflow_metadata_only
  and not disable_tts and tts_client` and fires
  `asyncio.create_task(run_tts_remote(orion_response_text, tts_client, tts_q))`.

Because the orion branch's `continue` exits the loop iteration before the
classic branch's code ever runs, **the TTS block was structurally
unreachable from an Orion-mode turn**, full stop -- not conditionally
skipped, never invoked at all.

## Architecture touched

| Seam | Change |
| --- | --- |
| `services/orion-hub/scripts/websocket_handler.py` | new `extract_unified_turn_final_text()` helper; TTS trigger added inside the orion branch, after the affect post-leg, before `continue` |
| `services/orion-hub/tests/test_orion_unified_turn_tts.py` | new -- real unit tests for the extraction logic, plus real-source control-flow checks (this repo has no full WebSocket TestClient harness for this file, matching `test_websocket_agent_claude_routing.py`'s own established convention) |

## Files changed

- `services/orion-hub/scripts/websocket_handler.py`:
  - `extract_unified_turn_final_text(frames)`: pulls the `type == "final"`
    frame's `llm_response` out of `run_unified_turn`'s return value.
    Deliberately does NOT read a `turn_error` frame's `partial_draft` --
    real assistant-authored text the browser does render as a bubble, but
    speaking an error-path partial aloud is a different, untested product
    decision, left out on purpose rather than folded in silently.
  - The orion branch now captures `run_unified_turn`'s return value
    (`orion_turn_frames`), and after the affect post-leg (still gated on
    `websocket.client_state == CONNECTED`, same reasoning as that leg's own
    disconnect check -- synthesizing speech for a socket that's already
    gone is pure waste) computes `orion_will_tts` the same way the classic
    lane does and fires `run_tts_remote` into the same shared `tts_q`.
- `services/orion-hub/tests/test_orion_unified_turn_tts.py`: new, 16 tests.

## Schema / bus / API changes

None. No new channel, schema, or payload shape -- reuses the exact same
`orion:tts:intake` RPC and `TTSClient.speak()` the classic lane already
uses.

## Env/config changes

None.

## Tests run

```text
services/orion-hub/tests/test_orion_unified_turn_tts.py          16 passed

# whole hub suite, branch vs fresh origin/main, identical comparison
branch:        64 failed, 1629 passed, 4 skipped
origin/main:   64 failed, 1613 passed, 4 skipped

Exact same failure COUNT on both. Diffed the two FAILED sets directly:
one test differs in each direction, both in
tests/test_substrate_mutation_manual_route_routing.py (a file already
confirmed order-dependent-flaky earlier this session, on unrelated PRs).
Ran BOTH differing tests together in isolation: 2 passed. Not a
regression -- the same known test-ordering pollution, not this patch.
```

## Evals run

```text
No eval harness exists for orion-hub's voice/TTS path specifically.
Covered by the live WS verification below instead, which is closer to a
real behavioral eval than a unit test could be for this seam.
```

## Docker/build/smoke checks

```text
docker compose --env-file <primary>/.env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml build
  => built clean, image sha256:21b2c0abdf52...

docker compose ... up -d   (REAL redeploy of the live orion-athena-hub container)
  => boot clean: "Application startup complete." / "Uvicorn running..."

# Real WebSocket turn against the live redeployed Hub, mode=orion,
# disable_tts=false, text_input (not voice-triggered, to keep this
# verification independent of the affect-capture cost the mic path
# carries -- see 2026-08-26's memory on that):
[frame] {"type": "final", "llm_response": "Test received. I'm here."}
[frame] {"state": "idle"}
[probe] GOT audio_response, len=230888
  meta={'content_type': 'audio/wav', 'duration_sec': 3.61, 'backend':
  'coqui', 'synthesis_ms': 1406, 'gpu_enabled': True}

Hub log for that turn:
  voice.tts.decision corr=057f8a46-... response_len=43 disable_tts=False
    has_tts_client=True will_tts=True mode=orion
  voice.tts.start text_len=43
  [hub.bus.tts] Sending TTS request to orion:tts:intake
```

Note on the first probe attempt: it appeared to fail (no audio observed),
but the actual cause was the TEST HARNESS closing the connection the
instant it saw the `{"state": "idle"}` frame -- `run_tts_remote` is
fire-and-forget by design (same as the classic lane), so the synthesis
task was still in flight when the probe's own socket closed. A real
browser client stays connected for the session's lifetime and does not
have this problem. Fixed the probe to wait up to 90s after `idle` before
disconnecting, which is what produced the successful run above.

## Review findings fixed

Not yet run through `/code-review` as of this report -- see status below.

## Restart required

```bash
# Already deployed live during verification. For a future redeploy:
cd <this-worktree>
docker compose --env-file <repo-root>/.env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low -- `run_tts_remote` is fire-and-forget (`asyncio.create_task`,
  never awaited), matching the classic lane's existing pattern exactly. If
  the WS handler's own message loop moves on to a new turn before synthesis
  finishes, this is the same behavior the classic lane already has, not a
  new risk this patch introduces.
- Severity: low -- a turn_error frame's `partial_draft` is deliberately NOT
  spoken (see `extract_unified_turn_final_text`'s docstring). If Juniper
  wants error-path partials read aloud too, that is a real, separate
  follow-up, not silently included here.

## PR link

<filled in after push>
