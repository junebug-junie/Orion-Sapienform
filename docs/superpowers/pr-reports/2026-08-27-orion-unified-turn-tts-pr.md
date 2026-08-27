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
- Wired a TTS trigger into the orion branch, then (post-review) extracted
  the gate/dispatch logic both lanes now share into one function,
  `dispatch_tts_reply()`, so the two lanes cannot silently diverge again.

## Outcome moved

Before: an Orion-mode chat turn (the lane microphone input feeds into) had
**no code path to speech at all**. After: it speaks its final reply exactly
like the classic lane does, gated the same way, through one shared
function both lanes call.

Live-verified end to end against the real production Hub, twice (once
pre-review-fix, once post-refactor): a real WS turn in `mode: "orion"`
with `disable_tts: false` produced a real `audio_response` frame each time.

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
| `services/orion-hub/scripts/websocket_handler.py` | new `extract_unified_turn_final_text()` and `dispatch_tts_reply()`; both lanes now route through the shared dispatch function |
| `services/orion-hub/tests/test_orion_unified_turn_tts.py` | new -- real behavioral tests for both helpers, plus real-source control-flow checks for the parts this repo has no full WebSocket TestClient harness for |
| `config/metrics/metric_definitions.lock.json` | re-locked (unrelated housekeeping, see Review findings section) |

## Files changed

- `services/orion-hub/scripts/websocket_handler.py`:
  - `extract_unified_turn_final_text(frames)`: pulls the `type == "final"`
    frame's `llm_response` out of `run_unified_turn`'s return value.
    Deliberately does NOT read a `turn_error` frame's `partial_draft` --
    real assistant-authored text the browser does render as a bubble, but
    speaking an error-path partial aloud is a different, untested product
    decision, left out on purpose rather than folded in silently.
  - `dispatch_tts_reply(...)`: the shared gate + fire-and-forget dispatch
    (with GC-safety, see Review findings) both the orion and classic lanes
    now call. `extra_gate`/`log_extra` let the classic lane fold in its own
    `workflow_metadata_only` exclusion without this function needing to
    know that concept exists.
  - The orion branch now captures `run_unified_turn`'s return value
    (`orion_turn_frames`), and after the affect post-leg (still gated on
    `websocket.client_state == CONNECTED`, same reasoning as that leg's own
    disconnect check) calls `dispatch_tts_reply(...)`.
  - The classic lane's pre-existing "4. TTS" block now calls the same
    function instead of hand-rolling its own copy of the gate.
- `services/orion-hub/tests/test_orion_unified_turn_tts.py`: new, 24 tests.
- `config/metrics/metric_definitions.lock.json`: re-locked against latest
  `main` -- unrelated to this PR's actual diff (`change_count: 0`), fixes
  a stale committed merge-base pointer that was failing CI (see Review
  findings section).

## Schema / bus / API changes

None. No new channel, schema, or payload shape -- reuses the exact same
`orion:tts:intake` RPC and `TTSClient.speak()` the classic lane already
uses.

## Env/config changes

None.

## Tests run

```text
services/orion-hub/tests/test_orion_unified_turn_tts.py          24 passed
  (16 initial + 8 added during the review-fix pass: dispatch_tts_reply's
  real gate/dispatch behavior, the GC-safety fix, and the classic-lane
  shared-function regression proof)

# whole hub suite, branch (post-review-fix, post-rebase) vs fresh origin/main
branch:        63 failed, 1638 passed, 4 skipped
origin/main:   64 failed, 1613 passed, 4 skipped

Diffed the two FAILED sets directly: zero failures unique to the branch;
one origin/main-only failure
(test_substrate_mutation_manual_route_routing.py::test_routing_apply_succeeds_for_auto_promote_and_can_rollback),
the same file already confirmed order-dependent-flaky earlier this
session on unrelated PRs -- it simply didn't flake on this particular run.
Zero new regressions.
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
  -f services/orion-hub/docker-compose.yml build && up -d
  => built clean, boot clean ("Application startup complete.")

# Real WebSocket turn against the live redeployed Hub, mode=orion,
# disable_tts=false, text_input (not voice-triggered, to keep this
# verification independent of the affect-capture cost the mic path
# carries -- see 2026-08-26's memory on that):

# Pre-review-fix:
[frame] {"type": "final", "llm_response": "Test received. I'm here."}
[probe] GOT audio_response, len=230888
  meta={'duration_sec': 3.61, 'synthesis_ms': 1406, 'gpu_enabled': True}
Hub log: voice.tts.decision corr=057f8a46-... will_tts=True mode=orion
         voice.tts.start / [hub.bus.tts] Sending TTS request to orion:tts:intake

# Post-review-fix (rebuilt + redeployed with the extracted dispatch_tts_reply()):
[frame] {"type": "final", "llm_response": "Say a short test sentence back to me."}
[probe] GOT audio_response, len=178280
  meta={'duration_sec': 2.78, 'synthesis_ms': 1263, 'gpu_enabled': True}
```

Note on the first probe attempt during initial verification: it appeared
to fail (no audio observed), but the actual cause was the TEST HARNESS
closing the connection the instant it saw the `{"state": "idle"}` frame --
`run_tts_remote` is fire-and-forget by design (same as the classic lane),
so the synthesis task was still in flight when the probe's own socket
closed. A real browser client stays connected for the session's lifetime
and does not have this problem. Fixed the probe to wait up to 90s after
`idle` before disconnecting, which is what produced both successful runs
above.

## Review findings fixed

High-effort `/code-review` run. 3 findings, all real, all fixed.

- **Finding**: `test_orion_lane_calls_run_tts_remote` asserted
  `source.count("run_tts_remote(") >= 2`, which was ALREADY true on
  `origin/main` before this fix (2: the `async def run_tts_remote(`
  definition plus the pre-existing classic-lane call -- verified via
  `git show origin/main:... | grep -c run_tts_remote`). A future revert of
  the orion-lane call would leave the count at 2, still satisfying `>= 2`,
  and this test would keep passing while the exact bug it exists to catch
  came back.
  - Fix: replaced with real behavioral tests against the actual shared
    dispatch function -- `test_dispatch_fires_when_all_conditions_are_met`,
    `test_dispatch_does_not_fire_when_any_condition_fails`,
    `test_the_orion_lane_actually_calls_the_shared_dispatch_function`
    (verifies the SPECIFIC orion-branch call site, not a substring count).

- **Finding**: ~30 lines of TTS gating logic (the `will_tts` computation,
  the decision log line, the `create_task(run_tts_remote(...))` dispatch)
  duplicated between the classic lane and the new orion lane -- exactly the
  shape that produced the bug this PR fixes in the first place (one lane
  wired, one lane not, nothing enforcing they stay in sync).
  - Fix: extracted `dispatch_tts_reply()`, called by BOTH lanes.
  - Evidence: `test_classic_lane_also_routes_through_the_shared_dispatch_function`,
    `test_extra_gate_can_suppress_dispatch_even_with_everything_else_ok`.

- **Finding**: `asyncio.create_task(run_tts_remote(...))` was fire-and-forget
  with no reference retained anywhere. asyncio holds only a WEAK reference
  to a running task, so it could be garbage-collected mid-synthesis with
  nothing surfaced -- and this PR was about to replicate that same
  pre-existing classic-lane pattern into a second call site rather than
  fix it.
  - Fix: `_TTS_DISPATCH_INFLIGHT` set + done-callback inside
    `dispatch_tts_reply` itself, so BOTH lanes get the fix from one place.
    Same shape already applied to
    `services/orion-whisper-tts/app/cuda_watchdog.py`'s `_INFLIGHT` set
    earlier the same day (PR #1901).
  - Evidence: `test_dispatched_task_is_strongly_referenced_then_released`.

**Self-caught while writing the new tests, not from the review**: the
shared gate's `bool(text and ...)` truthiness check treated a
whitespace-only string (`"   "`) as real text to speak -- Python's bare
truthiness on a non-empty string. This was a genuine, PRE-EXISTING bug in
the classic lane's own original gate (`bool(orion_response_text and ...)`,
no `.strip()`), never exercised until a real parametrized behavioral test
tried a whitespace-only value. Fixed with `.strip()` truthiness for both
lanes at once, since they now share one gate.

**Separately, CI (`Static repo gates` / `check_definition_drift.py`) was
failing** for a reason unrelated to this PR's actual diff: the committed
`config/metrics/metric_definitions.lock.json` carried a stale merge-base
pointer whose own drift had already been resolved upstream by a separate
PR's re-lock commit -- but that re-lock only escapes the gate's detection
when comparing `HEAD` to itself (confirmed: a clean checkout of
`origin/main` passes with an explicit "not verified, HEAD is the merge
base" note); any branch built on top of it still fails until its own
`--update` run. Rebased onto latest `main` and ran
`check_definition_drift.py --update`; `change_count: 0` confirms this
PR introduces no real metric-lineage drift.

## Restart required

```bash
# Already deployed live during verification. For a future redeploy:
cd <this-worktree>
docker compose --env-file <repo-root>/.env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low -- `run_tts_remote` is fire-and-forget, matching the
  classic lane's existing pattern; the GC-safety gap that shape had is now
  fixed (see Review findings).
- Severity: low -- a turn_error frame's `partial_draft` is deliberately NOT
  spoken (see `extract_unified_turn_final_text`'s docstring). If Juniper
  wants error-path partials read aloud too, that is a real, separate
  follow-up, not silently included here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1905
