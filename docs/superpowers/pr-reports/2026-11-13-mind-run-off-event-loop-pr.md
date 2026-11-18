## Summary

- orion-mind's `/v1/mind/run` now runs the synchronous `run_mind` in a worker thread (`asyncio.to_thread`), one run at a time.
- Before, it ran on the event loop and froze the whole service for each run (minutes): `/health` timed out, Prime's ticks, the curiosity replica and the Claude-tool consent responder all stalled.
- The run lock is released when the thread ends, not when the request does, so a cancelled request cannot start a second run beside a live one.
- Prime's "Juniper first" docstring updated: ticks now run during a turn; `JUNIPER_TURNS.active()` (thread-safe) keeps Prime from bidding while a Juniper turn is live.

## Outcome moved

Live 2026-11-13 19:46-19:52Z, after the #3340 deploy: `orion-mind` was `unhealthy` (FailingStreak 7, "Health check exceeded timeout (10s)") while two world-pulse reads ran; a manual `curl /health` hung 30 s; Prime logged `physiology_bid_unavailable err=TimeoutError`. A consent ask arriving in such a window waits out its 40 s and reads as "no". Present since `mind_run` was added (2026-05); Prime joined the loop 2026-10-30.

## Current architecture

`async def mind_run` called `run_mind(...)` directly. run_mind's LLM calls (`llm_client._run_blocking`) spin a thread + fresh loop per call and join it, blocking the main loop.

## Architecture touched

- `services/orion-mind/app/main.py`: `_run_mind_sync` + `_MIND_RUN_LOCK`; handler awaits a shielded worker-thread future.
- `services/orion-mind/app/prime_loop.py`: docstring only.

## Files changed

- `services/orion-mind/app/main.py`: the fix.
- `services/orion-mind/app/prime_loop.py`: docstring now true.
- `services/orion-mind/tests/test_mind_run_off_event_loop.py`: 4 regression tests.

## Schema / bus / API changes

None. Behavior change: non-Juniper runs (world-pulse reads) no longer pause Prime/replica as a side effect; they now bid alongside (GPU pool ranks them).

## Env/config changes

None.

## Tests run

```text
tests/test_mind_run_off_event_loop.py: 4 passed (also with -p no:asyncio)
services/orion-mind/tests: 386 passed
Mutation: calling _run_mind_sync on the loop -> health test FAILS; async-with lock release -> cancellation test FAILS
```

## Evals run

No orion-mind latency eval exists. Live check after deploy: `docker inspect -f '{{.State.Health.Status}}' orion-mind` stays healthy through a world-pulse read; `curl /health` answers in < 1 s mid-run.

## Docker/build/smoke checks

Not deployed from this branch.

## Review findings fixed

- Finding: cancelled request released the lock while its thread still ran (runs could overlap).
  - Fix: lock released by the worker future's done-callback; awaiter shields it.
  - Evidence: `test_a_cancelled_request_keeps_the_next_run_waiting_for_its_thread` (fails on the async-with form).
- Finding: module-level asyncio.Lock could carry one test's loop into another.
  - Fix: autouse fixture gives each test a fresh lock.
  - Evidence: 4 passed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: shutdown waits for a running thread (up to Docker's stop timeout). Mitigation: same as before (the frozen loop also waited).
- Issues opened/closed: none.

## PR link

(this PR)
