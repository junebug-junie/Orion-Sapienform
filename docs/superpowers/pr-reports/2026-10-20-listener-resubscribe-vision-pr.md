# Vision listeners come back after a Redis drop

Depends on PR #2786 (`feat/substrate-listener-health`). That PR adds
`run_subscription`, the reconnecting listener this patch uses. Merge #2786 first.
This branch is stacked on it.

## Summary

- Eleven bus listeners in seven services used to stop for good after one Redis
  error. The process stayed up and looked healthy. Each one now reconnects
  with backoff (1 s, doubling to 60 s).
- Three services (frame-router, vision-host, world-model) had a fake retry.
  It caught the error, slept 1 s, then read the same broken connection again,
  forever. That loop is gone. They now open a new connection.
- Each listener has a unique name (for example `vision-window.consume_rpc`).
  The name shows up in logs, and it is recorded in the process's listener
  health state.
- Work that used to run in the background still runs in the background. Work
  that used to run one message at a time still runs one message at a time.
- One regression test per service. Each test fails on the old loops and passes
  on the new ones.

## Outcome moved

Failure mode removed: a vision service that silently stops hearing the bus
after one Redis hiccup. Before, the only fix was a restart. Now the listener
logs the failure and subscribes again on its own.

## Current architecture

Each service opened one subscription at startup and read it in a plain loop.
`async with bus.subscribe(...)` ended on the first `ConnectionError` or
`TimeoutError` from redis, and the task died with it. Nothing restarted it.

## Architecture touched

Listener loops only. No bus channel, schema, env key, or HTTP route changed.
Each loop body moved into a small handler method, and
`orion.core.bus.resilience.run_subscription` now owns the subscribe/read/retry.

| Service | Listener (name) | Handler shape |
|---|---|---|
| vision-frame-router | `_frames_loop` (`vision-frame-router.frames`) | starts a task, returns None |
| vision-frame-router | `_reply_loop` (`vision-frame-router.replies`, glob pattern) | starts a task, returns None |
| vision-scribe | `_consume` (`vision-scribe.consume`) | starts a task, returns None |
| vision-scribe | `_consume_rpc` (`vision-scribe.consume_rpc`) | starts a task, returns None |
| vision-council | `_consume` (`vision-council.consume`) | starts a task, returns None |
| vision-council | `_consume_rpc` (`vision-council.consume_rpc`) | starts a task, returns None |
| vision-window | `_consume` (`vision-window.consume`) | awaited inline |
| vision-window | `_consume_identity` (`vision-window.consume_identity`) | awaited inline |
| vision-window | `_consume_rpc` (`vision-window.consume_rpc`) | starts a task, returns None |
| vision-edge | `run_detector_loop` (`vision-edge.detector`) | awaited inline, no stop event (stopped by task cancel) |
| vision-host | `_consume_loop` (`vision-host.intake`) | starts a task, returns None |
| world-model | `_consume_loop` (`world-model.intake`) | starts a task, returns None |

Messages look the same to the handlers. `iter_messages` and `run_subscription`
both pass the raw redis dict, filtered to `message`/`pmessage`.

Side effect: before, an unexpected error inside a loop body ended the
listener. Now `run_subscription` logs it and reads the next message.

## Files changed

- `services/orion-vision-frame-router/app/main.py`: two listeners, fake retry removed.
- `services/orion-vision-scribe/app/main.py`: two listeners.
- `services/orion-vision-council/app/main.py`: two listeners.
- `services/orion-vision-window/app/main.py`: three listeners.
- `services/orion-vision-edge/app/detector_worker.py`: detector listener.
- `services/orion-vision-host/app/main.py`: intake listener, fake retry removed.
- `services/orion-world-model/app/main.py`: intake listener, fake retry removed.
- `services/*/tests/test_listener_resubscribes.py` (seven files): regression tests.
- `docs/superpowers/pr-reports/2026-10-20-listener-resubscribe-vision-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: listeners reconnect after a Redis error instead of stopping.
- Compatibility notes: every touched Dockerfile already copies the in-repo
  `orion/` package (`COPY orion ./orion`), so the images get `run_subscription`
  from the same commit.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

Each fake bus fails the first subscribe, drops the second session mid-stream,
and only works on the third. The listener must still see both messages.

```text
PYTHONPATH=.:services/<svc> .venv/bin/python -m pytest services/<svc>/tests -q
                         branch                     origin/main baseline
vision-frame-router      109 passed                 108 passed
vision-scribe            12 passed                  11 passed
vision-council           125 passed                 124 passed
vision-window            170 passed                 168 passed
vision-edge              18 passed, 1 error         17 passed, 1 error
  (error = test_camera_reconnect.py collection: Settings needs SOURCE; same on main)
orion/core/bus/tests/test_resilience.py  8 passed

torch suites, run inside the existing service images (worktree mounted read-only):
world-model              61 passed, 2 failed        60 passed, 2 failed
  (both failures in test_device_strategy_and_dtype.py; same on main)
vision-host              369 passed, 1 skipped      368 passed, 1 skipped

New tests against the pre-patch code: all fail (frame-router 1, scribe 1,
council 1, window 2, edge 1, host 1, world-model 1).
```

## Evals run

```text
None. This is transport plumbing, not cognition quality. The regression tests
are the check. No eval harness exists for these listeners.
```

## Docker/build/smoke checks

```text
No image build. Tests for vision-host and world-model ran inside the current
images with the source mounted read-only. Live reconnect after a real Redis
drop: UNVERIFIED.
```

## Review findings fixed

A code review ran in a subagent. It found no blockers. It confirmed that every
moved `continue` became a `return` in the same place, that lock and try
boundaries did not move, and that the handler shapes are right.

- Finding: the edge test left `app.utils` and other modules imported against
  its stub context in `sys.modules`. A later test could depend on that order.
  - Fix: the test snapshots `sys.modules` first and removes every new `app.*` module afterwards.
  - Evidence: edge suite 18 passed (same pre-existing collection error as main).
- Finding: the frame-router test checked `patterns=True` but not the channel name.
  - Fix: it now asserts the exact channel and the patterns flag on all three subscribe calls.
  - Evidence: frame-router suite 109 passed.
- Finding (not fixed, recorded): no CI workflow runs the tests for vision-host,
  -edge, -window, -council, -scribe or -frame-router. CI runs world-model
  tests only, in `orion-gpu-pool-tests.yml`. The new tests here ran locally.
  The vision-host and world-model tests ran inside the service images.
- Finding (not fixed, pre-existing): bare `asyncio.create_task(...)` calls keep
  no reference to the task they start. Python can then delete a task while it
  is still running. This was true before this patch. It is out of scope here.

## Restart required

Rebuild and restart each touched service from the clean primary `main`
checkout after merge (the post-pull rebuild covers this):

```bash
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
scripts/safe_docker_build.sh orion-vision-scribe up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-vision-window up -d --build
scripts/safe_docker_build.sh orion-vision-edge up -d --build
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-world-model up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `services/orion-vision-edge/app/routes.py` still has a bare
  subscribe loop. It is a per-request debug stream (server-sent events). It
  ends with the HTTP request, and the browser reconnects. Not converted.
- Mitigation: none needed. It is not a long-lived listener.

- Severity: low
- Concern: these services do not expose `subscription_health()` on their
  `/health` yet. A listener that keeps failing is visible in logs only.
- Mitigation: follow-up, if wanted. Same pattern as #2786.

## PR link

Not opened (branch pushed only).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
