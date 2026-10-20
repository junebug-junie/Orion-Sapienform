# Hub bus listeners come back after a Redis drop

**Depends on:** the `feat/substrate-listener-health` PR (it adds
`orion.core.bus.resilience.run_subscription` with several-channel and pattern
support). This branch is cut from it. Merge that PR first.

## Summary

- Nine hub listeners now reopen their bus subscription after Redis drops,
  instead of dying quietly.
- Before this patch, one Redis error ended each listener for good. The hub
  stayed up and looked healthy, but the panel or loop behind that listener
  stopped getting new data until a restart.
- Each listener now runs under `run_subscription`. It waits 1 s, then doubles up
  to 60 s, and tries again. It logs each failure and records liveness under a
  unique name (`hub.<listener>`).
- Each site keeps its own decode, filter, and handler logic. The dead loop code
  is deleted.
- Hub `/health` now lists every listener. It reads `degraded` while any
  listener is retrying, so a dead listener can no longer hide behind `ok`. It
  still answers HTTP 200, because every current caller checks only that it is
  reachable.
- One new test file proves every converted listener resubscribes after its
  first subscribe fails, and that `/health` shows a listener that is down.

## Outcome moved

Failure mode removed: a hub listener that hits one Redis
`ConnectionError`/`TimeoutError` no longer stops for the life of the process.
Live incident 2026-10-20: a bare `async with bus.subscribe(...)` loop died this
way while its process kept reporting healthy.

## Current architecture

`services/orion-hub/scripts/main.py` starts each listener as a background task.
Each task ran:

```python
try:
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            await handle(msg)
except Exception:
    logger.error("... loop failed")   # task ends here, never restarts
```

## Architecture touched

orion-hub listener loops and the hub `/health` body. No bus channel, schema,
or env key changed.
`orion/core/bus/resilience.py` is used as-is (not edited).

Message shape is unchanged for every handler. `iter_messages` passed through the
raw redis dict when `type` was `message` or `pmessage`. `run_subscription`
passes the same raw dict with the same filter.

| Listener | Name | Channels | Handler shape | Stop |
|---|---|---|---|---|
| `cognition_trace_cache._run` | `hub.cognition_trace_cache` | 1 | async, existing `_handle_message` | cancel |
| `signals_inspect_cache._run` | `hub.signals_inspect_cache` | 1 pattern (`patterns=True`) | async, existing `_handle_message` | cancel |
| `embodiment_outcome_cache._run` | `hub.embodiment_outcome_cache` | 1 | sync, new `_handle_message` (inline body moved out, same logging) | cancel |
| `biometrics_cache._run` | `hub.biometrics_cache` | 3, passed as a list | async, existing `_handle_message` | cancel |
| `bus_synaptic_trigger_notifier._run` | `hub.bus_synaptic_trigger_notifier` | 1 | async, existing `_handle_metacog_trigger` | cancel |
| `vision_frame_cache._run` | `hub.vision_frame_cache` | 1 | async, existing `_handle_message` | cancel |
| `notification_cache._run` | `hub.notification_cache` | 1 | async, existing `_handle_message` | cancel |
| `curiosity_investigation._turn_request_loop` | `hub.curiosity_turn_request` | 1 | sync `_spawn`: starts a task, returns None | `self._stop` |
| `curiosity_investigation._run_state_loop` | `hub.curiosity_run_state` | 1 | async, existing `_handle_run_state` | `self._stop` |

The turn-request handler must return None. `run_subscription` awaits what a
handler returns, so returning the task would make each turn wait for the one
before it.

## Files changed

- `services/orion-hub/scripts/cognition_trace_cache.py`: listener uses `run_subscription`.
- `services/orion-hub/scripts/signals_inspect_cache.py`: same, pattern subscription.
- `services/orion-hub/scripts/embodiment_outcome_cache.py`: same; message body moved into a sync `_handle_message`.
- `services/orion-hub/scripts/biometrics_cache.py`: same, three channels.
- `services/orion-hub/scripts/bus_synaptic_trigger_notifier.py`: same.
- `services/orion-hub/scripts/vision_frame_cache.py`: same.
- `services/orion-hub/scripts/notification_cache.py`: same.
- `services/orion-hub/scripts/curiosity_investigation.py`: both durable listeners use `run_subscription` with the investigation's stop event.
- `services/orion-hub/scripts/api_routes.py`: `/health` adds `listeners_ok`, `listeners_down`, `listeners`; `status` is `degraded` while a listener is down.
- `services/orion-hub/tests/test_hub_listeners_resubscribe.py`: new regression tests.
- `docs/superpowers/pr-reports/2026-10-20-listener-resubscribe-hub-pr.md`: this report.

## Schema / bus / API changes

- Added: hub `/health` fields `listeners_ok`, `listeners_down`, `listeners`.
- Removed: none
- Renamed: none
- Behavior changed: listeners retry instead of ending. A handler error is now
  logged by `run_subscription` (`<name> handle_failed`) and the next message is
  read; this matches the old per-message `try/except` at the sites that had one.
  At the sites without one (the caches), an exception in a handler used to end
  the listener; now it skips that message.
- Compatibility notes: on hub shutdown, cache tasks now end with
  `CancelledError` instead of catching it. Every `stop()` already awaits the
  task inside `except asyncio.CancelledError`. Hub `/health` keeps `status` and
  `service` and still returns HTTP 200; `status` can now read `degraded`.
  Callers found (`verify_mind_llm_e2e.sh`, `smoke_memory_cognition_loop_e2e.sh`,
  `run_self_sense_eval.py`) check only that it is reachable.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=.:services/orion-hub python -m pytest -q -p no:cacheprovider \
  services/orion-hub/tests/test_hub_listeners_resubscribe.py
11 passed

# Every hub test file that imports a touched module (41 files + vision/notifier tests)
baseline (main @ 6b48ef9aa):  1115 passed   -> no pre-existing failures
branch, same files + new file: 1126 passed  (1115 + 11 new; no regressions)

python -m pytest -q orion/core/bus/tests/test_resilience.py
8 passed
```

Mutation checks (each reverted afterwards):

- Old `notification_cache._run` loop restored: its case fails (the listener logs
  "loop failed" and never delivers a message).
- Turn-request handler made to `return task`: the no-serialize test fails.

## Evals run

```text
None. This is a transport-resilience fix with no quality dimension to score;
the regression tests above cover the behavior.
```

## Docker/build/smoke checks

```text
Not run. No dependency, compose, or env change. Live resubscribe after a real
Redis drop is UNVERIFIED until the hub is redeployed.
```

## Review findings fixed

A code review ran in a subagent. It found no blocking issues. It confirmed
that the message shape, the turn fire-and-forget, stop/cancel behavior, and the
embodiment refactor are all correct, and that the test fails on the old code.

- Finding (should-fix): hub `/health` did not read `subscription_health()`, so a
  listener stuck in backoff still looked healthy.
  - Fix: `/health` now reports listener state and reads `degraded` while one is down.
  - Evidence: `test_hub_health_reads_degraded_while_a_listener_is_down` passes.
- Finding (minor): no test ran the refactored sync embodiment handler body.
  - Fix: added `test_embodiment_handler_records_outcome_fields_and_skips_undecodable`.
  - Evidence: passes; checks the five recorded fields and that an undecodable message adds nothing.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after both PRs merge.

## Risks / concerns

- Severity: low
- Concern: hub `/health` can now read `degraded`. A future caller that treats
  any status other than `ok` as down would see the hub as down while a listener
  retries.
- Mitigation: it still answers HTTP 200. The callers found today check only
  that it is reachable.
- Severity: low
- Concern: five other hub listeners were out of scope (`room_claude_relay`,
  `collapse_mirror_chat_reply`, `harness_step_relay`, `gpu_pool_routes`,
  `curiosity_urgent`). Each already wraps its subscribe in its own retry loop,
  but none reports to `/health`.
- Mitigation: follow-up to move them onto `run_subscription` for one retry
  policy and `/health` visibility.

## PR link

Not opened (by request). Branch: `fix/listener-resubscribe-hub`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
