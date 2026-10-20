# Shared bus listeners resubscribe after a Redis drop

Depends on PR #2786 (`feat/substrate-listener-health`). That PR adds
`run_subscription` and `subscription_health` in `orion/core/bus/resilience.py`.
Merge #2786 first. This branch is stacked on it.

## Summary

- Four listeners in the shared `orion` package now survive a Redis drop. Before, each one
  died on the first Redis error and its process stayed up, looking healthy.
- Each listener now uses `run_subscription`. It opens the subscription again with a backoff
  (1 s, doubling to 60 s), logs the failure, and records live state that `/health` can read.
- Each site keeps its own decode, filter, and handler logic. Each site also keeps its own way to stop.
- The concept-induction debug view now shows when the intake subscription has failed,
  how many times, and whether it is open now.
- One regression test per module. Each test fails the first subscription and checks that a
  later message still arrives.

## Outcome moved

Failure mode removed: one Redis `ConnectionError`/`TimeoutError` ended these listeners for good.
This happened live on 2026-10-20 in `orion-substrate-runtime`. These four listeners had the same bug:

- Concept induction stopped taking in chat, metacog, and world-pulse events. The service stayed up.
- World-pulse and concept-induction stopped tracking the active goal. Their capability checks then used an old goal.
- The sql_router writer thread stopped writing rows.
- A model-replay eval missed every pool grant after a drop. Each hold then waited until its timeout.

## Current architecture

Each site had its own `async with bus.subscribe(...)` loop. That loop used `iter_messages` or
`get_message`. There was no outer retry, so any exception ended the loop. The concept-induction
worker then went to `stop_reason=failed` and nothing restarted it. The goal-state listener only
logged "stopped".

## Architecture touched

Shared package only: `orion/core/sql_router`, `orion/autonomy`, `orion/spark/concept_induction`,
`orion/evals/model_replay`. No service code, bus channels, schemas, or env keys changed.

How each handler runs (`run_subscription` awaits the handler before it reads the next message):

| Site | Handler shape | Listener name |
|---|---|---|
| `sql_router/worker.py` `start_worker_thread` | inline async: sync DB write, then await publish. Same as before. | `sql-router-writer:<channel>:<Model>` |
| `autonomy/goal_state_listener.py` | inline async `_handle_bus_message`. Same as before. | `goal-state-listener:<channel>` |
| `spark/concept_induction/bus_worker.py` `ConceptWorker.start` | inline async `_handle_intake_message`. The old loop body moved into it unchanged. Event order matters for the window. | `concept-induction-intake` |
| `evals/model_replay/pool_hold.py` `BusControlTransport` | inline sync `_on_pool_event`. It only resolves futures. | `model-replay-pool-events` |

None of these sites started work in the background, so none of them returns a task.

Who uses each module:

- `sql_router/worker.py`: **nothing imports `start_worker_thread` today.** notify-digest and
  biometrics import only `orion.core.sql_router.db`. The worker is converted anyway, so a future
  caller gets the safe version.
- `goal_state_listener`: `services/orion-world-pulse/app/main.py` and
  `services/orion-spark-concept-induction/app/main.py`. The call signature did not change.
  Both services still stop it with the stop event or a task cancel.
- `ConceptWorker`: `services/orion-spark-concept-induction/app/main.py`. It still runs
  `start()` as a task and cancels it on shutdown. `/debug/concept-induction` shows
  `worker_liveness_status()`.
- `BusControlTransport`: `scripts/run_bonsai_replay_eval.py`. It opens one transport per run.

## Files changed

- `orion/core/sql_router/worker.py`: the subscribe loop is replaced by `run_subscription`. The message body moved into `handle`. Publish now uses `publish_with_reconnect`.
- `orion/core/sql_router/tests/test_sql_router_worker_resubscribe.py`: new regression test. The thread survives a mid-stream drop and a failed resubscribe.
- `orion/autonomy/goal_state_listener.py`: the loop is replaced by `run_subscription`. The docstring now says it resubscribes.
- `orion/autonomy/tests/test_goal_state_listener.py`: new regression test.
- `orion/spark/concept_induction/bus_worker.py`: `start()` uses `run_subscription`. The loop body is now `_handle_intake_message`. `stop()` sets a stop event, so the loop also ends when `stop()` is called without a cancel. `worker_liveness_status()` adds the live subscription state to the old fields.
- `orion/spark/concept_induction/tests/test_concept_induction.py`: `FakeStreamingBus` now gives a `get_message` pubsub for each scripted session, because `iter_messages` is no longer used. New regression test.
- `orion/evals/model_replay/pool_hold.py`: `_listen` uses `run_subscription`. `__aenter__` waits until the listener is subscribed. Before, it waited on a one-time ready event.
- `orion/evals/model_replay/tests/test_pool_hold.py`: new regression test.

## Schema / bus / API changes

- Added: `/debug/concept-induction` → `worker_liveness` gets these new keys:
  - `bus_subscribed`: is the subscription open now.
  - `bus_subscription_failures_total`: how many times the subscription has failed.
  - `total_events_failed` and `last_handler_exception`: envelopes that the handler could not process.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `worker_last_exception` is now set when the intake subscription fails, even while it retries.
    Before, it was set only when the worker died.
  - `bus_consumer_started` means "this run has subscribed at least once". This matches its old meaning.
  - A `handle_envelope` error no longer kills the concept-induction worker. It is counted and logged
    (`concept_induction_worker_handle_failed`), and the next message is read. Before, one bad envelope ended intake for good.
  - With the bus disabled, `ConceptWorker.start()` still fails at once with `RuntimeError("Bus disabled")`. It does not retry forever.
- Compatibility notes: no payload or channel changes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=<worktree> .venv/bin/python -m pytest -q -p no:cacheprovider \
  orion/core/bus/tests/test_resilience.py orion/core/sql_router/tests orion/autonomy/tests \
  orion/spark/concept_induction/tests orion/evals/model_replay/tests
  -> 413 passed, 2 failed (stable across repeated runs)
cd services/orion-spark-concept-induction && pytest tests   -> 12 passed
cd services/orion-world-pulse && pytest tests               -> 155 passed, 2 failed
cd services/orion-substrate-runtime && pytest tests/test_health_listeners.py -> 3 passed
pytest tests/test_execution_dispatch_bus_catalog.py         -> 3 passed
```

The 4 failures are the same before this branch and on `origin/main` (6b48ef9aa):
`test_falkor_materialization.py::test_materialize_populates_write_through_cache`,
`::test_hub_store_sees_spark_writes_only_after_rehydrate`, and world-pulse
`test_world_pulse_pipeline.py::test_run_world_pulse_with_fixture_fetch`,
`::test_pipeline_fixture_mode_without_network`. This branch did not cause them.

The sql_router regression test was also run against the old `worker.py`. It fails there, so it catches the bug.

## Evals run

```text
None. This is a transport-resilience fix. Behavior is covered by the regression tests above.
These modules have no eval harness for bus transport.
```

## Docker/build/smoke checks

```text
Not run. No dependency, compose, or env change. Live resubscribe after a real Redis drop is UNVERIFIED until the services are rebuilt.
```

## Review findings fixed

A code review ran in a subagent. It found no must-fix problems. These findings were fixed:

- Finding: when `BusControlTransport.__aenter__` timed out, the listener kept trying to resubscribe forever and the bus stayed open. `__aexit__` does not run when `__aenter__` fails.
  - Fix: a failed `__aenter__` now calls `__aexit__`. That cancels the listener, waits for it to end, and closes the bus.
  - Evidence: `test_bus_transport_enter_timeout_stops_the_listener_and_closes_the_bus`.
- Finding: the transport's readiness check could pass on an old "subscribed" row left by an earlier transport.
  - Fix: `__aexit__` now waits for the cancelled listener. Readiness also requires `last_subscribed_at` to be after this transport opened.
  - Evidence: `test_bus_transport_grant_listener_resubscribes_after_a_drop`.
- Finding: the sql_router test patched `asyncio.sleep` for the whole process.
  - Fix: the test now patches only the worker's `run_subscription` backoff (`max_backoff_sec=0.01`).
  - Evidence: the test passes, and the suite passed 3 times in a row.
- Finding: a `handle_envelope` error was no longer visible in the concept-induction debug view.
  - Fix: errors are now counted in `total_events_failed` and saved in `last_handler_exception`. Each one is logged as `concept_induction_worker_handle_failed`.
  - Evidence: the regression test makes the first envelope fail and checks both fields.
- Finding (nit): a read that failed while `stop()` closed the bus would show up as `worker_last_exception`.
  - Fix: a failure is copied into that field only if it happened before `stop()` was called.
  - Evidence: the regression test still sees the real mid-stream `ConnectionError`.
- Finding (nit): the sql_router bus was built outside the worker thread's event loop, and publish had no reconnect.
  - Fix: the bus is now built inside the thread. Publish now uses `publish_with_reconnect`.
  - Evidence: the sql_router test checks both publishes.
- Not fixed (nit): in the sql_router worker, a failed `bus.connect()` at startup still kills the thread. The old code did the same. Nothing calls this worker today.

## Restart required

Rebuild the two services that run these listeners, after #2786 and this PR merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-spark-concept-induction up -d --build
scripts/safe_docker_build.sh orion-world-pulse up -d --build
```

The model-replay eval and sql_router worker need no restart. The eval picks up the change on its next run. Nothing runs the sql_router worker.

## Risks / concerns

- Severity: low
  - Concern: a grant published while the pool listener is reconnecting is missed.
  - Mitigation: that hold's `wait_granted` times out, which is the same as any lost grant today. The old code missed every later grant.
- Severity: low
  - Concern: the concept-induction worker no longer dies on a handler error, so a handler bug that fails every message now shows up only in logs.
  - Mitigation: `total_events_failed` and `last_handler_exception` in `/debug/concept-induction` show it. Each failure is also logged with a traceback.
- Severity: low
  - Concern: listener names are unique per process, not per instance. Two `ConceptWorker`s or two open `BusControlTransport`s in one process would share one health row.
  - Mitigation: each service runs one of each. #2786 logs a warning when a name is registered twice.

## PR link

Not opened (stacked on #2786).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
