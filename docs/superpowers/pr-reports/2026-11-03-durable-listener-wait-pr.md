## Summary

- A deploy no longer costs readings. Sometimes a durable run sends a request while the service that answers it is restarting. The run now hands its GPU seat back, waits, and sends the request again. Before this patch, the run failed for good.
- The wait is 60 s, then 120 s, then 240 s, then 300 s for each later wait. A run may wait at most 6 times, about 22 minutes in total. After that, the run fails as it did before. Urgent runs may wait at most 2 times, 10 s then 20 s.
- A wait does not use up one of the run's normal attempts. A counter on the run (`listener_waits`) records each wait. Each wait also writes a `run.listener_wait` event.
- Only one of the two failure types is sent again: "published to nobody". In that case no worker received the request, so a second send cannot do the work twice. The other type is "listener gone while awaiting its reply". There, the worker received the request and may have started work, so it keeps today's path and uses up one attempt.
- The patch covers every durable workflow whose work step sends a listener-checked request (`require_listener=True`). `reverie.visual` and `dream.carry` already retried these failures without using up an attempt.

## Outcome moved

Live on 2026-10-31, 16 reading runs failed with `RpcNoListener` in four bursts: 04:28, 07:58, 22:29 and 23:20 UTC. Each burst lined up with a service redeploy.

Over 14 days, 26 reading runs failed this way. In 10 of them the request was published to nobody. In 16 the listener disappeared while the run waited for its reply.

The reading queue's own retry rescued most seeds, but each rescue used up one seed attempt. One seed, `digest_item:8411d608...`, failed for good: its last attempt was an `RpcNoListener`. Over the same 14 days, 1 `curiosity.investigate` run and 1 `self_sense_eval` run also failed this way.

The eval replays one deploy window with 16 readings:
- With the wait off: 16 of 16 fail. This matches the incident.
- With the wait on: 16 of 16 complete, and no run uses up an attempt.

## Current architecture

- PR #3022 made every request from durable-runs fail fast when nobody is listening (`require_listener=True` on all 7 of the runner's RPCs). This stopped runs from holding a GPU seat for hours.
- After that fast failure, each workflow's normal failure path ran:
  - `reading.turn` went straight to terminal `failed`.
  - curiosity and self-sense used up an attempt, with a 30 s or 60 s backoff and 3 attempts.
  - compactor retried immediately, 3 times.
- None of these paths can outlast a redeploy that takes minutes.

## Architecture touched

- `orion/core/bus/async_service.py`: a new error type, `RpcPublishedToNobody`, a subclass of `RpcNoListener`. The bus raises it when the request is published.
- `services/orion-durable-runs/app/admitted_graph.py`: new `published_to_nobody()` and `listener_wait()`. `failed_attempt` calls `listener_wait` first.
- `AdmissionRuntime.register`: keeps a parked run waiting until `hold.retry_at`. `_hold_ready` ignores the pool's "released" hint while a run is parked.
- The 5 graphs that do not use `failed_attempt` (reading, compactor digest, journal compose, episode distill, orion_day) now call `listener_wait` themselves.
- `runner._run_turn`: for an admitted turn (one that holds a GPU lease), a "published to nobody" error is now raised, not returned as `ok=False`. Turns without a lease still return `ok=False`, as before.

## Files changed

- `orion/core/bus/async_service.py`: the `RpcPublishedToNobody` type.
- `orion/core/bus/tests/test_rpc_no_listener.py`: checks the type of both failure cases. This file is now wired into CI (`transport-baseline-tests.yml`); until now no workflow ran it.
- `services/orion-durable-runs/app/admitted_graph.py`: the wait policy.
- `services/orion-durable-runs/app/admission_runtime.py`: the park in `register` and the hint guard in `_hold_ready`.
- `services/orion-durable-runs/app/reading_graph.py`, `compactor_digest_graph.py`, `journal_compose_graph.py`, `episode_distill_graph.py`, `orion_day_graph.py`: the wait call in each work step's error handler.
- The state definitions of 9 graphs: a new `listener_waits` key. LangGraph silently drops keys that the state does not declare.
- `app/runner.py`: the raise for admitted turns.
- `app/settings.py`, `.env_example`, `docker-compose.yml`: 3 new keys.
- `README.md`: a new section for this behavior.
- `tests/test_listener_wait.py`: 29 tests.
- `tests/test_listener_wait_postgres.py`: 2 tests on the real runtime with real Postgres.
- `evals/deploy_window_listener_wait.py`: the deploy-window replay.
- `evals/listener_wait_mutations.py`: the mutation eval.
- `.github/workflows/orion-durable-runs-tests.yml`: runs both evals, and also triggers on `async_service.py`.

## Schema / bus / API changes

- Added: the `RpcPublishedToNobody` exception, a Python type with no change to the data sent on the bus. The `run.listener_wait` resource event. The `ResourceEventV1.event` field is free text, so no schema change is needed.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Admitted durable runs replay a step whose request was published to nobody. The replay does not use up an attempt.
  - For self-sense, an unheard question used to become an empty answer at once. Now the run waits and asks the question again. After the wait limit, the question goes through the normal per-question attempts before the run gives it up.
- Compatibility notes:
  - `RpcPublishedToNobody` is a subclass of `RpcNoListener`. Every existing `except RpcNoListener` still catches it.
  - Checkpoints written before this patch have no `listener_waits` key. The code reads a missing key as 0.

## Env/config changes

- Added keys (in `services/orion-durable-runs`), all on by default:
  - `DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES=6`. Set it to 0 to turn the wait off.
  - `DURABLE_RUNS_LISTENER_WAIT_BASE_SEC=60.0`
  - `DURABLE_RUNS_LISTENER_WAIT_MAX_SEC=300.0`
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The 3 keys are on lines 230-232 of the primary checkout's `services/orion-durable-runs/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs: pytest tests (with ORION_ADMISSION_TEST_DSN, a disposable postgres:16)
  -> 774 passed, 1 skipped
orion/core/bus/tests -> 131 passed
check_env_template_parity.py -> PASS
git diff --check -> clean
```

## Evals run

```text
evals/deploy_window_listener_wait.py -> PASS
  wait off: 16 failed / 0 completed (incident reproduced)
  wait on: 0 failed / 16 completed; attempt == 1 on every run
evals/listener_wait_mutations.py -> 20/20 mutants killed (only test failures count as kills)
evals/deploy_order_skew.py -> PASS (regression check of the neighbouring pool backoff)
```

## Docker/build/smoke checks

```text
Not run. Docker is not needed for this change, which adds no dependencies, ports or new
startup steps. The live behavior after deploy is UNVERIFIED. To check it after the next Hub
redeploy: count the run.listener_wait rows in durable_resource_events, and confirm that no
reading.turn run failed with a turn_exception:RpcNoListener error containing "published to nobody".
```

## Review findings fixed

- Finding: urgent runs ignored the urgent budget and could wait about 22 minutes.
  - Fix: urgent runs now wait at most 2 times, 10 s then 20 s.
  - Evidence: `test_urgent_runs_wait_within_the_urgent_budget`, and the "urgent budget ignored" mutant is killed.
- Finding: the error-chain walk followed an implicit `__context__`. A request that was delivered and then lost its listener could look replay-safe because of an earlier unheard request in its context.
  - Fix: the walk stops at a plain `RpcNoListener` and respects `__suppress_context__`.
  - Evidence: `test_a_delivered_request_chained_to_an_earlier_unheard_one_is_not_replay_safe`, and the matching mutant is killed.
- Finding: the counter declaration was guarded only for reading, and self-sense had no wiring test.
  - Fix: a parametrized test checks that all 9 graph states declare the counter, plus a self-sense replay scenario.
  - Evidence: the "self-sense state drops counter" mutant is killed.
- Finding (nit): the pool's own "released" hint woke a parked run once for nothing.
  - Fix: `_hold_ready` ignores hints while a run is parked.
- Finding (nit): the mutation eval counted a crash as a kill.
  - Fix: only pytest exit code 1 counts as a kill.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run this from the clean primary `main` checkout after merge. orion-hub and the other bus users only gain the new exception subclass, so they need no restart for this behavior.

## Risks / concerns

- Severity: low
  - Concern: some requests still fail during a deploy, because they are sent outside a work step's GPU seat. These are compactor `finalize`, the reverie step RPCs and the dream-carry step RPCs.
  - Mitigation: they already retry: finalize through the driver's bounded resume, and reverie and dream through their own backed-off deferral. Follow-up only if live data shows losses.
- Severity: low
  - Concern: when the listener disappears while the run waits for its reply, the run still uses up one attempt. This was 16 of the 26 reading failures over 14 days.
  - Mitigation: this is deliberate, because the worker may have done work. For readings, the queue's next attempt usually arrives during the same deploy and is then published to nobody, which now waits.
- Severity: low
  - Concern: all runs parked by one deploy ask the pool again at the same moment, because the wait has no jitter.
  - Mitigation: the pool's queue absorbs it.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
