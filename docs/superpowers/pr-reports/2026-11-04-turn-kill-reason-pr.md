## Summary

- A killed harness turn now says why it stopped. The turn record (`HarnessRunV1`, stored as `harness_turn_trace.run_artifact`) gets one new field, `stop_reason`. It holds `cancel:<reason>` when a harness cancel killed the turn. Otherwise it holds the motor's own error code (`fcc_timeout`, `fcc_stream_stalled`, `fcc_nonzero_exit`, ...). Otherwise it is null.
- The governor's cancel listener now passes the cancel's reason to the motor (`cancel_fcc_turn(cid, reason)`). The motor keeps the reason until the turn ends, then puts it on the `-9` exit frame.
- orion-prediction marks a turn's self-bets void `interrupted` when a known outside cancel stopped it (Juniper's stop, a client disconnect, a websocket error, durable-runs stopping the attempt, worker recovery). It no longer marks them `failed_to_execute`. Timeouts, stalls, context-ceiling kills, a nonzero exit with no cancel, empty text, and unknown or absent reasons all stay `failed_to_execute`.
- The self-monitor leaves `interrupted` turns out of `turn.run`. They count as neither failed nor upheld.
- durable-runs used to send `durable_attempt_stopped` in every case. Now, when the attempt runs out of its own time budget, it sends `durable_attempt_timeout`, which stays a failure. A lost or recalled hold, run control, and worker shutdown still send `durable_attempt_stopped`.

## Outcome moved

In the last 7 days, 262 of 2,115 turns read as "Orion didn't run" (`turn.run` failed). Live: 139 of them had `exit_code=-9` and no reason. About 87 were GPU-pool recalls and about 40 were chat stops. After deploy, those turns read `interrupted` and drop out of the `turn.run` failure count. This is the same mistake as the false "broken" alarm fixed in #3078: an outside interruption was counted as Orion failing.

## Current architecture

- `orion/harness/fcc_motor.py::cancel_fcc_turn(correlation_id)` SIGKILLed the process and dropped the reason.
- `runner.py` kept only `exit_code=-9`.
- `orion-prediction/app/turn_self_bets.py::observe()` voided every non-zero exit as `failed_to_execute`.
- `orion/self_monitor/collect.py` counted a turn as failed when all of its rows were `failed_to_execute`.

## Architecture touched

- Motor (`fcc_motor.py`): a bounded per-correlation_id cancel-reason map (max 512 entries). The turn takes its entry in its `finally`. The `fcc_nonzero_exit` frame carries `metadata.cancel_reason`.
- New seam: `orion/harness/stop_reason.py`. It builds the string (`turn_stop_reason`) and holds the one allowlist of outside cancels (`is_outside_stop`).
- Runner: `HarnessMotorResult.stop_reason`. Governor: copies it to `HarnessRunV1.stop_reason` on all 6 motor-backed record paths.
- Cancel listener: passes the reason only when the sender named one. An unnamed cancel would otherwise take the schema default `client_disconnect` and read as an outside stop.
- Prediction `observe()`, self-monitor `turn.run` SQL, and the `turn_self_bet` adapter docstring.
- durable-runs `admission_runtime.execute`: a timeout gets its own cancel reason.

## Files changed

- `orion/harness/stop_reason.py`: new. Builds the value and holds the outside-cancel allowlist.
- `orion/harness/fcc_motor.py`: `cancel_fcc_turn(reason=)`, the reason map, and `cancel_reason` on the exit frame.
- `orion/harness/runner.py`: `stop_reason` on `HarnessMotorResult`.
- `orion/schemas/harness_finalize.py`: `HarnessRunV1.stop_reason` (optional, default None).
- `services/orion-harness-governor/app/bus_listener.py`: copies `stop_reason` onto the record.
- `services/orion-harness-governor/app/cancel_listener.py`: passes the named reason.
- `services/orion-harness-governor/README.md`: a "Why a turn stopped" section.
- `services/orion-prediction/app/turn_self_bets.py`: the `interrupted` void reason and docstring list.
- `orion/self_monitor/collect.py`, `orion/self_monitor/adapters.py`: drop `interrupted` from `turn.run`.
- `services/orion-durable-runs/app/admission_runtime.py`: `durable_attempt_timeout` on a time-budget stop.
- Tests: `orion/harness/tests/test_fcc_motor_cancel.py`, `test_fcc_warm_pool.py`, `test_stop_reason.py`; governor `test_harness_run_cancel.py`, `test_interrupted_resumable.py`; prediction `test_turn_self_bets.py`; `orion/self_monitor/tests/test_self_monitor_record.py`; durable-runs `test_urgent_preemption.py`.

## Schema / bus / API changes

- Added: `HarnessRunV1.stop_reason: str | None = None`. Added a new `prediction_turn_self_bet.void_reason` value, `interrupted`.
- Removed: none.
- Renamed: none.
- Behavior changed: the `harness.run.cancel.v1` reason now reaches the record. durable-runs sends `durable_attempt_timeout` on a time-budget stop.
- Compatibility notes:
  - `HarnessRunV1` is a plain `BaseModel` (`extra` is ignore, not forbid), and sql-writer stores the raw payload. Old consumers are therefore safe, and no consumer-first deploy is needed.
  - `void_reason` is plain `text`. Checked live: the only constraint on `prediction_turn_self_bet` is the primary key, so no migration is needed.
  - **No backfill.** Old rows have no `stop_reason` and stay `failed_to_execute`. Absent means unknown, and unknown never loosens.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/harness/tests                                  708 passed (before the review round); warm/cancel/stop_reason 31 passed after
orion/self_monitor/tests                             126 passed
services/orion-harness-governor tests                101 passed
services/orion-prediction tests/test_turn_self_bets  50 passed
services/orion-durable-runs tests                    697 passed, 123 skipped
services/orion-prediction full suite: 20 failures in test_backtest/test_causal/test_events/test_heat_balance.
  These fail the same way on clean main with the local venv. They are unrelated.
Static gates (orion-static-gates.yml): definition_drift, metric_lineage (--gate, --prompt-semantics),
  inner_state_registry, sentience_instruments --static-only, env_key_single_source, body_economy_contract,
  scripts_dir_no_stdlib_shadow, async_routes_not_blocking: all PASS.
  tests/test_no_personal_terms.py, tests/test_agent_trace_schema_registry.py: pass.
```

## Evals run

```text
No new eval. The seam is a settle rule, so it is covered by gate tests.
Live read-only check: the new turn.run SQL runs on the live Postgres (7 days: 1853 ran, 262 failed;
none interrupted yet, because no record carries stop_reason before deploy).
```

## Docker/build/smoke checks

```text
Not run. No compose/Dockerfile/requirements change. The live path is UNVERIFIED until the deploy.
Check after deploy: SELECT run_artifact->>'stop_reason', count(*) FROM harness_turn_trace
  WHERE created_at > <deploy> AND (run_artifact->>'exit_code')::int < 0 GROUP BY 1;
then SELECT void_reason, count(DISTINCT correlation_id) FROM prediction_turn_self_bet
  WHERE settled_at > <deploy> GROUP BY 1;
```

## Review findings fixed

- Finding (material): durable-runs sent `durable_attempt_stopped` even when the attempt's own time budget ran out, so a slow turn would read as `interrupted`.
  - Fix: the `except TimeoutError` path sets `durable_attempt_timeout`. That reason is not in the outside allowlist.
  - Evidence: `test_an_attempt_that_ran_out_of_its_own_time_cancels_with_a_timeout_reason`, plus an assertion on the preempt test. The prediction test checks that `cancel:durable_attempt_timeout` stays `failed_to_execute`.
- Finding (material): a cancel with no reason took the schema default `client_disconnect` and would read as an outside stop.
  - Fix: the cancel listener passes the reason only when the sender set it (`model_fields_set`). The motor records an unnamed cancel as `unspecified`, which stays a failure.
  - Evidence: `test_a_cancel_that_names_no_reason_is_not_read_as_a_disconnect`.
- Finding: the warm-pool retry change had no test.
  - Fix: on inspection, a cancel's kill already sets `WarmTurn.killed`, so the warm path never retries a cancelled turn as a spawn. The extra guard was redundant, so I removed it. A test now pins the real behavior: no respawn, and `cancel_reason` is carried. That test fails on main.
  - Evidence: `test_a_cancel_during_warm_acquire_is_honored_not_retried_as_a_spawn`.
- Finding: the "every reason sent today" comment missed `outreach_agent_lane_timeout`.
  - Fix: it is listed as deliberately excluded, together with `durable_attempt_timeout` and `unspecified`.
- Not fixed (nits):
  - `manual_migration_prediction_turn_self_bet_v1.sql`'s header comment does not list `interrupted`. I left it alone so that a comment edit does not re-trigger the post-pull migration applier.
  - `_PENDING_CANCEL` is unbounded. This predates this PR.
  - A cancel that arrives between a held final and its resume spawn is lost. This also predates this PR.
  - Two new tests (adapter returns None on `interrupted`; a finished turn with a cancel reason is still graded) also pass on main. They are pins, not regressions.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The self-monitor host runs in orion-durable-runs (`app/main.py`), so the durable-runs rebuild above also covers the `turn.run` change.

## Risks / concerns

- Severity: low. Concern: `durable_attempt_stopped` still covers run control (operator cancel) and worker shutdown as well as GPU recalls. All three are outside stops, so this is intended.
- Severity: low. Concern: a new cancel producer with a new reason stays `failed_to_execute` until someone adds the reason to `OUTSIDE_CANCEL_REASONS`. That is the safe direction.
- Severity: low. Concern: about 50 of the old unknown kills will stay unexplained. There is no backfill.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3105
