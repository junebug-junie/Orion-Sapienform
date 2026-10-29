## Summary

- A durable run can now run one step as one harness turn (the same `claude -p` motor chat and curiosity use) and keep the turn's final text, with the turn's correlation id, as the step's result.
- New reusable seam `services/orion-durable-runs/app/harness_step.py`: refusal, timeout, empty reply or a reply for another turn is a retryable step failure. A step never completes empty.
- New hand-triggered smoke workflow `harness.echo_smoke` (one step, agent lane, GPU pool hold) plus a trigger script.
- No new bus channel and no Hub change. The step reuses the existing durable-run -> Hub turn RPC (`orion:curiosity:turn:request`, `CuriosityTurnRequestV1`).
- This is "Patch 3" of the reminiscence design (PR #2888). `reminiscence.episode` hops will sit on this seam.

## Outcome moved

Before: each durable workflow that ran a harness turn handled the turn's outcome its own way. Nothing reusable gave "this step's text, or a retried failure". After: one helper plus a live smoke prove the path end to end. The smoke is hand-triggered and the live run is UNVERIFIED until Juniper deploys and triggers it.

## Current architecture

Existing-mechanism check: the spec (PR #2888) says "No durable run dispatches a harness turn today". **That is wrong.** Three durable workflows already run harness turns through Hub:

- `curiosity.investigate` and `self_sense_eval` use `Runner._run_turn` -> `orion:curiosity:turn:request` -> Hub `CuriosityInvestigation._turn_result_for` -> `_generate` -> `execute_unified_turn` -> harness governor.
- `reading.turn` uses its own reading-turn channel.

Verified live on 2026-10-29: `self_sense_eval` run `20261029T001121Z-cd2a05` completed. Its question turn `12c13174-b7d1-583b-b99a-63f11f158621` has `motor_boot`, `harness_dispatch` and `finalize` rows in `cockpit_turn_sighting`.

What was missing is a generic "step = harness turn" contract that fails loudly on empty. Curiosity reads its result back in later nodes. Self-sense records an empty answer as a valid (empty) measurement. `compactor.digest` makes plain cortex verb calls, not harness turns.

## Architecture touched

- orion-durable-runs: a new graph and seam module, wired into `AdmissionRuntime` (graphs, `WORK_NODES`, finish detail, `run.started` step id, `_cancel_harness`, `GET /runs/<id>`).
- Contract: an additive `DurableWorkflowV1` value `harness.echo_smoke`, plus a new brief `HarnessEchoSmokeBriefV1` in the `DurableRunRequestV1.brief` union.

## Files changed

- `orion/schemas/harness_step_run.py`: new. The smoke's workflow name, constants and brief. The brief has a required `smoke` literal, so it cannot be confused with another union member.
- `orion/schemas/durable_run.py`: adds the workflow literal, the brief union member, and the workflow↔brief and admission-required validation.
- `orion/schemas/registry.py`: registers `HarnessEchoSmokeBriefV1`.
- `services/orion-durable-runs/app/harness_step.py`: new. The seam: `run_harness_step`, `HarnessStepFailed`, `harness_step_correlation_id`, `step_detail`. The docstring records the Hub path's hard constraints.
- `services/orion-durable-runs/app/harness_smoke_graph.py`: new. `resource_request -> resource_wait -> harness_step -> finish`, with `retry_wait` backoff.
- `services/orion-durable-runs/app/admission_runtime.py`: wires in the new graph.
- `services/orion-durable-runs/scripts/harness_echo_smoke.py`: new. A stdlib trigger that submits, polls and prints PASS/FAIL plus the `cockpit_turn_sighting` query.
- `services/orion-durable-runs/tests/test_harness_smoke_graph.py`, `test_harness_smoke_postgres.py`: new.
- `services/orion-sql-writer/tests/test_durable_run_state_sql_shape.py`: consumer test for the new workflow value.
- `services/orion-durable-runs/README.md`: adds a "Harness steps" section.
- `.github/workflows/orion-durable-runs-tests.yml`: triggers on the new schema file.

## Schema / bus / API changes

- Added: `DurableWorkflowV1` value `harness.echo_smoke`; `HarnessEchoSmokeBriefV1` (`smoke`, `prompt` ≤2000 chars, `timeout_sec` 60..3600).
- Removed: none.
- Renamed: none.
- Behavior changed: `GET /runs/<id>` for this workflow adds `harness_steps`, `error`, `attempt`, `retry_at` and `harness_turn_meta`.
- Compatibility notes: the change is additive on `extra="forbid"`/`Literal` models. Deploy orion-sql-writer, orion-hub, orion-actions and orion-cortex-orch before triggering the first smoke, or an old validator drops the smoke's state rows. No existing producer sends the new value.

## Env/config changes

- Added keys: none. No feature flag: the workflow only runs when someone triggers it by hand.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-durable-runs && PYTHONPATH=<worktree> ORION_ADMISSION_TEST_DSN=<disposable postgres:16> pytest tests -q
  677 passed, 1 skipped   (includes 21 new tests: 17 graph/contract tests, 4 against real Postgres + in-process GPU pool)
services/orion-sql-writer: pytest tests/test_durable_run_state_sql_shape.py -q    6 passed
services/orion-cortex-orch: pytest tests/test_durable_run_dispatch.py tests/test_compactor_durable_submit.py -q   28 passed
pytest tests/test_no_personal_terms.py tests/test_agent_trace_schema_registry.py orion/schemas/tests -q   81 passed
scripts/check_scripts_dir_no_stdlib_shadow.py, scripts/check_env_key_single_source.py: clean
Mutation checks: disabling the empty-text guard fails 5 tests; removing the recovery cancel fails the Postgres recovery test.
```

## Evals run

```text
No offline eval: a deterministic "echo" has no quality dimension to score. The runtime eval is the
hand-triggered live smoke below (not run: no deploy from a worktree). Status: UNVERIFIED live.
```

## Docker/build/smoke checks

```text
Not run: no compose/Dockerfile/requirements change. Live smoke, after Juniper deploys from main:
  python3 services/orion-durable-runs/scripts/harness_echo_smoke.py
Exit 0 PASS, 1 FAIL, 2 submit refused or unreachable, 3 still running. It prints the turn's
correlation id and the cockpit_turn_sighting query.
```

## Review findings fixed

- Finding: the seam cannot opt out of Hub's lookup gate (`MIN_HARNESS_STEPS`), so a pure writing step would fail as `no_lookup` on every attempt.
  - Fix: recorded as a hard constraint in the `harness_step.py` docstring and the README, with the follow-up named (an optional `require_lookup` on `CuriosityTurnRequestV1`, consumer-first). Not built here: it is a Hub contract change and Patch 4's first prerequisite.
  - Evidence: `app/harness_step.py` module docstring; README "Harness steps".
- Finding: the recovery test was circular (both sides of the assertion came from the same dict).
  - Fix: added a real Postgres test. A driver dies mid-turn; a fresh driver runs `_recover`. The test asserts that the dead turn's actual id was cancelled and that the replay got a new id. Also added graph tests for the take-back and deadline paths.
  - Evidence: `test_a_worker_that_dies_mid_step_has_its_turn_cancelled_by_the_checkpoint_derived_id` passes, and fails when the cancel branch is removed.
- Finding: a small `timeout_sec` gave Hub a 1-second turn.
  - Fix: brief `timeout_sec` is now `ge=60`.
- Finding: in the trigger script, connection errors shared exit code 1 with FAIL, and one failed poll killed the watcher.
  - Fix: an unreachable submit now exits 2. Poll errors are logged and polling continues.
  - Evidence: `--url http://127.0.0.1:1` exits 2.
- Finding: two README statements were wrong (where the lookup gate lives; Hub's `_run_lock`).
  - Fix: reworded.
- Finding: one claude session per `(run_id, source_tag)`.
  - Fix: documented. Steps that need separate sessions need distinct tags.
- Not fixed (UNVERIFIED): the smoke prompt assumes Bash is available to the motor in this turn context. The reviewer traced that it should be, but the first live run decides.

## Restart required

orion-durable-runs is the service that runs the new workflow. The state consumers need the new schema before the first smoke. The post-pull rebuild from main covers them. If you deploy by hand, run one line per service, consumers first:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low. Concern: the smoke uses a real agent-lane GPU hold and one harness turn (minutes of GPU). Mitigation: hand-triggered only, background priority, 2 h deadline.
- Severity: medium (for Patch 4, not this patch). Concern: the lookup gate blocks pure writing steps. Mitigation: documented as a hard constraint; the fix is named.
- Severity: low. Concern: a smoke triggered before the consumers are deployed loses its state rows in sql-writer. Mitigation: deploy order above; the run itself still completes and `GET /runs/<id>` shows the result.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2958

🤖 Generated with [Claude Code](https://claude.com/claude-code)
