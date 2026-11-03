# Retire the rung 6 prediction expectation tuner

## Summary

- Orion no longer tunes how surprised they expect to be. The 2026-10-16 tuner
  (PR #2715, spacing fix #2798) is removed as one unit: producer, assessor,
  trial hook, apply and rollback branch, control-surface accessors, Hub wiring,
  evals and tests.
- The substrate runtime stops writing `substrate_prediction_raw_reading` and
  stops reading the tuned setting. Execution, chat and street-handoff surprise
  use their fixed constants again. The table and its migration stay (no DROP).
- `prediction_expectation_patch` joins `RETIRED_MUTATION_CLASSES`, the same way
  `graph_consolidation_param_patch` was retired. The 16 stored proposals still
  load. Nothing can decide or apply a new one.
- The generic mutation pipeline stays, as the future effector of the
  self-monitor organ (design PR #2827). The Hub's scheduled cycle still reads
  signals and queues proposals for operator review, but now applies nothing.
- Seven env keys are removed, plus `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` from
  the runtime only.

## Outcome moved

A self-revision loop that could not carry its own evidence is gone, with
nothing left ticking behind it. Live record, 2026-10-16 to 2026-11-01
(`substrate_mutation_proposal` / `_decision` / `_adoption` in Postgres):

- 16 proposals (the brief said 9; 7 more arrived after it was written).
- 15 rejected: 12 `trial_failed`, 3 `trial_inconclusive`.
- **1 adopted, not 0:** execution `alpha` 0.2 -> 0.05 on 2026-10-26 05:21 UTC,
  settled, never rolled back. It was live until this patch. After the restart,
  execution surprise uses alpha 0.2 again. This is the intended effect of
  "go back to the constants", but it is a real behavior change, not a no-op.

The raw reading was mostly zeros (execution 14,238 rows, chat 1,917,
street_handoff 12, as of 2026-11-03), so the trial had little to work with.

## Current architecture

- `orion-substrate-runtime` wrote one raw reading per execution, chat and
  street-handoff tick to `substrate_prediction_raw_reading`, pruned it hourly,
  and read `prediction_expectation.<domain>` from the shared control surface
  (cached 60 s) to override the EWMA smoothing rate and variance floor.
- The Hub's scheduled mutation cycle ran `PredictionExpectationTuner`: it
  replayed the readings, proposed a candidate, trialed it on held-out readings,
  auto-promoted it (`auto_promote_allowlist={"prediction_expectation_patch"}`,
  `allowed_classes={"prediction_expectation_patch"}`), and monitored it after
  adoption. The manual cycle used the same tuner for trials and monitoring.
- `prediction_expectation_patch` was the only class any unattended path could
  apply.

## Architecture touched

- Substrate runtime: `app/worker.py`, `app/store.py`, `app/settings.py`,
  compose, `.env_example`, README.
- Shared library: `orion/substrate/prediction_error.py`,
  `street_attention.py`, `mutation_{apply,contracts,control_surface,decision,queue,trials,worker}.py`,
  `orion/core/schemas/substrate_mutation.py` (comment only).
- Hub: `scripts/api_routes.py` (scheduled + manual cycle, readiness summary),
  `scripts/autonomy_constitution.py`, compose, `.env_example`.
- CI: `.github/workflows/street-attention-tests.yml` no longer runs deleted
  tests.

**Scheduled cycle decision.** With every auto-promotable class retired,
`allowed_classes` became empty. The old applier read an empty set as "no
filter", so passing `set()` would have opened apply to every class. The
applier now reads `None` as "every class" and any explicit set (empty
included) as "only these". The scheduled cycle passes `set()`. I kept the rest
of the cycle: it is not dead wiring. It still turns review telemetry into
signals and queues operator-review proposals (recall, cognitive lanes). It
just cannot apply anything.

## Files changed

- `orion/substrate/prediction_expectation.py`: deleted (assessor, no consumer left).
- `orion/substrate/prediction_expectation_tuner.py`: deleted (Hub producer).
- `orion/substrate/prediction_error.py`: `expectation` parameter removed from
  `execution_prediction_error` / `chat_prediction_error`; constants only.
  `execution_raw_reading` / `chat_raw_reading` kept (used internally, and by
  `tests/test_chat_substrate_reducer.py`).
- `orion/substrate/street_attention.py`: comments; the constants are the expectation now.
- `orion/substrate/mutation_contracts.py`: class added to `RETIRED_MUTATION_CLASSES`;
  contract kept (like `routing_threshold_patch`) so stored rows still score/display.
- `orion/substrate/mutation_decision.py`: `auto_promote_allowlist` default is empty.
- `orion/substrate/mutation_apply.py`: apply, noop and rollback branches removed.
- `orion/substrate/mutation_control_surface.py`: `get_/set_prediction_expectation` removed.
- `orion/substrate/mutation_trials.py`: `prediction_expectation_evaluator` hook removed.
- `orion/substrate/mutation_worker.py`: corpus entries removed.
- `orion/substrate/mutation_queue.py`: `has_queued_proposal_for_surface` and
  `latest_proposal_at` removed (only the tuner called them).
- `orion/core/schemas/substrate_mutation.py`: literal kept, marked retired.
- `orion/substrate/scripts/smoke_mutation_v21.py`: carrier is now
  `field_topology_weight_patch` with a smoke-only engine that flips only the
  final "class not auto-promotable" step.
- `services/orion-substrate-runtime/app/{worker,store,settings}.py`: raw-reading
  write, hourly prune, control-surface read and cache removed; street handoff
  uses `STREET_HANDOFF_EWMA_ALPHA` / `STREET_HANDOFF_MIN_VARIANCE`.
- `services/orion-substrate-runtime/{.env_example,docker-compose.yml,README.md}`: keys and docs.
- `services/orion-hub/scripts/api_routes.py`: tuner, summary field, manual-cycle
  wiring and imports removed; applier empty-set semantics; readiness summary
  back to "nothing changes without your approval".
- `services/orion-hub/scripts/autonomy_constitution.py`: row is `retired`, apply forbidden.
- `services/orion-hub/{.env_example,docker-compose.yml}`: keys removed.
- `services/orion-sql-db/manual_migration_prediction_raw_reading_v1.sql`: header
  now `REQUIRED-BY: none` (no service needs it); DDL unchanged.
- Tests: deleted `orion/substrate/tests/test_prediction_expectation.py`,
  `services/orion-substrate-runtime/tests/test_worker_prediction_expectation.py`,
  `services/orion-substrate-runtime/evals/{run,test}_prediction_expectation_live_check.py`.
  Retargeted `test_mutation_v21.py`, `test_noop_mutation_guard.py`,
  `test_mutation_unpromoted_goal_regression.py`, `test_street_attention.py`,
  `test_worker_street_attention.py`, and three Hub tests.
- `docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`: "Retired 2026-11-03" under rung 6.

## Schema / bus / API changes

- Added: none.
- Removed: Hub scheduled-cycle summary field `prediction_expectation_proposals`
  and phase `prediction_expectation_tuner`.
- Renamed: none.
- Behavior changed: `prediction_expectation_patch` is retired (decide rejects
  with `mutation_class_retired`, apply returns None). Autonomy constitution
  `live_apply_surfaces` is now empty.
- Compatibility notes: `MutationClassV1` keeps the literal, so the 16 stored
  proposals parse. Proven by
  `test_retired_prediction_expectation_history_loads_but_never_applies`.
  The control-surface row `prediction_expectation.execution` (alpha 0.05) stays
  in Postgres; nothing reads it.

## Env/config changes

- Added keys: none.
- Removed keys:
  - orion-substrate-runtime: `SUBSTRATE_PREDICTION_RAW_READINGS_ENABLED`,
    `SUBSTRATE_PREDICTION_RAW_READING_RETENTION_HOURS`,
    `SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED`,
    `SUBSTRATE_PREDICTION_EXPECTATION_REFRESH_SEC`,
    `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` (runtime only; the parity gate now
    shows the runtime no longer reaches the control surface).
  - orion-hub: `SUBSTRATE_PREDICTION_EXPECTATION_TUNER_ENABLED`,
    `SUBSTRATE_PREDICTION_EXPECTATION_WINDOW_HOURS`,
    `SUBSTRATE_PREDICTION_EXPECTATION_INTERVAL_MIN`.
- Renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced: the sync script only adds keys, so the removed lines
  were deleted by hand from the primary checkout's
  `services/orion-substrate-runtime/.env` (5 lines) and
  `services/orion-hub/.env` (3 lines). Root `.env` had none.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests                         5 failed, 1156 passed, 49 skipped
  main baseline                               4 failed, 1177 passed, 49 skipped
  same 4 pre-existing felt_state failures; 5th (test_metacog_trend_signals::
  test_induction_query_returns_the_newest_row_per_node) is order-dependent:
  passes in isolation and in 2 of 3 full reruns on this branch
services/orion-substrate-runtime/tests+evals  13 failed, 560 passed, 1 error
  main baseline                               13 failed, 566 passed, 1 error
  identical failure set; -6 passed = deleted tuner tests
services/orion-hub/tests/test_substrate_mutation_*.py + review_runtime_hub_debug
  + autonomy_runtime_ui_panel                 97 passed (main 96)
services/orion-hub/tests (full)               4466 passed, 146 skipped, 0 failed
tests/test_chat_substrate_reducer.py          15 passed
scripts/check_env_template_parity.py          PASS (94 services)
scripts/check_control_surface_store_parity.py 3 services checked, all configured
scripts/check_migration_required_by.py        OK
scripts/check_sql_migrations_applied.py --file manual_migration_prediction_raw_reading_v1.sql  applied
```

## Evals run

```text
services/orion-substrate-runtime/evals (pytest)  run with the runtime tests above
The tuner's own live check (run_prediction_expectation_live_check.py) is deleted
with the tuner. No new eval: this patch removes a loop, it adds no behavior.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-substrate-runtime/docker-compose.yml config  ok, 0 retired keys
docker compose ... -f services/orion-hub/docker-compose.yml config                ok, 0 retired keys
orion/substrate/scripts/smoke_mutation_v21.py (via test_smoke_script_trace_and_invariants)  passed
Docker image build: not run (scripts/safe_docker_build.sh needs a worktree .env). UNVERIFIED.
```

## Review findings fixed

Review by an orion-repo-agent subagent on the branch diff vs origin/main.
It found no must-fix items. It confirmed: Hub boot cannot crash on the stored
proposals; the settled adoption is never acted on (the monitor skips anything
not `applied`); the migration header change does not affect the post-pull
applier (the table is already applied).

- Finding: the autonomy constitution said "9 proposals, 0 adoptions" but the live DB has 16 and 1.
  - Fix: comment and served `rationale` now say 16 proposals, 1 adoption (execution alpha 0.2 -> 0.05, settled), 15 rejected.
  - Evidence: `services/orion-hub/scripts/autonomy_constitution.py`; Hub debug tests pass.
- Finding: the docs cited the self-monitor organ spec path, which exists only in open PR #2827.
  - Fix: README and ladder doc now cite "design PR #2827, not yet merged". The ladder doc is also edited by #2827, so re-check it after whichever merges second.
  - Evidence: `services/orion-substrate-runtime/README.md`, ladder doc.
- Finding: the carrier engines bypass the real allowlist branch, so "no class auto-promotes" was not locked in.
  - Fix: `test_no_mutation_class_auto_promotes_with_the_default_policy` (every class in `CONTRACTS`, passing trial, default engine, none auto-promotes) and `test_allowlisted_class_still_auto_promotes_when_a_policy_opts_in`.
  - Evidence: `test_mutation_v21.py` 52 passed.
- Finding: no committed PR report.
  - Fix: this file.
- Finding (CI, after rebasing onto main with #3041/#3058): `test_route_engine_reuse.py` listed the deleted tuner's `postgres_reading_reader` as a known per-call engine, and the gate fails on stale entries.
  - Fix: entry removed.
  - Evidence: `test_route_engine_reuse.py` 16 passed.
- Nits fixed: the migration body is marked historical; an `api_routes.py` comment that pointed at "allowed_classes below" is corrected.
- Nits left: `STREET_HANDOFF_DOMAIN` is kept as a log label. The test and the smoke each keep their own carrier helper (small, and the smoke is a standalone script). Renaming or retiring `SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED` is left for the existing env-rename follow-up.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: execution surprise changes at restart. The one adopted setting
  (alpha 0.05) stops applying and alpha 0.2 returns, so execution prediction
  error reacts faster to change than it did since 2026-10-26.
- Mitigation: this is the decided return to the calibrated constants. Watch
  `node:substrate.execution` prediction_error for a day after deploy.

- Severity: low
- Concern: no class can auto-promote now, so the generic auto-promote/apply
  path is only exercised by tests through a carrier engine.
- Mitigation: intended. The self-monitor organ will bring a real effector class.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3050

🤖 Generated with [Claude Code](https://claude.com/claude-code)
