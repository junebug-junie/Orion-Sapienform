# PR: Rung 6 rebuilt — Orion tunes how surprised they expect to be

## Summary

- Rung 6 of the substrate ladder (self-revision) was marked DONE but did nothing. Its module was deleted with `self_state` on 2026-07-22. What still ran was a loop on its own noise: ~65,000 signals a day from ~30 real graph reviews, one hardcoded patch, 930 of 930 trials inconclusive, 0 adoptions since 2026-09-03.
- Orion now revises their own surprise settings. Those are the smoothing rate (`alpha`) and variance floor (`min_variance`) that turn each raw execution or chat reading into a prediction error.
- The evidence is real raw readings. The substrate runtime records each one, and replaying them under another setting is an exact counterfactual.
- Changes are governed. A candidate is picked on older data and must win on newer data it never saw. Orion applies it. Later readings are replayed under both settings, and the change rolls back if the new one is confidently worse.
- `graph_consolidation_param_patch` and the `world_ontology` review signal are retired outright.

## Outcome moved

The rung-6 loop can now finish with a real verdict (adopt, reject, or roll back) based on evidence. Before, every trial was "inconclusive" by construction. Prediction error, the signal attention, reverie and curiosity all read, gets calibration Orion maintains themself, not hand-picked constants.

## Current architecture (before)

- `orion/substrate/prediction_error.py`: execution and chat surprise used hardcoded `alpha=0.2`, with floors of 1e-10 and 3e-5 ("no domain-specific reason found yet to diverge").
- Hub mutation scheduler (every 30 s):
  - It read the newest review telemetry with no cursor, so each review became a signal about 2,000 times.
  - "A review executed" counted as pressure on `graph_consolidation`.
  - The proposal was always the same constant patch.
  - The trial needed reviews that resolve, and 0 did in 10 days.
  - Any adoption only staged a profile that nobody activates.

## Architecture touched

- **`orion/substrate/prediction_expectation.py` (new, pure):**
  - Replay matching `orion/bus/ewma.py`.
  - Gaussian log score (strictly proper, so going numb loses).
  - Assess on the older 2/3 of the window, holdout on the newer 1/3.
  - Paired gains with batch-means standard errors.
  - 30-reading warm-up; minimum effect 0.01 nats per reading.
- **`orion/substrate/prediction_expectation_tuner.py` (new):**
  - A Hub producer that feeds the existing governed pipeline: trial, decision, apply, monitor, rollback or settle.
  - Proposals for one domain are at least 24 h apart.
  - The post-adoption delta is the upper confidence bound of (adopted − previous).
- **Mutation pipeline:**
  - New class `prediction_expectation_patch` with its contract (low risk, auto-promote). The trial needs a holdout lower bound > 0 and a mean gain ≥ 0.01.
  - Apply/rollback write the control surface `prediction_expectation.<domain>`.
  - Strict reads: apply refuses when the live value can't be read.
  - A failed trial read reads as inconclusive and doesn't crash the cycle.
  - `add_proposal(due_at=)`.
  - Per-domain rollback window: execution 6 h, chat 24 h.
- **Hub `api_routes.py`:**
  - The scheduled and manual cycles both trial and monitor the new class.
  - Review-telemetry ids are deduped, and marked seen only after a cycle consumes them.
  - The constitution and readiness summary state that one surface now self-applies.
- **`orion-substrate-runtime`:**
  - Records raw readings.
  - Reads the adopted expectation, cached for 60 s.
  - Rebuilds a fallen-back store.
  - Keeps the last good value on a read failure.
  - Has a kill switch.

## Files changed

- `orion/substrate/prediction_expectation.py`, `prediction_expectation_tuner.py`: the rung-6 core and producer.
- `orion/substrate/prediction_error.py`: `expectation=` kwarg; `execution_raw_reading` / `chat_raw_reading` split out.
- `orion/substrate/mutation_{apply,contracts,control_surface,decision,detectors,proposals,queue,scoring,trials,worker}.py`: new class wired in; graph_consolidation retired (evaluator, staging branch, defaults and maps deleted).
- `orion/core/schemas/substrate_mutation.py`: class literal.
- `services/orion-hub/scripts/api_routes.py`, `autonomy_constitution.py`: tuner wiring, dedupe, honest constitution.
- `services/orion-substrate-runtime/app/{worker,store,settings}.py`: raw-reading writes, expectation reads.
- `services/orion-sql-db/manual_migration_prediction_raw_reading_v1.sql`: additive table + index (passes the post-pull auto-apply allowlist).
- `.env_example` / `docker-compose.yml` for hub and substrate-runtime.
- Tests (see below), `orion/substrate/scripts/smoke_mutation_v21.py`, the eval, the ladder doc, and the runtime README.

## Schema / bus / API changes

- **Added:** mutation class `prediction_expectation_patch`; table `substrate_prediction_raw_reading`; control-surface keys `prediction_expectation.execution` / `.chat`; `RuntimeControlSurfaceStore.read()` (strict) with `ControlSurfaceReadError`.
- **Removed:** the graph_consolidation evaluator, staging branch and proposal defaults; `PatchApplier.policy_store`.
- **Behavior changed:**
  - `graph_consolidation_param_patch` is retired, so it is rejected and never applied.
  - `world_ontology` telemetry produces no signal.
  - The readiness page and constitution say one surface self-applies.
- **No bus channel changes.**

## Env/config changes

- **Added keys, hub:** `SUBSTRATE_PREDICTION_EXPECTATION_TUNER_ENABLED=true`, `..._WINDOW_HOURS=72`, `..._INTERVAL_MIN=60`.
- **Added keys, substrate-runtime:** `SUBSTRATE_PREDICTION_RAW_READINGS_ENABLED=true`, `SUBSTRATE_PREDICTION_RAW_READING_RETENTION_HOURS=192`, `SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED=true`, `SUBSTRATE_PREDICTION_EXPECTATION_REFRESH_SEC=60`, `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL` (required by `check_control_surface_store_parity.py`).
- **Local `.env` synced:** yes, with `sync_local_env_from_example.py --all-keys orion-substrate-runtime orion-hub`. All 8 keys were verified present in the primary checkout's `.env`.
- **Skipped keys:** none.

## Tests run

```text
orion/substrate/tests                        1138 passed, 3 failed (test_felt_state_self_definition_lane.py — also fails on main)
  incl. test_prediction_expectation.py       16 passed: replay == live EWMA, numbness loses, holdout passes/rejects,
                                              full loop adopt -> live value moves -> world shifts -> rollback,
                                              apply refuses on read failure, failed trial read = inconclusive,
                                              tiny gain fails, 24 h proposal spacing, retired class rejected
orion-hub tests -k mutation|autonomy|substrate|constitution|readiness
                                             278 passed, 8 failed (identical 8 fail on main)
  incl. scheduled cycle adopts a self-tuned expectation end to end; telemetry marked seen only after consumption
orion-substrate-runtime tests + evals        472 passed, 14 failed + 9 errors (identical on main)
  incl. test_worker_prediction_expectation.py (records reading, uses adopted value, cache, kill switch, keeps last value on read failure)
```

## Evals run

```text
services/orion-substrate-runtime/evals/test_prediction_expectation_live_check.py   3 passed
Synthetic behaviour (batch-means SE): steady data -> slower alpha, holdout lb +0.071; drifting data (2500) -> holdout lb -0.13 (rejected);
adopted-then-drift -> post-adoption upper bound -0.33 (rolls back); numb floor -> -4.65 nats/reading (loses).
Live eval (run after deploy): POSTGRES_URI=... python services/orion-substrate-runtime/evals/run_prediction_expectation_live_check.py
```

## Docker/build/smoke checks

```text
CI gates: control_surface_store_parity, definition_drift, env_key_single_source, metric_lineage (+prompt-semantics),
migration_required_by, settings_defaults hub, sentience_instruments, async_routes, scripts_dir_no_stdlib_shadow,
system_health_producers — all exit 0.
apply_post_pull_migrations.py --dry-run: "would apply (single transaction, lock_timeout=10s)".
smoke_mutation_v21.py: ok=true, real apply on an isolated control surface.
Docker build not run in this session. Live path UNVERIFIED until deployed.
```

## Review findings fixed

- **Finding: confidence bounds too narrow.** Neighbouring per-reading differences share EWMA memory, so the naive SE overstated confidence. Retries and flip-flop made it worse.
  - Fix: batch-means SE (blocks of 2/alpha readings, at least 8 blocks), minimum effect 0.01 nats per reading, proposals at least 24 h apart.
  - Evidence: `test_a_confident_but_tiny_gain_does_not_pass_the_trial`, `test_proposals_for_a_domain_are_spaced_a_day_apart`, the synthetic table above.
- **Finding: the manual cycle rejected rung-6 proposals and settled adoptions unmeasured.**
  - Fix: the manual cycle now wires the tuner's evaluator, corpus and post-adoption deltas.
- **Finding: a failed read became "live = default", so a rollback could restore the wrong value.**
  - Fix: strict `read()`. Apply and the tuner refuse or skip when the live value can't be read.
  - Evidence: `test_apply_refuses_when_the_live_value_cannot_be_read`.
- **Finding: the runtime could pin the constants silently for its whole life.**
  - Fix: rebuild the store when it isn't Postgres, warn, and keep the last good value on a read failure.
  - Evidence: `test_a_failed_read_keeps_the_last_value_read_not_the_defaults`.
- **Finding: one failed trial read killed every cycle.**
  - Fix: the evaluator is guarded and returns inconclusive.
  - Evidence: `test_a_failed_trial_read_is_inconclusive_not_a_crash`.
- **Finding: the constitution and readiness page said nothing self-applies.**
  - Fix: new constitution row and invariants, retired row, corrected summary; constitution test updated.
- **Finding: runtime tests could reach the live DB.**
  - Fix: `is True` checks; stub store in the tests.
- **Finding: chat is never measured after adoption.**
  - Fix: per-domain window (chat 24 h); proposal and monitoring in separate guards. Chat is still likely to settle unmeasured (see risks).
- **Finding: telemetry was marked seen before consumption.**
  - Fix: marked only after `run_cycle` returns without lock or kill-switch notes.
- **Finding: the retired class was only partly killed.**
  - Fix: the evaluator and its test, the staging and rollback branches, the defaults, the corpus entries, the zone map and the factory entry are all deleted. Readiness still shows the historical graph-consolidation activity (read-only history).
- **Finding: no connect timeout on the tuner's reader.**
  - Fix: `connect_timeout=5`, `statement_timeout=15s`.
- **Finding: post-adoption check re-read the whole window (efficiency).**
  - Fix: it now reads from adoption time minus 6 h, not the full 72 h window.

## Restart required

```bash
# From the clean primary main checkout after merge (post-pull applies the additive migration first):
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium.**
  - Concern: this is the first live, unattended self-modification of a cognition signal. Changing alpha shifts the distribution of execution/chat prediction error that attention, the reverie percentiles and curiosity read, and their 7-day histories will mix regimes.
  - Mitigation: bounded grid, held-out trial, automatic rollback, at most one proposal per domain per day, kill switch `SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED=false`.
- **Severity: medium.**
  - Concern: the post-adoption check compares settings by replay. It cannot see whether the runtime actually used the adopted value. The runtime now rebuilds a fallen-back store and warns, but which expectation each reading was scored with is not recorded.
  - Mitigation: live eval plus `prediction_expectation_store_not_postgres` warnings in the logs.
- **Severity: low.**
  - Concern: chat is sparse. It may never reach enough readings to propose, and an adoption may settle unmeasured after 24 h.
  - Mitigation: none beyond the window. The live eval reports chat's reading count.
- **Severity: low.**
  - Concern: whether any setting beats `alpha=0.2` on live execution data is UNVERIFIED. The loop may correctly conclude "keep it". That is a real verdict, not a failure.
- **Severity: low.**
  - Concern: the Gaussian log score is very sensitive to spikes. On heavy-tailed real execution readings it may favour larger variance floors (honest calibration, but lower sensitivity to small deviations).
  - Mitigation: floor changes are only proposed when they score better; ties keep the live floor.

## PR link

<link>
