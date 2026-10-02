## Summary

Step 1 ("producer and schema") of the approved proposal in PR #2474
(`docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md`, approved by Juniper 2026-10-02, 24 h p50/p90 fields included).

- The substrate runtime can now keep a history of every `node:substrate.*` prediction-error reading. It adds a row only when a node's reading actually moved (its `observed_at` advanced), and prunes rows older than 7 days.
- A new pure function turns that history into a size report for each node: the current value and its age, the 7-day and 24-hour usual range (p50/p90), how the current reading ranks against the last 7 days, whether it has been rising, settling, or flat, and a band (quiet / usual / high / unusual).
- The report rides on each broadcast loop as a new optional field, `OpenLoopV1.magnitude`. It is descriptive only. Ranking never reads it, and a test pins that the winner is the same with or without it.
- Everything is behind `SUBSTRATE_PE_HISTORY_ENABLED`, which defaults to off. The migration is written but **not applied**.
- Reverie's prompt, `reverie.py`, and broadcast ranking are untouched. Those are steps 2 and 3.

## Outcome moved

Nothing changes at runtime until the flag is turned on. Once it is, every broadcast tick stores a persisted trace (`substrate_attention_broadcast_log.projection_json -> frame.open_loops[].magnitude`) that says how big the winning error was against its own past. That trace is the input step 2 needs so reverie can call a normal reading normal. A new eval command is the gate before step 2.

## Current architecture

`_attention_broadcast_tick` (substrate-runtime) snapshots the FalkorDB graph and builds `OpenLoopV1` loops from `dynamic_pressure`. Each loop carries only a label and a fixed "novel or unresolved" string. Size, usual range, and direction exist nowhere for 4 of the 9 winning domains, and the 5 that do have a baseline use a mean/SD EWMA, which is the wrong summary for zero-inflated data (see the spec).

## Architecture touched

- `orion-substrate-runtime`: broadcast tick (history writer, in-memory 7-day window, hourly prune), store methods, settings/env.
- Shared schema: `PredictionErrorMagnitudeV1`, plus the additive `OpenLoopV1.magnitude` (consumer-first, see Restart required).
- `orion/substrate/attention_broadcast.py`: an optional `magnitude_by_node_id` argument, attached after loop building.
- New Postgres table `substrate_node_prediction_error_history` (manual migration).

## Files changed

- `orion/schemas/attention_frame.py`: `PredictionErrorMagnitudeV1` and the optional `OpenLoopV1.magnitude`.
- `orion/schemas/registry.py`: registers `PredictionErrorMagnitudeV1` (verified via `resolve()` in a test).
- `orion/substrate/prediction_error_magnitude.py` (new): pure history-to-magnitude function. Does not import `compute_prediction_error_trend`.
- `orion/substrate/attention_broadcast.py`: attaches magnitude to loops by `source_refs` node id when supplied.
- `services/orion-substrate-runtime/app/worker.py`: `_prediction_error_magnitudes()`, called from `_attention_broadcast_tick` only when the flag is on. Fail-open.
- `services/orion-substrate-runtime/app/store.py`: save (idempotent on PK), fetch, and prune for the history table.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new keys and a rollout section.
- `services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql` (new, not applied).
- `scripts/sync_local_env_from_example.py`: adds `SUBSTRATE_PE_HISTORY_` to `SYNC_PREFIXES`, so default syncs on other hosts pick the keys up.
- `services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py` and its test (new): the 24 h live sanity gate to run before step 2.
- Tests: `orion/substrate/tests/test_prediction_error_magnitude.py`, `services/orion-substrate-runtime/tests/test_worker_pe_history.py`, `services/orion-substrate-runtime/tests/test_store_pe_history.py`.

## Schema / bus / API changes

- Added: `PredictionErrorMagnitudeV1` (`schema_version="prediction_error.magnitude.v1"`, plus `value, age_sec, p50_7d, p90_7d, p50_24h, p90_24h, percentile_now, n_readings_7d, median_1h, median_prior_24h, trend, band`). `OpenLoopV1.magnitude: PredictionErrorMagnitudeV1 | None = None`.
- Removed / renamed: none.
- Behavior changed: none while the flag is off.
- Compatibility notes: **consumer-first.** `OpenLoopV1` is `extra="forbid"`, and an old consumer rejects a payload carrying `magnitude`. Old payloads without the field still validate (tested). No bus channel changes. The projection crosses the bus only nested inside `HubAssociationBundleV1` on `orion:thought:request`.
- Range/percentile fields are `None` when there is no data. `band`/`trend` read `insufficient_history` below 200 readings, and `trend` also reads `insufficient_history` when the last hour or the prior 24 h has no readings (for example, a stale node).

## Env/config changes

- Added keys (orion-substrate-runtime): `SUBSTRATE_PE_HISTORY_ENABLED=false`, `SUBSTRATE_PE_HISTORY_RETENTION_HOURS=168`, `ORION_REVERIE_PE_TREND_MIN_DELTA=0.01`.
- Removed / renamed keys: none.
- `.env_example` updated: yes, `services/orion-substrate-runtime/.env_example`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The default run added only `ORION_REVERIE_PE_TREND_MIN_DELTA` and skipped the two `SUBSTRATE_PE_HISTORY_*` keys, which matched no prefix. A dry-run showed `--all-keys orion-substrate-runtime` would add exactly those two keys and nothing else, so I ran it. All three are now present in `/mnt/scripts/Orion-Sapienform/services/orion-substrate-runtime/.env`. The prefix is now in `SYNC_PREFIXES`.
- Skipped keys requiring operator action: none.
- `ORION_REVERIE_PE_MAGNITUDE_ENABLED` (orion-thought) is step 2 and is not added here.

## Tests run

```text
# new tests
pytest orion/substrate/tests/test_prediction_error_magnitude.py            -> 20 passed
(services/orion-substrate-runtime) pytest tests/test_worker_pe_history.py tests/test_store_pe_history.py \
    tests/test_worker_attention_broadcast_tick.py evals/test_pe_magnitude_live_sanity.py -> 28 passed
# ranking test mutation-checked: making an 'unusual' magnitude raise salience fails it (1 failed), reverted.

# regression
(services/orion-substrate-runtime) pytest tests --ignore=tests/test_grammar_consumer_integration.py
    -> 385 passed, 13 failed, 9 errors; failure set IDENTICAL to main (f67a40b3b) baseline
       (pre-existing: cursor_reset_auth, quarantine_truth, etc.; grammar_consumer_integration fails to collect on main too)
pytest orion/substrate/tests -> 881 passed, 3 failed (test_felt_state_self_definition_lane.py, also failing on main)
consumer-side: orion/thought, orion/reverie, orion/hub association, tests/test_top_down.py, voluntary attention,
    attention schema surface, system one, inner_state_registry gate, measure_ast_hot_reducer -> 120 passed
(services/orion-thought) pytest tests -k "reverie or mind_light or broadcast" -> 174 passed
(services/orion-hub) test_attention_loops_api + test_attention_card_legibility -> 10 passed

# gates
check_definition_drift.py --gate PASS; check_metric_lineage.py --gate PASS; check_env_template_parity.py PASS;
check_sentience_instruments.py --static-only OK; check_inner_state_registry.py OK; check_env_key_single_source.py OK;
check_compose_no_relative_mounts.py PASS; check_service_hostname_refs.py OK
check_service_env_compose_parity.py orion-substrate-runtime: rc=1 on main baseline too, none of the new keys flagged
resolve("PredictionErrorMagnitudeV1") is PredictionErrorMagnitudeV1 (asserted in test)
compute cost: 11 nodes x 17,280 readings (full 7 d at tick cadence) = 0.18 s per tick, inside asyncio.to_thread
```

## Evals run

```text
pytest services/orion-substrate-runtime/evals/test_pe_magnitude_live_sanity.py -> 3 passed (evaluator flags flat,
    sparse and missing domains on synthetic data)
Live run of evals/run_pe_magnitude_live_sanity.py: NOT RUN -- the table does not exist yet (migration unapplied).
It is the 24 h gate before step 2 (see Restart required step 5). It also prints the per-domain band distribution
(review nit 8: zero-inflated domains put any nonzero reading at >= ~0.75 percentile, i.e. 'usual'/'high' by construction --
check that before step 2's calm gate relies on band).
The spec's reverie calm-fixture eval (acceptance check 2) belongs to step 2.
```

## Docker/build/smoke checks

```text
Not run. Per task instructions: no deploy, no container restarts, no migration applied.
Live path UNVERIFIED until the deploy steps below are run.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...feat/pe-magnitude-history`. It found no blockers.

- Finding (should-fix): the "magnitude never changes ranking" test used one node, so it could not fail.
  - Fix: added a test with 3 competitors, where the lowest-pressure loop gets an `unusual` magnitude. It asserts loop order, salience, candidate actions, and the winner are identical.
  - Evidence: a mutation that lets `unusual` raise salience fails the new test.
- Finding (should-fix): a failed seed read (for example, flag on before the migration) re-ran a 7-day read and logged a full traceback every tick.
  - Fix: retry after 5 minutes. The first failure logs a traceback and later ones log one line.
  - Evidence: `test_seed_failure_backs_off_instead_of_retrying_every_tick`.
- Finding (should-fix): the README rollout list did not say why it differs from the spec's consumer list, which also names attention-runtime.
  - Fix: the README and this report state that attention-, proposal-, and feedback-runtime read the projection as raw SQL/JSON and were checked.
  - Evidence: parse sites listed under Restart required.
- Finding (nit): a producer clock running ahead could pin the last-seen `observed_at` in the future and silently drop correct readings afterwards.
  - Fix: readings more than 5 minutes in the future are skipped with a warning.
  - Evidence: `test_far_future_reading_is_not_recorded_and_does_not_pin_last_observed`.
- Finding (nit): retention had no lower bound, so 0 would delete rows just written.
  - Fix: `ge=25` (the 24 h and prior-24 h windows need 25 h). Documented in `.env_example`.
  - Evidence: `test_retention_below_25_hours_is_rejected`.
- Not changed (nits, accepted):
  - Per-tick recompute cost: 0.18-0.21 s off the event loop.
  - Single-transaction batch insert: values are finite and timezone-aware before insert.
  - Seed sort: the PK `(node_id, observed_at)` already serves `ORDER BY node_id, observed_at`.

## Restart required

Run in this order. Each step is one line.

1. Apply the migration (Juniper):
```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql
```
2. Rebuild the consumers that pydantic-validate the broadcast, **before** the producer. Thought goes first, because hub forwards the projection to thought over `orion:thought:request`:
```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```
3. Rebuild the producer with the flag still off, then confirm the broadcast still parses in thought (look for no `attention broadcast read failed` lines):
```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```
4. Flip the flag: set `SUBSTRATE_PE_HISTORY_ENABLED=true` in `services/orion-substrate-runtime/.env`, then:
```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d
```
5. After 24 h, and before step 2:
```bash
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney python services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py
```

Services that parse the broadcast (from a repo search, confirmed at the parse sites):
- **orion-thought**: `app/broadcast_reader.py:85` (`AttentionBroadcastProjectionV1.model_validate` on the projection table; on failure reverie silently reads "no broadcast", which looks like calm) and `app/bus_listener.py` (`StanceReactRequestV1`, which nests the broadcast). **Must rebuild.**
- **orion-hub**: `orion/hub/association.py:42` (`_parse_broadcast`). **Must rebuild.**
- **orion-substrate-runtime**: validates its own stored projection (`store.load_attention_broadcast`). Rebuilt in step 3.
- Safe (raw JSON / SQL jsonb, no rebuild needed): orion-attention-runtime, orion-feedback-runtime, orion-proposal-runtime, orion-cortex-exec, hub observability routes.

## Risks / concerns

- Severity: medium. Concern: consumer-first ordering. If the flag is flipped before thought and hub are rebuilt, reverie goes silent, which looks like calm. Mitigation: the flag defaults off, the order above, and the README section.
- Severity: low. Concern: all 9 domains are UNVERIFIED on live data. The table does not exist yet, so nothing confirms per-domain coverage or that trend labels aren't degenerate. Mitigation: the step-5 eval gates step 2.
- Severity: low. Concern: band cut points (0.5/0.9/0.99) and the 200-reading minimum are function parameters, not env keys. Spec env list honored. Mitigation: promote them to env keys if the 24 h check suggests retuning.

## PR link

REVIEW_PLACEHOLDER_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
