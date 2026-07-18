## Summary

- Adds `substrate_attention_broadcast_log`, an append-only companion table to the singleton-upsert `substrate_attention_broadcast_projection` table (one row, overwritten every ~30s broadcast tick, no queryable history) -- closing the exact gap PR #1196's own replay script documented in its module docstring and Phase 1 report.
- New store writer `save_attention_broadcast_history()` (`services/orion-substrate-runtime/app/store.py`) appends one digest-keyed row per tick and prunes rows past a configurable retention window, mirroring the existing `save_coalition_dwell()`/`save_brain_frame()` pattern in the same file.
- Wires the write into `_attention_broadcast_tick()` (`services/orion-substrate-runtime/app/worker.py`) in its own fail-open `try/except`, alongside the existing `save_coalition_dwell()` call -- a history-log failure can never break the primary broadcast tick.
- `scripts/analysis/measure_ast_hot_reducer.py` now joins broadcast rows per-tick by nearest-preceding timestamp (same two-pointer pattern already used for `self_state`), instead of pinning one static snapshot to every replayed tick. New `fetch_broadcast_history_rows()` mirrors `fetch_self_state_rows()`.
- New setting `ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS` (default `168.0`, 7 days) added to settings, `.env_example`, and `docker-compose.yml`'s environment passthrough.
- `orion/sentience_striving_program/README.md`'s Phase 1 status note updated to say the structural gap is closed, while being explicit that the acceptance check will likely still report NOT MET immediately post-deploy (no backfill is possible; the log only accumulates forward from deploy time).

## Outcome moved

The AST/HOT reducer replay script (`scripts/analysis/measure_ast_hot_reducer.py`) can now perform a genuine historical search for `voluntary_override` events across the GWT-dispatch/broadcast lane, instead of being structurally limited to joining every replayed tick to the single most-recent snapshot. This does not itself flip the Phase 1 acceptance check to MET -- see "Risks / concerns" below -- but it removes the structural blocker the previous PR's own report named as the reason it couldn't be met.

## Current architecture

Before this patch, `substrate_attention_broadcast_projection` was the only persisted record of `AttentionBroadcastProjectionV1` -- a Postgres table with `PRIMARY KEY (projection_id)`, upserted every tick by `save_attention_broadcast()`. Any historical `voluntary_override` event that had occurred was overwritten in place by the time any replay/analysis script ran. `measure_ast_hot_reducer.py`'s `replay_reducer()` accepted a single `broadcast_row` and passed the same value to every call across the whole replay window.

## Architecture touched

- `services/orion-substrate-runtime` (app code, settings, docker-compose, tests)
- `services/orion-sql-db` (new manual migration file, this repo's hand-written-Postgres-migration convention -- no ORM/alembic here)
- `scripts/analysis/measure_ast_hot_reducer.py` and its tests (read-only analysis script, no writes/events/flags of its own)
- `orion/sentience_striving_program/README.md` (Phase 1 status note)

No bus channels, `orion/schemas/registry.py`, or `orion/schemas/attention_frame.py` were touched -- this is a pure Postgres storage seam, same as the `substrate_coalition_dwell_log`/`substrate_brain_frame_log` precedent it mirrors, neither of which is registered in the schema registry or bus channels either.

## Files changed

- `services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql`: new migration, `CREATE TABLE IF NOT EXISTS substrate_attention_broadcast_log` + index on `generated_at DESC`.
- `services/orion-substrate-runtime/app/settings.py`: new `attention_broadcast_log_retention_hours: float` setting (default 168.0).
- `services/orion-substrate-runtime/app/store.py`: new `save_attention_broadcast_history()` method, immediately after `save_attention_broadcast()`.
- `services/orion-substrate-runtime/app/worker.py`: `_attention_broadcast_tick()` now also calls `save_attention_broadcast_history()` in its own fail-open `try/except`.
- `services/orion-substrate-runtime/.env_example`: new `ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS=168.0` key with comment.
- `services/orion-substrate-runtime/docker-compose.yml`: new key added to the `environment:` passthrough list (found missing during `check_service_env_compose_parity.py` -- fixed in the same patch, not left as a gap).
- `services/orion-substrate-runtime/tests/test_store_observability_writers.py`: new tests for row shape/prune and digest idempotency.
- `services/orion-substrate-runtime/tests/test_worker_attention_broadcast_tick.py`: new tests asserting the history write happens once per tick and fails open.
- `scripts/analysis/measure_ast_hot_reducer.py`: `replay_reducer()` signature changed to accept `broadcast_rows: list[...]` and do a per-tick two-pointer join; new `fetch_broadcast_history_rows()`; `render_report()`/`run()` updated with new broadcast-coverage stats and a corrected MET/NOT-MET narrative; module docstring rewritten to describe the fix and the "no backfill possible" caveat; old `fetch_broadcast_row_count()`/`fetch_latest_broadcast_row()` kept as a secondary corroborating diagnostic on the singleton table, not removed.
- `scripts/analysis/tests/test_measure_ast_hot_reducer.py`: updated for the new 4-tuple `replay_reducer` return signature; new test proving the two-pointer join actually advances per tick (two broadcast rows straddling two field ticks join to their own distinct nearest-preceding row).
- `orion/sentience_striving_program/README.md`: Phase 1 status note updated with the "structural gap closed, acceptance check still likely NOT MET immediately post-deploy" framing.

## Schema / bus / API changes

- Added: `substrate_attention_broadcast_log` Postgres table (plain storage, not a registered schema or bus channel -- same as its sibling append-only log tables).
- Removed: none.
- Renamed: none.
- Behavior changed: none to the existing singleton table, its writer, `AttentionBroadcastProjectionV1`, or any of its existing consumers (`load_attention_broadcast`, `felt_state_reader.py`, orion-thought/reverie code) -- all untouched, verified via diff review.
- Compatibility notes: purely additive; the new table is a new INSERT target only, nothing reads it in production code yet except the offline analysis script.

## Env/config changes

- Added keys: `ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS` (default `168.0`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `services/orion-substrate-runtime/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran from repo root; this worktree has no `services/orion-substrate-runtime/.env` present at all (fresh worktree, no operator env files checked out), so the sync script reported `skip orion-substrate-runtime: no .env` for every service, not just this one. **Operator action needed**: on Juniper's real machine (where the live `.env` files exist), re-run `python scripts/sync_local_env_from_example.py` from repo root after pulling this branch to pick up the new key.
- skipped keys requiring operator action: none beyond the standing `PUBLISH_CORTEX_EXEC_GRAMMAR` skip-list entry (unrelated to this patch).

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest \
  services/orion-substrate-runtime/tests/test_store_observability_writers.py \
  services/orion-substrate-runtime/tests/test_worker_attention_broadcast_tick.py \
  scripts/analysis/tests/test_measure_ast_hot_reducer.py \
  orion/substrate/tests/test_attention_self_model.py -q
35 passed, 2 warnings (pre-existing pydantic protected-namespace warnings, unrelated)

PYTHONPATH=. .venv/bin/python -m pytest services/orion-substrate-runtime/tests -q \
  --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
16 failed, 134 passed
  (baseline on unmodified main, same command: 16 failed, 130 passed -- identical
  16 failures, confirmed pre-existing/environment-dependent, not caused by this
  patch; net +4 tests added, all pass)

python scripts/check_service_env_compose_parity.py orion-substrate-runtime
15 pre-existing keys missing (BRAIN_FRAME_*, EMBODIMENT_*, GRAMMAR_EVENT_CHANNEL,
  CHANNEL_GOAL_PROPOSAL -- all pre-existing, unrelated to this patch); the new
  ORION_ATTENTION_BROADCAST_LOG_RETENTION_HOURS key is NOT in this list (fixed
  in docker-compose.yml in the same patch)

python scripts/sync_local_env_from_example.py
  ran clean; no .env files present in this worktree to sync (see Env/config
  changes note above)
```

`test_grammar_consumer_integration.py` requires a live Postgres connection and fails identically on unmodified main (confirmed via `git stash`); excluded as pre-existing/environment-dependent, not a regression from this patch.

`scripts/check_env_template_parity.py`, `scripts/check_schema_registry.py`, and `scripts/check_bus_channels.py` named in root CLAUDE.md sec 17 do not exist in this repo as of this patch -- `scripts/check_service_env_compose_parity.py` is the actual live equivalent for env/compose parity and was run instead. No schema-registry or bus-channel change was made (plain Postgres table, same as the sibling log tables), so those checks would not have applied even if present.

## Evals run

No dedicated eval harness exists for `services/orion-substrate-runtime` or `scripts/analysis` beyond the unit tests above. The closest thing to an eval for this change is a live re-run of `scripts/analysis/measure_ast_hot_reducer.py --window-hours 48` against real Postgres after deploy -- documented as a required follow-up in "Restart required" below, not run in this session (no live Postgres/Docker access in this environment).

## Docker/build/smoke checks

Docker was not run in this session -- this sandboxed environment has no live Postgres or Docker daemon reachable for `orion-substrate-runtime`. Deterministic non-Docker checks (pytest, env/compose parity script) were run instead, per root CLAUDE.md sec 8's fallback instruction. `docker compose config` was not run for the same reason; the compose file's YAML structure was verified by direct diff review only (one line added inside the existing `environment:` list, consistent with sibling entries).

Attempted `scripts/safe_graphify_update.sh` after code changes per repo convention: it detected the known destructive incremental-update bug (2026-07-14 incident, node count would have dropped ~92%) and auto-refused, restoring `graphify-out/graph.json`/`manifest.json` to their pre-update state -- working as designed, nothing to commit from that step. Leftover `graphify-out/GRAPH_REPORT.md` diff and untracked `graph.html` from the refused run were manually reverted/removed to keep the branch clean.

## Review findings fixed

- Finding: (nit) `save_attention_broadcast_history()`'s digest hashes `generated_at` + `projection_id`, but `projection_id` is a constant default (`"substrate.attention.broadcast.v1"`) across all ticks, so it contributes no real independence to the idempotency key -- a future reader could mistakenly assume it does.
  - Fix: added an inline comment in `services/orion-substrate-runtime/app/store.py` explaining `projection_id` is effectively decorative today (kept for forward-compat), and that `generated_at` alone is sufficient because real ticks are always `attention_broadcast_interval_sec` apart (default 30s).
  - Evidence: commit `f0c3977a`; re-ran the full affected test suite after the fix (35 passed).

All 10 review constraints (singleton table/writer/schema untouched, sibling dwell/brain-frame tables untouched, fail-open write, idempotent digest, retention-pruning correctness + no SQL-injection risk given the trusted-config value, two-pointer join off-by-one/tie-break correctness, migration no-conflict, env parity across settings/.env_example/docker-compose, honest MET/NOT-MET reporting with compensating evidence preserved, and real (non-tautological) test coverage) were independently verified by a code-review subagent against the actual diff and code in context, not just the PR description. No blockers or should-fix items were found.

## Restart required

```bash
# 1. Apply the new migration against the live Postgres instance:
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql

# 2. Sync local .env with the new key (run on the machine with the real
#    services/orion-substrate-runtime/.env file):
python scripts/sync_local_env_from_example.py

# 3. Rebuild and restart orion-substrate-runtime so the new setting and code
#    path take effect:
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build

# 4. Confirm the tick is writing history rows (wait ~30-60s after restart,
#    then, e.g.):
psql "$POSTGRES_URI" -c "SELECT count(*), max(generated_at) FROM substrate_attention_broadcast_log;"

# 5. Re-run the replay script against real accumulated history once enough
#    time has passed (hours to days, not immediately after restart):
python scripts/analysis/measure_ast_hot_reducer.py --window-hours 48
```

## Risks / concerns

- Severity: low
- Concern: the acceptance check in `measure_ast_hot_reducer.py` (a real `voluntary_override` event found in the replay window) will almost certainly still report NOT MET if run immediately after this patch deploys -- `substrate_attention_broadcast_log` starts empty and can only accumulate forward from deploy time. The pre-patch singleton snapshots were overwritten in place and are genuinely unrecoverable; there is no backfill path.
- Mitigation: this is stated plainly in the script's own report output, its module docstring, and the `orion/sentience_striving_program/README.md` Phase 1 status note -- none of them claim MET prematurely. Re-run the script after several days of live 30s-cadence ticks (see "Restart required" step 5) to check whether the gap has actually closed with real accumulated data.

- Severity: low
- Concern: `docker compose config` / a live container smoke test were not run in this session (no Docker/Postgres access here).
- Mitigation: the compose diff is a single line inside an already-correct pattern (verified against 20+ sibling entries in the same file); risk of a YAML/syntax error is low, but Juniper should confirm `docker compose config` is clean before/during the restart steps above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/attention-broadcast-history-log
