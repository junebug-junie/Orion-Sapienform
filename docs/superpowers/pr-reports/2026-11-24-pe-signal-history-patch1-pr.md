## Summary

Patch 1 of the design in PR #3781 (`docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md`). Additive only: Orion's behavior does not change.

- New table `substrate_node_signal_history`. It holds the five nodes that write into the prediction-error slot but are not forecast misses. Each row has an honest `kind`: `alarm` (cabinet, strain, street) or `scene_change` (room camera, walkway camera).
- orion-substrate-runtime now writes those five nodes' readings to the new table too. The old prediction-error (PE) history writes do not change. Strain's direct event row is copied as well. The new table is pruned at the same 7-day retention.
- Read-both helpers (both tables, one row per node and time, new table wins) are ready for the later reader switch. No live reader uses them yet.
- A one-off backfill script copies the last 7 days of the five nodes from the PE table. Dry run by default. Not run against production.
- Flag `SUBSTRATE_SIGNAL_HISTORY_ENABLED` ships ON. The write fails open: without the table it logs and the tick goes on.

Juniper approved on 2026-11-24 (recorded on #3781): cameras stay an attention source (renamed "camera scene change"), street rhythm surprise later, old rows age out with no DELETE, and Patch 1 may land before #3732 and #3576.

## Outcome moved

The five non-PE signals get a home with an honest name. This is the step every later patch needs: readers can switch to the new table (with a full 7 days after the backfill) and the writers can then stop writing into the prediction-error slot. Nothing Orion reads or says changes in this patch.

What does NOT change: the PE table rows, the FalkorDB `prediction_error` key on the five nodes, the attention contests, strain news, self-readings, the node-history dig tool, the glossary, labels. The camera score is the same old value (the rescoring is a later patch). No new metric: the values are copies, so the metric gate in the design doc covers them.

## Current architecture

`services/orion-substrate-runtime/app/worker.py` writes all five nodes through `_write_prediction_error_node`. The attention-broadcast tick samples every `node:substrate.*` node into `substrate_node_prediction_error_history` when its `observed_at` moves (`_prediction_error_magnitudes`). Strain news also writes each event row directly so the restart seed can rebuild its budget. Every reader reads the PE table.

## Architecture touched

- Postgres: new table + index (additive migration, auto-applied by the post-pull rebuild).
- orion-substrate-runtime: two one-line dual-write calls and one prune block, plus one small helper `_dual_write_signal_history`. All in the existing PE-history paths.
- Shared module `orion/substrate/signal_history.py`: the node-to-kind map, the sample filter, the read-both merge and SQL.

## Files changed

- `services/orion-sql-db/manual_migration_node_signal_history_v1.sql`: new table, `REQUIRED-BY: orion-substrate-runtime`, additive DDL only.
- `orion/substrate/signal_history.py`: kinds, five ids, `signal_samples`, `merge_signal_rows`, insert/prune/read-both SQL.
- `services/orion-substrate-runtime/app/store.py`: `save_signal_history_samples`, `prune_signal_history`, `fetch_signal_history_read_both`, `count_signal_history_read_both`, `fetch_signal_event_times_read_both`.
- `services/orion-substrate-runtime/app/worker.py`: `_dual_write_signal_history`; called after the sampler's PE write and after strain's event row; prune beside the PE prune.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag, documented.
- `scripts/backfill_signal_history_from_pe.py`: one-off 7-day backfill, section 14 protocol.
- `config/substrate/prediction_error_readers.yaml`: registers the backfill as `not_a_reader` (CI `check_pe_alarm_readers.py`).
- Tests: `orion/substrate/tests/test_signal_history.py`, `services/orion-substrate-runtime/tests/test_store_signal_history.py`, `services/orion-substrate-runtime/tests/test_worker_signal_history_dual_write.py`, `tests/test_backfill_signal_history_from_pe.py`.

## Schema / bus / API changes

- Added: table `substrate_node_signal_history (node_id text, observed_at timestamptz, value double precision, kind text check in ('alarm','scene_change'), recorded_at timestamptz default now(), primary key (node_id, observed_at))`, index on `observed_at`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for readers. substrate-runtime writes extra rows.
- Compatibility notes: no bus or schema-registry change. No GRANT: the postgres role's default privileges already give `orion_readonly` SELECT on new tables (checked live, read-only: `pg_default_acl` = `{orion_readonly=r/postgres}`).

## Env/config changes

- Added keys: `SUBSTRATE_SIGNAL_HISTORY_ENABLED=true` (orion-substrate-runtime).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes, `/mnt/scripts/Orion-Sapienform/services/orion-substrate-runtime/.env` has the key.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-substrate-runtime python -m pytest -p no:cacheprovider -q \
  services/orion-substrate-runtime/tests orion/substrate/tests/test_signal_history.py \
  tests/test_backfill_signal_history_from_pe.py \
  --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
-> 609 passed, 5 skipped, 17 failed. All 17 failures reproduce on unmodified main in the same
   local env (cursor/quarantine/reducer-health/self-model/falkor tests); none touch this patch.
   test_grammar_consumer_integration.py needs a local Postgres on :5432 (pre-existing).
Focused suite after review fixes (new tests + test_worker_pe_history + test_strain_news_tick + test_store_pe_history-adjacent + test_pe_alarm_refile): 58 passed.
pytest tests/test_deploy_migration_gate.py tests/test_sql_migration_drift_gate.py -> 142 passed
python scripts/check_migration_required_by.py -> OK (1 new migration file)
python scripts/check_pe_alarm_readers.py -> OK (66 files, all accounted for)
python scripts/check_env_template_parity.py -> PASS
python scripts/check_integration_seam.py -> OK
git diff --check -> clean
git merge-tree vs feat/pe-retire-biometrics and feat/pe-retire-execution -> no conflicts
```

## Evals run

```text
No eval: Patch 1 adds no metric and changes no behavior. The design's acceptance proof for this
patch ("new-table rows match PE-table rows for the five ids over the same window") is built into
the backfill report (before_after.csv, verdict line).

Real-SQL check on a throwaway postgres:16-alpine container (NOT the live DB), 2026-11-24:
- migration applied twice (second run: "already exists, skipping"); CHECK rejects kind 'bogus'.
- 12,096 seeded PE rows (6 nodes x 2,016 over 7 days).
- backfill dry run: "DRY RUN: 10080 row(s) would be inserted; nothing written"; table count 0.
- backfill --apply: "OK: signal table matches the PE table for all five ids over the window",
  2,016 rows per node, 0 value mismatches; alarm 6,048 / scene_change 4,032.
- backfill --apply again: 0 targets, still OK (idempotent).
- store read-both: 5 ids only (execution excluded); a key in both tables returned the
  signal-table value; a PE-only key returned too; save twice on the same key = one row.
```

## Docker/build/smoke checks

```text
Live migration classification (read-only):
python3 scripts/apply_post_pull_migrations.py --dry-run --repo . services/orion-sql-db/manual_migration_node_signal_history_v1.sql
-> NOT APPLIED ... would apply (single transaction, lock_timeout=10s)
No image build run: no new dependency; the service imports only in-repo modules.
Live dual-write: UNVERIFIED until deploy.
```

## Review findings fixed

Code review ran in a subagent. No must-fix findings.

- Finding: if the PE write succeeded but the signal write failed, that reading was lost for good, because the sampler had already moved on.
  - Fix: a failed signal write is kept (newest 500 rows, about an hour) and sent again with the next write (`_signal_history_pending`, `_SIGNAL_HISTORY_PENDING_MAX`).
  - Evidence: `test_failed_signal_write_is_retried_with_the_next_write`, `test_pending_retry_buffer_is_capped`.
- Finding: the backfill's "matched" check compared row counts, so extra signal rows could hide missing keys.
  - Fix: a per-node `missing_keys` count (PE keys with no signal row) is now required to be 0. It is shown in `report.md` and `before_after.csv`.
  - Evidence: `test_extra_signal_rows_do_not_mask_missing_keys`; on a scratch Postgres, `missing_keys_after` = 0 for all five ids.
- Finding: the usage lines said `python3`, but host python3 has no sqlalchemy/psycopg2. The module docstring said "stdlib only", but the package `__init__` imports pydantic.
  - Fix: the usage now names the repo venv (`.venv/bin/python`). The docstring now says what the import really loads.
- Finding: the import and filter in the dual-write helper were outside the `try`.
  - Fix: both are now inside the `try`, so the whole helper fails open.
- Finding: the anomaly comment said NaN, null and future rows were snapshotted. They are only counted.
  - Fix: the comment now says they are counted.
- Finding: the read-both SQL could return other nodes' PE rows if a caller passed other ids. The Python merge drops them.
  - Fix: the store helper now keeps only the five signal ids.
- Not changed (nit): while the table is missing, every tick with new samples logs one warning. The post-pull rebuild applies the migration before the rebuild, so this window is short.
- Not changed (nit): the unit tests use a fake connection. The real SQL (DISTINCT ON, UNION ALL, ANY list binding, the missing-keys join) was run on a throwaway Postgres; results are under "Evals run".

## Restart required

The post-pull rebuild on the primary main checkout applies the migration first (auto-apply: additive DDL, valid REQUIRED-BY header), then rebuilds orion-substrate-runtime. No manual restart.

After that deploy, run the backfill (dry run first, then apply):

```bash
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/backfill_signal_history_from_pe.py
cd /mnt/scripts/Orion-Sapienform && .venv/bin/python scripts/backfill_signal_history_from_pe.py --apply
```

It uses `POSTGRES_URI` if set, else `postgresql://postgres:postgres@localhost:55432/conjourney`. Files land in `/tmp/signal-history-backfill/` (`progress.log`, `report.md`, `before_after.csv`, `snapshot-*.jsonl`). Expect about 15k target rows (5 nodes x ~3,080 rows per 7 days).

Live check after deploy:

```sql
select node_id, kind, count(*), max(observed_at) from substrate_node_signal_history group by 1, 2 order by 1;
```

## Risks / concerns

- Severity: low. Concern: a signal-table write can fail while the PE write succeeds (for example, the table is missing for a few minutes after deploy). That reading is not retried, so the new table can miss a few rows. Mitigation: the backfill is idempotent; re-run it before Patch 2 switches any reader. Its report shows any gap per node.
- Severity: low. Concern: the backfill must run AFTER the migration and AFTER the Patch 1 image is live, or the window between backfill and dual-write start is missing. Mitigation: order above; a re-run fills any gap.
- Severity: low. Concern: when the camera score is rescored later, the read-both union would mix old-scale PE rows and new-scale signal rows for the cameras. Mitigation: that patch must either backfill rescored values or read only the new table for cameras; noted for the camera patch.
- Issues opened: none. Issues closed: none. Refs #3781.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3800

🤖 Generated with [Claude Code](https://claude.com/claude-code)
