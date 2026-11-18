# Card a merged SQL migration that never reached the live database

## Summary

- The 10-minute substrate ladder watch now also checks that every hand-applied SQL migration merged in the last 30 days actually exists in live Postgres. If a table, column, index or sequence is missing, it raises a Hub card that names the migration file and gives the exact command to apply it.
- New pure module `orion/sql_migration_drift.py`:
  - A real SQL tokenizer that handles comments, quoted strings and identifiers, dollar-quoted `DO`/function bodies, and psql `\set` lines.
  - A replay of the whole migration history in the order files landed on main. A later `DROP` therefore explains an earlier `CREATE` instead of raising a false alarm.
  - A diff of that expected schema against the database's own catalog.
- Data-only migrations (UPDATE/INSERT with no schema objects, such as the #2400 chat-projection reset) are listed as **verify manually**. They are never reported as applied or as missing.
- `scripts/check_sql_migrations_applied.py`, the existing by-hand checker, becomes a thin CLI over the same module. The old regex version had three problems:
  - It saw only the first column of a multi-column `ALTER TABLE`.
  - Against live data today it raised four false alarms, all for objects a later migration dropped on purpose.
  - It never ran from any schedule.
- The drift-gate tests were not in CI before. They now run in the static-gates job.

## Outcome moved

Failure mode: a merged migration is not applied at deploy, and nothing notices until a service breaks.
- PR #2424: the `hardware_watch_incident` table was missing, and orion-hardware-watch crash-looped 13 times.
- PR #2400: the `last_value_observed_at` column was missing, and attention silently ran degraded.

Both incidents now replay as RED in tests, with a card naming the file. Live detection lag is at most 10 minutes after the merged code is in the deploy checkout.

## Current architecture

- Migrations in `services/orion-sql-db/manual_migration_*.sql` are applied by hand. There is no migration ledger table and no apply wrapper (checked: no `schema_migrations` or ledger anywhere in the repo).
- `scripts/check_sql_migrations_applied.py` existed as an on-demand regex checker (`make check-sql-migrations-applied`). Nothing scheduled it.
- `scripts/check_substrate_ladder_liveness.py` runs every 10 minutes from host cron in the primary checkout. It has two sections, rung freshness and schema skew. It sends one debounced Hub Pending Attention card per new red key through orion-notify, with the exit-4 escalation contract from PR #2392.

## Architecture touched

- **Placement:** the migration check is a third section of the existing ladder watch, not a sibling script. The watch already owns the read-only Postgres session, the debounce state, the orion-notify path and the exit-4 contract, and a sibling would duplicate all of them.
- **Report model:** `LadderReport` gains `migrations: Optional[DriftReport]`, which feeds `red`, `red_keys`, `green_keys`, `alert_message` and `to_dict`.
- **Debounce keys:** one key per migration file, in the form `migration:<file>`.
- **Window and sticky keys:** a file is red when its objects are missing and either:
  - it landed or changed on main in the last 30 days, or
  - it was already carded ("sticky keys" read from the watch's own state file).

  A carded file therefore stays red until it is applied. It does not age out silently with its card muted.

## Files changed

- `orion/sql_migration_drift.py`: new. Tokenizer, parser, replay, evaluation, git/catalog IO, alert text.
- `orion/substrate_ladder_liveness.py`: `LadderReport.migrations` and its wiring.
- `scripts/check_substrate_ladder_liveness.py`: migrations section on the same connection, sticky keys from the state file, human output, `--skip-migrations` and `--migration-days`.
- `scripts/check_sql_migrations_applied.py`: thin CLI over the module, plus `--since-days` (0 = all time) and `--ref`.
- `tests/test_sql_migration_drift_gate.py`: rewritten. Incident replays on the real corpus, parser tests on every real file, review regressions, and a merge-time git fixture.
- `tests/scripts/test_substrate_ladder_liveness.py`: wiring tests (card text, debounce/re-arm, cannot-check, sticky keys, JSON/human output).
- `.github/workflows/orion-static-gates.yml`: runs the drift-gate tests.
- `Makefile`: comments; `check-sql-migrations-applied` now uses `$(METRIC_PYTHON)`, because system python3 has no psycopg2. Adds `DAYS=`.

## Schema / bus / API changes

- Added: none (no bus channel, no schema registry entry, no DB object).
- Removed: none.
- Renamed: none.
- Behavior changed: `make substrate-ladder-watch` can now go RED (exit 1) and card for a missing migration object. The `--json` output gains a `migrations` key.
- Compatibility notes:
  - The in-file markers `ORION-MIGRATION-SUPERSEDED-BY` and `ORION-MIGRATION-NOT-A-MIGRATION` are unchanged.
  - New marker: `ORION-MIGRATION-ABSENT-OK: <object> <reason>`. The reason is required. It is the allowlist for objects that are intentionally absent for reasons outside the migrations (for example, dropped out of band). Drops made by a later migration need no marker.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## How it decides

1. Each migration file is split into statements. Comments, string literals and quoted identifiers can never create an object.
2. Each statement becomes zero or more effects: create or drop of a table, column, index or sequence. `RENAME` counts as a drop plus a create.
   - `ALTER TABLE t ADD a, ADD b` yields both columns.
   - Anything inside a `DO $$ ... $$` block, or under `ALTER TABLE IF EXISTS`, is *conditional*. It is reported, never alarmed on, and it can never overturn an unconditional expectation.
3. All files are replayed in the order they landed on main (`git log --first-parent`: the PR merge time, not the branch commit time). The result is the schema the repo expects right now.
4. That expected schema is compared with `pg_class`/`pg_attribute`/`pg_index`. A difference is blamed on the file that last set the object's expected state.
   - An invalid index (an interrupted `CREATE INDEX CONCURRENTLY`) counts as a failure.
   - A `DROP` that never ran counts as a failure.
5. A file with only UPDATE/INSERT/DELETE statements is DATA ("verify manually").

Known limits, stated in the module docstring:
- Columns declared inline in a restated `CREATE TABLE IF NOT EXISTS` are not tracked individually.
- `EXECUTE '<string>'` is not parsed.
- Views and functions are not tracked (none exist in the corpus).

## Live findings (2026-10-01, read-only)

The run used `check_sql_migrations_applied.py --since-days 0`, against the live catalog, over all 120 files and all time:

- **No merged migration is unapplied.** 111 applied, 2 superseded, 3 data-only, 2 settings-only, 2 not-a-migration (pg_dump files).
- The old checker's four "missing" findings were all false alarms. The new checker resolves each one from history:
  - `manual_migration_gateway_capacity_v1.sql`, `manual_migration_gpu2_elastic_v1.sql`: superseded. Everything they created was dropped by `manual_migration_gpu_pool_stage5_drop_legacy_tables.sql`.
  - `manual_migration_durable_resource_admission_v1.sql`: applied. Its legacy tables were dropped by stage 5.6, and `durable_admission_runs`/`durable_resource_events` are live.
  - `manual_migration_action_outcome_ledger.sql`: applied. Its old unique index was swapped by `manual_migration_action_value_control_arm.sql`.
- Since PR #2400 and PR #2424, both incident objects have been applied (they are present live).
- **Verify manually** (the schema cannot show whether these ran):
  - `manual_migration_chat_projection_pe_baseline_v3_reset.sql` (PR #2400, in the 30-day window).
  - Older: `manual_migration_receipt_retention_v2_aggressive.sql`, `manual_migration_reverie_theme_key_bare_id_backfill.sql`.
- Live RED path proven against the real database with a temporary probe migration declaring a non-existent table and column. It was deleted afterwards and never committed.
  - Watch (`check_substrate_ladder_liveness.py --skip-docker`, before the review fix that makes the watch skip uncommitted files): printed `RED migration manual_migration_zz_live_red_probe.sql: not applied to the live database -- column hardware_watch_incident.zz_probe_col, table zz_live_red_probe` plus `apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_zz_live_red_probe.sql`, then `RED`.
  - CLI after all fixes (`check_sql_migrations_applied.py --quiet`, which still includes uncommitted files): `RED [missing] manual_migration_zz_live_red_probe.sql`, with both objects listed, exit 1.

## Proposal (not built): a migration ledger for data migrations

The schema can never prove that an UPDATE ran. The honest fix is a tiny ledger:
- a table `orion_migration_ledger(file text primary key, sha256 text, applied_at timestamptz, applied_by text)`, and
- a wrapper `scripts/apply_sql_migration.sh <file>` that runs `psql -v ON_ERROR_STOP=1` and inserts the row in the same transaction.

This check would then report DATA files as applied or unapplied by ledger row, and flag a file whose sha changed after apply. It is not trivial and not consistent with current practice: it only works if every apply goes through the wrapper, which changes deploy habit and every runbook that pastes `docker exec -i ... psql < file`. Juniper's call.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest tests/test_sql_migration_drift_gate.py \
  tests/scripts/test_substrate_ladder_liveness.py tests/scripts/test_schema_skew_discovery.py -q
143 passed

Mutation checks (each must fail at least one test):
  multi-action ALTER first-only, DO-block not conditional, drop replay disabled, table-drop cascade
  disabled, E-string escapes, sticky red, conditional cascade, conditional override, first-parent,
  ALTER INDEX/SEQUENCE rename, glued AS$$/DO$$ -> all caught.

Static gates run locally (all exit 0): git diff --check, check_metric_lineage --gate,
check_definition_drift --gate, check_inner_state_registry, check_scripts_dir_no_stdlib_shadow,
check_service_hostname_refs, check_sentience_instruments --static-only,
check_system_health_producers, check_control_surface_store_parity, check_async_routes_not_blocking
```

## Evals run

```text
No eval harness applies: this is a deterministic gate, not a quality measure. The live run over
the real 120-file corpus against live Postgres (above) serves as the end-to-end check.
```

## Docker/build/smoke checks

```text
No container changes. Live smokes (read-only, from the worktree):
make substrate-ladder-check            -> GREEN, migrations: 31 file(s) in window, all present (~20s total, migrations ~0.3s)
make check-sql-migrations-applied-quiet -> exit 0, 1 verify-manually file in window
check_substrate_ladder_liveness.py --skip-docker with probe migration -> RED naming the file + apply command
```

## Review findings fixed

The review ran in a subagent against `origin/main...804c8cb85`. It found no blockers; five should-fix findings and five nits.

- Finding: a red file went quiet after 30 days, and its debounce key was never released.
  - Fix: the watch passes its delivered `migration:*` keys as sticky, so the file stays red until applied. Green now requires proof the file was applied. Out-of-window, never-carded drift prints `warn old migration`.
  - Evidence: `test_a_carded_file_stays_red_after_the_window_until_applied`, `test_the_watch_keeps_a_carded_migration_red_past_the_window`; mutation caught.
- Finding: the window and replay order used branch commit time, which on the real corpus is up to 10 days before the merge.
  - Fix: `git log --first-parent --diff-merges=first-parent`.
  - Evidence: `test_commit_times_use_the_merge_time_not_the_branch_commit_time` (a temp repo with a real merge); mutation caught.
- Finding: a conditional `DROP TABLE` in a DO block cascaded and hid a missing column.
  - Fix: conditional effects never override an unconditional expectation and never cascade.
  - Evidence: three regression tests; both mutations caught.
- Finding: an unparsed `ALTER INDEX/SEQUENCE ... RENAME TO` was a permanent false RED.
  - Fix: parsed as a drop plus a create.
  - Evidence: `test_alter_index_and_sequence_rename_are_tracked`.
- Finding: the real-corpus tests replayed by file name and derived the live state from the same parser.
  - Fix: added a real first-parent-order corpus replay (skips on shallow history; CI fetches full history), and hand-written expected objects for the incident files and the stage-5.6 drop.
  - Evidence: `TestRealCommitOrder`, `TestIncidentFilesDeclareWhatAHumanReadThemToDeclare`.
- Nits fixed:
  - `ALTER TABLE IF EXISTS` is now conditional.
  - `AS$$`/`DO$$` with no space is recognised.
  - Catalog reads use `pg_catalog` instead of privilege-filtered `information_schema`.
  - The watch ignores never-committed files.
  - The inline-`CREATE TABLE`-column limit is documented in the module docstring.
- Not changed: file contents are read from the working tree while history comes from HEAD. That is consistent for the cron (the primary checkout on main is what is deployed). A migration merged to origin/main but not yet pulled is checked once it is pulled, which is also when it can be deployed.

## Restart required

```text
No restart required. The cron entry already runs `make substrate-ladder-watch` from the primary
checkout every 10 minutes; once this merges and the primary checkout is pulled, the next tick
includes the migrations section. No container rebuild.
```

## Risks / concerns

- Severity: low
- Concern: a future migration in a shape the parser does not recognise could produce a false RED card. Examples are a rename via `EXECUTE`, or a drop done by a script rather than a migration file.
- Mitigation: the card names the file and the objects. `ORION-MIGRATION-ABSENT-OK: <object> <reason>` in the file silences exactly that object, with a reason on record. `--skip-migrations` turns the section off.

- Severity: low
- Concern: a data-only migration (like the #2400 reset) is still only listed, never verified.
- Mitigation: listed as "verify manually" in the watch log every tick. The ledger proposal above is the real fix.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2455

🤖 Generated with [Claude Code](https://claude.com/claude-code)
