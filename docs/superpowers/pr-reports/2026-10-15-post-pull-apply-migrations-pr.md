## Summary

When `git pull` brings in a new SQL migration, the pull now applies it to the database *before* it rebuilds any service. It does this only when the migration is plainly additive and safe.

- New `scripts/apply_post_pull_migrations.py` runs inside `scripts/rebuild_services_from_git_diff.sh`, after the migration list is known and before the first `safe_docker_build.sh ... up`.
- It acts only on changed files that the existing drift checker reads as NOT APPLIED. It applies them in filename order, with `lock_timeout=10s`, `ON_ERROR_STOP=1`, and one wrapping transaction when the file allows it.
- The safety filter is an allowlist of additive DDL with an inert remainder (`auto_apply_refusals` in `orion/sql_migration_drift.py`). Everything else is left for a human, with the manual command printed.
- It writes only from the same place a live deploy may run from: the clean primary `main` checkout, synchronized with `origin/main`, on a host where `orion-athena-sql-db` is running.
- Kill switch: `ORION_POST_PULL_APPLY_MIGRATIONS=0`. Every attempt, refusal and hold is logged to `~/.orion/applied_migrations.log`.

## Outcome moved

Before: a pull that carried a migration started the rebuilds at once. The deploy gate then refused every consumer, and Juniper had to apply the migration by hand and deploy again ("the second I git pull i cant run that migration ... happens all the time").

After: for the common case (new table, new index, new column), the migration lands first and the consumers rebuild normally. Of the 15 existing migrations that declare `REQUIRED-BY`, 12 would have been auto-applied as new migrations. The other 3 were correctly refused: a `DROP TABLE`, a `DO` block with a backfill `UPDATE` (PR #2701's own file), and `DROP CONSTRAINT`.

## Current architecture

Post-merge hook → `rebuild_services_from_git_diff.sh` → `rebuild_affected_services.py --plan/--migrations` → `safe_docker_build.sh <svc> up -d --build`. The wrapper refuses `up` while a migration that the service declares is unapplied. PR #2700 made migrations rebuild nothing and print a read-only post-pull report. Nothing ever applied a migration.

## Architecture touched

- Post-pull flow: a new apply step runs before the read-only report and before the rebuilds. Its exit code is ignored, so the rebuilds always run. The deploy gate still refuses any consumer whose migration did not land.
- `orion/sql_migration_drift.py`: adds `auto_apply_refusals` and `transaction_incompatible`. Both are pure and reuse the existing tokenizer, `_ALTER_TABLE` and `_split_top_level_commas`.
- No service, bus, schema or env-template change.

## Files changed

- `scripts/apply_post_pull_migrations.py`: new applier. It checks the checkout, the kill switch, the live state, classification, the container/host, the apply itself, and re-verifies after applying.
- `orion/sql_migration_drift.py`: auto-apply allowlist and transaction detection.
- `scripts/rebuild_services_from_git_diff.sh`: calls the applier (with `--dry-run` under `--dry-run`) before the report and the rebuilds.
- `scripts/check_sql_migrations_applied.py`, `scripts/safe_docker_build.sh`, `CLAUDE.md` §8: wording only. The text no longer says migrations are only ever applied by hand.
- `tests/test_post_pull_apply_migrations.py`: filter, classify, ordering, holds, checkout gate, kill switch, host gates.
- `tests/test_post_pull_apply_migrations_postgres.py`: the real rebuild script, applier and checker against a throwaway Postgres.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: apply-before-rebuild order, dry-run passthrough, and no applier call without migrations.
- `.github/workflows/post-pull-migrations.yml`: CI job with a Postgres service container.

## Safety rules (what auto-applies)

A changed file is applied only if **all** of these hold:

1. The checker says NOT APPLIED (MISSING). Files that are APPLIED, SUPERSEDED or NOT-A-MIGRATION are skipped. DATA, CONDITIONAL and UNKNOWN files are never auto-applied, because the schema cannot show whether they ran.
2. It is not a `*_rollback.sql` file, has no `-- DESTRUCTIVE` line, and has a valid `ORION-MIGRATION-REQUIRED-BY` header (services or `none <reason>`). There is no marker error, and the header does not name an apply script.
3. Every statement is one of these: `CREATE TABLE`, `CREATE [UNIQUE] INDEX`, `CREATE SEQUENCE`, `COMMENT ON`, `ALTER TABLE ADD COLUMN | ADD CONSTRAINT | ALTER COLUMN SET DEFAULT`, `INSERT ... VALUES (...) ON CONFLICT DO NOTHING`, or a balanced `BEGIN`/`COMMIT` of the file's own.
4. Nothing inert-breaking appears anywhere in the file:
   - no function call other than `now()` (calls inside `CREATE TABLE`, or in an index on a table that does not exist live yet, are allowed, because no existing row is touched)
   - no `GENERATED`, `CASCADE`, `EXECUTE`, `SELECT`, `RETURNING`, `CALL` or `COPY`
   - no `SET`/`RESET`, so a file cannot lift the 10 s lock timeout
   - no double quote outside comments, because a quoted identifier can spell an allowed prefix
   - no `DO` or function body
   - no psql meta-command except `\set ON_ERROR_STOP on`
5. On a table that already exists live, the change must not hold a long write lock:
   - an index must be `CONCURRENTLY`
   - a constraint must be `NOT VALID`
   - a new column must not carry `PRIMARY KEY`, `UNIQUE`, `REFERENCES` or `CHECK`

**Order and stops.** Files run in filename order. A refused file, a `DESTRUCTIVE` file, or an unverifiable file holds every later file in the set. So does the first failed apply. Rollback files hold nothing.

**How it applies.** `psql -X -q -v ON_ERROR_STOP=1 [--single-transaction] -f -`, with `PGOPTIONS=-c lock_timeout=10s -c statement_timeout=15min`. psql receives the exact text that was scanned. `--single-transaction` is dropped only for `CREATE INDEX CONCURRENTLY` or the file's own `BEGIN`/`COMMIT`.

**Where it applies:**

- Only the primary checkout, on `main`, clean, with HEAD equal to `origin/main`. A git failure counts as a problem.
- The classifier lists no migrations on circe (from #2700).
- The default command requires `orion-athena-sql-db` to be running locally. This is what stops hosts that have no service list, such as hecate.
- `ORION_PG_HOST` must be local, so the database written is the database verified.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the post-pull flow can now write to the live database (additive DDL only, with the gates above). Juniper approved this on 2026-10-15.
- Compatibility notes: the deploy gate is unchanged and still refuses consumers of anything left unapplied.

## Env/config changes

- Added keys: `ORION_POST_PULL_APPLY_MIGRATIONS` (default on; `0` turns it off), `ORION_APPLIED_MIGRATIONS_LOG` (optional log path), `ORION_MIGRATION_APPLY_PSQL` (optional psql prefix, used by tests). These are host-shell variables for the hook, not service env, so no `.env_example` changed.
- Removed / renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_post_pull_apply_migrations.py tests/test_post_pull_apply_migrations_postgres.py \
  tests/scripts/test_rebuild_services_from_git_diff.py tests/scripts/test_rebuild_affected_services.py \
  tests/test_deploy_migration_gate.py tests/test_sql_migration_drift_gate.py
  -> 295 passed (Postgres tests ran against a throwaway local initdb cluster)
Clean CI-like venv (pytest, psycopg2-binary, PyYAML only): new tests pass.
Mutation check: dropping PGOPTIONS from the apply command makes the real-Postgres test fail
  (the INSERT records lock_timeout=0 instead of 10s).
python scripts/check_migration_required_by.py --base origin/main -> OK
bash -n scripts/rebuild_services_from_git_diff.sh; git diff --check -> clean
```

## Evals run

```text
No eval harness exists for the post-pull scripts. Calibration against the real corpus is pinned
as a test instead (test_corpus_calibration): the known guarded wipe
(manual_migration_action_value_control_arm.sql), the stage-5 table drops, the DESTRUCTIVE retire
file, #2701's DO-block backfill and walkway_privacy_v2's DROP CONSTRAINT are refused; the plain
additive migrations pass as new migrations.
```

## Docker/build/smoke checks

```text
# Dry run on the real PR #2701 merge (read-only against the live DB):
$ scripts/rebuild_services_from_git_diff.sh --dry-run --base 5aa69bcb7
rebuild_affected_services: mesh host: athena
rebuild_services_from_git_diff.sh: hand-written SQL migration(s) changed -- no rebuild runs these; safe ones are applied first, before any rebuild:
post-pull migrations: 1 changed file(s), checked against the live database [dry-run: nothing is applied]
  - services/orion-sql-db/manual_migration_vision_ask_names_v1.sql: APPLIED
  - services/orion-sql-db/manual_migration_vision_ask_names_v1.sql (required by: orion-sql-writer, orion-hub)
      live state: APPLIED
      note: +1 data statement(s) the schema cannot confirm
rebuild_services_from_git_diff.sh: affected services:
  - orion-hub
  - orion-sql-writer
  - orion-substrate-runtime     <- from this branch's own orion/ change; HEAD is the branch, not 2b070a0e8
[dry-run] would rebuild: ...

# Dry run with two scratch files plus the real one (read-only):
  - manual_migration_zz_dryrun_probe.sql: NOT APPLIED ... would apply (single transaction, lock_timeout=10s)
  - manual_migration_zz_dryrun_probe2.sql: NOT auto-applied -- not safe to auto-apply: function call f() in: ALTER TABLE orion_ask ADD COLUMN
# Same probe, real (non-dry) run from this linked worktree:
post-pull migrations: not applying anything -- this is a linked worktree, not the primary checkout. ... rc=2

# Live default docker-exec path, scratch probe (created, checked, dropped, confirmed gone):
cmd: docker exec -i -e PGOPTIONS=-c lock_timeout=10s -c statement_timeout=15min orion-athena-sql-db psql -U postgres -d conjourney -X -q -v ON_ERROR_STOP=1 --single-transaction -f -
apply ok: True
session lock_timeout recorded by the INSERT: 10s
drop: DROP TABLE
exists after drop: ''
```

## Review findings fixed

Two code-review passes ran in subagents. The first draft used a denylist. The second used a prefix-only allowlist.

- Finding: the denylist missed `EXECUTE '...'` in DO blocks, psql `\i` / `\gexec` / `\!`, `ALTER COLUMN TYPE`, `WHERE` inside subqueries, CTE `DELETE`, `MERGE`, `CASCADE`, and `SELECT f()` running a function body.
  - Fix: replaced it with an allowlist (rules 3–4 above).
  - Evidence: every probe is a refused case in `test_anything_off_the_allowlist_is_refused`.
- Finding: the applier wrote to the live DB from any checkout. A feature-branch pull would apply unreviewed migrations.
  - Fix: `checkout_problem` uses the live-deploy rule (primary checkout, main, clean, `origin/main`).
  - Evidence: `test_only_the_clean_synchronized_main_checkout_writes`, `test_a_linked_worktree_never_writes`, and the rc=2 probe above.
- Finding: `DESTRUCTIVE` and DATA files did not hold later files.
  - Fix: both now hold later files.
  - Evidence: `test_rollback_holds_nothing_but_destructive_and_refused_files_hold_later_files`, `test_unverifiable_data_file_holds_later_files`.
- Finding: function calls rode after an allowed prefix. A volatile `DEFAULT`, a `CHECK` or an index expression runs once per existing row.
  - Fix: calls are refused except `now()`, or inside `CREATE TABLE` / an index on a table that does not exist live yet. `GENERATED` is refused.
  - Evidence: parametrized refusals plus `test_fresh_table_gets_plain_indexes_and_constraints_existing_one_does_not`.
- Finding: a quoted identifier could spell an allowed prefix (`ALTER TABLE "t add column x" DROP COLUMN secret`).
  - Fix: any double quote outside comments is refused.
  - Evidence: refused cases in the tests.
- Finding: `DROP CONSTRAINT`, `DROP DEFAULT`, `DROP/SET NOT NULL` and `SET lock_timeout = 0` were allowed.
  - Fix: removed from the allowlist. An unbalanced `BEGIN` is refused.
  - Evidence: refused cases in the tests.
- Finding: a non-concurrent index, or a validated constraint, on a big live table holds a write lock for minutes.
  - Fix: on a table that exists live, the index must be `CONCURRENTLY` and the constraint `NOT VALID`.
  - Evidence: the long-lock refused cases; the live catalog is passed in from the checker's read.
- Finding: the write target and the read target could differ.
  - Fix: `-U`/`-d` come from `ORION_PG_USER`/`ORION_PG_DB`, and a non-local `ORION_PG_HOST` refuses.
  - Evidence: `test_default_command_refuses_when_the_checker_reads_another_host`, `test_default_prefix_passes_lock_timeout_and_writes_the_database_it_reads`.
- Finding: the file was read twice, and a git failure read as a clean checkout.
  - Fix: psql gets the scanned text via stdin. `_git` returns `None` on failure, and that counts as a problem.
- Finding: refusals and holds were not logged.
  - Fix: `REFUSED`, `HELD` and `NEEDS-HUMAN` lines are now written to the durable log.
- Finding: the docker-exec path was untested.
  - Fix: verified live with the scratch probe above (the session saw `lock_timeout=10s`; the probe was dropped and confirmed gone).

## Restart required

```text
No restart required. Takes effect on the next `git pull` in the primary checkout with the post-merge
rebuild enabled (.orion/post-merge-rebuild-enabled or ORION_POST_MERGE_REBUILD=1).
```

## Risks / concerns

- Severity: medium
  - Concern: this is the first automated writer to the live database from the pull path.
  - Mitigation: an allowlist, not a denylist. It runs only from the clean synced main checkout. Lock timeout 10 s. Single transaction. Stop on first failure or refusal. The checker re-verifies, and every apply is logged. Kill switch `ORION_POST_PULL_APPLY_MIGRATIONS=0`.
- Severity: low
  - Concern: the allowlist is deliberately narrow. Migrations with `DO` blocks, backfills, `DROP CONSTRAINT` or `SET NOT NULL` still need a human. PR #2701's own migration is one of these.
  - Mitigation: the refusal prints the reason and the exact manual command. Widen the allowlist only with a test per new shape.
- Severity: low
  - Concern: a `CREATE INDEX CONCURRENTLY` that fails runs outside a transaction, so it can leave an INVALID index.
  - Mitigation: the checker reads INVALID, and the next pull refuses the file with the drop-and-rerun instruction.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2703

🤖 Generated with [Claude Code](https://claude.com/claude-code)
