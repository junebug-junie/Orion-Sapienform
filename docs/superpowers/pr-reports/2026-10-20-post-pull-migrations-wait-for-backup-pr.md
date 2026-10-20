## Summary

When `git pull` brings a new SQL migration while the nightly backup is running, the pull no longer gives up on it. It waits for the backup to finish, then applies the migration, then rebuilds the services that were waiting for it.

- Before each file, the applier asks Postgres (read-only) whether a backup session is running: a `pg_dump` or `pg_dumpall` session on this database that is inside a transaction (an idle, orphaned backend holds no locks and is ignored). While one runs, it waits. It polls every 30 s, prints a progress line, and waits at most `ORION_MIGRATION_BACKUP_MAX_WAIT_S` in total, counted from the first wait (default 90 min, or `--max-backup-wait`; 0 means a running backup fails the file at once). Past that, the file fails and holds the rest, exactly as a failed apply does today.
- While psql runs, a watcher reads `pg_blocking_pids()` for the apply session (it is now named `orion-post-pull-migrate-<pid>` through `PGAPPNAME`). A `lock_timeout` failure is retried (3 times, backoff 30/60/120 s, each after waiting for the backup again) only when every blocker seen was a backup and the file ran in one transaction. Any other failure, including a lock timeout behind a non-backup session, fails and holds as before.
- `lock_timeout` stays 10 s. It is never raised.
- The rebuild script no longer makes every rebuild wait for the backup. Its first applier pass runs with `--defer-if-backup` (exit 3 when it deferred files). Every rebuild then runs as usual. After that, a second pass waits for the backup and applies. Then the script rebuilds again only the services that failed and are named in those files' `REQUIRED-BY` headers (when the second pass exits 0 or 1; the deploy gate re-refuses any whose file still did not land).
- Every wait, retry and deferral is written to `~/.orion/applied_migrations.log` (`BACKUP-WAIT`, `BACKUP-WAIT-DONE`, `BACKUP-WAIT-EXCEEDED`, `RETRY`, `DEFERRED`). The rebuild log gets `DEFERRED-APPLY` and `RETRY-AFTER-MIGRATION`.

## Outcome moved

Incident 2026-10-17 ~03:56 UTC: the nightly `pg_dump` held AccessShareLocks for about 45 min. The applier hit `lock_timeout` on `manual_migration_vision_motion_story_strip_v2.sql`. `orion-sql-writer` and `orion-hub` then stayed refused by the deploy gate until someone applied the file by hand and deployed again.

With this patch, the same pull defers that file, deploys every other affected service at once, waits for the backup, applies the file, and rebuilds `orion-sql-writer` and `orion-hub`. No human step.

## Current architecture

Post-merge hook (foreground) -> `rebuild_services_from_git_diff.sh` -> `apply_post_pull_migrations.py` (PR #2703: allowlisted additive files only, `lock_timeout=10s`, the first failure holds the rest) -> read-only checker report -> per-service `safe_docker_build.sh ... up`. The deploy gate refuses `up` for any service whose required migration is unapplied. A lock timeout was a plain failure, whatever held the lock.

## Architecture touched

- `scripts/apply_post_pull_migrations.py`: backup look, wait loop with one deadline for the whole run, blocker watcher thread, bounded retry, `--defer-if-backup` (exit 3), `--max-backup-wait`, `--required-by` (prints the consumers named in the files' headers).
- `scripts/rebuild_services_from_git_diff.sh`: the per-service loop body is now the function `rebuild_one` (same body; `continue` became `return 0`). Adds `apply_deferred_migrations` and the consumer retry.
- No service, bus, schema, compose or `.env_example` change.

Item 5 of the task asked whether the inline wait makes the consumer retry moot. A plain inline wait would: the apply would finish before any rebuild. But it would also hold every unrelated rebuild behind the backup for up to 90 min, so unrelated services would not "keep going". The two-pass shape keeps them going, and so it needs the retry. The retry is narrow: only services in `FAILED` that a deferred file names in `REQUIRED-BY`, once, and only after the second pass actually ran (exit 0 or 1).

## Files changed

- `scripts/apply_post_pull_migrations.py`: backup wait, blocker watch, retry, deferral, new flags, docstring.
- `scripts/rebuild_services_from_git_diff.sh`: two-pass apply and consumer retry.
- `tests/test_post_pull_apply_migrations.py`: fake-DB cases for wait, max-wait, retry, non-backup blocker, mixed blockers, no blocker seen, no-transaction file, deferral, failed backup look, dry run, `--required-by`.
- `tests/test_post_pull_apply_migrations_postgres.py`: real-Postgres cases with a real session named `pg_dump` (or not) holding a real lock.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: two-pass ordering, refused-consumer retry, failed second pass, only-consumers retry, migration-only pull, dry run.
- `CLAUDE.md` (AGENTS.md): section 8 sentence on post-pull apply.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the applier waits for a running backup; exit code 3 (only with `--defer-if-backup`); apply sessions carry `application_name=orion-post-pull-migrate-<pid>`.
- Compatibility notes: an older applier without the flags is never paired with the new rebuild script (both ship in the same pull).

## Env/config changes

- Added keys: `ORION_MIGRATION_BACKUP_MAX_WAIT_S` (optional, read from the shell that runs `git pull`; default 5400). Documented in the applier docstring and AGENTS.md section 8, like the existing `ORION_POST_PULL_APPLY_MIGRATIONS` and `ORION_APPLIED_MIGRATIONS_LOG`.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no. No service reads it; the applier's existing keys are not in any `.env_example` either.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none

## Tests run

```text
pytest -q -rs tests/test_post_pull_apply_migrations.py tests/scripts/test_rebuild_services_from_git_diff.py tests/test_post_pull_apply_migrations_postgres.py
176 passed (real-Postgres lane ran on a throwaway local cluster, no skips)

Mutation checks (then reverted):
- blocker classification forced to "backup"      -> 4 failed
- pre-apply backup wait removed                    -> 4 failed

All 34 run steps of .github/workflows/orion-static-gates.yml, run locally -> 34 OK
```

## Evals run

```text
No eval harness for this host script. The real-Postgres lane is the behavioral proof:
a session named pg_dump holds ACCESS SHARE on the altered table; the applier waits, then
applies (3.5 s); a pg_dump that outlasts --max-backup-wait fails and holds; a pg_dump that
appears after the pre-apply look causes a real lock_timeout, the watcher names it through
pg_blocking_pids, and the retry lands; a non-backup holder fails at once with no retry.
```

## Docker/build/smoke checks

```text
No Docker change. Read-only live check against orion-athena-sql-db with the live role:
running_backups() -> []   (no backup in this window, expected)
BLOCKER_SQL            -> []   (both queries parse and are permitted)
The nightly backup (scripts/backup/orion_backup_databases.py) runs pg_dumpall in the
container, so its sessions are named pg_dumpall / pg_dump, as the incident report says.
Live wait path: UNVERIFIED until a pull lands during the nightly backup.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. Safety verdict: no path applies a file today's code would refuse, nothing raises `lock_timeout`, and the retry is idempotent (single-transaction files only).

- Finding (should): a deferred file was stranded when a later file was refused (applier returned 1, so the second pass never ran).
  - Fix: the applier returns 3 whenever it deferred anything; the second pass reports the refusal again.
  - Evidence: `test_a_deferred_file_keeps_exit_3_when_a_later_file_is_refused`.
- Finding (should): an idle or orphaned `pg_dump` backend would cause a 90 min wait and a failure for a file that applies today.
  - Fix: `BACKUP_SQL` counts only sessions with `xact_start IS NOT NULL` (a running pg_dump holds its snapshot transaction for the whole dump).
  - Evidence: real-Postgres `test_idle_pg_dump_backend_is_not_waited_for` (idle `pg_dump` session, `max_wait_s=0`, applies at once).
- Finding (should): `pg_basebackup` takes no relation locks, so waiting on it only cost time.
  - Fix: dropped it and the `datname IS NULL` arm.
  - Evidence: `test_backup_look_only_counts_sessions_inside_a_transaction`.
- Finding (should): one failed file in the second pass blocked the retry of every consumer.
  - Fix: the retry runs when the second pass exits 0 or 1; exit 2 (not allowed / UNKNOWN) still retries nothing.
  - Evidence: `test_partly_failed_second_pass_still_retries_the_consumers`, `test_deferred_apply_that_cannot_run_leaves_the_consumer_refused`.
- Finding (should): `git pull` can block in the foreground for the wait.
  - Fix: a stderr line names the bound before the wait; stated under Risks.
  - Evidence: asserted in `test_deferred_apply_that_cannot_run_leaves_the_consumer_refused`.
- Finding (nit): the budget started at run start, and retry backoff could sleep past it; the "exceeded" line overstated the wait.
  - Fix: the budget starts at the first real wait, backoff is clamped to it, and the line prints the real total waited.
  - Evidence: `test_retry_backoff_never_sleeps_past_the_wait_budget`.
- Findings (nit): max-wait 0 semantics, `PGAPPNAME` with a custom docker-exec psql command, empty-SERVICES exit code.
  - Fix: documented in the docstring and a script comment.
  - Evidence: n/a (docs).
- Finding (nit): the unit fake counted lock failures across all files.
  - Fix: it counts per file now.
  - Evidence: n/a (test harness).
- Not fixed (cosmetic): one extra progress line can print in the last instant before the deadline.

## Restart required

```text
No restart required. The next `git pull` on athena runs the new applier and rebuild script.
```

## Risks / concerns

- Severity: low
- Concern: a pull during the backup now blocks the terminal that ran `git pull` for up to 90 min, after every other rebuild has finished.
- Mitigation: Ctrl-C is safe (each apply is bounded by `lock_timeout`; a deferred file is simply unapplied). `ORION_MIGRATION_BACKUP_MAX_WAIT_S` shortens the wait.

- Severity: low
- Concern: an orphaned `pg_dump` backend (seen 2026-08-16..18, PR #1710) reads as a running backup, so the applier waits the full 90 min and then fails and holds.
- Mitigation: bounded wait; the outcome is the same as today's failure, 90 min later, and the log names the pid.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
