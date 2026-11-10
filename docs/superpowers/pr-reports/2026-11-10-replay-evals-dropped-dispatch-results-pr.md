## Summary

- Two offline replay evals still read `substrate_dispatch_results`, which was dropped on 2026-11-10 (#3162). They would crash in live mode.
- Leisure replay (`--live`) and disk-rule replay (`--export`) now take `--dispatch-db`. It names a database restored from the pre-drop backup.
- If that table is not there, each eval stops with exit code 2 and prints the exact restore commands. If the table is empty, the eval also stops.
- `--export` never changes the committed fixture when it fails. It writes a temporary file, then swaps it in.
- The drop-migration headers now show a `pg_restore --list` command that works.

## Outcome moved

No orphaned readers of the dropped table are left. Before: live/export mode crashed with a psql error. After: it stops with a clear message, or it replays from a restored copy.

## Current architecture

The leisure-gate verdicts and the image-prune runs were saved only in `substrate_dispatch_results`. The successors keep them nowhere durable: the Hub leisure timer's gate verdicts and the hardware-watch disk rule's answers go only to logs and `/health`. `want_ledger` saves the economy's decision on the shadow bid, not the doer's gate verdict. No act-ledger (`substrate_action_outcomes`) rows exist for retro leisure or image prune. So no live table can replace the history. The backup is now the only source.

## Architecture touched

These are offline evals only. No runtime service changes.

## Files changed

- `services/orion-hub/evals/retro_leisure_self_start_replay.py`: adds `--dispatch-db`, the presence check, and the empty-history check. Exit code 2 carries the restore recipe.
- `services/orion-hardware-watch/evals/run_image_prune_replay_eval.py`: the same for `--export`, plus the empty-restore guard and the atomic fixture write.
- `services/orion-hub/tests/test_retro_leisure_replay_live_source.py`: new tests.
- `services/orion-hardware-watch/tests/test_image_prune_replay_export_source.py`: new tests.
- `.github/workflows/retro-memory-tests.yml`: runs the new hub test (PR and push paths).
- `services/orion-sql-db/manual_migration_retire_*_drop.sql`: comment-only `pg_restore --list` fix. The `/dev/stdin` form fails with "did not find magic string", and the feedback header also lacked `-i`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: eval CLI only (`--dispatch-db`)
- Compatibility notes: fixture mode (CI) is unchanged

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-hub: pytest tests/test_retro_leisure_replay_live_source.py tests/test_retro_leisure_self_start.py -> 36 passed
services/orion-hardware-watch: pytest tests -> 114 passed
scripts/check_sql_migrations_applied.py -> every schema object present and valid
scripts/check_migration_required_by.py -> OK
```

## Evals run

```text
retro_leisure_self_start_replay.py (fixture) -> 9/9 checks
run_image_prune_replay_eval.py (fixture) -> rc 0, all gates PASS
retro_leisure_self_start_replay.py --live (real DB, table gone) -> rc 2, restore message
run_image_prune_replay_eval.py --export (real DB, table gone) -> rc 2, fixture untouched
pg_restore --list via stdin on both backups -> 6 TABLE DATA each
```

The `--dispatch-db` path against a really restored database is UNVERIFIED. Testing it needs a `createdb` on the live Postgres server, and I did not run that.

## Docker/build/smoke checks

```text
Not applicable (offline evals; no image change).
```

## Review findings fixed

- Finding: `--export` from an empty restore would overwrite the fixture with zero runs.
  - Fix: raise when no runs come back. Write to `.tmp`, then `os.replace`.
  - Evidence: `test_export_from_an_empty_restore_exits_2_and_keeps_the_fixture`
- Finding: the new hub test was missing from the workflow push paths.
  - Fix: added it.
  - Evidence: workflow diff
- Finding: a test assertion compared the SQL with itself.
  - Fix: assert the DB list only.
  - Evidence: test file
- Finding: `--dispatch-db` was ignored without `--live`/`--export`.
  - Fix: `ap.error`.
  - Evidence: two refusal tests

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: live replay now needs a manual restore into a scratch DB.
- Mitigation: the error message prints the two commands.

## PR link

(added after open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
