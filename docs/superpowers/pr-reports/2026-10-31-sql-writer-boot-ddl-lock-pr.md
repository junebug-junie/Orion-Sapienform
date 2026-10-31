## Summary

- orion-sql-writer re-ran 115 `ADD COLUMN` / `CREATE INDEX` / `CREATE TABLE IF NOT EXISTS` statements in one transaction on every boot. It now skips the ones the catalog already satisfies, so a current schema takes no table lock at boot.
- New shared helper `orion/db/guarded_ddl.py` (`GuardedDDL`) reads `information_schema` and `pg_indexes` once.
  - It skips only shapes it can judge exactly: ADD COLUMN IF NOT EXISTS (one or several clauses, nothing else), CREATE [UNIQUE] INDEX IF NOT EXISTS, CREATE TABLE IF NOT EXISTS.
  - Anything else runs: other ALTER shapes, several statements or comments in one string, and tables made earlier in the same run.
- All six sql-writer boot blocks route their DDL through it. The boot log reports `ran=` / `skipped=`.

## Outcome moved

- ADD COLUMN IF NOT EXISTS takes ACCESS EXCLUSIVE even when the column exists, and sql-writer's engine sets a `lock_timeout`.
- Behind a long reader such as a pg_dump, the whole bootstrap block therefore rolled back with a single warning, and the writer served without the columns it guards. The code comments already recorded it "seen rolling back whole on a LockNotAvailable".
- Real sql-writer lifespan on a throwaway Postgres 16, with a session holding a read lock on `chat_history_log`:
  - **this branch:** `boot DDL ran=0 skipped=115`, lifespan up in 0.3 s
  - **main:** `chat_message migration warning: (psycopg2.errors.LockNotAvailable) canceling statement due to lock timeout`, so the whole block rolled back, after 10.4 s

## Current architecture

The `lifespan` bootstrap ran every DDL statement through `conn.exec_driver_sql` inside one `engine.begin()`, with one warn-only handler.

## Architecture touched

`orion/db/guarded_ddl.py` (new, shared), `services/orion-sql-writer/app/main.py`, `.github/workflows/orion-sql-writer-tests.yml`.

## Files changed

- `orion/db/guarded_ddl.py`, `orion/db/__init__.py`: the helper.
- `orion/db/tests/test_guarded_ddl.py`: what is skipped and what always runs.
- `services/orion-sql-writer/app/main.py`: 119 ALTER/CREATE calls in 6 `engine.begin()` blocks go through `ddl(...)`. UPDATE / DELETE / retention / checks are unchanged.
- `.github/workflows/orion-sql-writer-tests.yml`: runs `orion/db/tests`.
- `services/orion-sql-writer/tests/test_boot_ddl_guarded.py`: pins the wiring.

## Schema / bus / API changes

- None. The same statements apply to a database that lacks them.

## Env/config changes

- None.

## Tests run

```text
orion/db/tests                      6 passed
services/orion-sql-writer/tests     1161 passed, 179 skipped (incl. 3 new wiring tests)   (main: 1157 passed, 1 failed, 179 skipped)
AST scan: 119 ddl() calls, none with several statements, comments or params
Throwaway Postgres 16, real sql-writer lifespan (bus off):
  empty DB                    ran=29 skipped=86, 1.6 s (the second run skipped all 115, so the schema is complete)
  schema present              ran=0  skipped=115, 0.3 s
  read lock on chat_history_log, this branch   ran=0 skipped=115, 0.3 s
  read lock on chat_history_log, main          LockNotAvailable -> block rolled back, 10.4 s
```

## Evals run

```text
n/a (boot schema)
```

## Docker/build/smoke checks

```text
Deploy: orion-sql-writer post-pull rebuild. Live check: its log line "boot DDL ran=0 skipped=115".
```

## Review findings fixed

- Self-check: a string with several statements or a comment is never skipped, so a satisfied first statement cannot hide the rest. Test added.
- Finding (subagent): the "missing table under ALTER TABLE IF EXISTS" skip gained nothing (Postgres takes no lock there) and was the one skip that depended on `current_schema()` matching search_path resolution.
  - Fix: a table missing from the snapshot is never skipped.
  - Evidence: test updated.
- Finding: nothing pinned the `main.py` wiring. A lost `ddl = ...` line would NameError, the error would be swallowed, and the block would roll back.
  - Fix: `services/orion-sql-writer/tests/test_boot_ddl_guarded.py` (AST) checks that no ALTER/CREATE stays on `conn.exec_driver_sql`, that every block defines `ddl` first, and that every guarded statement is a skippable shape.
  - Evidence: mutation-checked. A bypassed call fails 1 test; a removed `ddl =` line fails 1 test.
- Finding (nit): quoted mixed-case identifiers were lowercased.
  - Fix: unquoted names fold to lower case, quoted names keep their case, and catalog names are compared as stored.
  - Evidence: `test_quoted_names_keep_their_case`.
- Checked fine by the reviewer:
  - exactly 119 calls rewritten, and no non-DDL call converted
  - every `ddl()` string is a skippable shape
  - `ddl` is defined first in all 6 blocks
  - the log line is reachable only after the block
  - skip-on-name matches Postgres's own `IF NOT EXISTS`
  - live index definitions match
  - read-only replay against the live catalog: ran=0, skipped=123 over all six blocks

## Restart required

```bash
# athena: post-pull rebuild of orion-sql-writer (automatic on git pull of primary main)
```

## Risks / concerns

- Severity: low. Concern: a CREATE INDEX IF NOT EXISTS whose name exists with a different definition is skipped. Mitigation: that was already a no-op in Postgres; behavior unchanged.
- Severity: low. Concern: a genuinely new column still takes the lock once, on the deploy that adds it.
- Note: `orion/world_pulse_read/pull.py` is already guarded (Hub checks the columns, with a 3 s lock_timeout). `queue.ensure_seed_queue_schema` (with its DROP/ADD CONSTRAINT) has no production caller.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3024

🤖 Generated with [Claude Code](https://claude.com/claude-code)
