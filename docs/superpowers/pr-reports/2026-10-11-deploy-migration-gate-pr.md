# Deploy gate: refuse `up` while a required SQL migration is unapplied

## Summary

- `scripts/safe_docker_build.sh <svc> up` now asks the live database (read-only) whether every migration the service needs is applied, and refuses if one is missing or if it cannot tell.
- A migration says who needs it with one header line: `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs` (or `none <reason>`). Same `ORION-MIGRATION-*` marker family the drift watch already uses.
- `scripts/check_sql_migrations_applied.py --service <svc>` is the check (exit 0 applied / 1 missing / 2 unknown). It reuses `orion/sql_migration_drift.py`; nothing duplicated.
- New CI check `scripts/check_migration_required_by.py`: every newly added `manual_migration_*.sql` must declare; `-- DESTRUCTIVE` files may only declare `none`.
- Backfilled 8 migration headers.

## Outcome moved

Twice (10-10 PR #2594, 10-11 PR #2605) orion-durable-runs was deployed before its migration was applied: it started, then failed every step with `UndefinedTable`. That deploy is now refused before docker runs, with the exact apply command printed.

## Current architecture

`safe_docker_build.sh` already gated deploys on the shared checkout, env drift and hostname refs. `check_sql_migrations_applied.py` and the 10-minute `substrate-ladder-watch` reported unapplied migrations after the fact (30-day window, Hub card). Nothing linked a migration to the service that needs it, and nothing stopped a deploy.

## Architecture touched

- Deploy wrapper (section 2c): gates only when an argument is exactly `up`. `build`, `config`, `logs` and `ps` pass. It uses the main checkout's `.venv` python (for psycopg2), the same lookup as the Makefile's `METRIC_PYTHON`.
- Drift module: `parse_required_by`, `is_destructive`, `required_for`, `deploy_gate`, `has_unconditional_effects`. The deploy verdict has no recency window: a requirement does not age out.
- Nothing is ever applied automatically.

## Files changed

- `orion/sql_migration_drift.py`: REQUIRED-BY parsing and the deploy verdict.
- `scripts/check_sql_migrations_applied.py`: `--service` mode; `open_connection()` raises `ConnectError` instead of exiting.
- `scripts/safe_docker_build.sh`: section 2c gate and the override.
- `scripts/check_migration_required_by.py`: the CI declaration check.
- `.github/workflows/orion-static-gates.yml`: runs the CI check and the tests.
- `tests/test_deploy_migration_gate.py`: 33 tests.
- `services/orion-sql-db/manual_migration_{temporal_self_event_v1,temporal_self_v1}.sql`: orion-durable-runs.
- `manual_migration_field_dominance_run_v1.sql`: orion-attention-runtime, orion-durable-runs.
- `manual_migration_regulation_history.sql`: orion-dream, orion-sql-writer.
- `manual_migration_{system_one_appraisal_v1,vision_organ_substrate_loop,storage_write_substrate_loop}.sql`: orion-substrate-runtime.
- `manual_migration_retire_streak_tick_v1.sql`: `none` (DESTRUCTIVE retire).
- `AGENTS.md` section 8: four-line description.

## Schema / bus / API changes

- Added: the `ORION-MIGRATION-REQUIRED-BY` SQL header marker.
- Removed / Renamed: none.
- Behavior changed: `safe_docker_build.sh ... up` can now refuse.
- Compatibility notes: no bus or schema changes. Older untagged migrations are grandfathered; only new files must declare.

## Env/config changes

- Added keys: none in any `.env_example`. `ORION_ALLOW_UNAPPLIED_MIGRATION=1` is a per-command shell override, like `ORION_ALLOW_ENV_DRIFT`.
- Removed keys / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest tests/test_deploy_migration_gate.py tests/test_sql_migration_drift_gate.py tests/scripts/test_safe_docker_build_heartbeat.py -q
95 passed
python3 scripts/check_migration_required_by.py   -> OK (0 new migration files vs origin/main)
python3 scripts/check_env_template_parity.py     -> exit 0
git diff --check                                  -> clean
```

## Evals run

```text
No eval harness: this is a deterministic deploy gate. The live smoke below is the behavioural check.
```

## Docker/build/smoke checks

Live, read-only, against the real DB. A fake `docker` binary was put on PATH so nothing deployed.

```text
$ scripts/safe_docker_build.sh orion-durable-runs up -d
migration gate [orion-durable-runs]: all 3 required migration(s) applied: manual_migration_field_dominance_run_v1.sql, manual_migration_temporal_self_event_v1.sql, manual_migration_temporal_self_v1.sql
FAKE docker (nothing deployed): compose ... up -d

# with an untracked test migration requiring orion-durable-runs whose table does not exist:
migration gate [orion-durable-runs]: MISSING manual_migration_zz_smoke.sql: not applied to the live database -- table zz_smoke_never_applied -- apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_zz_smoke.sql
REFUSING to bring up orion-durable-runs: a SQL migration it needs is not applied (apply command above).

# ORION_PG_PORT=1 (DB unreachable):
migration gate [orion-durable-runs]: UNKNOWN -- cannot read the live database (could not connect to localhost:1/conjourney: ... Connection refused ...) ...
REFUSING to bring up orion-durable-runs: whether its required SQL migrations are applied is UNKNOWN (see above).

# build only: no gate, docker called.
```

`--service` also passes live for orion-attention-runtime, orion-dream, orion-sql-writer and orion-substrate-runtime.

## Review findings fixed

- Finding: false pass when a later migration re-declares the same table. Blame goes to the last creator, so the required file read APPLIED.
  - Fix: `deploy_gate` also blocks on findings against objects the required file itself creates.
  - Evidence: `test_deploy_gate_blocks_when_a_later_file_redeclares_the_same_table`.
- Finding: `field_dominance_run_v1` is also needed by orion-durable-runs. `temporal_self_sources.py` reads it inside a single transaction, so a missing table aborts the whole read.
  - Fix: header now names both services.
  - Evidence: live `--service orion-durable-runs` lists 3 files.
- Finding: REQUIRED-BY on a data-only file could never be enforced.
  - Fix: CI rejects it and asks for `none <reason>` instead.
  - Evidence: `test_ci_required_by_on_data_only_file_is_rejected`.
- Finding: a superseded required file passed without checking its successor.
  - Fix: the gate follows `SUPERSEDED-BY`.
  - Evidence: `test_deploy_gate_follows_superseded_by_to_the_successor`.
- Finding: a required name missing from the report was skipped silently.
  - Fix: it now blocks.
  - Evidence: `test_deploy_gate_required_file_absent_from_report_blocks`.
- Finding: a renamed-in migration escaped the CI check.
  - Fix: `--no-renames`.
  - Evidence: `test_ci_renamed_in_migration_must_declare`.
- Finding: `none` matched as a prefix (`none-svc`).
  - Fix: negative lookahead.
  - Evidence: `test_none_is_not_a_prefix_match`.
- Not changed (nit): only a literal `up` argument is gated. `run`/`create`+`start` bypass the gate. Documented.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: coverage is only as good as the declarations. orion-durable-runs also reads tables from older, untagged migrations (chat_turn, curiosity_run, ...). The gate does not cover those.
  - Mitigation: new migrations must declare. Tag older ones as incidents surface.
- Severity: low.
  - Concern: the gate connects to `localhost:55432` (ORION_PG_* overrides). On a host without that DB, a service with declared migrations is refused as UNKNOWN. All declared services run on athena with the DB today. Services with no declared migrations never connect.
  - Mitigation: `ORION_ALLOW_UNAPPLIED_MIGRATION=1`.
- Severity: low.
  - Concern: a required data-only file can only warn, never block.
  - Mitigation: CI now refuses such declarations.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2612
