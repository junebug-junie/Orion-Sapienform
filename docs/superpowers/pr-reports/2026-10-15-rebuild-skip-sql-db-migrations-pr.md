# fix(rebuild): a hand-applied SQL migration never recreates orion-sql-db

## Summary

- A pull that only adds or edits a `services/orion-sql-db/manual_migration_*.sql` file now rebuilds nothing. Before, it recreated the production Postgres container.
- `orion-sql-db` now rebuilds only when its `docker-compose.yml` (or a future `Dockerfile`) changes. Its README, `.gitignore`, `.env_example`, and scratch `collapse_dump*.sql` files rebuild nothing. The container never reads them.
- A file that a compose mount or `COPY` of `orion-sql-db` itself brings into its container still rebuilds it.
- The post-pull script now lists each changed migration: who needs it, whether the live database already has it, and the exact apply command. Rollback and `-- DESTRUCTIVE` files are labelled and get no apply command.

## Outcome moved

Failure mode removed: a migration-only merge restarted Postgres under the whole fleet.

Incident (2026-10-15 22:27 UTC): PR #2698 added one migration file. The post-merge rebuild recreated `orion-athena-sql-db`. Postgres logged "database system shutdown was interrupted ... automatic recovery in progress", then every Orion service logged "not yet accepting connections". The resolver's own answer for that path was `services=['orion-sql-db'] reason='direct change in services/orion-sql-db/'`.

After this patch, the same path resolves to `services=[]`, reason `hand-applied SQL migration: no container consumes it, so no rebuild (apply it by hand with docker exec ... psql; never by recreating the DB container)`.

## Current architecture

- `scripts/rebuild_services_from_git_diff.sh` (run by the opt-in `post-merge` hook) asks `scripts/rebuild_affected_services.py --plan` which services changed, then runs `safe_docker_build.sh <svc> up -d --build` for each.
- The resolver's direct rule sent every path under `services/<svc>/` to `<svc>`. Skip rules were only repo-level prefixes and exact root files.
- `orion-sql-db` runs the stock `postgres:15` image. It has no Dockerfile. It mounts only its data volume (`${VOLUMES_ORION_POSTGRES_PATH}`). Nothing in the container reads the migration files; they are applied by hand with `docker exec ... psql`.
- `safe_docker_build.sh` already refuses `up` for a service whose required migration (`-- ORION-MIGRATION-REQUIRED-BY:`) is not applied. It refused `orion-sql-writer` and `orion-hub` correctly during the incident. The post-pull output did not say why until that refusal.

## Architecture touched

- Rebuild resolver mapping (`scripts/service_rebuild_paths.yaml`): two new keys.
  - `hand_applied_migration_globs`: changed migrations are skipped and reported. `*` does not cross a `/`.
  - `service_rebuild_only`: for a stateful service, the only files inside `services/<svc>/` that trigger a rebuild. `orion-sql-db: [docker-compose.yml, Dockerfile]`.
- Resolver (`scripts/rebuild_affected_services.py`): applies both rules before the direct `services/<svc>/` rule. Neither rule skips a path that a compose bind mount or Dockerfile `COPY` of the owning service brings in (own directory included, via `build_dependency_sources(include_own=True)`). Another service's mount rebuilds that other service only. New `--migrations` output mode and `migrations` field in `--json`. On a host with an include allowlist that does not run `orion-sql-db` (circe), no migrations are reported.
- Migration checker (`scripts/check_sql_migrations_applied.py`): new `--post-pull-report PATH...` mode. One read-only run covers every changed file. It reuses the deploy gate's parsing, live-state read, and `apply_command()` (header apply scripts included). Always exits 0.
- Post-pull script (`scripts/rebuild_services_from_git_diff.sh`): calls that report once. If the checker is missing or fails, it prints UNKNOWN and a raw apply line, and still skips rollback files.

Why not `mesh-utilities/common/exclude_services.txt`: that file also drives the `up_all_services*.sh` bring-up sweeps. Listing `orion-sql-db` there would stop bring-up-all from starting the database. A real compose change to `orion-sql-db` (for example `shared_buffers`) needs a recreate, and the compose comments already say so. So this patch uses the narrow skip, not a blanket "never auto-recreate". A separate "never auto-recreate stateful services" rule does not exist in the repo today; it is a follow-up question for Juniper, not built here.

## Files changed

- `scripts/service_rebuild_paths.yaml`: `hand_applied_migration_globs`, `service_skip_globs`, with the incident as the rationale.
- `scripts/rebuild_affected_services.py`: migration and rebuild-only rules, the owner-mount guard, `--migrations`.
- `scripts/check_sql_migrations_applied.py`: `--post-pull-report` mode (`post_pull_report()`).
- `scripts/rebuild_services_from_git_diff.sh`: prints the report for changed migrations.
- `tests/scripts/test_rebuild_affected_services.py`: resolver regression tests.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: fake classifier answers `--migrations`; tests for the printed block.
- `tests/test_deploy_migration_gate.py`: tests for `post_pull_report()`.

## Schema / bus / API changes

- Added: none (no bus, schema, or HTTP change).
- Removed: none.
- Renamed: none.
- Behavior changed: `rebuild_affected_services.py` has a new `--migrations` mode; `--json` gains a `migrations` key. `check_sql_migrations_applied.py` has a new `--post-pull-report` mode.
- Compatibility notes: existing `--plan`, `--list-only`, `--json` keys unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py -q
71 passed

.venv/bin/python -m pytest tests/test_deploy_migration_gate.py -q
44 passed

.venv/bin/python -m pytest tests/test_sql_migration_drift_gate.py services/orion-kev/tests/test_orion_kev_contract.py -q
66 passed
```

New regression tests:

- `test_migration_only_change_rebuilds_nothing`: the exact incident path resolves to no services, with the apply-by-hand reason.
- `test_sql_db_container_inputs_still_rebuild`: `docker-compose.yml` and `Dockerfile` under `orion-sql-db` still resolve to `orion-sql-db`.
- `test_files_the_db_container_never_reads_rebuild_nothing` (`collapse_dump.sql`, `README.md`, `.gitignore`), `test_sql_db_env_template_does_not_recreate_postgres`, `test_rollback_migration_rebuilds_nothing`.
- `test_a_bind_mounted_sql_file_still_rebuilds_despite_the_skip_glob`, `test_a_mounted_migration_glob_path_still_rebuilds`: the owner's mount wins over the skip.
- `test_another_services_mount_does_not_drag_postgres_along`, `test_migration_glob_star_does_not_cross_a_directory`, `test_migrations_are_not_reported_on_a_host_without_the_database`.
- `test_migrations_flag_prints_paths`, `test_migrations_flag_omits_a_deleted_migration`.
- Report: missing (consumers + apply command), applied, DB down (UNKNOWN, never APPLIED), live-state crash (UNKNOWN, not NOT APPLIED), DESTRUCTIVE and rollback (no apply command), data-only (UNVERIFIABLE, not APPLIED).
- Shell: `test_migration_only_pull_builds_nothing_and_names_the_apply_command`, `test_failed_checker_reports_unknown_and_never_offers_a_rollback`, `test_no_migrations_prints_no_migration_block`.

## Evals run

```text
No eval harness exists for the rebuild resolver (scripts/). The deterministic tests above are the gate.
Live dry-run against the real PR #2698 diff is below.
```

## Docker/build/smoke checks

Dry run against the real incident diff (read-only; the checker opens a read-only DB session):

```text
$ scripts/rebuild_services_from_git_diff.sh --dry-run --base d585456a0^1
rebuild_affected_services: mesh host: athena
rebuild_services_from_git_diff.sh: hand-applied SQL migration(s) changed -- no rebuild applies these:
  - services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql (required by: orion-sql-writer, orion-hub)
      live state: APPLIED
rebuild_services_from_git_diff.sh: affected services:
  - orion-hub
  - orion-memory-consolidation
  - orion-sql-writer
  - orion-substrate-runtime
```

`orion-sql-db` is gone from the list. (The migration was applied by hand after the incident, so it reads APPLIED.)

Live report for an unapplied scratch migration, a rollback file, and a DESTRUCTIVE file (scratch file deleted after):

```text
  - services/orion-sql-db/manual_migration_zz_scratch_probe.sql (required by: orion-hub)
      live state: NOT APPLIED -- the services above stay refused by safe_docker_build.sh until it is
      manual_migration_zz_scratch_probe.sql: not applied to the live database -- table zz_scratch_probe_never
      apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_zz_scratch_probe.sql
  - services/orion-sql-db/manual_migration_episode_memory_v1_rollback.sql (required by: none declared)
      rollback file: undoes another migration; never applied on deploy. Do not run it unless you are backing that migration out.
  - services/orion-sql-db/manual_migration_retire_streak_tick_v1.sql (required by: none (destructive retire of superseded streak ticks; operator-approved only, never a deploy dependency))
      DESTRUCTIVE: operator-approved only; no apply command is printed. Read the file and decide by hand.
```

Database unreachable (`ORION_PG_PORT=1`):

```text
  - services/orion-sql-db/manual_migration_zz_scratch_probe.sql (required by: orion-hub)
      live state: UNKNOWN (could not connect to localhost:1/conjourney: ... Connection refused ...; no answer is not 'applied')
      apply if not yet applied: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_zz_scratch_probe.sql
```

## Review findings fixed

Code review ran in a subagent on the first commit (140ef293d).

- Finding (must): the glob also matched `*_rollback.sql`, and the script then printed an apply command for a DROP script.
  - Fix: the report labels rollback files "never applied on deploy" with no command; the shell fallback skips them too.
  - Evidence: `test_post_pull_report_never_offers_destructive_or_rollback`, `test_failed_checker_reports_unknown_and_never_offers_a_rollback`.
- Finding (should): checker exit 0 was printed as APPLIED, but exit 0 also covers data-only/unknown/conditional files. A crash inside the live-state read printed NOT APPLIED.
  - Fix: verdict now comes from the checker's own per-file status in `post_pull_report()`. Only `APPLIED` prints APPLIED; data-only prints UNVERIFIABLE; any live-state error prints UNKNOWN.
  - Evidence: `test_post_pull_report_data_only_is_unverifiable_not_applied`, `test_post_pull_report_live_state_crash_is_unknown_not_missing`.
- Finding (should): the "consumed" guard counted any service's mount, so another service mounting a sql-db file would rebuild Postgres.
  - Fix: only the owning service's mount/COPY counts.
  - Evidence: `test_another_services_mount_does_not_drag_postgres_along`.
- Finding (should): README, `.gitignore`, and `.env_example` under orion-sql-db still recreated Postgres.
  - Fix: replaced the `.sql` skip glob with `service_rebuild_only` (only `docker-compose.yml`/`Dockerfile` trigger). `.env_example` gets the env-sync message.
  - Evidence: `test_files_the_db_container_never_reads_rebuild_nothing`, `test_sql_db_env_template_does_not_recreate_postgres`.
- Finding (should): migrations were reported on circe, which has no such database.
  - Fix: with a host include allowlist, a migration is reported only if its owning service is allowed there.
  - Evidence: `test_migrations_are_not_reported_on_a_host_without_the_database`.
- Finding (should): "consumers stay refused" was printed for DESTRUCTIVE and `REQUIRED-BY: none` files.
  - Fix: the refusal clause prints only when the file names services and is not DESTRUCTIVE; DESTRUCTIVE files get no apply command.
  - Evidence: `test_post_pull_report_never_offers_destructive_or_rollback`.
- Finding (nit): fnmatch `*` crosses `/`. Fix: `_glob_match` requires the same depth. Evidence: `test_migration_glob_star_does_not_cross_a_directory`.
- Finding (nit): one checker run per migration. Fix: one `--post-pull-report` run for all files.
- Finding (nit): DB-down fallback ignored header apply scripts. Fix: the report now uses `apply_command()` in every case, which names the header script.
- Finding (nit): `lru_cache` key differs for the default argument. Harmless; left as is.

## Restart required

```text
No restart required. The post-merge hook reads these scripts from the checkout on the next pull.
```

## Risks / concerns

- Severity: low
  - Concern: a real `orion-sql-db` compose change still recreates Postgres automatically on pull.
  - Mitigation: intended (the compose file says such changes need a recreate). Whether stateful services should ever auto-recreate is a separate decision for Juniper.
- Severity: low
  - Concern: if the checker script itself is missing or crashes, the shell fallback prints the raw `psql` form, which ignores a header apply script.
  - Mitigation: the fallback line says to read the header first and never offers a rollback file. A database outage alone does not hit this path; the checker still prints the exact command.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
