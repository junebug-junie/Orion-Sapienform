## Summary

Fixes #3924. Orion's goals are now stored. Before this, the only record of a goal was one log line in the attention runtime, and a restart erased it.

- orion-sql-writer subscribes to `orion:memory:goals:proposed` and writes one row per goal to a new table, `field_goal_provenance_log`.
- A row holds the key fields as columns, the full validated payload as jsonb (`payload_json`), the bus envelope id, and the time sql-writer stored it (`received_at`).
- The table is insert-only on `artifact_id`. The producer sets that id once per goal, so a re-delivered goal is skipped, not stored twice.
- `orion/bus/channels.yaml` names orion-sql-writer as a consumer again. The 2026-07-18 removal note is replaced with what the consumer is now.
- Additive migration `services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql` (`-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`).
- Also fixed: `tests/test_autonomy_goals_bus_catalog.py` failed on main because its `GoalProposalV1` fixture still passed `drive_origin`, a field retired on 2026-08-11.

## Outcome moved

The goal-mix re-measure that was impossible on 2026-11-25 (#3295) can now run against stored data instead of container logs. Goal history survives restarts.

## Current architecture

- Producer: `services/orion-attention-runtime/app/worker.py` `_publish_goal` publishes `FieldGoalProvenanceV1` (`orion/schemas/field_goal.py`) on `orion:memory:goals:proposed`, kind `memory.field_goals.proposed.v1`, and logs `field_goal_provenance_published ... streak=N`.
- Consumers: orion-substrate-runtime, orion-spark-concept-induction, orion-world-pulse. All three use goals in memory. None stores them.
- orion-sql-writer was listed as a consumer until 2026-07-18 and removed because it had no code for this channel.

## Architecture touched

- orion-sql-writer only (one route, one model, one table). No change to the producer or the other consumers.
- Contract registry: `orion/bus/channels.yaml` consumer list for `orion:memory:goals:proposed`.
- Integration gate: this is a sink, not a new producer of experience. `scripts/check_integration_seam.py` passes unchanged.

## Files changed

- `services/orion-sql-writer/app/models/field_goal_provenance_log.py`: new `FieldGoalProvenanceLogSQL` model (PK `artifact_id`, named indexes on `received_at`, `ts`, `(field_target_id, ts)`).
- `services/orion-sql-writer/app/models/__init__.py`: export the model.
- `services/orion-sql-writer/app/worker.py`: `MODEL_MAP` entry, `INSERT_ONLY_MODELS`, `_normalize_field_goal_provenance_payload` (builds `payload_json` from the schema's own fields only), and a handler branch that passes only `envelope_id` from the envelope, so the envelope's fresh `correlation_id` cannot overwrite the goal's own.
- `services/orion-sql-writer/app/settings.py`: default route, default subscribe list, and an `effective_subscribe_channels` guard so a stale operator `.env` still subscribes.
- `services/orion-sql-writer/.env_example`: `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` gain the channel and route.
- `services/orion-sql-writer/README.md`: one paragraph on the new table.
- `services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql`: `CREATE TABLE IF NOT EXISTS` + three `CREATE INDEX IF NOT EXISTS`.
- `orion/bus/channels.yaml`: add `orion-sql-writer` to `consumer_services`; replace the stale removal note.
- `services/orion-sql-writer/tests/test_field_goal_provenance_log_sql_shape.py`: new tests.
- `config/metrics/metric_definitions.lock.json`: re-locked; the channel's declared consumers changed (`check_definition_drift.py`, `routing_changed`, intended).
- `tests/test_autonomy_goals_bus_catalog.py`: drop the retired `drive_origin` key; assert sql-writer is a listed consumer.

## Schema / bus / API changes

- Added: table `field_goal_provenance_log`; orion-sql-writer as a consumer of `orion:memory:goals:proposed`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for the producer or the other consumers. `FieldGoalProvenanceV1` is unchanged.
- Compatibility notes: the producer's streak count is not on the schema, so it is not stored. Adding it needs a consumer-first migration (the schema is `extra="forbid"` on three consumers). Filed as #3928.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, values of `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` (one entry added to each).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The default run does not cover these two keys, so `--all-keys --force orion-sql-writer` was run after a check that the local values differed from the template only by this patch's one entry each. Both entries are now in `/mnt/scripts/Orion-Sapienform/services/orion-sql-writer/.env`; no other key changed.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-sql-writer: pytest tests/test_field_goal_provenance_log_sql_shape.py -> 7 passed
  - subscribed + routed (code default, .env_example, stale-env guard), MODEL_MAP, insert-only
  - channels.yaml names orion-sql-writer as consumer
  - migration columns == model columns, index names match create_all, auto_apply_refusals == [], REQUIRED-BY parses
  - one row per goal through handle_envelope; columns, envelope_id, correlation_id not overwritten, payload_json == published JSON
  - re-delivered goal (same artifact_id, new envelope) -> still 1 row
  - malformed payload (salience 7.0, no field_target_id) -> no row, sql_writer_validation_failed logged, one fallback write, no raise
services/orion-sql-writer: pytest tests -> 1373 passed, 285 skipped
tests/test_autonomy_goals_bus_catalog.py -> 4 passed (1 failed on main before the fixture fix)
tests/test_bus_reply_channel_catalog_coverage.py tests/test_deploy_migration_gate.py -> passed
scripts/check_integration_seam.py -> OK (16 organs)
scripts/check_metric_lineage.py --gate -> PASS
scripts/check_definition_drift.py --gate -> PASS after --update (first CI run failed on the intended consumer change)
scripts/check_migration_required_by.py -> OK (1 new migration file)
scripts/check_env_template_parity.py -> PASS
git diff --check -> clean
```

## Evals run

```text
None. This is a storage sink with no quality dimension to grade; the write-path tests above
exercise the real handler. The live acceptance check is below and is UNVERIFIED until deploy.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer config: parity + hostname gates OK; compose config
itself needs the repo-root .env, which only exists in the primary checkout. No image build:
requirements and Dockerfile are unchanged.
```

Pre-deploy live baseline (read-only, 2026-11-25): attention runtime restarted 06:27:57Z; `field_goal_provenance_published` lines from 06:54:34Z to 07:09:46Z = ~420 goals in 15 minutes, all `field_target_id=world:chat`, streak 400+.

Goal-mix query (by target, last 7 days):

```sql
SELECT field_target_id, target_kind, count(*) AS goals,
       round(avg(salience_score)::numeric, 3) AS mean_salience,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM field_goal_provenance_log
WHERE ts > now() - interval '7 days'
GROUP BY field_target_id, target_kind
ORDER BY goals DESC;
```

Live check after deploy (one hour, rows vs log lines; run after the writer has been up for over an hour):

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT count(*) FROM field_goal_provenance_log WHERE received_at > now() - interval '1 hour'"
docker logs --since 1h orion-athena-attention-runtime 2>&1 | grep -c field_goal_provenance_published
```

The two numbers should match (allow a few at the window edges). UNVERIFIED: not deployed.

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. It found no material bug in the production code.

- Finding (should-fix): the redelivery test passed even with dedup broken. Two mutations (model removed from `INSERT_ONLY_MODELS`, so `merge` overwrites; second write raising into the fallback log) both kept `count()==1`.
  - Fix: the test now uses two envelopes and asserts the stored `envelope_id` and `received_at` are the first envelope's, the fallback log is never called, and `_write_row` returns `False` (idempotent skip) on a repeat.
  - Evidence: `test_redelivered_goal_does_not_double_insert` passes; both mutations now fail it (a merge rewrites `envelope_id`; a raise records a fallback).
- Finding (nit): a payload with no `artifact_id` would get a fresh id minted by the schema on every delivery, so redeliveries would be stored twice.
  - Fix: the handler branch refuses such a payload to the fallback log.
  - Evidence: new `test_payload_without_artifact_id_is_refused_not_stored` (0 rows, 2 fallback writes).
- Note: the `drive_origin` fixture edit in `tests/test_autonomy_goals_bus_catalog.py` repairs a test already failing on main (see Summary).

## Restart required

The post-pull rebuild applies the migration first (it is on the additive allowlist), then rebuilds orion-sql-writer. If it does not, apply by hand and rebuild from the clean primary main checkout:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: volume. ~1,700 goals/hour live (~40k rows/day, roughly 40 MB/day) with no pruning.
- Mitigation: filed #3928 to decide retention. The table is cheap to drop or prune; nothing else reads it yet.

- Severity: low
- Concern: the issue asked for streak, but streak is not on the schema, so the table cannot store it.
- Mitigation: #3928 (consumer-first schema field add). `payload_json` will carry it once the schema has it.

Issues opened: #3928. Issues closed: #3924 (by this PR).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3929

🤖 Generated with [Claude Code](https://claude.com/claude-code)
