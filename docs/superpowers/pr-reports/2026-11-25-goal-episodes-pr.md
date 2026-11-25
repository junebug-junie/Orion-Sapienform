## Summary

Follow-up to #3929 for #3924, approved by Juniper on 2026-11-25. Orion's goals are now stored as episodes. An episode is one run of Orion wanting the same target. Before, the table stored one row for every goal message.

- The attention runtime re-sends the goal it holds about every 2 seconds. Live, 974 goal messages in 35 minutes were 2 runs of wanting (`world:chat`, then `world:reading`). One row per message stored mostly repeats.
- orion-sql-writer now folds goals into `field_goal_episode`. A goal joins the newest episode when that episode is for the same target and its `last_ts` is within 10 minutes. Otherwise the goal opens a new episode. `repeat_count` counts the goals folded in.
- A goal that is delivered twice is not counted twice. A small table of seen `artifact_id`s (`field_goal_episode_artifact`, kept for 48 h) catches repeats.
- One Postgres advisory lock covers the whole fold, so two writes at the same time cannot lose a count or open two episodes.
- `field_goal_provenance_log` has no writer now (kill means kill, no dual-write). The table and its rows stay until Juniper drops them (command below).

## Outcome moved

- Goal history is one row per want, not one row per tick. The goal mix by target and the time spent on each target come from a short query.
- Streak and storage growth (#3928) are solved: `repeat_count` replaces the streak, and episodes grow with changes of want, not with tick rate.

## Current architecture

#3929 (merged and deployed 07:30Z): route `memory.field_goals.proposed.v1` -> `FieldGoalProvenanceLogSQL`, insert-only, one row per message, table `field_goal_provenance_log`. At 07:47Z the live table had 0 rows. The writer subscribed at 07:33:58Z, and the last goal was published at 07:29:12Z. Nothing was lost, because nothing was sent in that window.

## Architecture touched

orion-sql-writer only: one route, two models, two tables, one fold module. No change to the producer, the schema, or the other consumers. The integration gate is unchanged (this is a sink).

## Files changed

- `services/orion-sql-writer/app/field_goal_episode.py`: new. The fold rule, `GOAL_EPISODE_GAP` (10 min, with a comment on why), `ARTIFACT_DEDUP_WINDOW` (48 h), and the advisory lock.
- `services/orion-sql-writer/app/models/field_goal_episode.py`: new. `FieldGoalEpisodeSQL` and `FieldGoalEpisodeArtifactSQL`.
- `services/orion-sql-writer/app/models/field_goal_provenance_log.py`: deleted (no writer).
- `services/orion-sql-writer/app/models/__init__.py`, `app/worker.py`, `app/settings.py`: route to `FieldGoalEpisodeSQL`. A dedicated handler branch validates the goal, refuses a missing `artifact_id`, and folds in one transaction. The old normalizer and the insert-only entry are removed.
- `services/orion-sql-writer/.env_example`: the route value is `FieldGoalEpisodeSQL`.
- `services/orion-sql-db/manual_migration_field_goal_episode_v1.sql`: new and additive (2 tables, 3 indexes).
- `services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql`: the header is now `REQUIRED-BY: none retired ...`, plus a RETIRED line. The DDL is unchanged.
- `services/orion-sql-writer/tests/test_field_goal_episode_sql_shape.py`: new, replaces `test_field_goal_provenance_log_sql_shape.py`.
- `services/orion-sql-writer/tests/test_field_goal_episode_integration.py`: new. Real Postgres: migration, concurrency, A-B-A and gap.
- `.github/workflows/orion-sql-writer-tests.yml`: the integration test runs in the must-not-skip Postgres job.
- `orion/bus/channels.yaml`, `services/orion-sql-writer/README.md`, `tests/test_autonomy_goals_bus_catalog.py`: wording.

## Schema / bus / API changes

- Added: tables `field_goal_episode` and `field_goal_episode_artifact`.
- Removed: the writer of `field_goal_provenance_log` (the table is kept).
- Renamed: route target `FieldGoalProvenanceLogSQL` -> `FieldGoalEpisodeSQL`.
- Behavior changed: one row per goal episode instead of one row per goal message.
- Compatibility notes: no bus or schema change. The channel consumer list is unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: the `SQL_WRITER_ROUTE_MAP_JSON` value (one route target).
- local `.env` synced: yes. I took a backup first, then checked that the route target was the only difference, then ran `sync_local_env_from_example.py --all-keys --force orion-sql-writer`. Only that line changed.
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_field_goal_episode_sql_shape.py -> 12 passed
  routing (code default, .env_example, stale-env guard), registry consumer, migration == models
  + index names + allowlist, GAP == 10 min, 420 repeats -> 1 episode repeat_count 420,
  gap > GAP -> new episode (exactly GAP folds), A -> B -> A -> 3 episodes, duplicate artifact
  -> no bump (and _write_field_goal_episode returns False), duplicate of an episode opener
  after another target -> no new episode, malformed -> fallback + validation log + no raise,
  no artifact_id -> refused, seen-artifact pruning
tests/test_field_goal_episode_integration.py (postgres:15, -m integration) -> 2 passed
  16 threads, 160 goals + 40 concurrent re-deliveries -> 1 episode, repeat_count 160, 40 duplicates
services/orion-sql-writer: pytest tests -> 1378 passed, 287 skipped
tests/test_autonomy_goals_bus_catalog.py tests/test_deploy_migration_gate.py -> 56 passed
check_definition_drift.py --update/--gate -> no definition change on this branch; PASS
check_integration_seam.py OK; check_metric_lineage.py --gate PASS; check_env_template_parity.py PASS
check_migration_required_by.py --base origin/main -> OK
```

Mutation check: each broken version must fail the tests. 11 of 11 were killed. The unchanged code passes in the same harness environment.

```text
no_dedup KILLED            no_gap KILLED            per_target_latest KILLED
no_count_bump KILLED       no_max KILLED            no_last_ts KILLED
gap_strict (< not <=) KILLED   no_prune KILLED      no_artifact_guard KILLED
dup_counts_as_written KILLED   no_advisory_lock (Postgres concurrency test) KILLED 3/3 runs
```

## Evals run

```text
None. The fold rule was replayed over the live producer log (below). The tests above run the real handler.
```

## Docker/build/smoke checks

```text
No image build: requirements and Dockerfile unchanged. Postgres DDL proven in a throwaway postgres:15 container.
```

Replay of the live producer log (06:27Z restart to 07:47Z) through the same 10-minute rule:

```text
goals 974 -> episodes 2
world:chat     06:54:34 -> 07:09:27  420 goals
world:reading  07:09:33 -> 07:29:12  554 goals
```

Size estimate: one episode row is about 1.5 KB (the first payload is about 0.8 KB). At the observed rate (a few episodes an hour at most), that is well under 1 MB a day. The seen-artifact table holds 48 h of ids: about 1,700/h x 48 h = about 80k rows of about 120 B, so about 10 MB, and it stays flat. Before, one row per message was about 40 MB a day.

Goal-mix query (episodes and total minutes by target, last 7 days):

```sql
SELECT field_target_id, target_kind, count(*) AS episodes, sum(repeat_count) AS goals,
       round(sum(extract(epoch FROM last_ts - first_ts)) / 60.0, 1) AS total_minutes,
       round(max(max_salience)::numeric, 3) AS max_salience
FROM field_goal_episode
WHERE first_ts > now() - interval '7 days'
GROUP BY field_target_id, target_kind
ORDER BY total_minutes DESC;
```

Live check after deploy: over one hour, the goals stored should about equal the goals published. Allow a few at the window edges, and count only episodes that overlap the hour.

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT coalesce(sum(repeat_count),0) FROM field_goal_episode WHERE last_ts > now() - interval '1 hour'"
docker logs --since 1h orion-athena-attention-runtime 2>&1 | grep -c field_goal_provenance_published
```

The first number counts every goal in an episode that touched the hour, so it can be higher when an episode started earlier. For an exact match, compare after an episode closes, or run it over the whole time since the deploy. UNVERIFIED: not deployed.

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. It found no must-fix bug. It re-ran the unit and Postgres tests, and its own lock mutant failed the concurrency test.

- Finding (should-fix): deploy order. The live `.env` now names `FieldGoalEpisodeSQL`. An old sql-writer image recreated from main before this merges would send goals to the fallback log. The reviewer suggested removing the route from `.env_example` so each image uses its code default.
  - Fix: tried and reverted. `tests/test_route_map_completeness.py` and `check_env_template_parity.py` both require the operator route map to match the code defaults; both failed. The hazard is stated in Risks and Restart instead.
  - Evidence: with the entry restored, the full suite passes (1378) and parity passes.
- Finding (nit): a late, older goal that folds in does not lower `first_ts`.
  - Fix: documented in the code. `first_*` means the first goal seen, so `first_ts`, `first_artifact_id`, and `first_payload_json` always describe the same goal.
- Finding (nit): dedup lasts only 48 h, so a replay of older goals would count again.
  - Fix: comment on `ARTIFACT_DEDUP_WINDOW`. No such replay path exists today.
- Finding (nit, pre-existing): a goal wrapped in a nested envelope is refused by the `artifact_id` guard.
  - Not changed. The producer never nests, and the old code behaved the same way.

## Restart required

The post-pull rebuild applies the new migration first (it is additive), then rebuilds orion-sql-writer. If it does not, run these from the clean primary main checkout:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_field_goal_episode_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Juniper only, after she has looked at the old table's data. This is a destructive step, so it is not run by the PR:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DROP TABLE IF EXISTS field_goal_provenance_log;"
```

After the DROP, delete `services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql` in a small follow-up PR.

## Risks / concerns

- Severity: low
- Concern: the local `.env` already names `FieldGoalEpisodeSQL` (parity requires it). If anything recreates the old sql-writer image before this merges (for example another PR's post-pull rebuild of sql-writer), it does not know that route. Goals then go to `bus_fallback_log` (about 1,700 rows/h) until this deploys.
- Mitigation: the post-pull rebuild follows the merge right away. Goals are re-sent every ~2 s, so the episode picture is not lost.

- Severity: low
- Concern: one advisory lock serializes every goal write.
- Mitigation: there are about 0.5 writes a second, and each one is a few small statements. The Postgres test runs 16 threads with no trouble.

- Severity: low
- Concern: a goal that arrives out of order, behind another target's episode, opens a short new episode instead of joining its own.
- Mitigation: the producer publishes in order from a single loop. Re-deliveries are caught by the seen-artifact table before the fold.

Issues: #3924 (done by #3929 and this PR). #3928 is commented and closed (streak and retention are solved by episodes; the DROP is listed above).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3940

🤖 Generated with [Claude Code](https://claude.com/claude-code)
