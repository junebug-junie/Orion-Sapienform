## Summary

Orion now acts on Juniper's "someone else" answers on walkway picture cards. Before this patch it only recorded them.

- Every crop the clusterer assigns now stores which individual it went to (`vision_crop_observation.individual_id`). Older crops stay empty. There is no backfill.
- When Juniper marks a picture "someone else", Orion moves that picture into a new individual. Close face matches move with it. Body-only crops never move with it. The side she confirmed keeps its id, its name, and its substrate node.
- A typed name becomes the new individual's label. Orion also adds a "same" link to an existing individual with that name. It never merges crops.
- Every split writes a "different" link (cannot-link) back to the source. The online matcher honours cannot-links.
- Every split is recorded with its exact before and after state in `vision_individual_split`. `scripts/undo_vision_split.py` reverses one split. It is a dry run by default.
- Kill switch: `VISION_SPLIT_ON_VERDICT_ENABLED`. It ships on.

## Outcome moved

Problem: a card about one person could show someone else's pictures under that person's name. Fix: after a split, the source's sightings no longer point at moved crops' thumbnails, so the next ask card cannot show that picture under the source's name.

This patch fixes the record, not recognition. The design doc's forward replay shows that new crops of the split-off person still join the big mixed individual by body embedding. Fixing that needs a separate change.

## Current architecture

- `services/orion-sql-writer/app/vision_individuals.py` clusters walkway crops into individuals every 60 s. It does not store crop membership.
- Picture verdicts went to `vision_ask_picture_verdict` as a record only.
- Pair verdicts (`vision_individual_pair_verdict`) linked or separated individuals. The matcher never read them.

## Architecture touched

- orion-sql-writer only. There are no bus or schema changes, and no new channels.
- New module `app/vision_split.py`. It holds the pure rules (`plan_split`, `verdict_groups`, `group_sightings`, `face_threshold_calibration`) and the DB pass (`apply_splits`, `split_one`, `undo_split`).
- `apply_batch` has a new cannot-link rule and returns `crop_individual` (the crop-to-individual map). `_one_stream` writes that map in the batch's own transaction.
- `run_one_individuals_cycle` runs `apply_splits` after `_apply_answers`, in its own transaction. Each split runs in its own savepoint.
- `_check_tables` now requires the new migration. A writer that is deployed before the migration backs off with a clear error. It does not crash-loop.

## Files changed

- `services/orion-sql-db/manual_migration_vision_split_v1.sql`: adds the nullable `individual_id` column, a CONCURRENTLY index, and the `vision_individual_split` table. It has no BEGIN/COMMIT.
- `services/orion-sql-writer/app/vision_split.py`: the split, undo, and calibration logic.
- `services/orion-sql-writer/app/vision_individuals.py`: membership write, cannot-link rule, migration check, split call, config fields.
- `services/orion-sql-writer/app/vision_individuals_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: the new keys.
- `scripts/undo_vision_split.py`: the undo CLI (`--list`, dry run, `--apply`).
- `services/orion-sql-writer/tests/test_vision_split.py`: pure tests and real-Postgres tests.
- `services/orion-sql-writer/evals/run_vision_split_replay_eval.py`: the read-only replay eval against live data.
- `services/orion-sql-writer/README.md`: documents the behaviour.
- `.github/workflows/orion-sql-writer-tests.yml`: runs the new tests in the unit lane and the Postgres lane, and applies the new migration statement by statement.

## Schema / bus / API changes

- Added: `vision_crop_observation.individual_id` (nullable), index `vision_crop_observation_individual_idx` (CONCURRENTLY), and table `vision_individual_split`.
- Removed: none.
- Renamed: none.
- Behavior changed: a 'different' picture verdict now moves crops. A 'different' pair verdict now affects face matching (the cannot-link rule).
- Compatibility notes: the new writer refuses to cluster until the migration is applied (`MigrationMissing` backoff). The migration is additive and passes `auto_apply_refusals` with no refusals, so the post-pull applier applies it without a transaction wrapper.

## Env/config changes

- Added keys: `VISION_SPLIT_ON_VERDICT_ENABLED=true`, `VISION_SPLIT_FOLLOW_MIN_FACE_COSINE=0.60`, `VISION_SPLIT_FOLLOW_MARGIN=0.10`, `VISION_SPLIT_MAX_MOVE_FRACTION=0.5`, `VISION_INDIVIDUALS_CANNOT_LINK_MARGIN=0.10`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes. The compose file passes the kill switch.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-sql-writer`: yes, all 5 keys were added.
- skipped keys requiring operator action: none. These keys were already diverged and were not changed: `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON`.

## Tests run

```text
PYTHONPATH=.:services/orion-sql-writer ORION_TEST_PG_URI=<throwaway postgres:16> pytest -q \
  services/orion-sql-writer/tests/test_vision_split.py (+ confirm_asks, individuals, faces_stationary,
  zones_thumbs_cursor, rhythm, walkway_reducers_eval, route_map_completeness, tests/test_post_pull_apply_migrations.py)
-> 268 passed (before review); after review fixes: 137 passed in the vision set incl. 32 split tests, 0 skipped
Mutation checks: disabling cannot-link -> 2 fail; dropping the follow margin -> 2 fail; skipping the exact
restore -> 1 fail; letting body-only crops follow -> 10 fail.
```

## Evals run

```text
python services/orion-sql-writer/evals/run_vision_split_replay_eval.py --live <live, READ ONLY txn>
replay fidelity: 40/40 live person individuals rebuilt exactly from 4988 crops (0 with stored membership)
  ask-9dd6a5ad source 62cb5fcc: moved 2 + 0 face followers, stay 754, body cos 0.977; live without backfill: 2 + 0
  ask-64490457 source f5242d40: moved 1 + 3 face followers, stay 271, body cos 0.943, face cos 0.744; live without backfill: 1 + 0
face threshold calibration: HOLD (same=2, different=7)
REPRODUCED
```

## Docker/build/smoke checks

```text
Migration applied to a throwaway postgres:16-alpine with psql (twice, idempotent) and with the CI
statement-by-statement path (twice). Not deployed (task scope: no deploy/restart).
Live path: UNVERIFIED until deploy.
```

## Review findings fixed

Code review ran in a subagent. It found 1 must-fix, 7 should-fix and 5 nits.

- Finding (must-fix): undo did not take the individuals advisory lock. A running cycle could overwrite a just-committed undo, or re-create the deleted individual.
  - Fix: `undo_split` takes `pg_advisory_xact_lock(INDIVIDUALS_LOCK_KEY)` first.
  - Evidence: `undo_split` now opens with that call.
- Finding: source sightings kept the moved crops' time span, and the source's seen fields never changed.
  - Fix: a sighting whose remaining crops are all known members gets exact `started_at`/`ended_at`/`dwell_sec`. With exact membership, the source's `first_seen_at`, `last_seen_at` and `distinct_days` are recomputed.
  - Evidence: `test_split_moves_the_named_picture...` asserts the shrunk span and the new seen fields.
- Finding: undo silently deleted Juniper's later name or pair verdicts about the new individual.
  - Fix: undo refuses in both cases. It now expires open asks about the new individual instead of only listing them.
  - Evidence: `test_undo_refuses_when_juniper_named_or_judged_the_new_individual_later`.
- Finding: undo hit a foreign-key error when the source had been pruned.
  - Fix: undo now refuses cleanly.
  - Evidence: there is no test for this case. It is a simple guard in `undo_split`.
- Finding: a failing split was retried every tick, with no end state.
  - Fix: a failure writes one `skipped` row with `reason="error: ..."`.
  - Evidence: `test_a_failing_split_leaves_one_error_row_not_a_retry_loop`.
- Finding: a majority refusal fired even when only face followers pushed the split over the limit.
  - Fix: Orion drops the followers and moves only the named pictures.
  - Evidence: `test_followers_that_push_a_split_over_the_limit_are_dropped_not_the_answer`.
- Finding: `face_blocked_by_cannot_link` also counted crops that the body rule sent to the same individual anyway.
  - Fix: it counts only crops whose destination actually changed.
  - Evidence: `test_cannot_link_counter_only_counts_crops_it_actually_rerouted`.
- Finding (nit): one crop could be both kept and moved on the same card.
  - Fix: the split is skipped with `reason=contradictory`.
  - Evidence: `test_a_picture_both_kept_and_moved_is_contradictory`.
- Finding (nit): the confirm-asks eval's comments listed the old migrations.
  - Fix: comments updated.
- Not changed, stated here instead:
  - The kill switch does not let the writer run without the migration. Deploy runs after the post-pull auto-apply, and `safe_docker_build.sh up` refuses while the migration is unapplied.
  - The first deploy acts on every past "someone else" answer in one cycle. That is intended: today it means asks 9dd6a5ad and 64490457.
  - The CI migration loader splits on `;` after stripping comments. That is enough for this file.
  - Legacy crops never supply a replacement thumbnail. The thumbnail becomes NULL rather than risk showing someone else.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull hook applies the migration (auto-apply safe). If it does not, apply it by hand first:
`docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_split_v1.sql`

## Risks / concerns

- Severity: medium
  - Concern: there is no membership backfill, so f5242d40 splits as 1 + 0 live, not 1 + 3. The 3 face followers are older than stored membership and cannot follow.
  - Mitigation: the eval prints both numbers. The backfill (design Missing question 3) needs its own approval under the AGENTS.md section 14 backfill protocol.
- Severity: medium
  - Concern: the split does not fix recognition. New crops of the split-off person still join the big individual by body embedding.
  - Mitigation: none in this patch. That is the next change. Watch new crops per name C individual for 7 days.
- Severity: low
  - Concern: the named new individual gets a label but no substrate entity node. The ask-answered listener only names individuals on the card.
  - Mitigation: follow-up. Undo has no node to retire.
- Severity: low
  - Concern: the source keeps `distinct_days`, `first_seen_at` and `last_seen_at`. They cannot be recomputed without full membership.
  - Mitigation: these values only drive ask eligibility, and a split removes at most 50% of crops.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
