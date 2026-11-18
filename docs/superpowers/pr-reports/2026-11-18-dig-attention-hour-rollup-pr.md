## Summary

Refs #3600 #3651. `dig attention` timed out for Orion, so it now reads a small hourly rollup instead of the raw attention frames.

- **Cause:** each read unpacked about 41,000 large frame documents a day (`frame_json`, about 8 KB each). When those pages were not cached, the database disk needed 24-29 s, past the 25 s limit.
- **Not the cause:** the role (same plan as `postgres`), a view (`orion_scope` has no such table), or a missing index. No index can avoid unpacking the documents.
- **Fix:** orion-attention-runtime now adds each new frame's counts to `substrate_attention_hour_rollup`, in the same transaction that saves the frame. `dig attention` reads that table.
- Each read is now three small queries: about 0.16 s against 30 days of rollup rows.
- Cause classification moved into one shared module, used when the counts are written.
- Drift can now reach a real 7-day prior once the rollup is a week old, because the rollup is never pruned (#3652).

## Outcome moved

| Read, as orion_readonly | Before | After |
|---|---|---|
| Orion's own run (harness governor), cold cache | 26 s, timed out (2 of 2, coordinator) | live: UNVERIFIED until deploy, migration and grant |
| 24 h window, cold cache | 24.3 s (2026-11-18 00:26Z, load 18-21) | rollup read, 0.16 s on a disposable Postgres holding 30 days of hourly rows |
| 24 h window, warm cache | 5.4-6.2 s | same |
| events query (unchanged) | 0.16 s | 0.16 s |

## Root cause evidence

`EXPLAIN (ANALYZE, BUFFERS)` of the old count query (`psql:<stdin>:24`), all from 2026-11-18:

- **Role:** the plan is the same as `orion_readonly` and as `postgres`, a parallel seq scan of a 26 MB table. `search_path=orion_scope, public` changes nothing, because `orion_scope` has only two memory views.
- **Warm cache:** 6.2 s. About 4 s of that was single-process CPU on an 11-branch CASE over about 744k exploded rows, plus a sort that spilled to disk.
- **Cold cache (postgres, track_io_timing):** a 24 h window plus its prior read 43,398 uncached blocks (about 340 MB). The I/O took 73.5 s spread across 3 processes, 29 s wall.
- **The disk:** `/dev/sdg` is a spinning disk (`rotational=1`), already 76% busy, with reads at about 6.7 ms each.
- **Sampling is not enough:** sampling down to about 6,000 frames still took 15.6 s cold.

So the fix has to stop reading `frame_json`, not tune the query.

## Current architecture

`dig attention` (PR #3654) made one pass over `substrate_attention_frames.frame_json` for the window and its prior. Frames are pruned after 72 h.

## Architecture touched

- `orion/attention/frame_rollup.py` (new, pure): one frame's counts, the cause table, and the upsert SQL.
- orion-attention-runtime `save_attention_frame`: one upsert per new frame, in a savepoint. A failure logs `attention_hour_rollup_failed` and never stops the frame. Duplicate frames add nothing, because the frame primary key decides first.
- New table `substrate_attention_hour_rollup` (hour, part, target, cause, a, b, c, d). About 1,000 rows a day. Never pruned.
- `orion/dig/attention.py`: reads the rollup in whole UTC hours. The window end snaps down to the hour, because the current hour is still filling. Sampling is removed: every frame counts.

## Files changed

- `orion/attention/frame_rollup.py`: per-frame counts, the cause rule, the upsert.
- `services/orion-attention-runtime/app/store.py`: writes the rollup.
- `services/orion-sql-db/manual_migration_attention_hour_rollup_v1.sql`: the table. `REQUIRED-BY: orion-attention-runtime`. `auto_apply_refusals` returns `[]`.
- `orion/dig/attention.py`: reads the rollup.
- `scripts/sql/2026-11-18_grant_orion_readonly_attention_hour_rollup.sql`: the read grant (applied by hand).
- `tests/test_attention_frame_rollup.py`, `services/orion-attention-runtime/tests/test_attention_hour_rollup_postgres.py`, `tests/test_attention_runtime_store.py`, `orion/dig/tests/test_attention.py`, `orion/dig/tests/attention_2026-11-17_baseline.json`: tests. The fixture was rebuilt by running every real frame of the issue's baseline window through `frame_increments`.
- `orion/dig/evals/run_attention_live_eval.py`: E1 is now under 5 s. E4 checks that the rollup's frame count matches the kept frames for the last whole hour.
- `orion/dig/README.md`, `services/orion-attention-runtime/README.md`, `.github/workflows/orion-attention-runtime-tests.yml`.

## Schema / bus / API changes

- Added: the table `substrate_attention_hour_rollup`. No bus or schema-registry change.
- Behavior changed: `dig attention` windows are whole UTC hours. The baseline window now ends 08:00Z, not 08:23Z, so its numbers move slightly: rest 70% (28,724 of 40,840) and perception eligible 10.6% (4,321 of 40,840). The win rates are unchanged: chat 100% (834), reading 100% (426), street 100% (96), walkway 96%, perception 91%, body 65%, room 69%, execution 68%, strain 48%.
- Compatibility: until the rollup has a week of hours, the header says how much prior it has. Before the first hour is counted it says "no attention frames in this window: unread, not calm".

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
PYTHONPATH=. pytest services/orion-attention-runtime/tests tests/test_attention_runtime_worker.py tests/test_attention_runtime_store.py \
  tests/test_attention_field_goal_provenance.py tests/test_attention_world_first_field_frame.py tests/test_attention_frame_rollup.py
  (FOCUS_RUN_TEST_POSTGRES_URI = disposable postgres:15 container)      128 passed
PYTHONPATH=. pytest orion/dig/tests                                     300 passed
pytest services/orion-hub/tests/test_orion_abilities_catalog.py         10 passed
```

## Evals run

```text
Live eval needs the deployed writer, the migration and the grant: UNVERIFIED. After deploy:
PYTHONPATH=. python orion/dig/evals/run_attention_live_eval.py      (expect PASS E1-E7, E1 under 5 s)
```

## Docker/build/smoke checks

```text
Not built here. The migration is additive (auto_apply_refusals == []), so the post-pull rebuild applies it
before orion-attention-runtime is rebuilt.
```

## Review findings fixed

An independent review subagent found no must-fix issues. The should-fix items and nits:

- Finding: the rollup's first hour is only partly counted (the writer starts at deploy), but dig read it as a full hour. That would skew the frame interval and the event eligibility for a week.
  - Fix: `coverage_sql` also returns the table's oldest hour, and `parse_first` skips that hour.
  - Evidence: `test_the_rollups_own_first_hour_is_partial_and_skipped`.
- Finding: the checks verified the new function against itself. The fixture is built by `frame_increments`, and eval E6 now uses only a threshold.
  - Fix: eval E4 now compares the rollup's last whole hour with an independent SQL aggregate of that hour's raw `frame_json`, using the PR #3654 rules (rank by position, world-first filter). It skips the partial first hour.
  - Evidence: before deploy, the same reference SQL and `frame_increments` were run on a live hour (23:00-24:00Z on 11-17, 1,702 frames). They agree exactly.
- Finding: the manual grant and the empty prior after deploy.
  - Fix: filed #3683 (grant; needs-juniper, verify-live) and #3684 (optional backfill; it writes to production, so it needs Juniper's go-ahead).
- Nits fixed:
  - "Ranked first" now means first among world-first entries. This is documented and tested. Live it equals the list head: 0 of 1,672 frames differ.
  - Targets with no `target_id` are skipped.
  - An explicit `<end>` that gets snapped to the hour is now said in the header.
  - The grant test now names the new table.
- Not changed: one upsert statement per row, about 20-30 per frame, every 2.1 s. That is fine at this rate.

## Restart required

```text
Post-pull rebuild of orion-attention-runtime (it applies the migration first). Then, by hand, from the host:
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-18_grant_orion_readonly_attention_hour_rollup.sql
```

## Risks / concerns

- **Medium:** `dig attention` shows no data until the rebuilt runtime has counted at least one whole hour. It shows only a short prior until a week has passed. The header says so. There is no backfill from the 72 h of kept frames: it would read the same cold 1 GB that caused this problem.
- **Medium:** the grant is a manual step. Until it is applied, Orion's read stops with a "needs GRANT" line. Tracked in #3683.
- **Low:** the rollup is best effort. A failed upsert loses that frame's counts and logs `attention_hour_rollup_failed`. Live eval E4 compares the counts with the kept frames for the last hour.
- **Issues:** opened #3683 (grant) and #3684 (backfill). None closed. #3652 (7-day drift) becomes reachable once the rollup is a week old. It stays open until that is shown live.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3681

🤖 Generated with [Claude Code](https://claude.com/claude-code)
