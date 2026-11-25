## Summary

- The heartbeat seat rollup counted one harness atom twice when its two bus copies fell in two different 5-minute buckets. It now counts each event id once, in the bucket of its first copy. The ledger does the same. (#3900)
- The cause is proven from code and live data. orion-harness-governor sends `exec_result_assembled` twice with one event id: at motor close, and again at finalize, 20 s to ~2.5 min later, stamped with the finalize time.
- `/health` now shows how late atoms arrive, per seat (p50, p95, max), plus late drops per seat and cross-bucket duplicates. (#3901)
- Each stored rollup row now has `late_count`: how many of its atoms arrived after the bucket ended, inside the flush grace. It is a new nullable column, added by an additive migration. (#3901)
- The replay eval now feeds the second harness copy that the bus really carries. Before this, it fed one ledger row per event id, so it could not see the bug.

## Outcome moved

The rollup and the ledger agree exactly again. On 11-25 04:15-05:15Z the replay found 4 cells where the rollup had 1 atom more than the ledger. After the fix it finds 0. Those 4 cells are the same 4 cells, with the same numbers, as the live `--mode table` mismatches from the deployed code.

## Current architecture

- `services/orion-heartbeat/app/seat_rollup.py` sums each seat's routed atoms per 5-minute bucket, keyed by `emitted_at`. It deduped event ids only inside one bucket (`_Bucket.event_ids`).
- `orion/grammar/ledger.py` (`_bulk_insert_events`, line 223) inserts `grammar_events` with `on_conflict_do_nothing(event_id)`. The first copy wins.
- `orion/harness/grammar_emit.py`: `build_harness_grammar_events` (line ~497) stamps every event in the motor batch with one `emitted_at = now()` at motor close. `build_harness_grammar_finalize_events` (lines 600-620) re-sends `exec_result_assembled` (same `atom_id`, so the same `event_id` from `_event`, line 476-478) and the first `exec_result_emitted`, with `emitted_at = now()` at finalize. The caller is `_emit_finalize_lifecycle_grammar` in `services/orion-harness-governor/app/bus_listener.py` (lines 305-350).
- No per-atom arrival lag was measured. Rows had no late count.

## The proven cause (#3900)

1. Harness `emitted_at` is the batch publish time, by design. Per-atom time goes in `observed_at` (`_put_atom`, `grammar_emit.py` lines 130-151). The ledger's `created_at` is `observed_at` (`orion/grammar/ledger.py` line 33-34, `_created_at`). That is why harness rows show `created_at` minutes before `emitted_at`. That gap is not a bug and does not cause the mismatch.
2. The finalize batch re-sends `exec_result_assembled` with the same event id and a later `emitted_at`.
3. The ledger keeps the motor copy. The old rollup kept the motor copy in its bucket and also counted the finalize copy when the finalize copy fell in the next bucket.

Live evidence, 11-25:

| Trace (correlation id) | Motor publish | Finalize publish | Mismatch cell |
|---|---|---|---|
| 45d7d6ef | 04:19:55 | 04:20:21 | 04:20 (45 vs 44) |
| 51679783 | 04:29:48 | 04:31:59 | 04:30 (64 vs 63) |
| 30fad397 | 04:59:54 | 05:00:18 | 05:00 (45 vs 44) |
| 9effdb1d | 05:09:46 | 05:10:18 | 05:10 (29 vs 28) |
| 96dd0453 | 05:33:51 | 05:35:31 | 05:35 (178 vs 177) |

Publish times come from the `harness_lifecycle_grammar_published` log lines of `orion-athena-harness-governor`. The mismatch cells come from `--mode table` over 03:40-05:40Z: 5 mismatches in 110 cells, all seat 4. The ledger rows for 45d7d6ef and 51679783 show `exec_result_assembled` at the motor time and `exec_result_emitted` at the finalize time. Every motor/finalize pair inside one bucket was already caught by the old per-bucket dedup (`duplicate_atoms_ignored` on `/health`).

## Why the fix is in the rollup, not the harness

The re-send is intended. The finalize step re-records `exec_result_assembled` with the real compliance verdict and step counts (see the comment in `_emit_finalize_lifecycle_grammar`). Raw bus consumers can read that update. Changing the harness would change what every grammar consumer sees. The rollup's job is to match the ledger, and the ledger keeps the first copy. So the rollup now keeps the first copy too. The harness is not changed.

Note, not fixed here: the ledger drops the finalize copy's refreshed content, because the first write wins. That is a separate question about the harness and ledger contract.

## Architecture touched

- orion-heartbeat: `app/seat_rollup.py`, `app/service.py`, `evals/seat_rollup_match.py`, `README.md`.
- Contract: `orion/schemas/heartbeat_seat_rollup.py` (new optional field), `orion/bus/channels.yaml` (comment).
- orion-sql-writer: the model gets a nullable `late_count` column.
- orion-sql-db: a new additive migration.

## Files changed

- `services/orion-heartbeat/app/seat_rollup.py`: one `event_id -> first bucket` map across buckets instead of a per-bucket set. Ids are kept 1 h after their bucket closes, with a hard cap of 200,000. A late or future first copy is remembered too. Per-seat lag window (last 4096 first copies). Per-row `late` counts. `late_by_seat`. `duplicates_cross_bucket`. `lag_stats()`.
- `services/orion-heartbeat/app/service.py`: no lag is recorded when `emitted_at` is missing (fallback to receive time). New `/health` fields.
- `services/orion-heartbeat/tests/test_seat_rollup.py`: regression test from the live 04:19:55/04:20:21 trace, plus tests for dedup across a published bucket, a late first copy, retention, the cap, lag percentiles and `late_count`.
- `services/orion-heartbeat/evals/seat_rollup_match.py`: `with_bus_copies()` re-feeds each harness trace's `exec_result_assembled` just before its `exec_result_emitted`, at that atom's `emitted_at`. `--no-bus-copies` turns it off.
- `orion/schemas/heartbeat_seat_rollup.py`: `late_count: Optional[int] = None`, and in `HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS`.
- `services/orion-sql-writer/app/models/heartbeat_seat_rollup.py`: nullable `late_count` column.
- `services/orion-sql-writer/tests/test_heartbeat_seat_rollup_sql_shape.py`: the v2 migration is checked (header, `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` only, no `NOT NULL`). A payload without `late_count` still validates. A measured value lands in the column.
- `services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v2_late_count.sql`: `ALTER TABLE heartbeat_seat_rollup_5m ADD COLUMN IF NOT EXISTS late_count integer;`, with header `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`.
- `services/orion-heartbeat/README.md`, `orion/bus/channels.yaml`: docs.

## Schema / bus / API changes

- Added: `HeartbeatSeatRollupV1.late_count` (optional, default `None`, `ge=0`). `/health` `seat_rollup`: `arrival_lag_by_seat`, `late_atoms_dropped_by_seat`, `duplicate_atoms_cross_bucket`.
- Removed: none.
- Renamed: none.
- Behavior changed: a repeated event id in a later bucket is now a duplicate, not a new count. `duplicate_atoms_ignored` now includes cross-bucket copies.
- Compatibility notes: this is a consumer-first change. The schema has `extra="forbid"`. An old sql-writer rejects a payload that has `late_count`. Deploy orion-sql-writer (after the migration) before orion-heartbeat. A new sql-writer accepts old heartbeat payloads (`late_count` reads `None`).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider services/orion-heartbeat/tests
  -> 209 passed
PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_heartbeat_seat_rollup_bus_catalog.py
  -> 4 passed
(cd services/orion-sql-writer) python -m pytest -q tests/test_heartbeat_seat_rollup_sql_shape.py
  -> 9 passed
Regression check on the old code (origin/main seat_rollup.py + service.py, new test):
  test_harness_finalize_copy_in_the_next_bucket_is_not_counted_twice -> FAILED (assert 2 == 1)
scripts/check_migration_required_by.py --base origin/main -> OK (1 new migration file)
scripts/check_definition_drift.py --gate -> PASS
scripts/check_metric_lineage.py --gate -> PASS
scripts/check_integration_seam.py --require-base -> OK
scripts/check_env_key_single_source.py -> OK
scripts/check_sentience_instruments.py --static-only -> All claims hold
scripts/apply_post_pull_migrations.py --dry-run (v2 file) -> NOT APPLIED, would apply (single transaction, lock_timeout=10s)
```

## Evals run

```text
services/orion-heartbeat/evals/seat_rollup_match.py --mode replay --hours 1 --end 2026-11-25T05:15:00Z

Before (origin/main rollup code, new eval):   60 cells, 4 mismatches, 14 bus copies fed -> MISMATCH
  04:20 seat 4: rollup 45 vs ledger 44
  04:30 seat 4: rollup 64 vs ledger 63
  05:00 seat 4: rollup 45 vs ledger 44
  05:10 seat 4: rollup 29 vs ledger 28
Before, --no-bus-copies (old eval behaviour): 60 cells, 0 mismatches (the old eval could not see the bug)

After (this branch):
  04:15-05:15Z: 7056 atoms fed, 14 copies, 60 cells, 0 mismatches, cross-bucket duplicates 4 -> EXACT MATCH
  04:00-05:00Z: 7106 atoms fed, 13 copies, 60 cells, 0 mismatches, cross-bucket duplicates 2 -> EXACT MATCH
  04:40-05:40Z: 7011 atoms fed, 11 copies, 60 cells, 0 mismatches, cross-bucket duplicates 3 -> EXACT MATCH

Live --mode table on the deployed (old) code, 03:40-05:40Z: 110 cells, 5 mismatches (the 5 rows in the table above).
```

The replay's lag numbers are the eval's fixed 0.05 s receive delay, not real lag. Real per-seat lag is UNVERIFIED until deploy.

## Docker/build/smoke checks

```text
Not run. No deploy (per task). No Dockerfile, compose, requirements or env change.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on `git diff origin/main...HEAD`. It found one material issue, three minor issues and three nits.

- Finding (material): the deploy was not consumer-first. The new heartbeat always sends `late_count`. The old sql-writer validates with `extra="forbid"`, so it rejects those rows. The migration header named only orion-sql-writer, so the gate let the heartbeat deploy before the column existed.
  - Fix: the header is now `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-heartbeat`. The deploy gate now refuses to start the heartbeat until the column exists. The deploy order is stated in the migration and in this report.
  - Evidence: `check_migration_required_by.py --base origin/main` OK. The shape test asserts the new header.
- Finding (minor): eval `with_bus_copies` grouped trace-less rows together, could put a copy before its first copy, and made one copy per `exec_result_emitted` row.
  - Fix: rows without a trace id are skipped. A copy is made only when it is not earlier than the first copy. There is one copy per trace.
  - Evidence: `test_eval_bus_copies_one_per_trace_never_ahead_of_the_first_copy`. The replay numbers did not change (14/13/11 copies, 0 mismatches).
- Finding (minor): an event id stamped far in the future stayed in the dedup map until the cap evicted it.
  - Fix: the prune also drops ids more than 1 h past the open bucket.
  - Evidence: `test_future_atoms_do_not_enter_the_lag_and_far_future_ids_are_pruned`.
- Finding (minor): future-dropped atoms put negative lag values into the arrival lag.
  - Fix: lag is recorded only for counted and late atoms.
  - Evidence: same test (`n == 0` after a future atom).
- Finding (nit): the late-atom log line used a hard-coded 300.
  - Fix: it now uses `HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC`.
  - Evidence: 211 heartbeat tests pass.
- Not changed (nits, no real cost): the cap eviction cost only matters when flushes stop. The dict copy once per 5 minutes is about 7,000 entries.

## Restart required

Apply the migration first, then sql-writer, then heartbeat, from the clean primary main checkout after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v2_late_count.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-heartbeat up -d --build
```

After deploy, check: `curl -s localhost:7251/health` shows `seat_rollup.arrival_lag_by_seat` and `duplicate_atoms_cross_bucket`. After 1 h, `seat_rollup_match.py --mode table --hours 1` shows 0 mismatches, and new rows have a non-null `late_count`.

## Risks / concerns

- Severity: low
  - Concern: deploy order. The post-pull rebuild starts services in name order, so orion-heartbeat comes up before orion-sql-writer. Between the two starts, the old sql-writer rejects rows that carry `late_count` (`extra="forbid"`). Those rows are lost. The channel is pub/sub, and a consumer rejection is not retried.
  - Mitigation: the migration now names orion-heartbeat, so the heartbeat never starts before the column exists. A lost row reads as "unknown", not 0. `scripts/demon_network_replay.py` falls back to `grammar_events` for any bucket with no rollup rows. To lose nothing, run the three commands above in that order.
- Severity: low
  - Concern: a harness finalize run with `emit_result=False` sends `exec_result_assembled` again without `exec_result_emitted`. The replay cannot rebuild that copy from the ledger, so the eval does not feed it. The live rollup still dedups it.
  - Mitigation: `--mode table` after deploy covers the live path.
- Severity: low
  - Concern: the ledger drops the refreshed finalize content of `exec_result_assembled` (first write wins). Not changed here.
  - Mitigation: noted above for a later decision.
- Issues opened: none. Issues closed: #3900, #3901 (`Fixes`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3907

🤖 Generated with [Claude Code](https://claude.com/claude-code)
