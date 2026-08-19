# Index the per-tick dedup lookups that were reading whole tables

## Summary

- Identified the query behind the last large sequential-scan source in `conjourney`. It was **not** a fourth consumer and **not** an anti-join: it was the proposal stage's own per-tick dedup guard filtering an **unindexed column**.
- Added `(source_field_tick_id, generated_at desc)` to `substrate_proposal_frames`. Three sibling frame tables have indexed that column all along; this one was simply missed.
- Fixed a **larger, previously invisible** offender found in the same sweep: `load_attention_frame_for_field_tick` filtered `frame_json ->> 'source_field_tick_id'` while the table carried an index on the real column, so the planner walked the `generated_at` index end to end de-TOASTing every blob.
- Indexed `memory_crystallization_{sources,claims}.crystallization_id` — 848,215 sequential scans against 6 index scans, while the sibling table on the same access path is indexed.
- Added `shared_preload_libraries=pg_stat_statements` to the sql-db compose command. **Not applied** — needs a Postgres restart, listed below.
- Added a regression gate on the query shape, mutation-tested against the real file.

## Outcome moved

Whole-database sequential rows read per second, sampled live over 120 s:

| | tuples/sec | vs. start of arc |
|---|---|---|
| Before PR #1745 (arc baseline) | 1,021,558 | — |
| After PR #1745 (pending markers) | 157,464 | −84.6% |
| **After this patch** | **31,505** | **−96.9%** |

`substrate_proposal_frames` no longer appears in the top-8 at all (was 128,164/sec, 81% of all remaining scan load).

Host I/O pressure at the moment of measurement, mid-decay from the old regime:

```
full avg10=0.22  avg60=2.62  avg300=13.02
```

`avg300` still carries the pre-patch regime; `avg10` is the new one.

Per-query, measured with `EXPLAIN (ANALYZE, BUFFERS)` on the live tables:

| query | before | after |
|---|---|---|
| proposal dedup guard | Parallel Seq Scan, **18,515 blocks (144 MB)**, 52.4 ms | Index Scan, **4 blocks**, 0.104 ms |
| attention dedup guard | Index Scan + filter, **553,906 buffers (~4.3 GB)**, 4,777 ms | Index Scan, **7 buffers**, 0.174 ms |

## Current architecture

Both substrate stages guard against re-doing work for a field tick they have already handled:

```
services/orion-proposal-runtime/app/store.py
  load_proposal_frame_for_field_tick(tick)   -> substrate_proposal_frames
  load_attention_frame_for_field_tick(tick)  -> substrate_attention_frames
```

Each runs once per tick, embedding a distinct `tick_<hash>` literal. Both returned correct results the whole time. Neither logged anything. The only symptom was disk.

## Architecture touched

No contract, schema, bus channel, or service boundary changed. One SQL predicate, three indexes, one Postgres startup flag.

## Files changed

- `services/orion-sql-db/manual_migration_field_tick_lookup_indexes.sql` (NEW): the three indexes, with the measurements that justify each.
- `services/orion-proposal-runtime/app/store.py`: attention lookup filters the column instead of the JSON key.
- `services/orion-sql-db/docker-compose.yml`: `-c shared_preload_libraries=pg_stat_statements`.
- `tests/test_proposal_runtime_store.py`: `TestFieldTickLookupsStayIndexEligible` + `_captured_sql` helper.
- `docs/superpowers/specs/PARKING-LOT.md`: the parked "fourth consumer" item marked resolved, with both of its wrong assumptions corrected in place.

## How it was actually found — and why it took so long

Worth recording, because the obstacle was the *instrument*, not the target.

`pg_stat_activity` is a **per-transaction cached snapshot**. A `DO` loop that samples it 600 times over 30 seconds without calling `pg_stat_clear_snapshot()` re-reads the same frozen instant 600 times and finds nothing. That is why 14 earlier rounds of sampling — and my own first 600-sample run in this session — returned zero rows, and why the parking-lot entry concluded the query was "too fast to catch." It was not. One `perform pg_stat_clear_snapshot()` per iteration caught it inside a single 30-second window.

The second reason it hid: each execution embeds a different `tick_<hash>` literal, so no two executions are textually identical. That defeats naive text grouping as well as sampling. It is precisely the shape `pg_stat_statements` normalises away.

And the attention-frame query was invisible to **every** measurement in this arc for a structural reason: it is an *index* scan, and `pg_stat_user_tables.seq_scan` / `seq_tup_read` do not count index scans. A 4.3 GB, 4.8-second lookup contributed exactly zero to every "rows scanned per second" number in this arc, including the ones in PR #1745. It surfaced only from reading the code path, not from the metrics.

## Schema / bus / API changes

- Added: three indexes (`idx_substrate_proposal_frames_source_field_tick_id`, `idx_mcr_sources_crystallization_id`, `idx_mcr_claims_crystallization_id`).
- Removed / renamed / behaviour changed: none.
- Compatibility: the attention predicate swap is result-identical. Verified live: `where source_field_tick_id is distinct from (frame_json ->> 'source_field_tick_id')` returns **0** across all 99,626 rows, and the column is populated on every one.

## Env/config changes

- Added keys: none.
- `.env_example` updated: not required, no env key changed.
- The only config change is the compose `command:` array, which is checked in.

## Tests run

```text
pytest tests/test_proposal_runtime_store.py -q
8 passed in 0.52s

# mutation test -- reverted the real store.py to the JSON form:
FAILED tests/test_proposal_runtime_store.py::TestFieldTickLookupsStayIndexEligible::test_attention_lookup_filters_on_the_column_not_the_json
1 failed, 7 passed in 0.44s
```

The gate is checked against the real file, not a synthetic fixture: a swap back to `frame_json ->> ...` returns byte-identical results and would otherwise ship silently.

## Evals run

None. Neither `orion-proposal-runtime` nor `orion-sql-db` carries an eval harness, and this patch changes no cognitive behaviour — the two queries return identical rows before and after. The relevant quality signal is the live `EXPLAIN` and rate measurements above, which are reported rather than asserted.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-db config          # compose renders the new -c flag
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
  Container orion-athena-proposal-runtime  Started

docker exec orion-athena-proposal-runtime grep -n 'source_field_tick_id = :field_tick_id' app/store.py
  app/store.py:87
  app/store.py:292      # both lookups column-based in the deployed image

docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_field_tick_lookup_indexes.sql
  CREATE INDEX x3, all three indisvalid = t, 4.1s
```

## Restart required

The three indexes are **already applied and live**. The proposal-runtime code is **already deployed**. Nothing below is needed for the performance win.

The one outstanding item is `pg_stat_statements`, which is postmaster-only and therefore needs a Postgres restart. It is **optional** — this patch answered its question without it. Run when convenient:

```bash
cd /mnt/scripts/Orion-Sapienform-pg-stat-statements
scripts/safe_docker_build.sh orion-sql-db up -d
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
```

## Risks / concerns

- **Severity: low.** Write cost of a 4th index on `substrate_proposal_frames` (~444k rows, actively inserted). Concern: index maintenance on every insert. Mitigation: the table takes roughly one insert per tick, and the read it replaces was 144 MB; the trade is not close. Reversible with a single `DROP INDEX`.
- **Severity: low.** `memory_crystallization_claims` is currently empty (0 rows), so its index is provisioning for a load path that is running 848k times but finding nothing. Worth a separate look at *why* that loader is called that often — it is not a defect this patch introduces, but it is unexplained.
- **Severity: informational.** `substrate_reduction_receipts` is now the largest remaining scan source at 22,574 tuples/sec. **Measured and deliberately not fixed**: `EXPLAIN` shows 3,248 of 3,254 buffers served from cache, so it costs CPU and buffer traffic, not disk, and is not part of the I/O ceiling this arc was chasing. Fixing it would need an expression index on a nested JSON path. Recorded here rather than done.

## PR link

<fill in>
