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
| After the two field-tick indexes | 31,505 | −96.9% |
| **After the receipts fix (final)** | **7,715** | **−99.2%** |

`substrate_proposal_frames` and `substrate_reduction_receipts` are both gone from the top-8 entirely. Nothing above 5,444/sec remains.

Host I/O pressure, `/proc/pressure/io`:

```
at the start of the D2 investigation:   full avg60 = 22.29
after PR #1745:                         full avg60 =  2.62   (avg300 still 13.02, mid-decay)
now:                                    full avg60 =  0.57   (avg300 1.70)
```

athena was I/O-stalled roughly 22% of wall time. It is now stalled well under 1%.

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
- **Severity: low.** Three of the four `created_at` indexes added by PR #1745 are near-unused: lifetime `idx_scan` is 1 (`proposal_frames`, 19 MB), 10 (`policy_decision_frames`, 19 MB), 13 (`feedback_frames`, 10 MB) against 98,979 (`execution_dispatch_frames`, 23 MB). ~48 MB of index doing almost nothing. Recorded rather than dropped: PR #1745 added all four as a set one day ago, and removing three of them is a decision about that patch's intent, not a drive-by cleanup on this one. Parked.

## Review findings fixed

Review ran in a subagent against the pushed branch. It cleared the three angles I was most worried about and found one real defect **in my own PR report**.

- **Finding (HIGH): the PR report justified skipping `substrate_reduction_receipts` with a claim that is false.** I wrote that fixing it "would need an expression index on a nested JSON path." It does not — the table has carried a real `reducer_name` column all along, making those two call sites *the identical defect this PR fixes*.
  - **Fix:** verified the claim myself rather than taking the review's word for it (0 disagreeing rows across all 9,345, the 4,138 NULLs coincide exactly, `max(jsonb_array_length(state_deltas)) = 1` so the `[0]` subscript hides nothing), then did the actual fix: swapped both call sites in `services/orion-attention-runtime/app/store.py:130,253` to `WHERE reducer_name = :reducer_id` and added `idx_substrate_reduction_receipts_reducer_name (reducer_name, created_at)` — one index serving both the DESC read and the ASC cursor read.
  - **Evidence:** `EXPLAIN` before: Seq Scan, 3,254 buffers, 37.4 ms, 9,948 rows discarded. After: `Index Scan Backward`, **2 buffers, 0.088 ms**, no Sort node. Deployed and verified: `orion-athena-attention-runtime` up, frames saving, no errors.
  - This is why the finding mattered beyond the sentence: left alone, a durable doc would have told the next agent an expensive fix was required when a one-line swap was.

- **Finding (MEDIUM): the compose comment explains what `shared_preload_libraries` buys but never mentions `CREATE EXTENSION`.** The flag only loads the `.so`; the view does not exist until the extension is created per-database.
  - **Fix:** added the exact `CREATE EXTENSION` command to the compose comment block, with a note on the failure it prevents. The compose file is what gets read months later; the PR report is not.
  - **Evidence:** `services/orion-sql-db/docker-compose.yml`, comment above the flag.

- **Finding (LOW): the migration justified the composite index with a case live data contradicts.** I argued it earns its keep for ticks that produced more than one frame. Max frames per tick is **1** across 544,401 ticks on both tables — the guard's entire job is preventing that case.
  - **Fix:** rewrote the justification to the true one (~8 bytes/entry to make the ORDER BY structurally free) and recorded the contradicting numbers in the file, rather than deleting the wrong reasoning silently.
  - **Evidence:** confirmed independently — the proposal plan on the composite has **no Sort node**; the attention plan on its bare index does.

- **Finding (LOW): "three sibling tables already index this column" reads as evidence of value; it is only evidence of convention.** Two of the three cited siblings have lifetime `idx_scan` of 1 and 0.
  - **Fix:** demoted it to context in the migration comment and pointed at the 18,515-block seq scan as the actual argument.

- **Finding (LOW): a test docstring claimed a composite index that half its subjects do not have.** `substrate_attention_frames` has a bare `(source_field_tick_id)` index.
  - **Fix:** docstring now states which table has which, and that attention's live plan does carry a Sort node.

- **Cleared, not defects:** the column swap (the review traced the single writer, found the column is `not null` and bound from the same pydantic attribute as the JSON, and noted the stronger argument I had missed — the attention runtime's *own* reader at `services/orion-attention-runtime/app/store.py:427` has always filtered the column, so the swap makes the two agree rather than inventing new semantics); `_captured_sql` shared state (no leak — `_engine` is a per-instance attribute and every test builds its own store); and the migration's transaction safety (no `BEGIN`/`DO` block, `CREATE INDEX CONCURRENTLY` is safe, VERIFY query re-run live).

- **Independently swept for the same defect elsewhere:** the review checked all 116 `->>` uses in `services/`, `orion/`, and `scripts/` against `information_schema.columns` on the live DB. `substrate_reduction_receipts` was the only miss. Every other JSON filter is genuinely JSON-only (no column equivalent) or operates on a `jsonb_array_elements` lateral where no column is possible. A targeted grep for `->> 'source_field_tick_id'` now returns only comment text.

## An error of mine, corrected mid-patch

After deploying proposal-runtime I saw the "no attention frame for this tick" rate jump from a ~16-23%/hour baseline to 50% in the partial hour, which is the exact failure the column swap could have caused. It did not.

```
missed_ticks  attn_exists_by_column  attn_exists_by_json
          86                     80                   80
```

Both predicates agree on every one of the 86, and the per-minute series decays back to baseline within five minutes of the restart (26/29 → 19/28 → 16/29 → 11/28 → 0/4). It is a restart race — proposal-runtime briefly running ahead of attention-runtime — not a predicate change. Recorded because the number looked exactly like the regression I was watching for, and "it decayed" alone would not have been enough to conclude that.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1751
