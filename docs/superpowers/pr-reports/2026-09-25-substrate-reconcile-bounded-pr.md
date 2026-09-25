## Summary

- Three background safety checks were reading most of the database every 15 minutes and never finding anything. They run in the policy, execution-dispatch and feedback runtimes, and each looks for work items whose "still needs processing" flag was cleared by mistake. Together they were the top three I/O statements on athena's Postgres, which is on a spinning disk. Every commit in the database, including the GPU pool's lease writes, queued behind them.
- The 15-minute check now looks only at rows from the last 2 hours. It is one short UPDATE that uses the `generated_at` index.
- The whole-history check still exists. It runs at most once a day, only during UTC hour 9 (03:00 MDT / 02:00 MST) by default. It is one read-only query whose results are streamed in batches of 5,000 into short UPDATEs, so no write transaction ever covers the whole history.
- The three copies were identical except for table names, so they now share one implementation: `orion/substrate/pending_marker_reconcile.py`.
- There are three new settings per service: the window, the full-sweep interval and the full-sweep hour. They are in `.env_example`, compose, `settings.py` and the README. The env sync script now visits these services and syncs their `*_RECONCILE_` keys.

## Outcome moved

**About 98% less disk read from these checks, and no more temp-file spill every 15 minutes.**

Before: `pg_stat_statements` since 2026-09-21. After: measured on the live DB with the shipped 2 h predicate, run as a read-only `SELECT count(*)` with `EXPLAIN (ANALYZE, BUFFERS)`.

| check | before, per run | after, per run (2 h window) |
|---|---|---|
| feedback (`substrate_execution_dispatch_frames`) | 13.5 s mean, ~561k blocks read, ~83.5k temp blocks written | 1,271 blocks read (20,575 cache hits), 0.91 s, no temp |
| dispatch (`substrate_policy_decision_frames`) | 5.5 s mean, ~496k blocks read, ~60k temp blocks | 2,740 blocks read (20,210 hits), no temp |
| policy (`substrate_proposal_frames`) | 5.5 s mean, ~179k blocks read, ~26k temp blocks | 22 blocks read (16,953 hits), 0.28 s, no temp |

- Per 15-minute cycle, disk reads drop from ~1.24M blocks (~9.4 GiB) plus ~170k temp blocks (~1.3 GiB) written, to ~4k blocks (~32 MB) with no temp.
- Per day, reads drop from ~96 x 9.4 GiB, to ~3 GB plus one full sweep (about one old run, ~9.4 GiB).
- The dispatch measurement took 26.7 s of wall time for only 2,740 reads. That is ~10 ms per random read on a disk that is already saturated, partly by the old checks this PR removes. Wall time should fall once they stop. Block counts are the reliable number here, and that is UNVERIFIED until deploy.
- In 72 h of `docker logs`, none of the old checks ever logged `*_pending_reconciled`. They re-queued nothing.

**Is 2 h wide enough?** The window counts from when the parent row was created, not from when its flag was cleared. So it only protects rows that get processed within 2 h of being created. I measured that delay on the live database (read-only):

- policy: p50 1.2 s, p99 41 s (last 24 h, 39,629 rows; measured by the reviewer)
- dispatch: p50 135 s, p99 217 s, max 250 s (last 6 h, 9,656 rows)
- feedback: p50 612 s, max 699 s (newest 400 rows; a wider sample hit the 2-minute statement timeout)

That leaves more than 10x margin. The migration note's "34 h behind" figure is from the August backlog. If a stage falls behind by more than 2 h again, a flag lost on those rows is caught by the daily full sweep instead (within about 24 h).

**Why the window is 2 h and not 24 h.** I measured 24 h first, and it was the wrong choice on this disk.

- A 24 h window read 37,602 / 33,425 / 22,698 blocks and took 213 s / 263 s / 113 s.
- The bounded check does one random index-and-heap probe per row, and random reads cost ~5-10 ms each on this HDD.
- A 24 h window would have been slower in wall time than the old sequential hash join.

**Why the full sweep is a hash-join SELECT and not chunked index probes.**

- Walking all history with per-row probes would be ~1.6M random reads, hours of disk time.
- One sequential hash anti-join is the cheap access pattern on a spinning disk.
- Doing it once a day as a read-only query, instead of 96 times a day as a write, is where the saving comes from.

## Current architecture

- Each runtime clears a `*_pending` flag in the same transaction that inserts the downstream frame, so the queue lookup reads the flag instead of doing an anti-join. This is ROADMAP D2, 2026-08-19.
- The safety net ran every `*_RECONCILE_INTERVAL_SEC` (900 s): `UPDATE parent SET x_pending = true WHERE NOT x_pending AND NOT EXISTS (child ...)`. It had no bound and ran as one write transaction over the whole of both tables. On 1.7M-row, 4-6 GB tables that meant seq scan + hash anti-join + temp spill (work_mem 4 MB).

## Architecture touched

- New shared module `orion/substrate/pending_marker_reconcile.py`. It holds `PendingMarkerSpec` (which table, flag and child foreign key) and `PendingMarkerReconciler` (the scheduling plus the three SQL shapes).
- Each service's `store.reconcile_*_pending(force=False, full=False)` keeps its name and signature, with `full` added. It delegates to the shared module. Worker call sites are unchanged.
- No bus or schema changes. No migrations: every index used already exists (see EXPLAIN).

## Files changed

- `orion/substrate/pending_marker_reconcile.py`: the shared bounded reconciler (new).
- `orion/substrate/tests/pending_marker_fake.py`: an in-memory DB that applies the statements using only the bounds each statement names, so the old unbounded shape fails the tests (new).
- `orion/substrate/tests/test_pending_marker_reconcile.py`: 21 tests for the shared module (new).
- `services/orion-{policy,execution-dispatch,feedback}-runtime/app/store.py`: delegate to the shared reconciler.
- `services/orion-{policy,execution-dispatch,feedback}-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the three new keys.
- `services/orion-{policy,feedback}-runtime/app/{main,worker}.py` and `services/orion-execution-dispatch-runtime/app/worker.py`: pass the new settings through.
- `services/orion-*-runtime/tests/test_reconcile_bounded.py`: 9 per-service tests (new).
- `services/orion-feedback-runtime/tests/test_pending_marker.py`: adapted to the new internals. The safety assertions are unchanged.
- `scripts/sync_local_env_from_example.py` and `tests/scripts/test_sync_local_env_from_example.py`: the three services had never been visited by the default sync. The script now visits them, adds the `POLICY_RECONCILE_`, `DISPATCH_RECONCILE_` and `FEEDBACK_RECONCILE_` prefixes, and has a test for it.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the 15-minute safety check covers only the last `*_RECONCILE_WINDOW_SEC` (2 h). A flag lost on an older row is caught by the daily full sweep, within 24 h instead of 15 min. The check can still only set flags to `true`, never clear them.
- Compatibility notes: none. The store method signatures only gained an optional `full`.

## Env/config changes

- Added keys, per service (`POLICY_` / `DISPATCH_` / `FEEDBACK_`):
  - `*_RECONCILE_WINDOW_SEC=7200`
  - `*_RECONCILE_FULL_SWEEP_INTERVAL_SEC=86400` (0 disables the full sweep)
  - `*_RECONCILE_FULL_SWEEP_HOUR_UTC=9` (-1 means any hour)
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, all three services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. All nine keys are present in `/mnt/scripts/Orion-Sapienform/services/orion-{policy,execution-dispatch,feedback}-runtime/.env` (checked with grep). Before this PR the sync never visited these three services and no prefix matched, so it would have added nothing. That is fixed here and covered by a test.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/test_pending_marker_reconcile.py         21 passed
services/orion-feedback-runtime/tests            36 passed  (FEEDBACK_POLICY_PATH set; pre-existing cwd-relative config)
services/orion-execution-dispatch-runtime/tests  72 passed  (EXECUTION_DISPATCH_POLICY_PATH set; same)
services/orion-policy-runtime/tests              12 passed
tests/scripts/test_sync_local_env_from_example.py + scripts/tests/test_check_env_template_parity.py + tests/test_grammar_event_producer_catalog.py   36 passed
```

I also ran all three service suites again with the wall clock pinned to 09:30 UTC (a pytest plugin that patches `datetime.now`): 36 / 72 / 12 passed.

I checked the race test by mutation. I removed the `NOT EXISTS` re-check from the batch UPDATE, and both `test_full_sweep_rechecks_each_candidate_before_requeueing` and `test_it_only_sets_the_marker_true[batch]` failed.

I checked streaming through the real driver (read-only, `generate_series`). With `stream_results` and `yield_per=5000`, 12,001 rows come back as partitions of 5000 / 5000 / 2001, and a second pooled connection works while the cursor is open.

I ran the new per-service tests against the old `store.py`: 8 of 9 fail in each service. The main one is `test_frequent_sweep_leaves_old_rows_to_the_full_sweep`. It fails because the old unbounded UPDATE re-queues the 30-day-old row (`assert 2 == 1`). That is the right failure reason, not an import or attribute error.

I checked the bind rendering through the real SQLAlchemy/psycopg2 driver against the live DB, in a read-only session using EXPLAIN only. `make_interval(secs => :window_sec)` renders as `now() - '02:00:00'::interval`, which is an index condition. `= ANY(:ids)` renders as `ANY ('{a,b}'::text[])` on the primary key.

## Evals run

```text
No eval harness exists for these three services. The live EXPLAIN / buffer measurements above
stand in for one. Follow-up: once deployed, a pg_stat_statements before/after snapshot.
```

## Static gates (CI list from .github/workflows/orion-static-gates.yml)

```text
All PASS: metric lineage, definition drift, inner-state registry, scripts stdlib shadow,
hostname refs, compose relative mounts, compose claude.json mounts, journal dispatch
registry, daily schedule collisions (report-only), sentience instruments, system_health
producers, control-surface store parity, async routes, chat route poachers.

check_service_env_compose_parity: policy and feedback OK. execution-dispatch reports 4
pre-existing missing keys that are not mine: EXECUTION_DISPATCH_STALENESS_MIN/MAX_SEC,
ORION_DISPATCH_ALLOCATOR_ENFORCE, ORION_DISPATCH_ALL_REFUSED_ALERT_TICKS.
All 3 new DISPATCH_RECONCILE_* keys are in compose.
```

## EXPLAIN evidence (live `conjourney`, read-only session, 2026-09-25)

Frequent sweep, as shipped (2 h). All three use the `generated_at` index and a nested-loop anti-join on the child's foreign-key index:

```text
feedback: Update on substrate_execution_dispatch_frames p  (cost=1.11..9340.65)
  ->  Nested Loop Anti Join
        ->  Index Scan using idx_substrate_execution_dispatch_frames_generated_at on substrate_execution_dispatch_frames p
              Index Cond: (generated_at >= (now() - '02:00:00'::interval))
              Filter: (NOT feedback_pending)
        ->  Index Scan using idx_substrate_feedback_frames_source_dispatch on substrate_feedback_frames c
dispatch: Update on substrate_policy_decision_frames p  (cost=0.98..9140.24)
  ->  Nested Loop Anti Join
        ->  Index Scan using idx_substrate_policy_decision_frames_generated_at ...
              Index Cond: (generated_at >= (now() - '02:00:00'::interval))
        ->  Index Scan using uq_substrate_execution_dispatch_source_policy on substrate_execution_dispatch_frames c
policy:   Update on substrate_proposal_frames p  (cost=0.98..19164.86)
  ->  Nested Loop Anti Join
        ->  Index Scan using idx_substrate_proposal_frames_generated_at ...
              Index Cond: (generated_at >= (now() - '02:00:00'::interval))
        ->  Index Scan using idx_substrate_policy_decision_frames_source_proposal on substrate_policy_decision_frames c
```

The same predicate as a read-only SELECT, with real buffers:

```text
feedback: Buffers: shared hit=20575 read=1271   rows probed 2916   Execution Time: 908 ms
dispatch: Buffers: shared hit=20210 read=2740   rows probed 3146   Execution Time: 26677 ms (saturated disk)
policy:   Buffers: shared hit=16953 read=22     rows probed 3182   Execution Time: 280 ms
```

Old statement, for comparison (feedback):

```text
Update on substrate_execution_dispatch_frames d  (cost=369204.01..864533.00)
  ->  Hash Anti Join
        ->  Seq Scan on substrate_execution_dispatch_frames d  (rows=1667525)
        ->  Hash -> Seq Scan on substrate_feedback_frames f  (rows=1780445)
```

Full sweep candidate scan (read-only, once a day): a sequential hash anti-join, as intended. Policy's table is small, so the planner picks a nested loop over a seq scan.

```text
feedback: Parallel Hash Anti Join  <- Parallel Seq Scan dispatch_frames / Parallel Seq Scan feedback_frames
dispatch: Parallel Hash Anti Join  <- Parallel Seq Scan policy_decision_frames / Parallel Index Only Scan dispatch source_policy
policy:   Nested Loop Anti Join    <- Parallel Seq Scan proposal_frames (270k rows) / Index Only Scan source_proposal
```

Full sweep batch UPDATE: `Index Scan using <parent>_pkey  Index Cond: (frame_id = ANY (...))`, then an anti-join probe on the child's foreign-key index. It is cheap per batch.

Indexes: every bound column already had an index (`idx_*_generated_at` on all three parents, and a foreign-key index on all three children). No migration was needed.

## Sibling search

I looked for the same anti-join pattern elsewhere in two ways.

- `rg "NOT EXISTS" services/*/app/store.py` plus a repo-wide search. The only `*_pending = true` re-queuers are the three fixed here.
- `pg_stat_statements` filtered to `NOT EXISTS`, ranked by blocks read. Below the three fixed statements, the next real ones are both already bounded (a cutoff plus a LIMIT batch) and small:
  - the receipt pruner `DELETE ... WHERE ctid IN (... NOT EXISTS ...)`: 999 calls, 36k blocks total
  - field-digester's `PRUNE_APPLIED_DELTAS_SQL`: 252 calls, 33k blocks total

  Nothing else has this problem.

## Review findings fixed

A code-review subagent found no blockers. It confirmed that no path clears a flag, and that the batch UPDATE re-checks both conditions. What it found, and what I did about each:

- Finding: the tests failed whenever CI ran between 09:00 and 09:59 UTC. The stores default to the hour-9 gate, so `run(force=True)` took the full-sweep path (the reviewer reproduced this: 8, 3 and 3 failures).
  - Fix: the per-service and legacy test helpers now default to `full_sweep_hour_utc=-1`. Tests that are about scheduling pin the clock explicitly.
  - Evidence: all three suites pass with the wall clock pinned to 09:30 UTC.
- Finding: the 2 h window counts from when the parent was created, not from when its flag was cleared. The migration note records a 34 h backlog in August.
  - Fix: I measured the live delay for all three stages (see Outcome moved), and documented in the module docstring and in this report that a backlog over 2 h falls back to the daily sweep.
  - Evidence: p99 is 41 s / 217 s, and feedback max is 700 s.
- Finding: the race test passed for the wrong reason. The fake always applied both guards, whatever the SQL said.
  - Fix: the fake now applies `NOT p.<marker>` and `NOT EXISTS` only when the statement contains them.
  - Evidence: the mutation check above.
- Finding: the full sweep was only considered on ticks where the frequent sweep was due. A long interval could keep skipping hour 9 forever.
  - Fix: the full sweep is now checked on every call, but never within one interval of process start, so a crash loop still cannot trigger it.
  - Evidence: `test_a_long_frequent_interval_cannot_skip_the_sweep_hour` and `test_no_full_sweep_before_one_interval_of_uptime`.
- Finding: with the hour gate on and a full-sweep interval of 3600 s or less, every tick in the hour ran a full sweep.
  - Fix: when gated, the minimum gap between full sweeps is at least one hour.
  - Evidence: `test_short_full_interval_still_runs_at_most_once_per_hour_window`.
- Finding: the full sweep loaded every candidate id into memory at once.
  - Fix: a server-side cursor (`stream_results`, `yield_per=5000`), with each batch re-queued in its own transaction.
  - Evidence: the real-driver check above.
- Finding: `run(full=True)` was silently ignored when the rate limit was active.
  - Fix: `full=True` now always runs.
- Finding: the dispatch worker read the new settings through `getattr` with duplicate defaults.
  - Fix: it now reads the settings attributes directly.
- Finding: "9 = 03:00 MDT" is only correct in summer.
  - Fix: changed to "03:00 MDT / 02:00 MST" everywhere.
- Not fixed, accepted:
  - A restart inside 09:xx UTC, after at least 15 minutes of uptime, repeats the full sweep. It is one extra read-only scan, and this is documented.
  - The full-sweep SELECT still holds one snapshot for the length of its scan, once a day. Stopping that would need chunking by random probes, which costs more on this disk.

## Restart required

Not deployed. After merge, from a worktree on up-to-date main:

```bash
scripts/safe_docker_build.sh orion-policy-runtime up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
```

After deploy, the first bounded sweep runs 15 minutes after boot, and the first full sweep runs in the next 09:xx UTC hour. Verify with:

```bash
docker logs --since 26h orion-feedback-runtime 2>&1 | grep -E "feedback_pending_(reconciled|full_sweep_done)"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select calls, mean_exec_time, shared_blks_read, temp_blks_written, left(query,80) from pg_stat_statements where query ilike '%make_interval(secs%' or query ilike '%_pending = % where p.frame_id = any%'"
```

## Risks / concerns

- Severity: low.
  - Concern: a flag lost on a row processed more than 2 h after it was created is now caught within about 24 h instead of 15 min.
  - Mitigation: the flag is cleared in the same transaction as the child insert, and the measured parent-to-child delay is at most about 12 minutes today. The full sweep still covers all history daily. The window is one env key if Juniper wants it wider.
- Severity: low.
  - Concern: the daily full sweep still costs about one old run of I/O (the seq-scan hash join, and possibly temp spill).
  - Mitigation: it runs once a day instead of 96 times, off-peak for Juniper, read-only, and only the batched UPDATEs write.
- Severity: low.
  - Concern: the hour-gated full sweep runs again if a service restarts during 09:xx UTC after at least 15 minutes of uptime.
  - Mitigation: that costs one extra read-only scan. A crash loop cannot trigger it, because the full sweep never runs within one interval (15 minutes) of process start.
- Severity: low.
  - Concern: the full sweep runs inside the worker tick, so that stage pauses for the length of the scan once a day.
  - Mitigation: it runs off the event loop (`asyncio.to_thread`), and the next tick picks the work back up.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
