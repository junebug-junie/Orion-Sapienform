## Summary

This is patch 3 of Temporal Self ([PR #2369](https://github.com/junebug-junie/Orion-Sapienform/pull/2369), rev 4). Patch 2 ([#2597](https://github.com/junebug-junie/Orion-Sapienform/pull/2597)) built a pure reducer that turns rows Orion already writes into one account of its day, made of **arcs**. An arc is a stretch of time Orion kept coming back to one subject, or one bounded process such as a conversation or a sleep. Nothing ran that reducer. This patch runs it live. Nothing is deployed.

- A new `chronicle` step runs on the same durable-runs thread as the arousal `regulate` step (#2594), right after it, every 2 minutes. It reads every source the reducer binds, up to 5 minutes ago. It folds the rows into arcs and stores the arcs, each closed local day, a "where am I in my day" frame, and the reducer's own state, all in one transaction.
- A restart or a failed write never folds a row twice and never drops one. The reducer state is stored in the same transaction as everything derived from it, and the in-memory state moves only after that transaction commits.
- Rows that arrive after their window was read are counted once, stored as `late_unfolded`, and never folded. Nothing is dropped silently.
- Five read-only routes: `GET /temporal-self/frame`, `/day/{day_id}`, `/arcs`, `/threads` (open threads) and `/cursors`.
- Retention runs every 6 hours. It also covers the arousal-transition rows #2594 left without one.
- An eval runs this live code path (the real SQL, store, late-row probe and body reads) over the 10-09 fixture day on a throwaway Postgres. It reproduces #2597's numbers exactly. Pointed read-only at the production tables, it does too.

## Outcome moved

Patch 2 could answer "did Orion's morning change Orion's afternoon?" only offline, from an exported file. After this patch, durable-runs answers it continuously, from the live tables. Every arc keeps the ids of the rows that built it.

The live code path reproduces patch 2's 10-09 day exactly:

| Run | Steps | Closed 10-09 equals the pure one-pass fold | Patch-2 gates | Late / skipped rows |
|---|---|---|---|---|
| fixture in throwaway Postgres, 600 s steps, restart mid-afternoon | 289 (92 s) | yes | all pass | 0 / 0 |
| fixture in throwaway Postgres, 120 s steps (the live tick) | 1,441 (448 s) | yes | all pass | 0 / 0 |
| **production tables read-only** (`--source-dsn`), 600 s steps | 289 (160 s) | yes | all pass | 0 / 0 |

On all three runs: evidence precision is 2,103 / 2,103. Arcs by kind are 49 attention, 44 interoception, 184 reverie, 8 curiosity, 2 conversation, 2 concern and 2 sleep. Process recall is exact. The labels pass 6/6. Returns, dwell, rest share (30.6%) and the frame at close all match #2597.

Body summaries match the eval's own rule on 107 of 107 arcs. On the production read it is 105 of 105, because two arcs (the 20.6 h curiosity run and an arc that started at 23:49 on 10-08) reach back before the fixture's body export window. Production has those earlier sensor rows and the fixture does not.

## Current architecture

- `orion/temporal_self/` (#2597): a pure `fold` / `advance_clock` / `build_frame` / `drain_closed_days`, with no caller.
- `services/orion-durable-runs` (#2594, live): the `temporal_self:orion:<local date>` thread, `ingest -> regulate -> done`, every 120 s and on each Juniper turn. Its only table is `temporal_self_event` (arousal transitions), with no retention.

## Architecture touched

- The thread becomes `ingest -> regulate -> chronicle -> done`.
  - A chronicle failure adds a `chronicle_failed` warning to the step. It never touches the regulation reading or its Redis projection; a test pins this.
  - The reducer state (up to ~1.2 MB of JSON at a busy midday, stored gzipped) lives in its own table, not in the LangGraph checkpoint. The checkpoint keeps a one-line summary.
- New modules:
  - `app/temporal_self_sources.py`: the reads, with `export_fixture_day.sql` as the reference for every query and flag column;
  - `app/temporal_self_store.py`: the one-transaction commit, the tolerant loader, retention and the route reads;
  - `app/temporal_self_chronicle.py`: the step.
- One additive migration.
- No bus channel and no new schema. The stored shapes are #2597's registered models.

### How a window is read

The reducer orders items by when they became **available** (a process at its end). Each query selects a superset by an indexed column, widened where available time can come after that column. Every row then goes through the reducer's own adapter, and only events whose available time is in `[lo, hi)` are kept. A window is never read twice, and SQL never restates an adapter's time rule.

### Read lag and late rows (measured live, 3 to 7 days to 10-10)

| Source | Written after its timestamp |
|---|---|
| broadcast, attention rows, GPU waits, thoughts, reverie chains, visual chains, dream cycles | p99 ≤ 1.4 s, max ≤ 16 s |
| chat row after its salience trace (the concern-raise check) | p50 70 s, max 141 s |
| metacog row after its trigger | p99 12.7 s, max 532 s |
| `episode_memory` after `occurred_at` | median 13 h |

`TEMPORAL_SELF_READ_LAG_SEC=300` covers everything except the metacog tail, memory episodes and abandoned visual attempts (`abandoned` replaces `active` about 5,400 s after `started_at`). A probe re-reads the previous 30 minutes (memory episodes: 3 days; visual deferrals: 3 h) for rows that were not there yet. Those rows are counted, not folded. Memory episodes are therefore always late in live operation: they are context only, and they are counted. A replay binds them.

## Files changed

- `services/orion-durable-runs/app/temporal_self_chronicle.py` (new): the step. It reads windows, probes for late rows, runs one fold per window, reads the body sensors per closed arc and commits once, and it runs retention.
- `services/orion-durable-runs/app/temporal_self_sources.py` (new): the per-source SQL and the body reads.
- `services/orion-durable-runs/app/temporal_self_store.py` (new): the commit, which refuses a moved watermark (a second writer); the tolerant state loader; retention; and the route reads.
- `services/orion-durable-runs/app/temporal_self_graph.py`, `temporal_self_driver.py`: the `chronicle` node and its summary in `/health`.
- `services/orion-durable-runs/app/main.py`: builds the chronicler and adds the 5 routes and `/health -> temporal_self_chronicle`.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: 9 keys.
- `services/orion-sql-db/manual_migration_temporal_self_v1.sql` (new): the arc, day, projection, cursor and state tables; `temporal_self_event.late_unfolded`; and three `CONCURRENTLY` indexes on source tables read every step (`substrate_attention_schema.generated_at`, `orion_metacog.timestamp`, and a partial index on `substrate_reverie_thought.expectation_scored_at`).
- `orion/temporal_self/body.py`: `body_window` (the eval's per-arc window rule, now shared by the eval and the driver).
- `orion/temporal_self/evals/run_arc_precision_eval.py`: `score()` split out of `evaluate()`, so the live replay scores with the same gates. Its output is unchanged.
- `services/orion-durable-runs/evals/temporal_self_live_replay.py`, `temporal_self_fixture_db.py`, `fixtures/temporal_self_source_tables.sql` (new):
  - the replay eval;
  - the fixture-to-tables loader;
  - the column types of all 24 source tables, read from the live database (constraints dropped).
- `services/orion-durable-runs/tests/test_temporal_self_chronicle.py`, `test_temporal_self_chronicle_postgres.py`, `temporal_self_world.py` (new).
- `services/orion-durable-runs/README.md`, `orion/temporal_self/README.md`.
- `.github/workflows/orion-durable-runs-tests.yml`: new paths, and the replay eval as a step.

## Schema / bus / API changes

- **Added:**
  - HTTP `GET /temporal-self/frame`, `/temporal-self/day/{day_id}`, `/temporal-self/arcs?day_id=&kind=&limit=`, `/temporal-self/threads`, `/temporal-self/cursors`;
  - `/health` gains `temporal_self_chronicle`, and `temporal_self.chronicle` (the last step's summary).
- **Tables** (`manual_migration_temporal_self_v1.sql`): `temporal_self_arc`, `temporal_self_day`, `temporal_self_projection`, `temporal_self_cursor`, `temporal_self_state`; the column `temporal_self_event.late_unfolded` (default false); the indexes `idx_substrate_attention_schema_generated_at`, `idx_orion_metacog_timestamp` and `idx_substrate_reverie_thought_scored_at`.
- **Removed / renamed:** none.
- **Behaviour changed:** the temporal-self thread does more work per step: about 0.3 s per window measured, and about 11 s for the one-time first-boot backfill of ~36 windows.
- **Compatibility:** no bus or schema change.
- **Deviations from the spec:**
  - `temporal_self_state` was added: cursors alone cannot restore open arcs and buffers, and the restart point has to be the reducer state;
  - `temporal_self_event` stores `ended_at` and `privacy_class` inside `payload_json`, because the table from #2594 has no such columns;
  - the spec's 10× per-source cadence warning is reduced to a single warning on the read watermark: every source is read up to the same watermark, so per-source cadence tables would be a list nobody acts on;
  - FalkorDB adapters (`:PriorRevision`, peer asks) are not built, as allowed by the task.

## Env/config changes

- **Added keys** (`services/orion-durable-runs`; all ship ON or at their measured default):
  - `TEMPORAL_SELF_CHRONICLE_ENABLED=true`
  - `TEMPORAL_SELF_ARC_MIN_TICKS=3`
  - `TEMPORAL_SELF_RETURN_WINDOW_MIN=30.0`
  - `TEMPORAL_SELF_CONVERSATION_RETURN_WINDOW_MIN=180.0`
  - `TEMPORAL_SELF_READ_LAG_SEC=300.0`
  - `TEMPORAL_SELF_BACKFILL_DAYS=1`
  - `TEMPORAL_SELF_EVENT_RETENTION_DAYS=30`
  - `TEMPORAL_SELF_ARC_RETENTION_DAYS=90`
  - `TEMPORAL_SELF_DAY_RETENTION_DAYS=365`

  The time zone reuses the existing `ORION_SITUATION_TIMEZONE`.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes, along with `settings.py` and the compose fallbacks.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py` (the worktree's script, run from the primary checkout root): yes. All 9 keys are in `/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env`.
- **Skipped keys requiring operator action:** none. The sync reported one existing divergence that this patch did not cause: `DURABLE_RUNS_GRAPH_HOST`.

## Metric quality gate

This patch adds no cognition metric. Every number the frame and the arcs carry was gated in #2597.

The new numbers are operational only: the read watermark lag, the late-row counts and the skipped count.
1. Provenance: wall clock minus the stored watermark, and `count(*)` of `late_unfolded` rows.
2. Independence: not applicable (they are not inputs to any model).
3. Theory anchor: none needed (they are counts).
4. Live sanity: 0 late and 0 skipped on all three replays.
5. Existing mechanism: none in this service.
6. Reversibility: one flag.

`python scripts/check_metric_lineage.py --gate` passes.

## Tests run

```text
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<throwaway pg16> PYTHONPATH=<worktree> python -m pytest tests -q
  466 passed, 1 skipped
    tests/test_temporal_self_chronicle.py           20 (in-memory store/reader: live == one-pass fold; restart mid-arc;
                                                       failed commit re-reads; row inside the lag folded; row after it
                                                       counted once, never folded; probe never reaches before origin;
                                                       first boot closes yesterday; invalid state resets to its day;
                                                       second writer refused then reloads; body once per closed
                                                       non-reverie arc; retention cadence; chronicle failure never
                                                       breaks regulate; + 6 review regressions: abandoned
                                                       deferral 90 min late, moved available time, reset
                                                       re-fold, step budget, node timeout, late thermal)
    tests/test_temporal_self_chronicle_postgres.py   9 (migration idempotent + regulate insert still works; the SQL reads
                                                       exactly what the adapters read, incl. in-flight visual attempts;
                                                       TEXT cabinet timestamps with either separator; Postgres run ==
                                                       in-memory run across a restart; late row stored unfolded; stale
                                                       commit writes nothing; tolerant loader; retention incl. arousal rows;
                                                       attempt inserted active, abandoned 90 min later)
    tests/test_temporal_self_regulate.py            unchanged, green
python -m pytest -q orion/temporal_self/tests                               73 passed
mutation: rewind the reloaded watermark by 4 min -> restart and second-writer tests fail (reverted)
mutation: drop the visual_deferral probe override and the moved-row filter -> both regressions fail (reverted)
static gates (every step of orion-static-gates.yml, run locally)              all green, incl. metric lineage,
                                                                              prompt semantics, definition drift,
                                                                              inner-state registry, async-route DB calls
services/orion-attention-runtime tests + attention runtime/store tests        98 passed, 5 skipped
orion-sql-writer unit set (triggered by services/orion-sql-db/**)              211 passed
scripts/check_sql_migrations_applied.py --file manual_migration_temporal_self_v1.sql
  parses all 9 objects; RED "missing" until the operator applies it (expected)
git diff --check                                                               clean
```

## Evals run

```text
python orion/temporal_self/evals/run_arc_precision_eval.py                       passed: true (unchanged)
ORION_ADMISSION_TEST_DSN=<throwaway> python services/orion-durable-runs/evals/temporal_self_live_replay.py
  --step-sec 600: passed (289 steps, restart mid-afternoon)   --step-sec 120: passed (1,441 steps)
  --source-dsn <production, READ ONLY transactions>:            passed (bodies: 105 compared, 2 outside the export window)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile -t ts-chronicle-check:tmp .     ok (throwaway tag, removed)
docker run --rm ts-chronicle-check:tmp python -c "import app.main ..."                   5 routes, flags read
docker compose --env-file <primary .env> --env-file <primary durable-runs .env> -f services/orion-durable-runs/docker-compose.yml config
                                                                                         all 12 TEMPORAL_SELF_* keys resolve
orion-durable-runs-durable-runs:latest                                                   untouched (sha256:cf64fd08…, the running image)
```

Live path: **UNVERIFIED**. Not deployed by design. The live checks are below.

## Review findings fixed

A code-review subagent reviewed `feat/temporal-self-live-chronology` at 0cee65477. It used probe scripts, and it read the production tables read-only. It found 1 blocker, 4 should-fix and 5 nits. Replay identity, restart, `[lo, hi)` against `advance_clock`, the stale-writer check and the SQL escaping were all checked and found correct.

- **Finding (blocker):** live, an `abandoned` visual attempt gets its outcome about 5,400 s after `started_at`, which is its time column. Until then it reads `active` and is skipped. A 30-minute probe never saw it again, so the main live deferral kind was lost with no count.
  - Fix: the late probe looks 3 h back for `visual_deferral`, so these rows are counted as late (`late_unfolded`).
  - Evidence:
    - `test_abandoned_deferral_finalised_ninety_minutes_later_is_counted_late` (in memory);
    - `test_attempt_inserted_active_then_abandoned_ninety_minutes_later_is_late_not_lost` (Postgres: inserted `active`, flipped by `UPDATE`);
    - removing the override fails both.
- **Finding:** the CPU-bound fold ran on the event loop that admission and `/health` share. It measured p95 0.23 s and max 0.56 s per window.
  - Fix: `fold` / `advance_clock` / `build_frame` / `drain_closed_days` and the state gzip now run in `asyncio.to_thread`.
- **Finding:** there was no time limit, so a hung read would hold every queued tick and the regulate checkpoint.
  - Fix:
    - every chronicle transaction gets `SET LOCAL statement_timeout = '30s'`;
    - a step stops starting windows after 30 s, though it always makes at least one;
    - the node wraps the call in `asyncio.wait_for(..., 120 s)`.
  - Evidence: `test_a_hung_chronicle_times_out_and_regulation_still_steps`, `test_step_budget_spreads_a_long_catch_up_over_steps`.
- **Finding:** a row whose available time moves later after it was folded would be folded twice. This has a live trigger: Hub's curiosity outcome upsert rewrites `completed_at = now()`.
  - Fix: an in-window event that is already stored with a different available time is dropped and counted (`moved_total` in health, plus a warning log). A re-fold after a reset reads the same rows at the same time, so it still folds them.
  - Evidence:
    - `test_a_row_whose_available_time_moves_later_is_not_folded_twice` (removing the fix fails it);
    - `test_reset_refold_still_folds_rows_it_had_stored`.
- **Finding:** the `substrate_attention_schema` index was built inside the transaction, which blocks attention-runtime writes while it builds. Several per-window reads were also seq scans: `orion_metacog` at 611 MB, `expectation_scored_at`, and the action-outcome `coalesce`.
  - Fix: three `CREATE INDEX CONCURRENTLY IF NOT EXISTS` statements after `COMMIT`:
    - `substrate_attention_schema(generated_at)`;
    - `orion_metacog(timestamp)`, with the read now using an indexed TEXT prefix a day wide plus the exact cast;
    - a partial index on `expectation_scored_at`.

    The action-outcome read now uses the existing `observed_at` index, via `UNION ALL` with a null branch.
  - Evidence: `psql -v ON_ERROR_STOP=1 <` applies the file twice on a fresh throwaway database, all 3 indexes are valid, and the migration checker tracks all of them.
- **Nits fixed:**
  - Routes return 503 naming the migration file when the tables are missing, instead of a 500.
  - Late deferrals now count in a closed arc's `thermal_refusals` (`test_late_deferrals_count_in_the_body_thermal_refusals`).
  - A re-fold clears a stored `late_unfolded` flag (`ON CONFLICT ... DO UPDATE ... WHERE late`).
- **Nits declined:**
  - **Probing less often than every step.** With the indexes it costs one more indexed read per source per step. Probing less often would delay when late rows are counted.
  - **Removing the slack on curiosity and dominance runs.** The slack is harmless, and it covers the clamp case where the end comes before the start.
  - **Gate counting in-flight rows older than the probe horizon.** Left as a follow-up.

## Restart required

Apply the migration first. Then deploy from the primary checkout on main after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_temporal_self_v1.sql && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

### Live checks after deploy

1. **Cursors advancing.** Run this twice, 2 minutes apart. The `read_watermark` should move forward and its `lag_sec` should stay at or under about 420:

   ```bash
   curl -s localhost:8124/temporal-self/cursors | jq '{w: [.cursors[]|select(.source_kind=="read_watermark")], late: .late_rows_last_24h, warnings}'
   ```

2. **One closed day.** The first step backfills from local midnight yesterday, so yesterday closes on the first step:

   ```bash
   curl -s localhost:8124/temporal-self/day/$(TZ=America/Denver date -d yesterday +%F) | jq '{day_id, arcs: (.arcs|length), warnings: .frame.warnings}'
   ```

3. **One arc with at least 2 returns whose refs resolve in two tables.**
   - By construction, an arc's *subject* evidence comes from one table: the broadcast log for attention, `field_dominance_run` for interoception.
   - The second table is its time-bound context. For example, metacog observations resolve in `orion_metacog`.
   - The query resolves both:

   ```sql
   WITH a AS (SELECT arc_id, kind, attention_returns, arc_json FROM temporal_self_arc WHERE attention_returns >= 2 AND kind IN ('attention','interoception') AND jsonb_array_length(arc_json->'context_event_ids') > 0 ORDER BY updated_at DESC LIMIT 1),
   ev AS (SELECT split_part(r, ':', 1) AS t, substr(r, strpos(r, ':') + 1) AS pk FROM a, jsonb_array_elements_text(a.arc_json->'evidence_refs') r),
   cx AS (SELECT split_part(c, ':', 1) AS k, substr(c, strpos(c, ':') + 1) AS pk FROM a, jsonb_array_elements_text(a.arc_json->'context_event_ids') c)
   SELECT (SELECT arc_id FROM a), (SELECT kind FROM a), (SELECT attention_returns FROM a),
     (SELECT count(*) FROM ev) AS evidence,
     (SELECT count(*) FROM ev JOIN substrate_attention_broadcast_log b ON ev.t = 'substrate_attention_broadcast_log' AND b.log_id = ev.pk)
       + (SELECT count(*) FROM ev JOIN field_dominance_run f ON ev.t = 'field_dominance_run' AND f.run_id = ev.pk) AS evidence_resolved,
     (SELECT count(*) FROM cx WHERE k = 'metacog_observation') AS metacog_context,
     (SELECT count(*) FROM cx JOIN orion_metacog m ON cx.k = 'metacog_observation' AND m.id = cx.pk) AS metacog_resolved;
   ```

   The answer is right when `evidence_resolved = evidence` and `metacog_resolved = metacog_context > 0`.

4. **The rest.** `curl -s localhost:8124/health | jq .temporal_self_chronicle` shows `error: null`. `/regulation/state` still answers.

## Risks / concerns

- **Medium: abandoned visual deferrals are always late live.** `abandoned` replaces `active` about 90 minutes after `started_at`, so these rows are counted (`late_unfolded`), never folded into an arc's constraints. Fixing that needs the attempt to carry a finish time the reducer can order by.
- **Medium: memory episodes are always late live.** They are written a median 13 h after `occurred_at`, so live arcs never bind them. They are counted (`late_unfolded`), and a replay binds them. A fix belongs to whoever owns the episode writer's timing, or to a later reducer rule that binds by write time.
- **Low: the first-boot backfill** runs about 36 windows. A step stops starting windows after 30 s, so the backfill spreads over a few 2-minute steps, and regulation runs before it on each one.
- **Low: `temporal_self_state` churn.** It is rewritten every 2 minutes, at about 100 KB gzipped. Autovacuum handles it, and the row is a singleton.
- **Low: the three new source-table indexes** are built `CONCURRENTLY`, without blocking writers. If one is interrupted it leaves an INVALID index; the migration header says to drop it and re-run.
- **Low: "two tables" in the live check** is subject evidence plus context, not two evidence tables. The reducer's lanes each draw their evidence from one table.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
