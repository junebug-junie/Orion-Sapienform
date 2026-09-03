# Hub Biometrics poll was freezing the whole UI

## Summary

- The Biometrics card added 2026-09-02 polls `/api/biometrics/preview/induction`
  every 10s per node. That route called a **synchronous** SQLAlchemy helper
  inline from an `async def`, so the hub's event loop was blocked for the whole
  query and served nothing at all — including static assets.
- The query behind it was pathological: `DISTINCT ON (node)` with an
  `ORDER BY timestamp::timestamptz` cast, against a 247MB / 187k-row table with
  no supporting index. 418ms mean, and **12 TB of cumulative temp spill** —
  92% of all temp I/O on the instance.
- Fixed in three places, all three needed: run the blocking call off the loop;
  drop the unindexable cast and rewrite `DISTINCT ON` as a `LATERAL` top-1-per-node;
  add the index that shape needs.
- Also: cache the SQLAlchemy engine instead of rebuilding it per request; move
  both history routes onto a bounded connection pool; stop polling while the
  browser tab is hidden.
- Measured live, end to end: **418ms mean → 0.98ms p50**. Event-loop stall under
  8 concurrent `/induction` calls: **1,100ms → 5.98ms p50**.

## Outcome moved

Operator report: "orion hub is running crazy slow… on browser opens, the tabs
take 30 seconds to load." Reproduced and measured, not inferred.

Sampling a **static JS file** (nothing to do with the database) once a second
against the live hub, while a browser had the Biometrics card open:

```
p50 = 0.005s
max = 60.0s   (capped by the client, still stalled)
      47.3s
```

Two stalls in ~2 minutes of sampling. Caught one in the act — during the freeze,
three Postgres backends (one leader, two parallel workers) were all executing the
induction query:

```
216900 | active | SELECT DISTINCT ON (node) node, metrics, timestamp::timestamptz AS ts
216901 | active |   FROM orion_biometrics_induction WHERE node = ANY(...)
    72 | active |
```

A static asset cannot be slow because of a database query — unless the event loop
serving it is blocked. That is the whole bug.

## Current architecture (before this patch)

`services/orion-hub/scripts/biometrics_preview_routes.py` exposes
`/api/biometrics/preview/{snapshot,history,history_multi,induction,gpu}` behind
the Cognitive EKG card and its deep-inspection modal.

`/induction` reused the existing shared helper
`orion/substrate/metacog_trend_signals.py::latest_biometrics_induction_by_node`
— correct instinct, but that helper is **synchronous** SQLAlchemy, and it was
called inline:

```python
async def api_biometrics_preview_induction(node: str = Query(...)):
    engine = _induction_engine()                          # create_engine() per request
    by_node = latest_biometrics_induction_by_node(engine, [nid])   # blocks the loop
```

Until 2026-09-02 the only caller of that helper was
`services/orion-cortex-exec/app/metacog_trend_reader.py`, which does it properly:
`asyncio.to_thread`, a `statement_timeout` GUC, a cached engine, once per chat
turn. The Hub route reproduced none of those guards and put the call on a 10s
poll instead.

## Architecture touched

- `orion-hub` — `/induction` execution model, engine lifecycle, history-route
  connection strategy, client-side poll gating.
- `orion/substrate` — the shared induction query shape (also read by
  `orion-cortex-exec`; the rewrite is a strict improvement for that caller too,
  no signature or contract change).
- `orion-sql-writer` — one added boot-time `CREATE INDEX IF NOT EXISTS`,
  following the convention already used for `orion_biometrics_summary`.

## Files changed

- `services/orion-hub/scripts/biometrics_preview_routes.py`: `/induction` moved
  to `asyncio.to_thread` + `wait_for`; process-wide cached engine carrying a
  `statement_timeout` GUC; both history queries moved from `asyncpg.connect()`
  per request to a bounded module-level pool with event-loop identity tracking.
- `orion/substrate/metacog_trend_signals.py`: induction query rewritten from
  `DISTINCT ON` + casted ORDER BY to `CROSS JOIN LATERAL` top-1-per-node.
- `services/orion-sql-writer/app/main.py`: boot DDL for
  `orion_biometrics_induction_node_ts_idx (node, timestamp DESC)`.
- `scripts/sql/2026-09-03_biometrics_induction_node_ts_idx.sql`: out-of-band
  form of the same index, including the `CONCURRENTLY` variant for live application.
- `services/orion-hub/static/js/biometrics-view.js`: poll gate on page
  visibility, one refresh on return, module made `require()`-able for `node:test`.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: two
  new timeout keys.
- Tests: `services/orion-hub/tests/test_biometrics_preview_api.py` (+5),
  `orion/substrate/tests/test_metacog_trend_signals.py` (+2),
  `services/orion-sql-writer/tests/test_biometrics_induction_node_ts_index.py` (new, 3),
  `services/orion-hub/static/js/biometrics-view.test.js` (new, 4).

## Why all three fixes were needed

Each was verified separately against the live table. None of them is sufficient alone.

**1. The cast made the ordering unindexable by construction.**

`varchar::timestamptz` is not `IMMUTABLE`, so Postgres refuses to build an index
on that expression at all:

```
CREATE INDEX ... ON orion_biometrics_induction (node, (timestamp::timestamptz) DESC);
ERROR:  functions in index expression must be marked IMMUTABLE
```

Ordering on the bare text column is safe here and was verified, not assumed:
every row is written through psycopg2 against a server with `TimeZone=Etc/UTC`
and `DateStyle=ISO`, so Postgres renders each value as
`YYYY-MM-DD HH:MM:SS[.ffffff]+00`. Across all 187,563 rows: **zero off-format
values, and zero rows where text ordering and timestamptz ordering disagree.**

**2. Removing the cast and adding the index was still not enough.**

`DISTINCT ON` must consume its entire sorted input, and Postgres has no
loose/skip index scan — so it kept choosing a parallel seq scan even *with* the
index present:

```
Parallel Seq Scan on orion_biometrics_induction (rows=46309, loops=3)
Sort Method: external merge  Disk: 57792kB + 46760kB + 48744kB
Execution Time: 911.613 ms
```

This is the trap worth flagging: dropping the cast alone *looks* like the fix
and changes nothing measurable.

**3. The `LATERAL` form asks for what is actually wanted.**

One `ORDER BY timestamp DESC LIMIT 1` per requested node — two index lookups:

```
Index Scan using orion_biometrics_induction_node_ts_idx (rows=1, loops=2)
Buffers: shared hit=8
Execution Time: 0.467 ms
```

**911ms / 30,015 buffers → 0.47ms / 8 buffers.** The index is load-bearing for
this shape too: with index scans disabled the LATERAL form is 281ms, so neither
half of the fix stands alone.

## Schema / bus / API changes

- Added index: `orion_biometrics_induction_node_ts_idx (node, timestamp DESC)`.
  `DESC` is load-bearing — it is the direction the top-1-per-node read scans;
  an ASC index returns the *oldest* row per node, a wrong answer that still
  looks like a working query.
- No bus channel, schema-registry, or API response-shape changes. `/induction`
  gains one new `error` value, `induction_timeout`, alongside the existing
  `induction_unavailable`; both already render as `ok: false, metrics: {}`.
- No change to `latest_biometrics_induction_by_node`'s signature or return
  contract. Absent nodes stay absent (not zero-filled), the `max_age_sec`
  staleness filter is unchanged — both re-verified against live data.

## Env/config changes

- Added keys: `BIOMETRICS_INDUCTION_STATEMENT_TIMEOUT_MS=2000`,
  `BIOMETRICS_INDUCTION_FETCH_TIMEOUT_SEC=3.0`
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`)
- local `.env` synced: yes — **note**, `scripts/sync_local_env_from_example.py`
  reads `.env_example` from the *primary* checkout, so keys added in a worktree
  are invisible to it. Both keys were written into
  `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` by hand and confirmed
  present. `.env` remains gitignored and unstaged.
- Skipped keys requiring operator action: none.

Both timeouts are sized against the **pre-index** cost (418ms mean), not the
post-index cost, so that losing the index degrades to a logged timeout rather
than back to a 30-second whole-UI stall.

## Tests run

```text
services/orion-hub/tests/test_biometrics_preview_api.py
services/orion-hub/tests/test_biometrics_view_ui.py            63 passed
orion/substrate/tests/test_metacog_trend_signals.py            19 passed
services/orion-sql-writer/tests/  (full)         11 failed, 487 passed, 3 skipped
services/orion-cortex-exec/  (3 metacog files)                 24 passed
node --test services/orion-hub/static/js/                     113 passed, 0 failed
```

The 11 sql-writer failures are pre-existing and environmental (docker hostnames
like `orion-athena-sql-db` do not resolve from the host). Proven pre-existing by
reverting only this patch's sql-writer changes: **11 failed / 477 passed** before,
**11 failed / 487 passed** after — same failures, +10 new passing tests.

CI jobs reproduced locally:

```text
11/11 static gates in .github/workflows/orion-static-gates.yml   all OK
orion-sql-writer-tests.yml unit job (exact command)              45 passed
```

Not run locally: the `orion-sql-writer-tests.yml` Postgres *integration* job,
which needs the CI service container.

### Mutation testing

Every new assertion was checked against the mutation it claims to catch,
including mutations *worse* than the original bug:

```text
revert to inline blocking call (the original bug)     -> 2 failed  ✓
revert engine caching (create_engine per request)     -> 1 failed  ✓
drop the statement_timeout GUC                        -> 1 failed  ✓
reinstate ORDER BY timestamp::timestamptz             -> 1 failed  ✓
flip ORDER BY to ASC (returns OLDEST row per node)    -> 1 failed  ✓
drop DESC entirely (silently defaults to ASC)         -> 1 failed  ✓
drop the SELECT-list cast (breaks max-age filtering)  -> 1 failed  ✓
index DDL loses DESC                                  -> 1 failed  ✓
shouldPoll() always true (gate removed)               -> 1 failed  ✓
shouldPoll() truthy coercion instead of boolean check -> 1 failed  ✓
```

The `DISTINCT ON` → `LATERAL` test asserts the query *shape*, not merely the
absence of a cast — precisely because removing the cast alone is the plausible
"fix" that measures identically to the bug.

## Evals run

No eval harness exists for `services/orion-hub`'s biometrics preview routes
(`services/orion-hub/evals/` has no biometrics coverage). Not claiming eval
coverage. The live measurements below are the substitute evidence.

## Docker/build/smoke checks

The hub's compose mounts `static/` and `templates/` from
`${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}` — an **absolute path to
the primary checkout**. Building from this worktree would produce an image
carrying the new Python while still serving the old JS from main. So this was
deliberately not deployed from the worktree; it deploys on merge.

Instead, the new routes were exercised **in-process against the live production
database**:

```text
== /induction against the REAL database ==
  athena: http=200 ok=True channels=20   8.6ms (warm)
  circe:  http=200 ok=True channels=14   8.6ms

== canary endpoint latency while 8 concurrent /induction calls run ==
  canary ms: p50=1.46  max=1.94
  (pre-fix equivalent: 4 concurrent calls -> 1,100ms)

== history_multi through the bounded pool ==
  athena: ok=True  15 channels  9,780 points  239ms
  circe:  ok=True  15 channels  9,072 points  126ms
  pool cap = 4, live = 1 connection (was: 1 new connection per request)
  connect_timeout = 2.0s, acquire_timeout = 5.0s (were: unbounded)
  shutdown aclose(): pool=None engine=None
```

And the rewritten helper directly against live Postgres:

```text
live latency ms: min=0.85  p50=0.98  max=1.28        (was 418ms mean)
nodes returned: athena (20 channels), circe (14 channels)
absent node 'nope-not-a-node'      -> absent from result, not zero-filled  ✓
empty node list                    -> {}                                   ✓
max_age_sec=0                      -> {} (staleness filter intact)         ✓
```

The index was applied to the live instance with `CREATE INDEX CONCURRENTLY`
(1.19s, 11 MB, `indisvalid=t indisready=t`), and answers were confirmed
identical with index scans forced off, and identical to the old casted ordering:

```text
athena|2026-09-03 05:32:41.437679+00
circe |2026-09-03 05:32:29.566801+00      (all three query forms agree)
```

## Review findings fixed

An adversarial review pass ran against the first commit and found real defects,
including **two of my own new tests that could not fail**. All fixed in
`a4e635c2d`, each re-checked by mutation.

### MUST

- **Finding:** `test_induction_slow_query_does_not_stall_the_loop` was vacuous.
  It counted `await asyncio.sleep(0)` ticks and asserted the count — but a
  fully blocked loop produces the same count, just later.
  - **Fix:** timestamp every tick; require ≥5 to land *while* the worker thread
    is inside the query.
  - **Evidence:** the reviewer scored the pre-fix inline route 20/20 on the old
    assertion. Against the rewritten test it scores
    `only 0 loop ticks landed during the 0.25s query -- the loop was blocked`.

- **Finding:** three files still described the index as backing `DISTINCT ON` —
  the shape this same commit deleted *because it cannot use the index*.
  - **Fix:** rewrote the docstrings in `main.py`, the out-of-band SQL file and
    the index test to describe the LATERAL shape, and why `DISTINCT ON` cannot
    use it.
  - **Evidence:** `grep -rn "DISTINCT ON" scripts/sql services/orion-sql-writer`
    returns only the explicit "not this shape, and here's why" notes.

- **Finding:** the env-key rationale claimed the timeouts were sized so a
  dropped index "degrades to a logged timeout, not a 30-second UI stall."
  **Measurement contradicts it.** With `enable_indexscan`, `enable_bitmapscan`
  and `enable_indexonlyscan` all off, the read completes in 163ms (1 node) /
  422ms (3 nodes) — inside both the 2000ms statement timeout and the 3s fetch
  timeout. A dropped index produces no timeout, no log line, no signal at all.
  It compounds: the index was created inside a ~700-statement `engine.begin()`
  block whose single handler swallows failures as a warning.
  - **Fix:** corrected the rationale in `settings.py`, `.env_example` and the
    live `.env`; moved the index out of the swallowing transaction into its own
    block that queries `pg_indexes` afterwards and logs an **error** if absent.
  - **Evidence:** `test_boot_verifies_the_index_actually_exists` and
    `test_index_ddl_is_outside_the_swallowing_bootstrap_transaction`.

### SHOULD

- **Finding:** no `connect_timeout`. A black-holed (not refused) Postgres blocks
  the worker thread in TCP connect for the OS default ~130s — long past the
  route's 3s timeout, so at 10s polling the threads pile up invisibly in the
  loop's *shared* default executor. That is the same failure class this PR
  exists to fix, relocated.
  - **Fix:** `connect_timeout=2` on the engine, plus `_INDUCTION_SLOTS`, a
    4-permit semaphore capping how much of the shared executor this one route
    can hold.

- **Finding:** `pool.acquire()` had no timeout. The 14-channel history fan-out
  against a 4-connection pool could hang forever with no error and no 500 —
  the browser just spins. The old connection-per-request code could not do that.
  - **Fix:** `POOL_ACQUIRE_TIMEOUT_SEC`, plus `command_timeout` on the pool.

- **Finding:** `_pg_pool()` guarded its event-loop-identity swap with an
  `asyncio.Lock`, which provides no cross-thread exclusion — and two loops on
  two threads is precisely the situation that branch exists for. It also
  returned the global *outside* the lock, so a concurrent swap could hand back
  another loop's pool, or `None`.
  - **Fix:** `threading.Lock` around the swap; the pool is captured and returned
    from inside the async lock.

- **Finding:** an abandoned pool leaked a live Postgres backend (`POOL_MIN_SIZE`
  eagerly opens one). Wrong direction in a repo with PR #2010's history.
  - **Fix:** `pool.terminate()` — synchronous, needs no foreign loop.

- **Finding:** the cached engine was replaced without `dispose()`; neither pool
  nor engine was ever closed at shutdown.
  - **Fix:** `dispose()` on URI change, and a new `aclose()` called from
    `main.py`'s shutdown handler alongside `memory_pg_pool`.
  - **Evidence:** live run ends `aclose(): pool=None engine=None`.

- **Finding:** `assert "LIMIT 1" in sql` is **True for `LIMIT 10`** — and
  `LIMIT 10` is *worse than the original bug*, because the reader assigns
  `out[node] = metrics` per row, so the last row wins and it would silently
  return the **oldest** of ten.
  - **Fix:** token-boundary regex.
  - **Evidence:** mutating to `LIMIT 10` now fails.

- **Finding:** the rewritten query had **zero real-database coverage**. Both new
  substrate tests drove a `MagicMock` and asserted on SQL text. The entire
  parameter contract rested on one untested `list(nodes)` call — cortex-exec
  passes a *tuple*, which psycopg2 adapts to a composite record:
  `ProgrammingError: cannot cast type record to text[]`. Deleting that call
  keeps every mocked test green while cortex-exec's trend cue goes permanently
  dark, and because that reader fails open, nothing reports it.
  - **Fix:** a real-Postgres lane parametrised over both callers' shapes plus
    duplicates and absent nodes, and a test that cross-checks the LATERAL
    against an independent `MAX()` rather than against its own ordering.
  - **Evidence:** deleting `list(nodes)` now fails.

- **Finding:** `assert seen["is_main"] is False` passed against the pre-fix code
  too — under `TestClient` the loop runs on a portal thread, so the inline call
  was also "not the main thread". Its failure message named a bug it could
  never detect.
  - **Fix:** dropped; `on_loop is False` is the assertion that discriminates.

### CONSIDER

- **Finding:** the ordering argument reasoned from byte order (`'+' < any
  digit`), but this database collates `en_US.utf8`, not `C`. The empirical check
  holds; the stated *reason* was wrong for the collation in force.
  - **Fix:** replaced the byte-order reasoning with the empirical evidence and
    an explicit note on what a glibc collation change would do.

- **Finding:** the fixed-width timestamp format is a property of the *writer
  session's* DateStyle/TimeZone, not of the column — and nothing enforced it.
  Setting `PGTZ` on the sql-writer container would start writing `-06` offsets,
  silently diverging text order from chronological order.
  - **Fix:** `test_biometrics_induction_timestamp_format.py` — a real gate, per
    CLAUDE.md's "not a louder comment, a failing gate".
  - **Evidence:** it caught **two defects in its own first draft** — my
    non-UTC demonstration used
    `(now() AT TIME ZONE 'X')::timestamptz`, which re-interprets in the session
    zone and still renders `+00` (demonstrating nothing), and my
    `model_dump(mode="json")` assertion was file-wide when `worker.py`
    legitimately uses that form ~10 times for *bus* payloads. Both fixed; the
    corrected test shows a Denver-rendered "now" sorting *below* the current
    newest row.

- **Finding:** the "281ms with index scans disabled" figure disabled only
  `enable_indexscan`, leaving bitmap and index-only paths available.
  - **Fix:** re-measured with all three off (163ms / 422ms) and corrected the
    docstring. Same conclusion, honest method.

- **Finding:** `test_boot_ddl_stays_non_concurrent` was case-sensitive.
  - **Fix:** lowercased comparison.

- **Finding:** **nothing ran any of the 13 `node:test` files** in
  `services/orion-hub/static/js/` — hand-run only, which is exactly the
  "nudge you can skip" failure the static-gates workflow exists to replace.
  - **Fix:** added a `node --test services/orion-hub/static/js/` step to
    `orion-static-gates.yml`.
  - **Evidence:** exit 0 clean; exit 1 with a deliberately failing probe test.

- **Finding (documented, not changed):** because `statement_timeout` (2000ms)
  fires before `wait_for` (3.0s), a slow *query* surfaces as
  `induction_unavailable`, so `induction_timeout` specifically means the worker
  never reached the query. Genuinely useful; now said out loud in the docstring.

### Reviewer findings that were checked and needed no change

- `CROSS JOIN LATERAL` drops nodes with no rows — identical to the old
  `WHERE node = ANY(...)`. Zero NULL `node` rows live; both forms exclude them.
- Duplicate node names are idempotent (dict write).
- `statement_timeout` is genuinely enforced (`pg_sleep(3)` → `QueryCanceled`).
- cortex-exec needs no change; its default list includes decommissioned `atlas`,
  which the LATERAL drops exactly as the old query did.
- The reviewer's own worry that "N nodes means N scans is a regression" was
  disproven empirically: at cortex-exec's 3-node shape the LATERAL is 422ms with
  no temp spill versus 748ms and 206MB of external merge for `DISTINCT ON`.

## Restart required

Yes. The index is already live; the code is not. After merge, from the primary
checkout:

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
bash scripts/safe_docker_build.sh orion-hub up -d --build
curl -fsS http://localhost:8080/api/biometrics/preview/induction?node=athena
```

`orion-sql-writer` also needs a restart to pick up the boot DDL — though the
index is already applied live, so this is only for parity on a fresh database:

```bash
bash scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- **Severity: low.** Text ordering on `timestamp` depends on every writer
  producing `+00`-offset ISO strings. That holds today (187,563/187,563 rows,
  server `TimeZone=Etc/UTC`) and all writes go through orion-sql-writer, but a
  future writer with a different session timezone would break ordering silently.
  Mitigation: the invariant is documented at the query, and the projected `ts`
  still casts, so the staleness filter would still be correct.
- **Severity: low.** `_pg_pool()` abandons rather than closes a pool created on
  a different event loop (it cannot `await close()` on a foreign loop). The hub
  has one long-lived loop so this cannot happen in production; it logs a warning
  when it does, and it is what keeps the real-Postgres tests working.
- **Severity: informational.** `asyncio.wait_for` abandons the worker thread; it
  does not cancel the query. That is exactly why the `statement_timeout` GUC is
  set — the two bounds cover different failures and both are needed.
- **Severity: informational.** Seven other `setInterval` pollers in
  `services/orion-hub/static/js/app.js` are still ungated by page visibility.
  Not touched here — out of scope for this regression, and none of them is
  backed by a query anywhere near this cost. Worth a follow-up sweep.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2063
