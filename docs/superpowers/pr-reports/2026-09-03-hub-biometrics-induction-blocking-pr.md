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
orion/substrate/tests/test_metacog_trend_signals.py            14 passed
services/orion-sql-writer/tests/  (full)         11 failed, 480 passed, 3 skipped
services/orion-cortex-exec/tests/test_metacog_trend_reader.py   8 passed
services/orion-cortex-exec/tests/test_metacog_trend_cue_prompt_render.py
services/orion-cortex-exec/tests/test_metacog_biometrics_fleet_watts.py  16 passed
node --test services/orion-hub/static/js/                     113 passed, 0 failed
```

The 11 sql-writer failures are pre-existing and environmental (docker hostnames
like `orion-athena-sql-db` do not resolve from the host). Proven pre-existing by
reverting only this patch's sql-writer changes: **11 failed / 477 passed** before,
**11 failed / 480 passed** after — same failures, +3 new passing tests.

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
  canary ms: min=2.21  p50=5.98  max=12.15
  (pre-fix equivalent: 4 concurrent calls -> 1,100ms)

== history_multi through the bounded pool ==
  athena: ok=True  15 channels  9,780 points  239ms
  circe:  ok=True  15 channels  9,072 points  126ms
  pool cap = 4, live pool = 1 connection (was: 1 new connection per request)
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

(populated from the review pass — see section below)

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

(filled in on open)
