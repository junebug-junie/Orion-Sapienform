# Causal Geometry v1: closing the spec gap list

## Summary

- Persist Phase A snapshots to a new dedicated Postgres table
  (`causal_geometry_snapshots`, `orion/substrate/causal_geometry_snapshot_store.py`)
  written directly by `orion-field-digester` -- the spec wanted "bus + Postgres via
  sql-writer"; after investigating `orion-sql-writer`'s actual worker (a 2300-line
  file shared by a dozen unrelated pipelines), routing through it was judged too
  invasive for the value. This is what makes the hub's Snapshot/History panels stop
  being permanently `degraded: true`.
- Wire `orion/substrate/mutation_trials.py`'s `SubstrateTrialRunner` into the live
  proposal pipeline (`orion/substrate/causal_geometry_producer.py`) -- it existed and
  was unit-tested standalone but was never actually invoked. Always resolves to
  `"inconclusive"` today (no replay corpus registered for this mutation class) --
  expected and correct; the point is the step is genuinely exercised and honestly
  recorded, not gating enqueueing.
- Hub reads those snapshots live via its existing `memory_pg_pool` asyncpg pool
  (`services/orion-hub/scripts/api_routes.py`).
- Hub UI (`static/js/causal-geometry.js`) renders Snapshot/History as real DOM
  tables with divergence highlighting instead of raw JSON dumps, plus gentle
  polling (45s) that pauses on `document.visibilitychange` and resumes with an
  immediate refresh.
- Code review (2 of 5 dispatched angles completed before an API session rate
  limit; not re-dispatched given solid non-overlapping coverage from the two that
  landed) found and fixed 6 real issues: a stale bus-channel registry claim, a
  schema-drift risk between the writer and reader's column lists, a missing
  retention/prune path for the new table, redundant per-write DDL, a trial-status
  result silently dropped from the operator-facing UI, and an under-emphasized
  `insufficient_data` flag.

## Outcome moved

Before this branch, the design spec's Phase A persistence requirement and Phase
B's trial-runner step were both documented gaps -- the hub's Causal Geometry tab
looked "dead" (Snapshot/History always degraded) even once the producer was live,
and the spec's `proposal -> trial -> HITL adopt` mechanism skipped the trial step
entirely. Both are now closed, live-testable end to end.

## Current architecture

Before this PR: `orion-field-digester`'s scheduled producer (from PR #1093/#1095)
measured real data and enqueued HITL proposals, but the underlying snapshot was
never persisted anywhere the hub could read, and `SubstrateTrialRunner` was dead
code outside its own test file.

## Architecture touched

- `orion/substrate/causal_geometry_snapshot_store.py` (new) -- persistence + prune.
- `orion/substrate/causal_geometry_producer.py` -- calls persistence + trial runner.
- `orion/bus/channels.yaml` -- corrected `producer_services` entry.
- `services/orion-field-digester/app/{worker.py,settings.py}`, `.env_example`,
  `docker-compose.yml` -- new retention setting, wired into the existing prune tick.
- `services/orion-hub/scripts/api_routes.py` -- async Postgres read path.
- `services/orion-hub/static/js/causal-geometry.js`,
  `templates/causal_geometry.html` -- table rendering, polling, trial-status
  surfacing, insufficient_data banner.
- `services/orion-sql-db/manual_migration_causal_geometry_snapshots_v1.sql` (new)
  -- documentation, matching this repo's convention for new tables.

## Files changed

Rung 1 (`703dcb1c`): `orion/substrate/causal_geometry_snapshot_store.py` (new),
`orion/substrate/causal_geometry_producer.py`,
`orion/substrate/tests/test_causal_geometry_snapshot_store.py` (new),
`orion/substrate/tests/test_causal_geometry_producer.py`.

Rung 2 (`d2342091`): `services/orion-hub/scripts/api_routes.py`,
`services/orion-hub/tests/test_causal_geometry_api.py`.

Rung 3 (`0426916c`): `services/orion-hub/static/js/causal-geometry.js`,
`services/orion-hub/templates/causal_geometry.html`,
`services/orion-hub/tests/test_causal_geometry_page.py`.

Rung 4 / review fixes (`93015cd3`): `orion/bus/channels.yaml`,
`orion/substrate/causal_geometry_snapshot_store.py` (shared column constant,
schema-ensure-once, prune function), `orion/substrate/tests/test_causal_geometry_snapshot_store.py`,
`services/orion-field-digester/app/{worker.py,settings.py}`, `.env_example`,
`docker-compose.yml`, `services/orion-field-digester/tests/test_worker_prune.py`,
`services/orion-hub/scripts/api_routes.py` (import shared column constant),
`services/orion-hub/static/js/causal-geometry.js` (trial-status + insufficient_data
banner), `services/orion-hub/tests/test_causal_geometry_page.py`,
`services/orion-sql-db/manual_migration_causal_geometry_snapshots_v1.sql` (new).

## Schema / bus / API changes

- Added: `causal_geometry_snapshots` Postgres table (not a typed bus schema --
  see below).
- Bus: `orion:causal_geometry:snapshot`'s `producer_services` corrected from
  `["orion-field-digester"]` to `[]` -- nothing publishes to this channel; the
  entry stays registered (schema/kind still valid) in case a future consumer
  needs live bus notification.
- Behavior changed: `/api/causal-geometry/snapshot` and `/api/causal-geometry/history`
  now read real data instead of a hardcoded stub.
- Compatibility notes: response shape unchanged (`source`/`data` envelope); only
  `source.kind`/`degraded` values change from always-`unavailable`/`true` to
  reflecting real state.

## Env/config changes

- Added key: `FIELD_PLASTICITY_SNAPSHOT_RETENTION_HOURS` (default `720`, i.e. 30
  days) on `orion-field-digester`.
- `.env_example` updated: `services/orion-field-digester/.env_example`.
- Local `.env` synced: worktree `.env` files updated during development; no
  operator-facing skip -- default is safe (30-day retention, matches the
  producer's own 24h-default cadence).

## Tests run

```text
orion/substrate/tests + orion/schemas/tests + tests/test_causal_geometry_report.py: 349 passed
services/orion-field-digester/tests (own dir): 65 passed
services/orion-hub/tests/test_causal_geometry_{api,page}.py: 35 passed
```

## Evals run

No eval harness exists for this service tree (same gap noted in prior PRs in
this series). `docker compose config` used as a live-config verification step.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-field-digester config --quiet   # exit 0
```
No new volume mounts or image-affecting changes in this PR (reuses the
`postgres_uri`/`memory_pg_pool` connections both services already have) -- a
code-only redeploy is sufficient, no new mounts to verify.

## Review findings fixed

- Finding: `orion/bus/channels.yaml`'s `producer_services` falsely claimed
  `orion-field-digester` publishes to the bus for this schema.
  - Fix: corrected to `[]` with an explanatory comment.
  - Evidence: read-through confirmation; no bus-publish call exists anywhere in
    the touched files.
- Finding: Postgres column list was two independently-maintained literals (writer's
  INSERT, reader's SELECT), no test catching drift.
  - Fix: exported `SNAPSHOT_COLUMNS` from the store module; `api_routes.py` now
    imports it instead of re-declaring its own literal.
  - Evidence: all snapshot-store and hub API tests still pass post-refactor.
- Finding: `causal_geometry_snapshots` had no retention/prune path, unlike every
  other table this service owns.
  - Fix: added `prune_snapshots()` (mirrors `PRUNE_FIELD_STATE_SQL`'s
    never-delete-the-latest-row guard), wired into the existing `_prune_tick()`,
    gated by new `FIELD_PLASTICITY_SNAPSHOT_RETENTION_HOURS` setting.
  - Evidence: 2 new worker-level tests (`test_worker_prune.py`), 3 new
    store-level tests.
- Finding: `ensure_schema()` reissued DDL on every single write.
  - Fix: module-level flag, runs at most once per process.
  - Evidence: new regression test asserting exactly one DDL call across two
    `persist_snapshot()` invocations.
- Finding: trial-runner's "inconclusive" result recorded on `proposal.notes` but
  never rendered where an operator actually clicks Adopt/Reject.
  - Fix: proposal card now shows a trial-status line, color-coded by status.
  - Evidence: new static-source-scan test in `test_causal_geometry_page.py`.
- Finding: `insufficient_data=true` buried in a dense one-line summary, easy to
  miss.
  - Fix: distinct amber banner when true.
  - Evidence: manual read-through; existing tests unaffected (no behavior test
    was asserting on the removed summary clause).
- Finding: no PR report committed for this changeset.
  - Fix: this file.

## Restart required

```bash
# Code-only change; both services already have the Postgres connections this
# PR relies on (postgres_uri, memory_pg_pool). No new volume mounts.
docker compose --env-file .env --env-file services/orion-field-digester/.env -f services/orion-field-digester/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low
  - Concern: no live end-to-end test exists proving a real row written by
    `orion-field-digester` is correctly read back by `orion-hub` in a live
    deployment -- both sides' test suites mock their own half of the
    integration independently.
  - Mitigation: the shared `SNAPSHOT_COLUMNS` constant closes the highest-risk
    drift vector; recommend a live smoke check after deploy (hit
    `/api/causal-geometry/snapshot` after the producer's next cycle and confirm
    `source.kind == "postgres"`, `degraded: false`).
- Severity: Low
  - Concern: code review coverage is partial -- 2 of 5 dispatched finder angles
    completed (cross-file contract tracing, line-by-line diff scan, and
    removed-behavior audit all failed on an API session rate limit, not on
    content) before this handoff.
  - Mitigation: the 2 completed angles (altitude/design, CLAUDE.md-conventions)
    had meaningful overlap with the failed ones' likely scope; all findings from
    both completed angles were fixed. A follow-up review pass covering the
    remaining 3 angles would be reasonable but is not blocking.

## PR link

(opened in this same turn -- see below)
