# Causal Geometry v1: closing the spec gap list

## Summary

- Persist Phase A snapshots so the hub's Snapshot/History panels stop being
  permanently `degraded: true`. **Correction mid-PR**: the first version of
  this routed `orion-field-digester` writing directly to Postgres, bypassing
  the bus entirely, on the reasoning that `orion-sql-writer`'s worker (2300
  lines, shared by a dozen unrelated pipelines) was too invasive to touch for
  the value. That reasoning was wrong -- the bus is this repo's mechanism for
  tracking load/failures across services, and a new write path silently
  bypassing it is a real observability regression, not a stylistic choice.
  Corrected (Rung 5, `63fe570d`): `orion-field-digester` now publishes
  `CausalGeometrySnapshotV1` on `orion:causal_geometry:snapshot`;
  `orion-sql-writer` consumes it via its standard `MODEL_MAP`/
  `DEFAULT_ROUTE_MAP` routing (a small, declarative addition -- one route-map
  entry, one model-map entry, one new SQLAlchemy model, not surgery on the
  shared worker file) and writes the table; `orion-hub` reads it via its
  existing `memory_pg_pool`.
- Wire `orion/substrate/mutation_trials.py`'s `SubstrateTrialRunner` into the
  live proposal pipeline (`orion/substrate/causal_geometry_producer.py`) -- it
  existed and was unit-tested standalone but was never actually invoked.
  Always resolves to `"inconclusive"` today (no replay corpus registered for
  this mutation class) -- expected and correct; the point is the step is
  genuinely exercised and honestly recorded, not gating enqueueing.
- Hub UI (`static/js/causal-geometry.js`) renders Snapshot/History as real DOM
  tables with divergence highlighting instead of raw JSON dumps, plus gentle
  polling (45s) that pauses on `document.visibilitychange` and resumes with an
  immediate refresh. Proposal cards now surface the trial-runner's status.
- Code review (2 of 5 dispatched angles completed before an API session rate
  limit; not re-dispatched given solid non-overlapping coverage from the two
  that landed) found and fixed 6 real issues before the bus-routing
  correction (see below); the correction itself was driven directly by
  operator review, not the automated pass.

## Outcome moved

Before this branch, the design spec's Phase A persistence requirement and
Phase B's trial-runner step were both documented gaps -- the hub's Causal
Geometry tab looked "dead" (Snapshot/History always degraded) even once the
producer was live, and the spec's `proposal -> trial -> HITL adopt` mechanism
skipped the trial step entirely. Both are now closed, live-testable end to
end, and routed through this repo's bus/sql-writer convention rather than
around it.

## Current architecture

Before this PR: `orion-field-digester`'s scheduled producer (from PR
#1093/#1095) measured real data and enqueued HITL proposals, but the
underlying snapshot was never persisted anywhere the hub could read, and
`SubstrateTrialRunner` was dead code outside its own test file.

## Architecture touched

- `orion/substrate/causal_geometry_bus_publish.py` (new) -- one-shot bus
  publish (connect/publish/close per cycle, mirrors
  `services/orion-collapse-mirror/app/routes.py`'s pattern).
- `orion/substrate/causal_geometry_producer.py` -- publishes instead of
  writing Postgres directly; calls the trial runner per proposal.
- `orion/schemas/causal_geometry.py` -- new shared
  `CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS` constant, the single source of truth
  for the table's column list on both the write and read side.
- `orion/bus/channels.yaml` -- `producer_services`/`consumer_services`
  restored to reflect the real (bus-routed) wiring.
- `services/orion-field-digester/app/{worker.py,settings.py}`, `.env_example`,
  `docker-compose.yml`, `requirements.txt` -- new `ORION_BUS_URL`/
  `ORION_BUS_ENABLED` settings (this service had zero bus-publish capability
  before) and the `redis` dependency; removed the (now-moot)
  snapshot-retention prune wiring, since this service no longer owns the
  table.
- `services/orion-sql-writer/app/models/causal_geometry_snapshot.py` (new) --
  `CausalGeometrySnapshotSQL`, registered in `MODEL_MAP`, `DEFAULT_ROUTE_MAP`,
  and `SQL_WRITER_SUBSCRIBE_CHANNELS` (settings default list AND
  `.env_example`'s literal -- missing the subscribe-list entry is the exact
  "route_map correct but Redis never delivers" failure mode this repo's
  `test_drive_audit_sql_shape.py` was written to catch after a prior
  incident).
- `services/orion-hub/scripts/api_routes.py` -- async Postgres read path,
  column list imported from the new shared schema constant.
- `services/orion-hub/static/js/causal-geometry.js`,
  `templates/causal_geometry.html` -- table rendering, polling, trial-status
  surfacing, `insufficient_data` banner.
- `services/orion-sql-db/manual_migration_causal_geometry_snapshots_v1.sql`
  (new) -- documentation, matching this repo's convention for new tables;
  rewritten mid-PR to describe the corrected bus-routed ownership.

## Files changed

Rungs 1-4 (`703dcb1c`, `d2342091`, `0426916c`, `93015cd3`): initial direct-
Postgres-write implementation, hub read path, hub UI polish, and a first
code-review-fix pass. Superseded by Rung 5 below wherever they touched the
now-deleted `causal_geometry_snapshot_store.py`.

Rung 5 / bus-routing correction (`63fe570d`):
- New: `orion/substrate/causal_geometry_bus_publish.py`,
  `orion/substrate/tests/test_causal_geometry_bus_publish.py`,
  `services/orion-sql-writer/app/models/causal_geometry_snapshot.py`,
  `services/orion-sql-writer/tests/test_causal_geometry_snapshot_sql_shape.py`.
- Deleted: `orion/substrate/causal_geometry_snapshot_store.py`,
  `orion/substrate/tests/test_causal_geometry_snapshot_store.py`.
- Modified: `orion/bus/channels.yaml`, `orion/schemas/causal_geometry.py`,
  `orion/substrate/causal_geometry_producer.py`,
  `orion/substrate/tests/test_causal_geometry_producer.py`,
  `services/orion-field-digester/{.env_example,app/settings.py,app/worker.py,
  docker-compose.yml,requirements.txt,tests/test_worker_causal_geometry_producer.py,
  tests/test_worker_prune.py}`, `services/orion-hub/scripts/api_routes.py`,
  `services/orion-hub/tests/test_causal_geometry_api.py`,
  `services/orion-sql-db/manual_migration_causal_geometry_snapshots_v1.sql`,
  `services/orion-sql-writer/{.env_example,app/models/__init__.py,
  app/settings.py,app/worker.py}`.

## Schema / bus / API changes

- Added: `causal_geometry_snapshots` Postgres table, owned by
  `orion-sql-writer` (not a typed bus schema of its own -- persisted via the
  existing `CausalGeometrySnapshotV1` schema already registered).
- Bus: `orion:causal_geometry:snapshot` now has a real producer
  (`orion-field-digester`) and consumer (`orion-sql-writer`), matching the
  channel's original registration from PR #1087 (a mid-PR detour temporarily
  zeroed both lists out, then this correction restored them).
- Behavior changed: `/api/causal-geometry/snapshot` and
  `/api/causal-geometry/history` read real data instead of a hardcoded stub.
- Compatibility notes: response shape unchanged (`source`/`data` envelope);
  only `source.kind`/`degraded` values change from always-`unavailable`/`true`
  to reflecting real state.

## Env/config changes

- Added keys (`orion-field-digester`): `ORION_BUS_URL`
  (`redis://100.92.216.81:6379/0`, the real tailscale address per root
  CLAUDE.md's bus convention), `ORION_BUS_ENABLED` (`true`).
- Added keys (`orion-sql-writer`): `orion:causal_geometry:snapshot` appended
  to `SQL_WRITER_SUBSCRIBE_CHANNELS`'s default list and `.env_example`
  literal; `causal.geometry.snapshot.v1` appended to
  `SQL_WRITER_ROUTE_MAP_JSON`.
- Removed key (`orion-field-digester`): `FIELD_PLASTICITY_SNAPSHOT_RETENTION_HOURS`
  (added in Rung 4, removed in Rung 5 -- this service no longer writes the
  table it would have pruned).
- `.env_example` updated: `services/orion-field-digester/.env_example`,
  `services/orion-sql-writer/.env_example`.
- Local `.env` synced during development in the worktree; live deploy will
  need the same sync (`python scripts/sync_local_env_from_example.py
  --all-keys` covers non-default-prefix keys like these).

## Tests run

```text
orion/substrate/tests + orion/schemas/tests + tests/test_causal_geometry_report.py: 345 passed
services/orion-field-digester/tests (own dir): 63 passed
services/orion-hub/tests/test_causal_geometry_{api,page}.py: 35 passed
services/orion-sql-writer/tests/test_causal_geometry_snapshot_sql_shape.py + test_route_map_completeness.py: 14 passed
services/orion-sql-writer/tests (full suite, excluding test_dream_model_constraints.py --
  a pre-existing SQLAlchemy table-registration collision confirmed identical on
  unmodified main via a fresh detached worktree): 19 failed / 130 passed,
  exactly matching the pre-existing baseline (120 passed + this PR's 10 new
  tests = 130) -- no new regressions.
```

## Evals run

No eval harness exists for this service tree (same gap noted in prior PRs in
this series).

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-field-digester config --quiet   # exit 0
```
`orion-sql-writer` and `orion-hub` config not re-validated live in this
session (no new volume mounts for either; `orion-sql-writer`'s new model adds
a table via `Base.metadata.create_all`, not a compose/env change beyond the
subscribe-channel and route-map keys already validated above).

## Review findings fixed (Rungs 1-4, before the bus-routing correction)

- Finding: `orion/bus/channels.yaml`'s `producer_services` didn't match
  reality at the time. Fix: corrected (superseded again by Rung 5's restore).
- Finding: Postgres column list was two independently-maintained literals.
  Fix: shared constant (carried forward into Rung 5's
  `CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS`, now genuinely shared via
  `orion.schemas.causal_geometry` instead of one service reaching into
  another's internals).
- Finding: no retention/prune path for the table. Superseded: `orion-sql-writer`
  now owns this table; `orion-field-digester`'s prune wiring was removed as
  part of the Rung 5 correction rather than carried forward. Revisit table
  growth if a future rung raises the producer's cadence significantly.
- Finding: `ensure_schema()` reissued DDL on every write. Superseded: Rung 5's
  `orion-sql-writer` model uses `Base.metadata.create_all` at service boot,
  not per-write DDL.
- Finding: trial-runner's "inconclusive" result never surfaced to the
  operator. Fix: proposal card shows a trial-status line, color-coded by
  status (unaffected by the Rung 5 correction, still in place).
- Finding: `insufficient_data=true` under-emphasized in the UI. Fix: distinct
  amber banner (unaffected by the Rung 5 correction, still in place).

## Review findings fixed (Rung 5, bus-routing correction)

- Finding: the original direct-Postgres-write design bypassed the bus, this
  repo's mechanism for tracking cross-service load/failures.
  - Fix: full rework described above.
  - Evidence: `orion/substrate/causal_geometry_bus_publish.py`'s tests confirm
    a real `OrionBusAsync.connect()`/`publish()`/`close()` cycle; grepped
    every touched file for bus-publish calls to confirm the old direct-write
    path is fully gone, not left as dead parallel code.
- Finding (self-caught via a real, non-mocked test): the generic sql-writer
  write path builds column values via `obj.model_dump()` (not
  `mode="json"`), so nested `datetime` fields inside `edges`/`divergence`
  (each `CausalGeometryEdgeV1` carries its own `window_start`/`window_end`)
  survive as native Python objects -- neither SQLite's `JSON` type (used in
  tests) nor psycopg2's default `JSONB` adapter (prod) can serialize a raw
  `datetime` nested inside a dict/list.
  - Fix: `_JSONSafeList`, a `TypeDecorator` wrapping the JSON/JSONB variant
    type, round-trips the value through `json.dumps(..., default=str)` /
    `json.loads(...)` in `process_bind_param` before it reaches the DB
    driver, regardless of which dump mode upstream code used.
  - Evidence: `test_insert_only_redelivery_keeps_one_row_and_first_write_wins`
    failed with `TypeError: Object of type datetime is not JSON serializable`
    against an in-memory SQLite engine before the fix; passes after.
- Finding: `SQL_WRITER_SUBSCRIBE_CHANNELS` (the Redis subscribe list) is a
  separate concern from `MODEL_MAP`/`DEFAULT_ROUTE_MAP` (the kind->model
  routing) -- registering the model/route alone does nothing if the
  subscriber never listens on the channel the message actually arrives on.
  - Fix: added `orion:causal_geometry:snapshot` to both the settings default
    list and the `.env_example` literal (which replaces the default entirely
    in live deployments).
  - Evidence: two new tests mirroring `test_drive_audit_sql_shape.py`'s
    `test_channel_is_in_settings_default_list`/
    `test_channel_is_in_env_example_subscribe_channels`, added specifically
    because this exact failure mode has hit this repo before.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env -f services/orion-field-digester/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-sql-writer/.env -f services/orion-sql-writer/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```
`orion-sql-writer` needs a rebuild (new model, needs `Base.metadata.create_all`
to run against a live Postgres to create the table) in addition to the other
two services already covered by earlier rungs.

## Risks / concerns

- Severity: Low
  - Concern: no live end-to-end test exists proving a real bus message
    published by `orion-field-digester` is correctly consumed by
    `orion-sql-writer` and read back by `orion-hub` in a live deployment --
    each service's test suite mocks its own half of the integration
    independently.
  - Mitigation: the shared `CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS` constant and
    the subscribe-channel tests close the two highest-risk drift/dead-wiring
    vectors found during this work. Recommend a live smoke check after
    deploy: enable `FIELD_PLASTICITY_PRODUCER_ENABLED`, wait one cycle,
    check `orion-sql-writer` logs for a `causal.geometry.snapshot.v1` write,
    then hit `/api/causal-geometry/snapshot` and confirm `source.kind ==
    "postgres"`, `degraded: false`.
  - Note: this same class of gap (mocked-both-sides, no live proof) is
    exactly what the direct-write version of this PR would have *avoided* by
    being simpler -- the bus-routed version is the architecturally correct
    choice, but it does have more moving parts to verify live. Worth being
    honest about that trade-off rather than only citing the correction's
    upside.
- Severity: Low
  - Concern: code review coverage for Rungs 1-4 was partial (2 of 5 dispatched
    finder angles completed before an API session rate limit); the Rung 5
    bus-routing correction itself was driven by direct operator review, not
    run through the same automated pass.
  - Mitigation: the correction's own test suite (14 new/updated tests across
    3 services) was written adversarially against the specific failure modes
    this repo has hit before (schema drift, dead subscribe-list wiring,
    JSON-serialization of nested datetimes) rather than only asserting happy
    paths -- a substitute for, not a replacement of, a full review pass.

## PR link

(to be opened after this report is committed -- see final response)
