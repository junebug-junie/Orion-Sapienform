# feat(sql-writer): storage-write organ -- Orion feels whether its own writes land

## Summary

- The SQL writer (the service that turns most bus traffic into Postgres rows) now counts its own
  write outcomes: committed, idempotent duplicate, or lost, and if lost, why (schema reject,
  constraint, serialization, database unavailable, timeout, other DB error, grammar queue shed).
- Once a minute it publishes one small report per table on the grammar bus (`sql_writer.storage:`
  traces). Counts and class names only, never payloads or error text. At most 34 events a minute,
  whatever the write volume, and it never counts its own reports, so it cannot feed back on itself.
- A new substrate reducer lane turns that into one number, `write_failure_pressure`: the worst
  table's share of writes that did not land over the last 10 minutes, with the same floor and
  2-failure hysteresis the RPC-delivery and inference lanes use.
- The field digester puts it on `node:substrate.storage_write`, and a new topology edge carries it
  into `capability:storage` `reliability_pressure`. If the writer goes quiet, the channel is dropped
  after 180 s: storage reads "unknown", never a held calm value.
- Replayed over real history: three calm hours read a measured 0.0 every minute; the 2026-09-26
  home-cooling outage reads 1.0 within a minute and holds for 72 minutes until the fix.

## Outcome moved

Before: `capability:storage` had only athena's disk and memory pressure, both feeding `pressure`.
Its `reliability_pressure` read **0.0 with no provenance at all** (live, 2026-10-02 04:5x UTC:
`capability_provenance['capability:storage'] = {"pressure": "node:athena"}`): a calm number with
nothing behind it. 863 home-cooling rows failed to serialize on 2026-09-26 and 182 cockpit hops
were rejected on 2026-09-07 while that number stayed 0.0.

After: the number is measured from the writer's own outcomes, names the table that set it, and
goes unmeasured (not calm) when the writer stops reporting.

## Current architecture

- `orion-sql-writer` (`app/worker.py`) consumes most bus channels. Non-grammar envelopes go through
  `handle_envelope` -> `_handle_envelope_body`; every lost write ends in `_write_fallback`
  (a `bus_fallback_log` row with the error text). Grammar events are persisted by sharded worker
  tasks (`_persist_grammar_event_envelope` / `_persist_grammar_trace_batch_envelope`), with their
  own timeout/fallback paths and a queue-full shed.
- What already watched it: `app/fallback_watch.py` (alerts on the *unrouted* backlog in a 24 h
  window, via notify), `app/route_coverage.py` (boot-time route drift), `/health` and
  `/grammar/truth` (reducer cursor lag), `scripts/check_postgres_connection_headroom.py`
  (connection count vs `max_connections`). None measures the share of writes that land, and none
  reaches the field.
- `capability:storage` edge: `node:athena` `disk_pressure`/`memory_pressure` -> `pressure` only.

## Architecture touched

- Producer: `orion-sql-writer` (`app/write_health.py` new; hooks in `app/worker.py`; publisher in
  `app/main.py`).
- Contract: `orion/schemas/storage_write_projection.py` (new, `extra="forbid"`), registry,
  `orion/bus/channels.yaml` (`orion-sql-writer` added as an `orion:grammar:event` producer).
- Reducer: `orion/substrate/storage_write_loop/` (new lane), wired into `orion-substrate-runtime`
  (`REDUCER_SPECS[6]`, own cursor `storage_write_grammar_reducer`, own projection table).
- Field: `orion-field-digester` (`storage_write` delta branch, `write_failure_pressure` channel,
  single-observer, expiring at 180 s), topology edge, glossary.
- Retention: sql-writer's `GRAMMAR_LANES` mirror gains the lane, so grammar retention protects
  unconsumed storage-write events like every other lane.

## Files changed

- `services/orion-sql-writer/app/write_health.py`: classifier, window recorder, per-envelope outcome (contextvar), window -> grammar events, publisher.
- `services/orion-sql-writer/app/worker.py`: hooks at `handle_envelope`, `_write`, `_write_fallback`, grammar persist helpers, queue-full shed, vision-crop success; `_write_family_for_kind`.
- `services/orion-sql-writer/app/main.py`: starts the publisher when the flag is on.
- `services/orion-sql-writer/app/grammar_ledger_handler.py`: records what each persist call actually did (thread-local outcome); return values unchanged.
- `services/orion-sql-writer/app/cockpit_turn_sighting_persist.py`, `app/harness_turn_trace_persist.py`: report a rejected row as a validation failure before returning `False`.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flag + window, docs.
- `services/orion-sql-writer/app/grammar_truth.py`: lane added to the retention lane table.
- `services/orion-sql-writer/tests/test_write_health.py`: classifier on the three real error texts, live-path hooks through the real `handle_envelope`, grammar path, self-exclusion, publisher.
- `services/orion-sql-writer/tests/test_grammar_retention_periodic.py`: lane mirror pinned.
- `services/orion-sql-writer/evals/storage_write_replay.py`, `evals/fixtures/storage_write_replay.json`, `evals/test_storage_write_replay_eval.py`: replay of real history (live pull or committed fixture) through the real emitter and reducer.
- `orion/schemas/storage_write_projection.py`, `orion/schemas/registry.py`: contract.
- `orion/substrate/storage_write_loop/{__init__,constants,extract,failure_window,reducer,pipeline}.py`: the lane.
- `services/orion-substrate-runtime/app/{worker,store,settings,grammar_truth}.py`, `.env_example`, `docker-compose.yml`, `README.md`: lane wiring.
- `services/orion-substrate-runtime/tests/test_worker_storage_write_tick.py` (new), `tests/test_worker_independent_reducers.py`: tick + poll task.
- `services/orion-sql-db/manual_migration_storage_write_substrate_loop.sql`: projection table + cursor row.
- `services/orion-field-digester/app/{ingest/state_deltas,tensor/channels,digestion/decay,settings,worker}.py`, `.env_example`, `docker-compose.yml`, `README.md`: field side.
- `services/orion-field-digester/tests/test_field_storage_write_perturbations.py`: delta -> field -> capability, expiry, seeding, gate.
- `config/field/orion_field_topology.v1.yaml`, `config/field/field_channel_glossary.v1.yaml`, `tests/test_field_channel_glossary.py`: edge, channel meaning.
- `config/metrics/metric_definitions.lock.json`: re-locked (new channel, producer routing).
- `scripts/sync_local_env_from_example.py`: three new prefixes (none matched an existing one).
- `scripts/check_substrate_projection_schema_drift.py`: new projection table covered.
- `tests/test_storage_write_substrate_reducer.py`: producer -> reducer round trips.
- `.github/workflows/orion-sql-writer-tests.yml`: new tests and eval in CI.

## Schema / bus / API changes

- Added: `StorageWriteFamilyStateV1`, `StorageWriteWindowCountV1`, `StorageWriteProjectionV1`
  (registered). Trace prefix `sql_writer.storage:<writer>:<window_id>`, atom roles
  `storage_write_window_observed` (one per table family) and `storage_writer_window_completed`.
  Table `substrate_storage_write_projection`, cursor `storage_write_grammar_reducer`.
  `orion-sql-writer` listed as an `orion:grammar:event` producer.
- Removed / renamed: none.
- Behavior changed: none while the flags are off. With them on, the writer publishes ~2-34
  grammar events a minute and the field gains one channel and one edge.
- Compatibility: the new models are new, nothing reads them yet besides this lane. Deploy order is
  consumer-first (migration, then substrate-runtime and field-digester, then the writer), so no
  window is published before something can reduce it. Every stage tolerates the others being off.

## Env/config changes

- Added keys: `SQL_WRITER_WRITE_HEALTH_ENABLED`, `SQL_WRITER_WRITE_HEALTH_WINDOW_SEC` (sql-writer);
  `ENABLE_STORAGE_WRITE_REDUCER`, `STORAGE_WRITE_GRAMMAR_BATCH_LIMIT` (substrate-runtime);
  `ENABLE_STORAGE_WRITE_FIELD_DIGESTION` (field-digester). All flags **off in code, on in
  `.env_example`** and in compose they default to `false` when unset.
- Removed / renamed keys: none.
- `.env_example` updated: yes (three services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, the primary
  checkout's three service `.env` files now carry the keys, flags `true`.
- Skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md 0A) -- `write_failure_pressure`

1. **Provenance.** Producer: `services/orion-sql-writer/app/write_health.py`. Committed/duplicate
   from `_write_row`'s return value at `worker._write`; failures from the error text passed to
   `worker._write_fallback` (every lost write on the envelope path ends there) or the exception
   escaping `handle_envelope`; grammar events from the persist helpers' own success / timeout /
   exception branches and the queue-full shed. Reading: `orion/substrate/storage_write_loop/
   failure_window.py::failure_reading` over the writer's windows.
2. **Independence.** Not a transform of anything in `capability:storage`: athena's
   `disk_pressure`/`memory_pressure` are host resource biometrics; a full disk would move both, but
   most failures seen live (schema rejects, serialization bugs) move only this one. Not
   `fallback_watch` (counts *unrouted* rows for an alert; unrouted is excluded here). Not the
   connection-headroom watch (connection count vs max: an upstream cause of one failure class,
   `db_unavailable` "too many clients", not the same measurement). Not `rpc_timeout_pressure`
   (the writer is a pub/sub consumer; it makes no RPC calls for these writes). Not grammar truth's
   cursor lag (that is reducers falling behind, not writes failing).
3. **Theory anchor.** Error-rate service-level indicator: good events / valid events for the
   storage path as its clients experience it (SRE "availability" SLI). Valid = writes the writer
   attempted; good = committed or already-present duplicate. Worst-table `max` because the
   dominant live failure mode (schema or serialization drift) is local to one table and would be
   diluted to ~1% in a pooled share. Floor and hysteresis reused from `rpc_delivery.hop_pressure`
   (`RpcDeliveryConfig`: 600 s, floor 10, min 2 failures), not re-picked.
4. **Live data.** Replay of real history through the real emitter and reducer
   (`evals/storage_write_replay.py --live`, pulled 2026-10-02):
   - Calm, last 3 h: 175 of 175 minutes measured, every one exactly 0.0, with >1,000 real writes in
     every 10-minute span. The rest state is reachable and is exactly 0 (a count ratio with a floor,
     not a mean of |z|-scores), and a 0 is only ever written when writes were actually attempted.
   - Non-calm, 2026-09-26 home-cooling serialization outage (863 rows lost, first loss 03:39:50,
     fix 04:52): 0.2 at 03:39, **1.0 from 03:40 to 04:51 (72 minutes)**, then 0.95 -> 0.03 as the
     failures leave the 10-minute span, 0.0 from 05:02. Before 03:39 the minute reads unmeasured,
     not calm (no writes yet).
   - Non-calm, 2026-09-07 cockpit validation rejects (182 rejects mixed with 191 commits): 237
     nonzero minutes between 08:11 and 18:46, peak 1.0; no minute fires on a single reject.
   - Not replayable: 2026-09-21 Postgres restart (21 grammar writes lost in one minute).
     `grammar_events` is retained ~3 days, so the denominator for that minute is gone.
   - Expected steady state: `bus_fallback_log` has had **0 rows since 2026-09-27**, so live this
     will read 0.0 almost all the time. That is a true reading (writes are landing), not a pinned
     field: it moved to 1.0 on both real incidents, and silence is handled by expiry, not by 0.
5. **Existing mechanism.** Checked (see Current architecture). Reused: `hop_pressure` /
   `RpcDeliveryConfig`, the grammar-lane reducer pattern (`llm_inference_loop`), the expiring
   single-observer channel pattern (`rpc_timeout_pressure`), the error text the writer already
   records for `bus_fallback_log`.
6. **Reversibility.** Three flags. Removing it means: flags off, drop the edge, add the channel to
   `RETIRED_NODE_CHANNELS`, drop one table. No training default, no manifest.

Scope limits, stated: failures swallowed with only a log line and no fallback row (side writes such
as evidence units after a successful primary write, a spark-meta patch that finds no row) are not
counted as failures; retention/maintenance statement timeouts are not writes of incoming data and
are not counted; HTTP-path writes (`api_notify.py`) are not instrumented.

## Tests run

```text
tests/test_storage_write_substrate_reducer.py + transport/energy CI step ......... 92 passed
sql-writer CI unit step + cockpit/harness/fallback-drain/retention suites (incl. tests/test_write_health.py, replay eval) .. 307 passed
services/orion-substrate-runtime/tests/test_worker_storage_write_tick.py ......... 5 passed
services/orion-field-digester/tests (full) ............ 262 passed, 6 skipped, 1 error (pre-existing: test_heartbeat_chassis FileNotFoundError, also on main)
services/orion-substrate-runtime/tests (full): 13 failures + collection errors, identical set on main (verified by running the same ids on the untouched main checkout)
services/orion-sql-writer/tests (full): 7 failures, identical set on main incl. the 5 notify suite-order failures
static gates: grammar producer catalog, field topology edges, field channel glossary, agent trace registry,
  env sync + parity, substrate ladder/schema skew/migration drift, substrate services declare requests,
  metric lineage --gate PASS, definition drift --gate PASS (after re-lock), inner state registry,
  hostname refs, compose mounts, journal dispatch, system health producers, control surface parity,
  async routes, chat poachers, circe worker refs, sentience instruments --static-only: all pass
Mutation checks: removing the _write_fallback hook fails 3 envelope tests; ignoring the ledger
handler's reported outcome fails 3 grammar tests.
```

## Evals run

```text
services/orion-sql-writer/evals/test_storage_write_replay_eval.py ..... 3 passed (fixture = live pull)
python services/orion-sql-writer/evals/storage_write_replay.py --live
  calm_recent_3h                        175 min, 175 measured, 0 nonzero, peak 0.0
  home_cooling_serialization_2026-09-26 180 min, 111 measured, 83 nonzero, peak 1.0 (03:40), 1.0 held 03:40-04:51
  cockpit_validation_2026-09-07         720 min, 308 measured, 237 nonzero, peak 1.0
```

## Docker/build/smoke checks

```text
docker build -f services/{orion-sql-writer,orion-substrate-runtime,orion-field-digester}/Dockerfile .   -> all three built
in-image import smoke:
  sql-writer:        import app.write_health, app.worker, app.main -> ok
  substrate-runtime: REDUCER_SPECS[6].reducer_key == storage_write -> ok
  field-digester:    write_failure_pressure in NODE_CHANNELS, EXPIRING {'write_failure_pressure': 180.0}, topology edge present -> ok
scripts/safe_docker_build.sh <svc> build: refused in the worktree (no .env there, gitignored); built directly instead.
Live rail: UNVERIFIED -- not deployed (production deploys are from main). Proof queries below.
```

## Review findings fixed

Review: `orion-repo-agent` subagent, target `git diff origin/main...HEAD`. One blocker, eight
should-fix, six nits. Fixed:

- Finding (blocker): grammar persists counted lost writes as committed. `persist_grammar_event` /
  `persist_grammar_trace_batch` swallow statement cancels (the server-side grammar statement
  timeout) and every `IntegrityError`, and return `False`/`0`, which the hook read as success.
  - Fix: the ledger handler now records what each call actually did (committed / duplicate /
    timeout / constraint, per event) in a thread-local; the worker runs the persist through
    `_grammar_persist_with_outcome` in the same executor thread and records that. Unique
    violations (23505) are duplicates; other integrity errors are `constraint`. Return values and
    every other caller (fallback drain) are unchanged.
  - Evidence: `test_real_ledger_handler_reports_what_return_false_meant` (5 cases through the real
    handler) and `test_real_ledger_batch_partial_dedupe_and_cancel`; ignoring the reported outcome
    fails 3 of them.
- Finding: cockpit-hop and harness-trace helpers return `False` for rejected rows (no fallback
  row), which read as an idempotent duplicate.
  - Fix: both helpers report the reject (`validation`) before returning `False`. Energy upserts
    always return `True` (checked).
  - Evidence: `test_cockpit_reject_is_a_validation_failure_not_a_duplicate`.
- Finding: "any failure beats committed" counted a landed row as lost when a post-commit step
  (memory-turn emit, publish) raised into the shared fallback handler; a side-table commit could
  upgrade a primary duplicate.
  - Fix: the envelope's own table decides. Precedence: primary committed > failure > primary
    duplicate > secondary committed > secondary duplicate > unrouted > skipped.
  - Evidence: `test_exception_after_the_primary_commit_does_not_count_as_lost`,
    `test_secondary_commit_does_not_upgrade_a_primary_duplicate`, updated precedence test.
- Finding: an exception while building a window killed the publisher task for good, silently.
  - Fix: each window is wrapped; a failure is logged and the loop continues.
  - Evidence: `test_publisher_survives_a_window_that_fails_to_build`.
- Finding: Postgres `DETAIL:` lines (row data) come before `[SQL:` and could pick the class; the
  plural pydantic message ("2 validation errors for") classified as `other`.
  - Fix: match only the first line, cut at `detail:` (also when newlines were flattened), needle
    `validation error`.
  - Evidence: `test_classifier_ignores_postgres_detail_and_reads_plural_validation` (5 cases).
- Nits fixed: vision crop counts `committed` only when a row was written; the family lookup is
  skipped when the organ is off; latency renamed `write_p50_ms`/`write_p95_ms` and documented as
  handler wall time, not commit latency; marker typo `(background on this error`.
- Not changed (stated instead): shutdown `CancelledError` is not classified (one envelope at
  shutdown); the last partial window is lost on shutdown; grammar-family and full-outage failures
  are under-reported because the report travels the path it reports on (documented in
  `write_health.py` and Risks).
- Also fixed before review: an unrouted kind that the evidence-unit adapter still wrote was
  reported under `unrouted`; it now reports under `evidence_units`
  (`test_unrouted_kind_written_as_evidence_units_reports_that_table`).

## Restart required

Deploy in this order, from the primary checkout on main after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_storage_write_substrate_loop.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Proof queries (`docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "..."`):

```sql
-- 1. the writer reports (one trace a minute)
select created_at, event_json->'atom'->>'summary' from grammar_events
 where source_service = 'orion-sql-writer' and trace_id like 'sql_writer.storage:%'
 order by created_at desc limit 5;
-- 2. the reducer's cursor moves
select * from substrate_reduction_cursor where cursor_name = 'storage_write_grammar_reducer';
-- 3. the reading and what set it
select generated_at, projection_json->'write_failure_pressure', projection_json->'reading'
  from substrate_storage_write_projection;
-- 4. the field: node channel, capability reliability, and its provenance
select generated_at,
       field_json->'node_vectors'->'node:substrate.storage_write',
       field_json->'capability_vectors'->'capability:storage'->'reliability_pressure',
       field_json->'capability_provenance'->'capability:storage'
  from substrate_field_state order by generated_at desc limit 3;
```

## Risks / concerns

- Severity: medium. Concern: during a full Postgres outage the writer's own report cannot be
  stored either (it is a grammar event, and the reducer reads grammar from Postgres), so storage
  reads "unmeasured", not "failing". Windows published during the outage are lost; only the first
  window stored after recovery shows failures, and only those counted in that minute.
  Mitigation: expiry makes it unknown, never calm. Making an outage read as failing would need a
  path that does not go through Postgres (bus-direct to the digester); deliberately not built here.
- Severity: low. Concern: steady state is 0.0 almost always (0 fallback rows in 5 days). Mitigation:
  replay shows it moving on both real incidents; absence is expiry, not 0.
- Severity: low. Concern: substrate-runtime and field-digester tests for this lane run locally, not
  in CI (same as the rpc_delivery and llm_inference lanes before it). Mitigation: the round-trip
  reducer test and the replay eval are in CI.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
