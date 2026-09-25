## Summary

- The transport reducer reads the bus observer's readings about every 10 seconds. It could write made-up "half healthy" numbers over the real `bus:athena` reading when one observer reading reached it in two pieces. That can no longer happen.
- Now a reading is written only when it is complete, from "tick started" through "tick completed" or "tick failed". If a piece arrives on its own, the reducer re-reads that tick from `grammar_events` and writes the whole thing. If the tick still isn't complete, it writes nothing, and the previous real reading stays in place.
- Each reading is written once. A reading older than the one already stored is never written over it.
- Regression tests use one real, unedited 13-event observer tick pulled from `grammar_events` on 2026-09-25. They cut it at every position. The transport unit tests now run in CI (before this, no workflow ran them).

## Outcome moved

Before this patch, the part of a tick that arrived without the Redis health check wrote `redis_ping_ok=None`. That becomes `stream_backlog_health=0.5`, `delivery_confidence=0.5` and `reliability_pressure=0.5` over `bus:athena`, and the field digester passes those values on to the field (replace mode). The part without the census read as `catalog_drift_pressure=0` ("no drift", when nothing was measured), with `streams_observed` too low.

- On main's code, 6 of the 12 possible cut points in the real tick write a made-up value (`test_without_a_loader_a_piece_is_held_not_fabricated`, cuts 4-9: `redis_ping_ok None`, `streams_observed 0/1`, `undeclared_active_count None`).
- With this patch, 0 of 12 do, both with and without the trace re-read.

### How often this happens live (measured 2026-09-25)

- In `substrate_reduction_receipts` for `transport_bus_reducer` on `bus:athena` over the last 7 days, 0 of 202 deltas have `stream_backlog_health=0.5`. All 202 are 1.0. This table only keeps a sample: about 200 receipts out of 24,634 real athena ticks. So this rules out a common event, not a rare one.
- `docker logs orion-athena-substrate-runtime` since the 03:56 UTC restart has 856 `transport_incident_signal` lines. None of them has `reliability_pressure`. All are `catalog_drift_pressure≈0.0109` (3/276).
- **Why it's rare.** sql-writer commits each trace in one transaction (`persist_grammar_trace_batch`), so a poll never sees half of a trace that is still being inserted. A split only happens when the reducer is catching up, because it reads at most 500 events per batch and 500 isn't a multiple of 13: every such batch boundary cuts one tick. The other case is poison isolation, which replays events one at a time. In both cases the made-up values are usually overwritten within the same batch, so the final projection hides them. The delta sent to the field digester does not.
- **Expected after deploy.** Still 0 half-health deltas, now guaranteed rather than just unobserved. After a catch-up, `transport_incomplete_window_held` and `observer window already applied` show up in the logs and receipt warnings where the old code would have written made-up values.

## Current architecture

- `services/orion-substrate-runtime/app/worker.py::_transport_tick` fetches `bus.transport:`-prefixed orion-bus grammar events, ordered by `(created_at, event_id)` and limited by `TRANSPORT_GRAMMAR_BATCH_LIMIT`.
- `orion/substrate/transport_loop/pipeline.py` groups them by trace_id `bus.transport:<node>:<window>`. `reducer.py` turns each group into a full `TransportBusStateV1` that replaces `buses[bus:<node>]` and emits a `transport_bus` state delta.
- Before this patch, each group was reduced alone. A group holding part of a tick produced a complete-looking state with default values.

## Architecture touched

- The transport reducer now has a whole-tick rule, can re-read a trace through a loader, and guards against writing the same window twice or an older window over a newer one.
- The substrate-runtime store has a new read-only method, `fetch_transport_trace_events(trace_id)`, which uses the existing `idx_grammar_events_trace_id` index.
- The worker passes that loader in.
- CI: the sql-writer workflow now runs the four transport unit test files.

Honest and simplest choice: `grammar_events` already holds the whole tick, so the reducer reads it back when it needs it. Nothing new has to be persisted, the schema doesn't change, and the cursor doesn't have to be held back.

## Files changed

- `orion/substrate/transport_loop/reducer.py`: whole-tick rule, trace re-read, write-once and no-older-window guards, logging.
- `orion/substrate/transport_loop/pipeline.py`: passes `load_trace_events` through.
- `orion/substrate/transport_loop/extract.py`: `_ATOM_ROLES` renamed to `ATOM_ROLES` (made public, since the reducer now uses it).
- `services/orion-substrate-runtime/app/store.py`: `fetch_transport_trace_events`.
- `services/orion-substrate-runtime/app/worker.py`: wires the loader.
- `services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py`: Postgres test for the loader.
- `tests/test_transport_split_window_no_fake_health.py` and `tests/fixtures/transport_bus_observer_trace_live_2026-09-25.jsonl`: regression tests plus the real trace they use.
- `tests/test_transport_substrate_{reducer,pipeline}.py` and `tests/test_transport_rpc_timeout_not_a_bus.py`: fixtures now include `bus_observer_tick_started`, as every real observer trace does.
- `.github/workflows/orion-sql-writer-tests.yml`: runs the transport unit tests.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: a `bus.transport` trace without both `bus_observer_tick_started` and a terminal atom (`bus_observer_tick_completed` or `bus_observer_tick_failed`) no longer writes a bus state. Live check: 24,634 of 24,634 observer traces over 7 days carry both, so real traffic isn't held.
- Compatibility notes: no payload shape changed. The field digester reads only `after.pressure_hints` plus the ids, and neither changed.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, since no template changed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/test_transport_split_window_no_fake_health.py tests/test_transport_substrate_reducer.py \
       tests/test_transport_substrate_pipeline.py tests/test_transport_rpc_timeout_not_a_bus.py -q
  57 passed
Same new test file against main's orion/ package: the no-loader cut cases, the tail-alone case and the
  head-without-census case fail on assertions (the made-up values). The loader cases fail on main with
  TypeError, because the parameter does not exist there.
Same file against this branch's first commit: the atomic-commit write-once and older-window cases fail (7 failed).
pytest -m integration services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
  (throwaway postgres:16 container, not the live DB): 3 passed
pytest services/orion-substrate-runtime/tests -m "not integration": 360 passed, 16 failed. The same 16
  names fail on unmodified main in this local environment (settings/.env/config path sensitive), so the
  set is identical before and after.
python scripts/check_definition_drift.py --gate: PASS
```

## Evals run

```text
No eval harness covers the transport reducer. The live-frequency measurement above (receipt sample and
container logs) is the behavioral check. The follow-up is the post-deploy live check below.
```

## Docker/build/smoke checks

```text
Not deployed (no production deploys in this task). No Dockerfile, requirements or compose change.
```

## Review findings fixed

- Finding (should): in production (a trace committed atomically), the head piece re-reads and writes the whole tick, and then the tail, which carries `tick_completed`, rewrote it with a fresher `observed_at`.
  - Fix: noop when the stored bus already holds this `source_trace_id`.
  - Evidence: `test_atomic_commit_shape_writes_each_window_exactly_once` checks all 12 cuts. It fails on the first commit and passes now.
- Finding (should): a failed trace re-read was recorded only in receipt warnings.
  - Fix: a `transport_trace_reload_failed` WARNING and a `transport_incomplete_window_held` INFO log line.
  - Evidence: `test_reload_failure_is_logged`.
- Finding (should): nothing stopped an older window from replacing a newer one.
  - Fix: skip when the stored `sample_window_id` is newer (the `YYYYMMDDTHHMMSSZ` format sorts by time).
  - Evidence: `test_older_window_never_replaces_a_newer_one`.
- Finding (nit): the loader cap of 200 rows could drop `tick_completed` from a very large trace.
  - Fix: cap raised to 2000, with a truncation WARNING.
- Finding (nit): the source service was a string literal.
  - Fix: now uses `TRANSPORT_SOURCE_SERVICE`, bound as a parameter.
- Finding (nit): the reducer imported a private name.
  - Fix: `ATOM_ROLES` is now public.
- Finding (nit): a held piece's events appear in two receipts.
  - Fix: documented in the reducer docstring.
- Not changed (nit): poison isolation re-reads the trace once per event it replays. This is bounded and only runs after a batch fails.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Post-deploy live check:

```bash
docker logs orion-athena-substrate-runtime 2>&1 | grep -E "transport_incident_signal.*reliability_pressure|transport_trace_reload_failed" | tail
# expect: nothing
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "
SELECT count(*) FILTER (WHERE (d->'after'->>'stream_backlog_health')::float = 0.5), count(*)
FROM substrate_reduction_receipts r, jsonb_array_elements(r.receipt_json->'state_deltas') d
WHERE r.reducer_name='transport_bus_reducer' AND r.created_at > now()-interval '1 day'
  AND d->>'target_id'='bus:athena';"
# expect: 0|<n>
```

## Risks / concerns

- Severity: low
  - Concern: if the trace re-read keeps failing (for example a missing grant), every split tick is held, and `bus:athena` keeps its last reading until an unsplit tick arrives. In steady state that's the next tick, about 10 seconds later.
  - Mitigation: a WARNING log line, and the next whole tick writes normally.
- Severity: low
  - Concern: a future emitter that stops sending `bus_observer_tick_started` would have every trace held.
  - Mitigation: both observer paths emit it today (`bus_observer.py`). The fixtures now pin that shape.
- Severity: info
  - Concern: separate from this patch, a trace committed later with an earlier `created_at` than the cursor would be skipped by the cursor.
  - Mitigation: not changed here, and not observed. Worth checking if cursor gaps show up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2333

🤖 Generated with [Claude Code](https://claude.com/claude-code)
