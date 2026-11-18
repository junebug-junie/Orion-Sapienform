## Summary

- The Postgres headroom alarm fired repeatedly on 2026-11-02 (265/300 used; `orion-hub` held 149). The cause was not the hub route modules fixed in #2993. It was shared code in `orion/substrate/`.
- Three substrate stores (review telemetry, review queue, policy profiles) built a new connection pool on every read or write and never closed it. The hub's autonomy loop reads review telemetry every 30 seconds, so every tick left one idle connection open until Python's full garbage collection ran.
- Each store now builds one pool on first use and reuses it (same pattern as `mutation_control_surface.py`).
- Review found a second leak of the same kind: the hub builds a new felt-state reader on every chat turn, and each built its own pool. Readers now share one pool per database address.
- The "no per-call pool" gate now also scans shared `orion/` code, and no longer treats a bare `self._engine = create_engine(...)` as cached.

## Outcome moved

Live evidence, 2026-11-03:

- New hub connections opened one every ~30 s (`backend_start` spacing), matching `SUBSTRATE_AUTONOMY_INTERVAL_SEC=30`.
- A 5-minute sample of queries on brand-new `orion-hub` connections: 9 x `SELECT payload_json::text FROM substrate_review_telemetry ...`, 4 x the mutation worker's leader lock (that one closes correctly, NullPool + dispose), plus driver handshakes.

Live eval (`services/orion-hub/evals/run_route_engine_pool_eval.py`, hub image, live database):

| store code | pools built per 25 reads | connections left open |
|---|---|---|
| main | 27 per store | 90 (FAIL) |
| this branch | 1 per store | 12 = 9 route modules + 3 stores (PASS) |

## Current architecture

- `GraphReviewTelemetryRecorder`, `GraphReviewQueue`, `SubstratePolicyProfileStore` are module-level singletons in `services/orion-hub/scripts/api_routes.py`. Each method (`_ensure_postgres_schema`, `_load_from_postgres`, `_persist_to_postgres`, `_insert_postgres`, `_trim_postgres`) called `create_engine(self.postgres_url)` and dropped it.
- SQLAlchemy engines hold reference cycles, so an abandoned pool is freed only by a full (generation 2) collection. In the large, long-lived hub process those are rare, so connections piled up between collections (149 at the alarm; 112 on 11-03 before a collection dropped it to 25).
- `orion/hub/association.py::_default_reader()` builds a `SubstrateFeltStateReader` per turn; its `__init__` called `create_engine`.
- The #2993 gate scanned only `services/orion-hub/scripts/`, so it could not see any of this.

## Architecture touched

- Shared `orion/substrate/` stores and felt-state reader (used by the hub; `review_schedule.py` builds an in-memory queue only). Hub test and eval. No bus, schema, API, or env change.

## Files changed

- `orion/substrate/review_telemetry.py`, `review_queue.py`, `policy_profiles.py`: per-instance `_engine()` with a cached pool (`pool_pre_ping=True`); all calls go through it.
- `orion/substrate/felt_state_reader.py`: `_shared_engine(url)`, an `lru_cache`d engine per database URL, used by every reader.
- `tests/test_substrate_store_engine_reuse.py`: each store builds one engine across repeated loads/persists; readers share one engine per URL.
- `services/orion-hub/tests/test_route_engine_reuse.py`: gate scans `orion/` with an allowlist; attribute/subscript storage counts as cached only under `if <same target> is None`.
- `services/orion-hub/evals/run_route_engine_pool_eval.py`: loads the three stores 25 times each (read-only), counts pools built, holds automatic GC off during the loop so the eval sees what the hub sees.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: each store keeps one bounded pool (5 + 10 overflow) for the process lifetime; felt-state readers share one. Reader result caches are still per reader, so association reads are unchanged.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
tests/test_substrate_store_engine_reuse.py + hub test_route_engine_reuse.py        20 passed
  with main's three store files restored:                                         4 failed
root/orion tests touching the three stores                                        166 passed
hub tests touching the three stores                                               102 passed, 1 skipped
hub tests touching felt_state_reader / association                                113 passed
orion-thought / orion-durable-runs tests touching felt_state_reader               16 / 2 passed
```

Pre-existing, not caused by this branch: 4 tests in `orion/substrate/tests/test_felt_state_{lived_answers,self_definition_lane}.py` and `services/orion-cortex-exec/tests/test_substrate_felt_state_reader.py::test_stale_row_not_injected` fail in the host venv. The same tests fail identically on `main`.

## Evals run

```text
python services/orion-hub/evals/run_route_engine_pool_eval.py   (live DB, hub image, host network)
  main store code:   27 engines/store, 90 held -> FAIL (exit 1)
  this branch:        1 engine/store,  12 held -> PASS (exit 0)
```

## Docker/build/smoke checks

```text
Eval ran inside the current orion-hub-hub-app image with the branch mounted read-only.
No image change; the fix ships in the shared orion/ package copied into the hub image.
```

## Review findings fixed

Code review ran in a subagent. Fix itself: no findings (construction happens at import, before threads; init order correct; single uvicorn process, no fork).

- Finding (material): `orion/hub/association.py` builds a felt-state reader per hub chat turn; each leaked a pool. The new gate exempted it, because it treated any `self.x = create_engine(...)` as cached.
  - Fix: `_shared_engine(url)` in `felt_state_reader.py`; gate now needs an `is None` guard on the same target.
  - Evidence: new test `test_felt_state_readers_share_one_engine_per_url`; gate run lists `felt_state_reader` before the fix and not after.
- Finding: `setdefault(url, create_engine(url))` builds a throwaway engine per call.
  - No change: the throwaway engine never connects, and both sites sit behind a `.get()`.
- Noted, not fixed (other service): `orion/reverie/referent_loader.py` builds a pool per orion-thought reverie tick. orion-thought holds 10 connections today. Allowlisted in the gate with a note.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build hub-app
```

Run from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low
- Concern: 5 remaining per-call sites in `orion/` (allowlisted), including the orion-thought reverie loader.
- Mitigation: the gate list can only shrink; follow-up if wanted.

- Severity: low
- Concern: the live drop is UNVERIFIED until the hub runs on this code for several hours; GC timing made the old leak come and go.
- Mitigation: watch `orion-hub` in `logs/orion-postgres-headroom.log`; it should stay near 15-20.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3039

🤖 Generated with [Claude Code](https://claude.com/claude-code)
