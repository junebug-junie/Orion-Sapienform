# PR report: real write-based change detection for `SubstrateGraphStore.snapshot()`

## Summary

- Follow-up to #1040 (merged), triggered by direct user feedback after observing post-deploy behavior: "build a proper change detector so it isn't mindlessly computing" — correctly identifying that #1040's blind TTL cache wasn't real change detection, and that post-deploy query volume hadn't meaningfully dropped (still ~30/min, matching pre-fix levels) because the dominant caller (a 5-second-forever tick) never finds itself still-fresh against a 2-second blind timer on its own solo cadence.
- Every real write now bumps a `self._write_generation` counter; `snapshot()` requires the generation to be unchanged since the last fetch before reusing the cache at all — a write is a known, certain change, never silently served stale data.
- A single remaining time threshold, `snapshot_force_refresh_ceiling_sec` (default 30.0s), is a periodic safety net for staleness from writes made by a *different* process this instance's own counter can't see.
- Removed the old `snapshot_cache_ttl_sec` entirely — code review proved it was dead code under every real default once the ceiling existed, and its docstring had drifted to directly contradict the new field's docstring.
- Code review (3 parallel agents, high effort) caught a real, independently-confirmed-by-all-three concurrency bug in my own first draft — fixed before this shipped. See "Review findings fixed" below.

## Outcome moved

Real, verifiable change detection instead of a blind timer that only reduced overlapping-caller waste, not the dominant caller's own query rate. This is the difference between "wait N seconds no matter what" and "only re-fetch when something is actually known to have changed" — directly addressing what was asked for, not a rebrand of the same mechanism.

## Current architecture (before this patch)

#1040's `snapshot()` served the cache if `elapsed < snapshot_cache_ttl_sec` (default 2.0s), with no awareness of whether anything had actually changed. A dominant caller ticking every 5 seconds forever never found itself within a 2-second window, so it re-queried on every single tick regardless of the cache existing at all.

## Architecture touched

`orion/substrate/graphdb_store.py` (the store class), `orion/substrate/tests/test_graphdb_store.py`, three services' `.env_example`, one service's `docker-compose.yml`, those same services' live `.env` files.

## Files changed

- `orion/substrate/graphdb_store.py`:
  - New `_write()` wrapper method (all three real write call sites — `upsert_node`, `upsert_edge`, `_upsert_identity` — now route through it instead of calling `self._update()` directly, since `_update()` is overridden per-backend in `SparqlSubstrateStore` but the write call sites aren't, so bumping the counter in one shared wrapper covers both backends without duplication — confirmed by reading `SparqlSubstrateStore`'s full class body).
  - `snapshot()` rewritten: requires `same_generation` before any cache reuse; `snapshot_force_refresh_ceiling_sec` is the sole remaining time threshold, only consulted when `same_generation` holds.
  - Removed: `snapshot_cache_ttl_sec` field (both `GraphDBSubstrateStoreConfig` and `SparqlSubstrateStoreConfig`), `_resolve_snapshot_cache_ttl_sec()`, both its call sites in `build_substrate_store_from_env()`.
- `orion/substrate/tests/test_graphdb_store.py`: rewrote the snapshot-cache test block (10 tests: same-generation reuse, same-generation survives elapsed time, a write forces an immediate re-fetch, the TOCTOU race fix specifically, ceiling forces a periodic refresh, `ceiling<=0` trusts forever, failure fallback, config defaults, env threading, invalid-env fallback). Removed tests referencing the deleted field.
- Three `.env_example` files + `services/orion-substrate-runtime/docker-compose.yml`: removed `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC`, kept/documented `SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC`. The compose change also folds in the still-unmerged `fix/substrate-runtime-snapshot-ttl-compose-env`'s ceiling-var passthrough directly (that PR's ttl-var fix is now moot since the field it fixed no longer exists) — makes this patch fully self-contained regardless of merge order.

## Schema / bus / API changes

None. `MaterializedSubstrateGraphState`'s shape is unchanged.

## Env/config changes

- Removed: `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC` (from `.env_example` × 3, `docker-compose.yml` × 1, live `.env` × 3).
- Unchanged/carried over from #1040: `SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC` (default `30.0`), already present in all three services' `.env_example`/live `.env` from that PR.
- Verified zero remaining references to the removed field anywhere in the repo or live `.env` files (`grep -rn` came back empty).

## Tests run

```text
$ python -m pytest orion/substrate/tests/test_graphdb_store.py -q
29 passed

$ python -m pytest orion/substrate/ -q --ignore=orion/substrate/experiments
310 passed
```
Both independently re-run after every round of review fixes, not just once at the end.

## Evals run

Not applicable — deterministic caching/concurrency fix, no eval surface. The TOCTOU race test (`test_snapshot_write_generation_captured_before_live_query_not_after`) is the closest equivalent — it doesn't just assert behavior, it reproduces the exact race condition review found and proves the fix resolves it.

## Docker/build/smoke checks

```text
$ docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
    -f services/orion-substrate-runtime/docker-compose.yml config
SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC: "30.0"
SUBSTRATE_STORE_BACKEND: sparql
```
Resolves correctly against the real live `.env`; old field confirmed absent.

**Not yet re-observed live in production** (this branch isn't deployed yet) — the honest, still-open question is whether real query volume actually drops once this ships, versus #1040 alone. Recommend re-running the same `docker stats`/Fuseki-log observation used throughout this investigation after deploy.

## Review findings fixed

- **CONFIRMED by all 3 parallel review agents independently — the most severe finding**: the write generation was captured *after* the live query completed (`self._last_snapshot_generation = self._write_generation` post-fetch), not before. Since `_write()` is deliberately not lock-protected against `snapshot()`, and this store is called from multiple OS threads in production (`orion-substrate-runtime`'s brain-frame tick runs via `asyncio.to_thread`), a write racing in *during* an in-flight query would get silently credited to data fetched *before* that write happened — permanently masking the write until the next unrelated write or the ceiling fired. This is exactly the failure mode the whole mechanism exists to prevent.
  - Fix: capture the generation *before* issuing the live query. A race now costs one possibly-redundant extra fetch on the next call (safe, wasteful-but-correct direction), never a falsely-trusted stale cache.
  - Evidence: new test `test_snapshot_write_generation_captured_before_live_query_not_after` directly simulates the race (a write happens mid-fetch, injected via a monkeypatched `_query_nodes`) and asserts the recorded generation is the pre-race value, proving the next call will see a mismatch and correctly re-fetch.
- **CONFIRMED by 2 of 3 agents**: `snapshot_cache_ttl_sec` was provably dead code under every shipped default (2.0s always subsumed by the 30.0s ceiling once `same_generation` gates both), and its own docstring claimed the exact "cross-process safety net" role the new ceiling field's docstring also claimed — a real, live contradiction, not just redundancy.
  - Fix: removed the field entirely rather than patch the docstring, per the reasoning above (same-day, before it's load-bearing anywhere).
- **CONFIRMED**: a new test called `upsert_edge(identity_key=None, ...)`, exercising (not catching) a real, pre-existing, out-of-scope bug — `upsert_edge` calls `_upsert_identity` unconditionally, unlike `upsert_node`'s guarded call, so a `None` identity key would write a bogus identity record rather than being safely skipped.
  - Fix: changed the test to use a real identity key, matching `upsert_edge`'s documented non-Optional signature. The underlying `upsert_edge` bug itself is left unfixed — out of scope for this patch, noted in a test comment for whoever next touches that method.
- Not fixed (considered, confirmed correct as-is): review verified the `_write()`-wrapper-in-base-class layering correctly covers both `GraphDBSubstrateStore` and `SparqlSubstrateStore` without duplication (confirmed by reading the full `SparqlSubstrateStore` class body — it only overrides `_update`, never the write call sites). Review also confirmed a harmless double-bump of the generation counter when `upsert_edge`/`upsert_node` internally call `_upsert_identity` (two `_write()` calls per logical operation) — not fixed, since the counter is only ever used for equality comparison, not exact counting, so a double-bump has no observable effect.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Medium — not yet observed reducing real production query volume; only unit-tested. The honest, still-open question from the last live status check (memory climbing again post-#1040-restart, query rate not meaningfully lower than pre-fix) is whether *this* patch actually closes that gap. Recommend a live re-check after deploy, same methodology as before (`docker stats`, Fuseki access log query counts, real triple-count as a data-growth control).
- Severity: Low — per-process write-generation tracking only catches writes made by *this instance's own* store object. Multiple real services each construct separate instances; the 30s ceiling bounds cross-instance staleness but a pure-reader instance that never writes relies entirely on that periodic ceiling refresh (or process restart) to ever see external changes. Documented explicitly in the config field's docstring, not a silent gap.
- Severity: Low — this patch supersedes the still-open `fix/substrate-runtime-snapshot-ttl-compose-env` PR (that PR fixed a compose gap for a field this patch deletes). Recommend closing that PR without merging once this one lands, to avoid confusion.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/substrate-snapshot-write-generation-cache?expand=1`
