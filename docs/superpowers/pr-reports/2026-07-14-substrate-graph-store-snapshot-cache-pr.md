# PR report: TTL cache for `SubstrateGraphStore.snapshot()` — root-causing Fuseki memory pressure

## Summary

- Root cause of a real, live memory-pressure incident (Fuseki hit 99.99% of its 40GB limit while the entire chat pipeline was completely idle): `GraphDBSubstrateStore.snapshot()` issued a full, uncached live SPARQL query on every call, with zero freshness check.
- At least 6 real call sites hit this, traced end to end with live evidence: `orion-substrate-runtime`'s brain-frame tick (every 5s, forever, `BRAIN_FRAME_INTERVAL_SEC`), its dynamics tick (30s), attention-broadcast tick, endogenous-curiosity tick, `beliefs_for_stance`'s warm-path check (used by `chat_stance.py`/`mind_runtime.py`/`projection_builder.py` on real chat/mind turns), and a rare fallback in `graph_cognition/views.py`.
- Fix: a `time.monotonic()`-based TTL guard reusing the class's pre-existing `self._cache` (previously only used as a failure fallback, never as a freshness-based skip). Default `2.0s`, configurable via `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC`.
- Full spec with the complete blast-radius trace at `docs/superpowers/specs/2026-07-14-substrate-graph-store-snapshot-cache-spec.md`.
- Code review (3 parallel finder agents, high effort) caught 4 real issues before this shipped — all fixed, see below.

## Outcome moved

A live, currently-escalating production resource-pressure issue (confirmed via `docker stats`: Fuseki climbed to 99.99% memory, 241% CPU spikes, entirely independent of real conversational activity) now has a targeted, root-cause fix instead of a resource-limit band-aid. A Fuseki memory/CPU bump was explicitly deferred pending this investigation — this patch is the actual fix, not a stopgap.

## Current architecture (before this patch)

```
snapshot() [orion/substrate/graphdb_store.py:217, pre-fix]
    try:
        nodes = self._query_nodes(limit_nodes=500)       # live SPARQL, every call, no exceptions
        edges = self._query_edges_for_node_ids(...)       # live SPARQL, every call, no exceptions
    except GraphDBSubstrateStoreError:
        return self._cache.snapshot()                     # cache used ONLY on failure
    self._refresh_cache(nodes=nodes, edges=edges)
    return self._cache.snapshot()
```

Called by a hard 5-second-forever tick loop (`_brain_frame_loop`) with no dependency on real activity, plus ~5 other periodic/per-turn callers — all sharing one process-singleton store instance (confirmed via `_get_substrate_graph_store`'s own docstring and every real caller's construction site), but each independently re-querying Fuseki on every single invocation regardless of how recently any of the others had already fetched the same data.

## Architecture touched

`orion/substrate/graphdb_store.py` (the fix itself), `orion/substrate/tests/test_graphdb_store.py` (new tests + one restored assertion), three services' `.env_example` (`orion-hub`, `orion-substrate-runtime`, `orion-cortex-exec` — the ones that document this config surface), those same three services' live `.env` files on this host, one new spec doc.

## Files changed

- `orion/substrate/graphdb_store.py`: `GraphDBSubstrateStoreConfig`/`SparqlSubstrateStoreConfig` gain `snapshot_cache_ttl_sec: float = 2.0`; `SparqlSubstrateStore.__init__` threads it through to the base config; new `_resolve_snapshot_cache_ttl_sec()` env resolver (`SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC`, falls back to `2.0` on unset or unparseable values with a logged warning); `GraphDBSubstrateStore.__init__` gains `_last_snapshot_at` and a `threading.Lock`; `snapshot()` wraps its whole body in that lock and checks the TTL before attempting a live query.
- `orion/substrate/tests/test_graphdb_store.py`: 8 new tests (TTL-hit reuses cache with zero new queries, TTL-expiry issues a new query, `ttl=0` disables caching entirely matching pre-fix behavior, failure-fallback behavior is unchanged, config defaults, env-var threading, invalid-env-value fallback) + 1 restored assertion (see Review findings below).
- `services/orion-hub/.env_example`, `services/orion-substrate-runtime/.env_example`, `services/orion-cortex-exec/.env_example`: new `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC=2.0` with a comment explaining the incident and pointing at the spec.
- `docs/superpowers/specs/2026-07-14-substrate-graph-store-snapshot-cache-spec.md` (new): full design doc — current architecture, the complete upstream/downstream blast-radius trace (every real caller, its cadence, and its actual freshness tolerance), the exact proposed fix, non-goals, and acceptance checks. Written and reviewed *before* any code was touched, per this repo's design-mode contract.

## Schema / bus / API changes

None. `MaterializedSubstrateGraphState`'s shape is unchanged; callers receive the exact same object type, just potentially a cached-and-reused instance rather than a freshly-fetched one within the TTL window.

## Env/config changes

- Added: `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC` (default `2.0`), documented in `.env_example` for the three services that expose substrate-store config.
- `.env_example` updated: yes, all three.
- Local `.env` synced: yes, in the same session, directly (not deferred as a PR-report TODO, per this repo's "env sync is mandatory" standing feedback) — verified present in `services/orion-hub/.env`, `services/orion-substrate-runtime/.env`, `services/orion-cortex-exec/.env` on this host.
- Skipped keys requiring operator action: none.

## Tests run

```text
$ python -m pytest orion/substrate/tests/test_graphdb_store.py -q
25 passed   # 17 pre-existing + 8 new

$ python -m pytest orion/substrate/ -q --ignore=orion/substrate/experiments
306 passed  # full substrate suite, no regressions
```
(`orion/substrate/experiments/hyperbolic_gpt/smoke_test.py` fails to collect on this environment — confirmed pre-existing and unrelated, an experimental module with its own missing optional dependency, not touched by this patch.)

## Evals run

Not applicable — this is a deterministic caching fix to internal data-access code, not a behavior surface with an eval harness. The live `docker stats`/Fuseki-log investigation that produced this patch's root-cause evidence (see the spec's "Current architecture" table) is the closest equivalent to an eval here.

## Docker/build/smoke checks

Not run as part of this patch — no Docker/compose/port/dependency changes. The live incident that motivated this fix, and its full evidence trail (query cadence matching `BRAIN_FRAME_INTERVAL_SEC=5.0`, memory climbing to 99.99%, zero correlation with cortex-exec activity), was already gathered during the investigation that preceded this spec and is documented there rather than repeated here.

**Recommended post-merge check** (not run yet, since this hasn't shipped to the live container): re-observe Fuseki's `docker stats` memory/CPU and query-log frequency over a real window after this deploys, to confirm the fix actually reduces load in production, not just in unit tests.

## Review findings fixed

- Finding: my own initial edit accidentally deleted a pre-existing, still-valid assertion (`assert "fuseki:3030/orion/update" in red`) from an unrelated test (`test_sparql_http_client_strips_credentials_from_redacted_update_url`) — an artifact of an imprecise `old_string` match that didn't include the full original test body.
  - Fix: restored the assertion exactly as it existed on `main`.
  - Evidence: `git show main:orion/substrate/tests/test_graphdb_store.py` confirmed the original content; test re-verified passing.
- Finding (CONFIRMED by 2 independent review agents): a real check-then-act race on `_last_snapshot_at`, plus increased exposure to `InMemorySubstrateGraphStore.snapshot()`'s unsynchronized `dict(self._nodes)` copy — `snapshot()` is called from multiple OS threads in this deployment (`orion-substrate-runtime`'s brain-frame tick runs via `asyncio.to_thread`), and this patch makes the cache-read path the *dominant* path (previously only a rare failure fallback), sharply increasing how often that unsynchronized copy is exercised concurrently with writers.
  - Fix: added a `threading.Lock` around the entire `snapshot()` body — serializes the TTL check, the live query, the cache refresh, and the cache read within this method, closing both the TOCTOU race and the concurrent-mutation exposure for this call path.
  - Evidence: all 25 tests still pass after the change; the lock is a standard, minimal, non-reentrant guard with no calls back into `snapshot()` inside the critical section (verified by reading the full method body).
- Finding: an `.env_example` comment I wrote claimed the new TTL "only matters if `SUBSTRATE_STORE_BACKEND=sparql`," but `build_substrate_store_from_env()` applies it to the `graphdb` backend too (which `orion-hub` explicitly supports).
  - Fix: corrected the comment.
- Finding: a hand-rolled `_CountingPost` test helper reimplemented `unittest.mock.MagicMock`'s free call-counting.
  - Fix: replaced with `MagicMock(side_effect=fake.post)`, using its built-in `.call_count` (a plain writable int, so the existing reset-before-counting pattern still works unchanged).
- Not fixed (considered, dismissed as correct scope): review flagged that the store's other query methods (`query_focal_slice`, `query_hotspot_region`, `query_provenance_neighborhood`, `query_concept_region`/`query_contradiction_region`) share the same "live query, cache only on failure, no TTL" pattern. Investigated their real callers (HTTP request handlers, per-turn query planning) — materially different risk profile than the traced 5s-forever tick loop incident. Deferring them is the spec's explicit, considered non-goal, not an oversight.
- Not fixed (accepted, documented honestly): review correctly noted the 2.0s default TTL does not reduce the brain-frame tick's *own* solo query volume, since 5s > 2s means that caller never hits a still-fresh cache on its own cadence alone. The fix's real, verified benefit is collapsing *overlapping* calls from *different* callers (brain-frame, dynamics, attention, curiosity, `beliefs_for_stance`) that fire within the same couple of seconds into one shared fetch — which is the pattern actually observed in Fuseki's access log (multiple query pairs within 1-3 seconds of each other, not evenly spaced at any single caller's interval). Stated plainly rather than overclaiming a bigger win than the mechanism actually delivers.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```
All three services construct their substrate store at process startup (`build_substrate_store_from_env()` / module-level singletons) — a restart is required for the fix to take effect on any already-running container.

## Risks / concerns

- Severity: Low — the 2.0s default is conservative relative to every real downstream freshness requirement traced (tightest is a 30s tick; `beliefs_for_stance`'s own design already tolerates far more staleness), but it has not yet been observed reducing load against live production traffic (only against unit tests). Recommend the post-merge live re-check described above before considering this fully closed.
- Severity: Low — the store's other query methods (see Review findings) share the same no-TTL pattern and remain unfixed by design; if any of their real callers turn out to have a tick-loop-like cadence rather than the request/turn-triggered pattern assumed, they'd warrant the same fix in a follow-up.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/substrate-graph-store-snapshot-cache?expand=1`
