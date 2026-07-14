# Spec: TTL cache for `SubstrateGraphStore.snapshot()`

## Arsonist summary

`GraphDBSubstrateStore.snapshot()` (`orion/substrate/graphdb_store.py:217`, inherited by the live `SparqlSubstrateStore` — `SUBSTRATE_STORE_BACKEND=sparql` in production) issues a full, uncached live SPARQL query against Fuseki on **every single call**, with zero freshness check. At least 6 real call sites hit this method, one of them (`orion-substrate-runtime`'s brain-frame tick) on a hard 5-second loop that runs forever regardless of real activity. Live-verified: Fuseki's memory climbed to 99.99% of its 40GB limit while the entire cortex-exec/chat pipeline was completely idle (zero log lines across all 4 containers) — this pattern is not tied to real conversational load at all, it is pure waste from an uncached read in a hot loop.

The class already has an in-memory mirror (`self._cache`, an `InMemorySubstrateGraphStore`) that gets refreshed after every successful live query (`_refresh_cache()`) — but it is currently used **only as a failure fallback**, never as a freshness-based skip-the-query optimization on the success path. The fix does not require new caching infrastructure, only a timestamp check in front of the existing cache.

## Current architecture

```
snapshot() [orion/substrate/graphdb_store.py:217]
    try:
        nodes = self._query_nodes(limit_nodes=500)       # live SPARQL, every call
        edges = self._query_edges_for_node_ids(...)       # live SPARQL, every call
    except GraphDBSubstrateStoreError:
        return self._cache.snapshot()                     # cache used ONLY on failure
    self._refresh_cache(nodes=nodes, edges=edges)
    return self._cache.snapshot()
```

Confirmed real callers of `.snapshot()` on this store (traced 2026-07-14, live evidence, not static-only):

| Caller | Location | Cadence | Notes |
|---|---|---|---|
| Brain-frame tick | `services/orion-substrate-runtime/app/worker.py:1223` (`_brain_frame_tick`) | Every 5s, forever (`BRAIN_FRAME_INTERVAL_SEC=5.0`) | **Confirmed primary source** — query signature (`limit_nodes=500`/`limit_edges=1000`) and cadence match Fuseki's access log exactly; fires independent of any real activity |
| Dynamics tick | `services/orion-substrate-runtime/app/worker.py:1112`-adjacent (`SubstrateDynamicsEngine`, via `orion/substrate/dynamics.py:74`) | Every 30s (`SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC=30.0`) | Writes activation/pressure/dormancy back to the graph |
| Attention-broadcast tick | `services/orion-substrate-runtime/app/worker.py` ~1112 | Periodic, gated by `enable_attention_broadcast` | Feeds `broadcast_projection_from_frame` |
| Endogenous-curiosity tick | `services/orion-substrate-runtime/app/worker.py:1421` | Periodic | Feeds goal-proposal candidates |
| `beliefs_for_stance` warm-path check | `orion/substrate/relational/layer.py:151` (`CognitiveUnificationLayer`) | Per real chat/mind turn (via `chat_stance.py`'s `_UNIFICATION_LAYER` singleton in cortex-exec, `mind_runtime.py` in cortex-orch, `projection_builder.py`) | **Second, related bug**: calls `self._store.snapshot()` unconditionally at the top of its own per-producer-TTL warm-path check, before that check even runs — defeating its own `freshness_ttl_sec` (60s-1800s) design at the entry point |
| `graph_cognition/views.py:312` | Fallback only, when a targeted query returns empty (`snapshot_fallback_due_to_empty_query_region`) | Rare | Low risk, not a hot path |

Ruled out as unrelated during this trace: `orion/spark/concept_induction/parity_evidence.py:187`'s `.snapshot()` is a different class (`ParityEvidenceStore`, in-process stats) with no relation to the substrate graph store — a false positive from name-only grep.

## Missing questions

- Exact live value of `SUBSTRATE_ATTENTION_BROADCAST`/endogenous-curiosity tick intervals were not pinned down to a precise number during the trace (confirmed periodic, not confirmed exact seconds) — not blocking, since the fix's TTL will be conservative relative to even the tightest known interval (30s dynamics tick).
- Whether `orion-cortex-orch`'s `mind_runtime.py` and `projection_builder.py` construct their own separate `SparqlSubstrateStore` instances (own cache, own TTL clock) or share the same process-level singleton pattern as `chat_stance.py`'s `_get_unification_layer()`. Does not change the fix (each instance gets its own cache either way, each independently bounded by the same TTL), only affects how many *independent* caches exist across the whole system post-fix. Not resolved here; not needed to ship this patch.

## Proposed fix

Add a monotonic-clock TTL guard in front of the existing live-query path in `GraphDBSubstrateStore.snapshot()`. No new cache object — reuse `self._cache`, which already mirrors the last successful fetch.

```python
def __init__(self, cfg: GraphDBSubstrateStoreConfig) -> None:
    self._cfg = cfg
    self._cache = InMemorySubstrateGraphStore()
    self._result_source_kind = "graphdb"
    self._last_snapshot_at: float | None = None          # NEW

def snapshot(self) -> MaterializedSubstrateGraphState:
    now_mono = time.monotonic()                            # NEW
    ttl = float(self._cfg.snapshot_cache_ttl_sec)           # NEW
    if (                                                    # NEW
        ttl > 0.0
        and self._last_snapshot_at is not None
        and (now_mono - self._last_snapshot_at) < ttl
    ):
        return self._cache.snapshot()
    try:
        nodes, _ = self._query_nodes(limit_nodes=500)
        node_ids = [n.node_id for n in nodes]
        edges, _ = self._query_edges_for_node_ids(node_ids=node_ids, limit_edges=1000)
    except GraphDBSubstrateStoreError:
        return self._cache.snapshot()
    self._refresh_cache(nodes=nodes, edges=edges)
    self._last_snapshot_at = now_mono                        # NEW
    return self._cache.snapshot()
```

`time` needs importing (not currently imported in `graphdb_store.py` — confirmed via the file's import block).

**New config field on `GraphDBSubstrateStoreConfig`:**

```python
@dataclass
class GraphDBSubstrateStoreConfig:
    endpoint: str
    graph_uri: str = DEFAULT_SUBSTRATE_GRAPH_URI
    timeout_sec: float = 5.0
    user: str | None = None
    password: str | None = None
    snapshot_cache_ttl_sec: float = 2.0                      # NEW
```

Threaded through `build_substrate_store_from_env()` via a new env var: `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC` (default `2.0`), following the same `os.getenv(...)`-in-the-factory pattern already used for every other field this function resolves. `SparqlSubstrateStoreConfig` (the subclass config, line 581) needs the same field added if it doesn't already inherit it structurally — verify at implementation time whether it's a separate dataclass or extends the base one.

**Why 2.0 seconds by default:** conservative relative to every real downstream requirement found in the blast-radius trace — the tightest periodic caller (dynamics tick) runs every 30s; `beliefs_for_stance`'s own explicit per-producer freshness design already tolerates 60-1800s of staleness. A 2-second cache does not reduce the brain-frame tick's own solo contribution (5s > 2s, so it never hits a live cache-fresh skip on its own cadence) — its real value is collapsing near-simultaneous calls from *different* callers (brain-frame, dynamics, attention, curiosity, `beliefs_for_stance`) that fire within the same couple of seconds into one shared fetch instead of N independent ones, which is the actual pattern observed in Fuseki's access log (multiple query pairs within 1-3 seconds of each other, not perfectly spaced at any single caller's own interval).

## Files likely to touch

- `orion/substrate/graphdb_store.py` — `GraphDBSubstrateStoreConfig` (new field), `GraphDBSubstrateStore.__init__`/`.snapshot()` (the fix), `SparqlSubstrateStoreConfig` if it needs its own copy of the field, `build_substrate_store_from_env()` (env resolution for the new field).
- `services/orion-substrate-runtime/.env_example` and any other service's `.env_example` that documents `SUBSTRATE_STORE_BACKEND`/related substrate-store env vars (check `services/orion-cortex-exec/.env_example`, `services/orion-cortex-orch/.env_example` too, since they also construct stores via the same factory) — add `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC=2.0`.
- `orion/substrate/tests/` (or wherever `graphdb_store.py`'s existing tests live — locate via the repo's own test discovery, don't assume a path) — new tests for the TTL behavior.

## Non-goals

- Not extending this TTL-cache pattern to the store's other query methods (`query_focal_slice`, `query_hotspot_region`, `query_provenance_neighborhood`, `query_concept_region`/`query_contradiction_region`) — these were not part of the blast-radius trace for this incident and may have different call patterns/freshness needs. A follow-up patch, not this one.
- Not fixing `beliefs_for_stance`'s deeper design issue (calling `.snapshot()` before checking whether anything is actually cold) beyond what this store-level cache incidentally mitigates. The store-level fix reduces the cost of that call but doesn't restructure the warm-path check itself.
- Not touching `BRAIN_FRAME_INTERVAL_SEC` or any other tick-loop interval — the fix is at the data-access layer, not the scheduling layer.
- Not bumping Fuseki's memory/CPU limits — this patch is the actual root-cause fix; a resource bump was explicitly deferred pending this fix during investigation.

## Acceptance checks

1. Unit test: two `snapshot()` calls within the TTL window issue exactly one live query (mock the HTTP/SPARQL client, assert call count).
2. Unit test: two `snapshot()` calls spaced further apart than the TTL each issue their own live query.
3. Unit test: `snapshot_cache_ttl_sec=0` disables caching entirely (every call is live) — preserves current behavior for anyone who explicitly wants it off.
4. Unit test: existing failure-fallback behavior (live query raises `GraphDBSubstrateStoreError` → returns `self._cache.snapshot()`) is unchanged.
5. Live check (post-deploy): re-run the same Fuseki log/`docker stats` observation from the investigation — confirm query frequency drops and memory usage stabilizes well below the 90%+ range seen before the fix, over a real observation window (not just immediately after restart).
6. Confirm `beliefs_for_stance` and brain-frame tick still produce correct-looking output after the change (no regression in `orion:substrate:brain_frame` payloads, no stale-belief symptom in a real chat turn) — spot-check, not a new automated test, since this is an existing-behavior-preservation check not a new feature.

## Recommended next patch

This one. Single, well-scoped, low-risk (additive cache, defaults preserve near-identical behavior modulo the intentional dedup), high-confidence root cause with live evidence at every step of the trace.
