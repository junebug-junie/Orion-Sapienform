# Hub: load concept graph stores in the background, not at import

## Summary

- Hub no longer waits for its three concept graphs to load before it starts. Before this change, Hub read every node and edge of the Orion, Self Atlas, and AI Town graphs into memory while Python was still importing `api_routes.py`. Uvicorn could not open port 8080, and no bus listener could start, until all three reads were done.
- Each graph store is now built once, in one background thread, in this order: Orion first (cognition reads it), then Self, then AI Town (the biggest, used only for interpretability).
- A caller that arrives before a store is ready gets a clear "not ready" answer. It never gets an empty graph that looks real:
  - Concept Atlas routes return their existing `substrate_store_unavailable` answer, with a new `store_warmup` field that names the store still loading and how long it has been loading.
  - Any other route that touches a store still loading returns HTTP 503 `substrate_store_warming` with `Retry-After: 30`.
  - The substrate status posture reports `semantic.kind = "warming"` and `degraded = true`.
- The golden-concept seed and the topic-foundry startup ingest now wait for the stores in a worker thread. Startup does not wait for them.
- Nothing changes once a store is ready. Callers get the same `FalkorSubstrateStore` object as before.

## Outcome moved

Measured from the live container log (`docker logs -t orion-athena-hub`, restart at 2026-10-20 18:00:19 UTC):

| Time (UTC) | Event |
|---|---|
| 18:00:19 | `Starting Hub...` (container start) |
| 18:01:40 | `falkor_substrate_hydrate_failed error=Query timed out` (81 s into the graph reads) |
| 18:02:24 | Self graph read finished (`duplicate_identities graph=orion_substrate_self`) |
| 18:04:16 | `Routes, WebSocket endpoint, and static mounts ready.` Uvicorn starts. |
| 18:04:18 | `Uvicorn running on :8080`. **3 min 59 s after start.** |

Where those 4 minutes went. All of it was synchronous module import, before uvicorn created its event loop. The loop was not blocked, because there was no loop yet: nothing could serve at all.

- **Graph reads: about 158 s.** I measured each read alone against live FalkorDB, read-only, from this worktree:
  - Orion `orion_substrate`: 19.0 s (7,165 nodes, 45,720 edges)
  - AI Town `orion_substrate_aitown`: 113.0 s (45,103 nodes, 146,117 edges, 12,785 duplicate nodes)
  - Self `orion_substrate_self`: 25.9 s (18,511 nodes, 47,064 edges)
- **Mutation store load: about 60-70 s. NOT fixed here.** I profiled the import inside the live container with the graph backend set to in-memory. Import still took 67 s, and 62 s of that was `api_routes` module code. cProfile shows `SubstrateMutationStore.__post_init__` -> `_load_from_postgres`, which parses about 1.97 million Postgres rows with pydantic at import (`orion/substrate/mutation_queue.py:1401`). Its own code comment says "550k-row", so this table has grown about 3.5x.
- **Everything else in the import: about 5 s.**

After this patch, the graph reads no longer delay serving. Expected time to serve: about 65-70 s, which is the mutation store load plus the rest of the import. This is UNVERIFIED on the live container, because I did not deploy. Expected store-ready times: Orion about 20 s after the import starts, Self about 45 s, AI Town about 165 s.

Live smoke of the background-load mechanism, against live FalkorDB with a read-only client (scratch script, not committed):

```text
start_background_warm returned after 0.001s
early caller got: {'available': False, 'reason': 'substrate_store_warming', 'store': 'orion', 'state': 'warming', 'error': None}
substrate_store_ready store=orion type=FalkorSubstrateStore secs=19.2
substrate_store_ready store=self type=FalkorSubstrateStore secs=26.1
substrate_store_ready store=aitown type=FalkorSubstrateStore secs=118.7
all ready t+164.1s; max event-loop tick gap 2.37s
orion nodes 7165 edges 45720
```

The event loop kept running during all three reads. The biggest pause between loop ticks was 2.37 s. That pause comes from the reader thread holding Python's global interpreter lock (GIL) while it parses rows. The loop is not blocked, but it can stall briefly.

## Current architecture

- `services/orion-hub/scripts/api_routes.py` built `SUBSTRATE_SEMANTIC_STORE`, `SUBSTRATE_SEMANTIC_STORE_AITOWN`, and `SUBSTRATE_SEMANTIC_STORE_SELF` as module globals at import. `FalkorSubstrateStore.__init__` calls `_hydrate_from_durable()` (`orion/substrate/falkor_store.py:463`), which reads the whole graph page by page.
- `scripts.main` imports `api_routes` at module level, so uvicorn could not finish importing the app until all three reads finished.
- The startup event awaited the golden-concept seed in a worker thread.

## Architecture touched

- Hub only. There are no bus, schema, env, or compose changes.
- Same fix shape as orion-substrate-runtime PR #2784 (`_get_substrate_graph_store` / `_start_substrate_graph_store_warm`): build once, under a lock, never on the event-loop thread, and fail open with an explicit not-ready answer.
- I kept the holder in Hub rather than in `orion/substrate`. Only Hub needs a proxy holder. Its stores were captured by objects at import (`GraphConsolidationEvaluator(store=...)`, `GraphReviewBootstrapper(semantic_store=...)`), so a plain getter would not work. substrate-runtime already has its own small getter, and moving both behind a shared abstraction now would add a layer for one caller.

## Files changed

- `services/orion-hub/scripts/substrate_store_warm.py` (new): the `WarmingSubstrateStore` holder (build once, background thread, refuses to build on the loop, retries a failed build after 30 s), `SubstrateStoreNotReadyError`, the 503 handler, `start_background_warm`, and `wait_all_ready`.
- `services/orion-hub/scripts/api_routes.py`: wraps the three stores in holders and starts the background load. In-memory backends build synchronously at import, with no I/O. Also: `substrate_store_warm_status()`, a seed that waits for the store and never seeds an unloaded store, a status posture that reports "warming", and `semantic_backend` that names the real store type.
- `services/orion-hub/scripts/concept_atlas_routes.py`: the store getter returns `None` while a store loads, so callers take their existing honest unavailable path. `_unavailable` attaches `store_warmup`.
- `services/orion-hub/scripts/main.py`: registers the 503 handler. Runs the seed as a background task. Makes the topic-foundry startup tick wait for the stores (up to 900 s, off the loop). Cancels the seed task on shutdown.
- `services/orion-hub/tests/test_substrate_store_warm.py` (new): 17 regression tests.

## Schema / bus / API changes

- Added: HTTP 503 `{"available": false, "reason": "substrate_store_warming" | "substrate_store_build_failed", "store", "state", "error"}` on routes that touch a store still loading.
- Added: an additive `store_warmup` list on Concept Atlas `substrate_store_unavailable` payloads.
- Behavior changed: during the first ~20 s to ~165 s after boot (see the next section), substrate routes report not-ready instead of not responding at all.
- Removed / Renamed: none.

## Which features return not-ready during warm-up, and for how long

Times are from the start of the import, using the smoke numbers above:

- Orion graph, about 0-20 s: Concept Atlas default graph routes, graph overview/hotspots, review smoke-check, the world-pulse read store write (counts as a failed attempt and is retried on the next 300 s tick), and the reading concept index (skips the pass).
- Self Atlas, about 0-45 s: Concept Atlas `?graph=self` routes and Self topic-foundry ingest.
- AI Town, about 0-165 s: Concept Atlas `?graph=aitown` routes and AI Town ingest.
- Because the mutation store load still takes about 60 s before uvicorn serves, the Orion and Self stores should normally be ready before the first HTTP request.
- Schedulers: the autonomy and review ticks already catch exceptions. A not-ready store shows up as a logged tick error, not a crash. The seed and the topic-foundry startup tick wait for the stores instead.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest services/orion-hub/tests/test_substrate_store_warm.py -q
  -> 17 passed (5 consecutive runs)

Hub tests that import api_routes / concept_atlas_routes / main / substrate stores (94 files):
  origin/main baseline : 18 failed, 1229 passed, 8 skipped
  this branch          : 18 failed (identical set), 1246 passed (+17 new), 8 skipped
  No new failures versus the baseline.
```

The 18 baseline failures already fail on main, and this patch does not touch them. Examples: `test_handle_chat_request_turn_effect.py::test_handle_chat_request_retries_memory_graph_suggest_on_timeout`.

My first full run showed 20 new failures. Their cause: `api_routes` is sometimes first imported from inside a running event loop (TestClient apps, lazy imports in async code), and the holder refused the in-memory build there. I fixed it: `build_blocking(allow_event_loop=True)` is now used only for the in-memory backend, which does no I/O. A regression assert covers this case.

## Evals run

```text
No eval harness covers Hub boot latency. The live smoke above (read-only, against live FalkorDB) is the evidence for the background-load mechanism.
Follow-up: a boot-latency eval (time from container start to /health) would make this regression observable.
```

## Docker/build/smoke checks

```text
No Docker build or deploy (instructed: do not deploy).
Live read-only measurements: per-graph read timings, the in-container cProfile of the import, and the background-load smoke (above).
```

## Review findings fixed

A code review ran in a subagent. It found no must-fix in the steady state. It found real problems at the boot and failure edges, and I fixed them:

- Finding (must-fix): if the first build failed (for example, FalkorDB not up yet on a full-stack start), every later `wait_ready` returned `None` at once and never retried. The golden-concept seed would then be skipped for that whole boot.
  - Fix: `wait_ready` now loops until its deadline, in short slices. It retries a failed build after the 30 s backoff.
  - Evidence: `test_wait_ready_retries_a_failed_build_until_deadline` (fails twice, then `wait_ready(2)` returns the store).
- Finding: a review-scheduler tick during warm-up would mark a queue item reviewed (using one of its cycles) before failing on the store.
  - Fix: the tick returns `skipped / substrate_store_not_ready` before any queue side effect. Both `execute-once` routes now return 503 before doing anything.
  - Evidence: `test_review_scheduler_tick_skips_while_semantic_store_loads`.
- Finding: the topic-foundry boot ingest was lost for a whole interval (86,400 s) if the store wait timed out.
  - Fix: on the startup tick, an ingest that comes back `substrate_store_unavailable` retries on the existing bounded startup backoff (`STARTUP_TICK_MAX_ATTEMPTS`).
- Finding: dunder probes (`copy`, `pickle`, mock spec) were forwarded to the store. They raised not-ready while loading, or copied the holder's lock.
  - Fix: dunder names raise `AttributeError` on the holder.
  - Evidence: `test_dunder_probes_are_not_forwarded_to_the_store`.
- Finding: shutdown could stall on uncancellable worker threads that block in `wait_ready` for up to 900 s.
  - Fix: a shutdown event, set in `shutdown_event`, releases every waiter.
  - Evidence: `test_shutdown_releases_waiters`.
- Finding: `test_store_is_built_exactly_once_under_concurrent_callers` could flake, because it joined the threads before releasing the builder.
  - Fix: the builder is now released on a timer before the joins. I ran the file 5 times in a row; all runs passed.
- Finding (nit): the ready event could be set outside the lock, which allowed a rare early wake-up.
  - Fix: the event is now set inside the locked state change.
- Finding (nit): code that checks capabilities on the store's type would see the holder's type. A docstring on the holder now says such callers must pass `resolve_ready_store(holder)`. No Hub path does this today.
- Finding (test gaps): two tests are added. One checks the module-level wiring: all three globals are holders, in the order orion, self, aitown. The other checks that `startup_event` creates the seed as a background task and does not await it.

Not changed: the topic-foundry startup tick still waits for the stores before its training trigger, not only before its ingest. This delays the boot training trigger by up to about 165 s. That is harmless, because training is asynchronous and deduplicated on topic-foundry's side, and splitting the tick would mean restructuring three graph branches. A build that hangs (no socket timeout) still shows as `warming` with `warming_for_sec` in `store_warmup`. That is not a regression, because the old import hung too.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

Run this from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: medium. Concern: Hub still takes about 60-70 s to serve, because `SubstrateMutationStore` parses about 2 million Postgres rows at import. Mitigation: listed as follow-up 1, not hidden.
- Severity: low. Concern: the background read holds the GIL in bursts, so the event loop can pause for up to about 2.4 s (measured). Mitigation: this is still far better than not serving at all. A real fix needs neighborhood reads (follow-up 3).
- Severity: low. Concern: a world-pulse read that lands in the first ~20 s spends one of its 3 attempts on `concept_atlas_store_unavailable`. Mitigation: the next 300 s tick retries it.

## Follow-ups

1. Move the `SubstrateMutationStore` Postgres load (about 2M rows, about 60 s, `orion/substrate/mutation_queue.py:1401`) out of import. Also add retention or compaction for the mutation tables, which have grown about 3.5x past the 550k rows the code comment expects.
2. The snapshot refresh ceiling (`SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC=30`) re-reads a whole graph inside `snapshot()` on whatever thread calls it. `concept_atlas_summary` is `async def` and calls `store.snapshot()` directly (`concept_atlas_routes.py`, `_concept_nodes`), so it can re-read on the event loop. Live log gaps of 9-13 s around the post-boot re-reads (18:06:03->18:06:16) match this.
3. Duplicate-identity cleanup: AI Town has 12,785 duplicate nodes and Self has 3,531. Move callers to neighborhood reads (`docs/plans/substrate/2026-10-06-reading-property-graph-design.md`) so no caller needs a full in-memory copy.
4. Read failures under load: during the live boot, one graph read timed out at 81 s (`Query timed out`), so that store started with an empty cache marked `last_hydrate_ok=False`, and its next `snapshot()` read again. This is existing behavior, not changed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2814

🤖 Generated with [Claude Code](https://claude.com/claude-code)
