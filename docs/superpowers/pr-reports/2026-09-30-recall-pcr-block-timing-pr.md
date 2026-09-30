# fix(recall): warm substrate store at boot; PCR collectors timed and under the deadline

## Summary

- The first belief recall after a recall restart took 9.5 seconds, and the recall's own timing record only explained about 0.2 of them. The missing ~9.3s was the belief-specific step (the "PCR collectors": `active_packet` and `concept_region`), which ran after the time-limited fetch, had no time limit of its own, and was never timed.
- Recall now builds the concept-graph store handle in the background when the service starts, so the first request no longer pays the slow first connection (6.25s measured live).
- The two belief collectors now run concurrently inside whatever is left of the recall's overall time limit. If they are still running at the limit, recall stops waiting, drops only their results, keeps everything else, and marks `deadline_hit`. They never fail the recall.
- The timing record now covers these steps (`pcr_collectors`, plus `pcr_active_packet` / `pcr_concept_region`), the eligible-belief count (`eligible_count`), self-hit suppression (`suppression`) and the shadow comparison (`shadow_compare`), so the stages add up to the total. A test pins that.
- `get_substrate_store()` now takes a lock while building, so the boot warmup and a request racing it cannot both build a store. A request waits at most 2s for that lock, then skips concept_region for that turn.
- A store whose startup load failed (FalkorDB not up yet) or came back empty is no longer kept for the life of the process. It is thrown away and retried later, waiting 5s, then 10s, 20s ... up to 5 minutes between tries, so a down FalkorDB is not hit on every turn. The "warmed" log line only appears for a store that really loaded.
- Recall's connection to FalkorDB now has timeouts (5s to connect, 30s per read), so a hung FalkorDB cannot pin a thread forever. The shared client keeps no timeout by default; only recall opts in.
- A concept_region lookup that recall gave up on no longer writes its "this concept was recalled" activation bump, because those results were thrown away.

## Outcome moved

Failure mode: an unbounded, invisible ~9s stall on the first purposeful recall after a restart. After this patch the stall is (a) usually gone because the store is already warm, (b) capped by the recall deadline if it does happen, and (c) visible in `recall_telemetry.timings_ms` when it happens.

## Current architecture

`services/orion-recall/app/worker.py::process_recall` runs intake, then a concurrent fetch under one deadline (80% of the caller's `deadline_ms`, else `RECALL_DEADLINE_MS_DEFAULT`), then windowing and self-hit suppression. For `recall_phase == "purposeful"` it then awaited `fetch_active_packet_fragments` and `asyncio.to_thread(fetch_concept_region_fragment_and_reinforce(q, store=get_substrate_store()))` in sequence, with no deadline and no timing, before fusion. `count_eligible_active` ran after fusion, also unbounded and untimed. `get_substrate_store()` built its singleton lazily on first call, with no lock.

## Architecture touched

- `orion-recall` worker: PCR block moved into `_run_pcr_collectors()`, which applies the remaining deadline budget.
- `orion-recall` lifespan: background warmup task.
- `orion-recall` substrate store singleton: lock around first construction; new `warm_substrate_store()`.
- Contract: `RecallDecisionV1.timings_ms` description text only (it is a free-form `Dict[str, int]`; no field added or changed).

## Files changed

- `services/orion-recall/app/worker.py`: new `_run_pcr_collectors`; PCR block, `count_eligible_active`, suppression and shadow compare now timed; collectors and eligible count bounded by the deadline.
- `services/orion-recall/app/substrate_store.py`: `_STORE_LOCK` double-checked init with a bounded request-path wait (`REQUEST_LOCK_TIMEOUT_S = 2.0`); failed/empty hydrate not cached, retried with exponential backoff (5s doubling to 300s); Falkor socket timeouts (connect 5s, read 30s); `warm_substrate_store(timeout_s=30)` logs `recall_substrate_store_warmed` only for a real, non-empty store and `recall_substrate_store_warmup_failed reason=...` otherwise.
- `services/orion-recall/app/collectors/concept_region.py`: `fetch_concept_region_fragment_and_reinforce(..., abandoned=threading.Event | None)` skips the reinforcement write when set.
- `orion/substrate/falkor_store.py` (additive): `FalkorSubstrateStore.last_hydrate_ok` (None/True/False) and `last_hydrate_node_count`; config fields `client_socket_timeout_s` / `client_socket_connect_timeout_s` (default None = old behaviour); `build_falkor_substrate_store_from_env` forwards them.
- `orion/substrate/graphdb_store.py` (additive): `build_substrate_store_from_env(falkor_socket_timeout_s=None, falkor_socket_connect_timeout_s=None)`, forwarded only in the direct `falkor` branch.
- `orion/graph/falkor_client.py` (additive): `RedisGraphQueryClient(socket_timeout=None, socket_connect_timeout=None)`, passed to `redis.Redis` only when set. Every existing caller passes neither, so none change.
- `orion/substrate/tests/test_falkor_store_hydrate_signal.py`: new.
- `services/orion-recall/tests/conftest.py`: autouse reset of the store singleton and backoff state between tests.
- `services/orion-recall/tests/test_concept_region_wiring.py`: stubs accept the new keyword arguments.
- `services/orion-recall/app/main.py`: lifespan starts the warmup as a background task when `RECALL_PCR_ENABLED` and `RECALL_CONCEPT_REGION_ENABLED` are both on; cancels it on shutdown if still pending.
- `services/orion-recall/tests/test_recall_pcr_block_timing.py`: new tests (below).
- `orion/core/contracts/recall.py`: `timings_ms` description lists the new keys.

## Schema / bus / API changes

- Added: new keys inside `timings_ms`: `suppression`, `pcr_collectors`, `pcr_active_packet`, `pcr_concept_region` (the last two only when that collector was planned), `eligible_count`, `shadow_compare`.
- Removed: none.
- Renamed: none.
- Behavior changed: purposeful recall can now return `deadline_hit=True` because of a slow PCR collector (previously it would just wait). `eligible_belief_count` in `recall_debug` reads 0 when the deadline had already passed (it is debug-only and does not set `deadline_hit`).
- Compatibility notes: `timings_ms` is a `Dict[str, int]` stored as jsonb; extra keys need no migration.

## Env/config changes

- Added keys: none. Removed / renamed: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template change).
- skipped keys requiring operator action: none.
- The warmup's wait cap is a code constant (`WARMUP_TIMEOUT_S = 30.0`), not a setting.

## Tests run

```text
cd services/orion-recall && python -m pytest tests -q -p no:cacheprovider
  baseline (main @ d3c09c9cb): 2 failed, 333 passed
  after first commit:          2 failed, 346 passed
  after review fixes:          2 failed, 356 passed
  the 2 failures are pre-existing and unchanged:
    test_recall_policy_harness::test_process_recall_diagnostic_contains_gating_suppression_and_selection
    test_recall_vector_amputation::test_worker_and_recall_v2_import_without_vector_adapter

tests/test_recall_pcr_block_timing.py: 13 passed
  against the OLD worker.py: 5 of them fail (both deadline-cut tests, the
  skip-when-deadline-passed test, the new-keys test, the stages-add-up test)

review round: 10 new recall tests + 7 in orion/substrate/tests/test_falkor_store_hydrate_signal.py.
  against the pre-review code: all 7 store/client tests fail; 9 of the 10 recall
  tests fail (the 10th, "in-time concept_region still reinforces", is the control
  and passes on both).
python -m pytest orion/substrate/tests orion/graph -q: 974 passed, 3 failed
  (the 3 are test_felt_state_self_definition_lane.py, failing identically without
  this branch's changes)
python -m pytest tests/test_recall_bounded_retrieval_contract.py -q: 10 passed
All steps of .github/workflows/orion-static-gates.yml run locally: all PASS
  (includes check_definition_drift.py --gate and check_metric_lineage.py --gate)
```

New tests:
- slow `concept_region` (sync, 1.5s `time.sleep` in a thread) with a 500ms deadline: recall returns in under 1s, the event loop kept turning, `deadline_hit=True`, fetch and `active_packet` candidates kept, `concept_region` dropped.
- slow `active_packet` (async, 10s): cut at the deadline, `concept_region` kept.
- deadline already spent by the fetch: collectors never start.
- failing collector does not fail the recall; the other collector's result is kept.
- `timings_ms` has every new key.
- stages add up: with 150ms / 300ms / 200ms stubbed collectors and eligible count, `total - sum(top-level stages) <= 50ms`.
- four threads racing `get_substrate_store()` build once.
- warmup calls `get_substrate_store` once, off the event-loop thread; returns False without raising on exception, on `None`, and on timeout (without waiting for the thread).
- lifespan starts the warmup once and still boots when it fails; skips it when concept_region is off.

## Evals run

```text
None. This is a latency/timing fix; the recall evals measure retrieval
quality, which this patch does not change when the collectors finish in time.
```

## Docker/build/smoke checks

```text
Not run: told not to deploy or restart. No dependency, compose, or env change.
```

## Review findings fixed

- Finding (SHOULD): a warmup against an unreachable FalkorDB cached an empty store for the process lifetime and logged "warmed". The store's hydrate swallows query errors, and concept_region never refreshes the cache.
  - Fix: `FalkorSubstrateStore` now records `last_hydrate_ok` / `last_hydrate_node_count` (additive, class-level defaults for `__new__`-built instances). Recall's `get_substrate_store` refuses to cache a store whose hydrate failed or returned 0 nodes, and retries after an exponential backoff (5s, 10s, 20s ... capped at 300s). "warmed" is logged only for a real store; otherwise `recall_substrate_store_warmup_failed reason=hydrate_failed|hydrate_empty|timeout|error`. Stores without the signal (in-memory) are accepted as before.
  - Evidence: `test_failed_hydrate_is_not_cached_and_warmup_does_not_log_warmed`, `test_empty_hydrate_is_not_cached`, `test_failed_build_backs_off_then_retries_and_caches`, `test_backoff_doubles_and_caps`, `test_store_without_hydrate_signal_is_accepted`, `test_hydrate_*` in `test_falkor_store_hydrate_signal.py`.
- Finding (SHOULD): the lock wait had no bound, so request threads leaked while FalkorDB hung, and the Falkor client had no socket timeouts. The warmup docstring also wrongly said a racing recall "waits under the recall deadline".
  - Fix: request callers use `_STORE_LOCK.acquire(timeout=2.0)` and return None on timeout (concept_region then returns nothing). Only the warmup waits longer (30s). `RedisGraphQueryClient` takes optional `socket_timeout` / `socket_connect_timeout` (default unchanged for every other caller), threaded through the store config and `build_substrate_store_from_env`. Recall passes connect 5s and read 30s. Docstring rewritten.
  - Evidence: `test_request_path_gives_up_on_a_held_lock`, `test_recall_builder_passes_socket_timeouts`, `test_redis_client_default_passes_no_socket_timeouts`, `test_redis_client_passes_socket_timeouts_when_set`, `test_env_builder_threads_timeouts_to_the_store_client`. I measured live from the host, read-only (`GRAPH.RO_QUERY`) against `orion_substrate`: the four hydrate reads (4,588 nodes, 10,000 edges) took 1.42s and 1.60s in total across two runs, each under 1s. So 30s per read is about 40 times the slowest query.
- Finding (SHOULD): an abandoned concept_region thread still wrote the reinforcement bump for fragments the recall had dropped.
  - Fix: the worker passes a `threading.Event` as `abandoned=`. It is set before cancelling at the deadline, and on any cancellation. The collector checks it immediately before `reinforce_matched_concepts`.
  - Evidence: `test_deadline_cut_concept_region_does_not_reinforce` uses the real fetch-and-reinforce with the fetch half blocking 0.8s and the reinforce half recording; the recall is cut at the deadline and nothing is reinforced. Control: `test_in_time_concept_region_still_reinforces`.
- Finding (NIT): no worker-level test of a recall racing a mid-hydration warmup.
  - Fix/Evidence: `test_recall_during_warmup_hydration_is_cut_at_the_deadline`. A thread holds the real `_STORE_LOCK` inside the real `get_substrate_store` with a 1s builder. The recall returns in under 0.9s with `deadline_hit=True` and keeps the fetch and active_packet candidates. The store is built exactly once, and the recall's thread later receives the warmed store already marked abandoned.
- Finding (NIT): `test_timed_stages_add_up_to_total` relied on a fixed 50ms wall-clock bound.
  - Fix: it now asserts the untimed remainder is at most 20% of total (before the fix it was close to 100%) and at least minus the number of stages (rounding only).
- Finding (NIT): shutdown could hang on a stuck warmup thread.
  - Fix/Evidence: with the socket timeouts, a hung FalkorDB bounds the hydration thread to about 5s connect plus 4 reads of up to 30s each, roughly 2 minutes worst case (seconds normally). Python 3.12's `asyncio.run` waits at most 300s for executor threads at shutdown, and `docker stop` SIGKILLs after its grace period, so shutdown is bounded either way. The lifespan still cancels a pending warmup task. Not tested against a real hung FalkorDB (UNVERIFIED).

## Restart required

```bash
scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- Severity: low. Concern: a recall that races the boot warmup waits on the store lock inside its own worker thread. Mitigation: that wait is capped at 2s, and the recall itself never waits past its deadline.
- Severity: low. Concern: a genuinely empty substrate graph is now treated as "not ready" and retried every 5 minutes at most. Mitigation: the live graph has 4,588 nodes, so empty means FalkorDB is not ready rather than that there is no data. The only cost is a periodic retry and a warning log.
- Severity: low. Concern: the hydrate-success check only covers the direct `falkor` backend. A `routed` backend (primary plus shadow) has no signal and is accepted as-is. Recall runs `falkor`.
- Severity: low. Concern: `active_packet` still makes some synchronous calls on the event loop (chroma query, graphiti adapter) inside `retrieve_active_packet`. The deadline cannot preempt a blocked loop. Live evidence says chromadb is not installed so that path returns immediately; not changed here.
- Severity: low. Concern: `warm_substrate_store` stops waiting after 30s, but the hydration thread keeps going. Mitigation: that thread is now bounded by the 5s connect and 30s read timeouts; if it later succeeds, the store is cached.
- UNVERIFIED: behaviour against a real hung or down FalkorDB (timeouts and backoff are covered only by stubbed tests).
- UNVERIFIED: that the warmup removes the first-recall stall in production. Proof would be a `recall_substrate_store_warmed elapsed_ms=... nodes=...` log line at boot and a first purposeful `recall_telemetry` row after restart with small `pcr_concept_region` and `total ~= sum(stages)`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2436

🤖 Generated with [Claude Code](https://claude.com/claude-code)
