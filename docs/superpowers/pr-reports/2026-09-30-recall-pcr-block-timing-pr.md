# fix(recall): warm substrate store at boot; PCR collectors timed and under the deadline

## Summary

- The first belief recall after a recall restart took 9.5 seconds, and the recall's own timing record only explained about 0.2 of them. The missing ~9.3s was the belief-specific step (the "PCR collectors": `active_packet` and `concept_region`), which ran after the time-limited fetch, had no time limit of its own, and was never timed.
- Recall now builds the concept-graph store handle in the background when the service starts, so the first request no longer pays the slow first connection (6.25s measured live).
- The two belief collectors now run concurrently inside whatever is left of the recall's overall time limit. If they are still running at the limit, recall stops waiting, drops only their results, keeps everything else, and marks `deadline_hit`. They never fail the recall.
- The timing record now covers these steps (`pcr_collectors`, plus `pcr_active_packet` / `pcr_concept_region`), the eligible-belief count (`eligible_count`), self-hit suppression (`suppression`) and the shadow comparison (`shadow_compare`), so the stages add up to the total. A test pins that.
- `get_substrate_store()` now takes a lock while building, so the boot warmup and a request racing it cannot both build a store.

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
- `services/orion-recall/app/substrate_store.py`: `_STORE_LOCK` double-checked init; `warm_substrate_store(timeout_s=30)`.
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
  after:                       2 failed, 346 passed
  the 2 failures are pre-existing and unchanged:
    test_recall_policy_harness::test_process_recall_diagnostic_contains_gating_suppression_and_selection
    test_recall_vector_amputation::test_worker_and_recall_v2_import_without_vector_adapter

tests/test_recall_pcr_block_timing.py: 13 passed
  against the OLD worker.py: 5 of them fail (both deadline-cut tests, the
  skip-when-deadline-passed test, the new-keys test, the stages-add-up test)

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

- Review is run by the orchestrator after this branch is pushed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- Severity: low. Concern: a recall that races the boot warmup waits on the store lock inside its own worker thread. Mitigation: the recall no longer waits past its deadline; the thread finishes in the background and holds one default-executor slot meanwhile.
- Severity: low. Concern: `active_packet` still makes some synchronous calls on the event loop (chroma query, graphiti adapter) inside `retrieve_active_packet`. The deadline cannot preempt a blocked loop. Live evidence says chromadb is not installed so that path returns immediately; not changed here.
- Severity: low. Concern: `warm_substrate_store` gives up waiting after 30s but the hydration thread keeps going (Falkor calls have no client timeout). Mitigation: if it later succeeds the store is populated anyway; if it hangs forever, requests behave as before this patch, except bounded by the deadline.
- UNVERIFIED: that the warmup removes the first-recall stall in production. Proof would be a `recall_substrate_store_warmed elapsed_ms=...` log line at boot and a first purposeful `recall_telemetry` row after restart with small `pcr_concept_region` and `total ~= sum(stages)`.

## PR link

(filled in on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
