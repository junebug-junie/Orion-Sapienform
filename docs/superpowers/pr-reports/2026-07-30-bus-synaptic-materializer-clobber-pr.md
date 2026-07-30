# Bus synaptic materializer metadata-clobber fix

## Summary

- Second, independent cause of the `node:substrate.bus_synaptic` false "Bus Anomaly Detected" alerts, found via live re-verification after PR #1498's write-skip-on-zero fix landed and the value still snapped back to a stale `1.0` within seconds.
- Traced via FalkorDB `CLIENT LIST`/`MONITOR` to `orion-cortex-exec-background`'s concept-induction pipeline, which re-materializes this node's identity every few seconds through `SubstrateGraphMaterializer.apply_record()`'s merge branch — far more often than the owning reducer's 30s tick.
- `reconcile.py`'s `merge_node()` always keeps `existing.metadata` for any key `incoming` doesn't provide (including `prediction_error`), and the merge branch's `upsert_node()` call had no `skip_metadata_keys` protection, so that read-time snapshot got durably re-persisted every pass — a self-reinforcing stale fixed point that kept beating the real reducer's fresher writes.
- Fix: pass `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS` on the merge branch's `upsert_node()` call, reusing the exact protection mechanism already proven for `SubstrateDynamicsEngine.tick()`'s identical bug class.
- Code review before merge caught a real regression this change surfaced (two test-double store wrappers didn't accept the new kwarg, breaking the Concept Atlas ingest route) — fixed in the same patch, confirmed against the existing test suite.
- Live end-to-end verified: rebuilt both affected services, watched the node track real values (0.73 → 0.076 → 0.046) across a 64s window with zero snap-back.
- Three related, same-bug-class gaps found in other services during review — documented, not fixed here (see "Known related risk").

## Outcome moved

The false "Bus Anomaly Detected" alert this session set out to fix is now genuinely resolved end to end — not just the first of two compounding causes. `node:substrate.bus_synaptic` reflects real, live reducer output continuously instead of a value any concept-induction pass could durably re-freeze.

## Current architecture

`orion-cortex-exec-background` runs concept induction, which materializes extracted concepts into the durable substrate graph via `SubstrateGraphMaterializer.apply_record()`. When an incoming concept's identity resolves to an already-existing node (the "merge" branch), `reconcile.py`'s `merge_node()` builds a merged node by preferring `existing.metadata` over `incoming.metadata` for any key already present, then `apply_record()` durably upserts that merged node. This path has no awareness of which metadata fields are "owned" by a different subsystem (a reducer like `_bus_synaptic_tick()`) versus genuinely part of the concept's own state — it treats every concept node identically.

## Architecture touched

`orion/substrate/materializer.py` (core fix). `services/orion-hub/scripts/concept_atlas_routes.py` and its test file (regression fix surfaced by the core fix, same PR). No contract/schema/bus changes.

## Files changed

- `orion/substrate/materializer.py`: `apply_record()`'s merge branch now passes `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS` to `upsert_node()`. The create branch is deliberately unchanged (no existing state to protect).
- `orion/substrate/tests/test_materializer_reducer_owned_metadata_protection.py` (new): asserts the create-branch call has `skip_metadata_keys=None` and the merge-branch call has `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`, via a spy on `store.upsert_node`. Confirmed red-before-green (fails against pre-fix code via `git stash`).
- `services/orion-hub/scripts/concept_atlas_routes.py`: `_CountingSubstrateStore.upsert_node()` now accepts and forwards `skip_metadata_keys` — it previously didn't, and defining its own `upsert_node` meant `__getattr__` never bailed out to the wrapped store, so the materializer's new call raised `TypeError` on this wrapper.
- `services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py`: `_FailAfterNUpsertsStore` test double gets the identical fix for hygiene (not currently exercised by the merge branch, but same latent gap).

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` changes; nothing to sync.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/test_materializer_reducer_owned_metadata_protection.py tests/test_cognitive_substrate_phase3_materialization.py orion/substrate/tests/test_reconcile.py -q
15 passed in 1.07s

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests -q --ignore=orion/substrate/relational/tests
490 passed, 16 warnings in ~9s   (489 on main + 1 new test; zero regressions)

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py -q
31 passed, 17 warnings in 6.50s   (4 of these failed with TypeError before the regression fix in this same patch)
```

Red-before-green check on the new materializer test:
```
$ git stash push -- orion/substrate/materializer.py && pytest orion/substrate/tests/test_materializer_reducer_owned_metadata_protection.py -q
1 failed, 1 passed   (the skip_metadata_keys assertion correctly fails against pre-fix code)
$ git stash pop
```

## Evals run

No dedicated eval harness for this narrow path; not applicable.

## Docker/build/smoke checks

```
$ bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build
... all 4 cortex-exec containers (cortex-exec, -chat, -spark, -background) rebuilt and restarted ...

$ bash scripts/safe_docker_build.sh orion-hub up -d --build
... rebuilt and restarted (picks up the concept_atlas_routes.py regression fix) ...
```

Live-verified post-deploy, directly against FalkorDB:
```
$ redis-cli GRAPH.QUERY orion_substrate "MATCH (n) WHERE n.node_id = 'node:substrate.bus_synaptic' SET n.prediction_error = 0.73 RETURN n.prediction_error"
20:12:49 prediction_error=0.73
20:12:57 prediction_error=0.07615
20:13:05 prediction_error=0.07615
20:13:13 prediction_error=0.07615
20:13:21 prediction_error=0.045842
20:13:29 prediction_error=0.045842
20:13:37 prediction_error=0.045842
20:13:46 prediction_error=0.045842
```
Real, varying, non-saturated values tracked continuously across a 64s window (spanning both the materializer's fast re-touch cycle and the reducer's 30s tick) — zero snap-back to the stale `1.0`, confirming both this fix and PR #1498 together close the incident.

## Review findings fixed

- Finding: `services/orion-hub/scripts/concept_atlas_routes.py`'s `_CountingSubstrateStore.upsert_node()` and `services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py`'s `_FailAfterNUpsertsStore.upsert_node()` didn't accept `skip_metadata_keys`, so the materializer's newly-added kwarg on the merge branch raised `TypeError` on every re-ingest of an already-existing concept/entity via the Concept Atlas topic-foundry route — degrading the whole ingest call to `available: false` (caught by the route's broad `except Exception`). Confirmed reproducible: 4 previously-passing tests failed before this fix.
  - Fix: both wrapper classes now accept and forward `skip_metadata_keys`.
  - Evidence: `pytest services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py -q` — 31/31 passing after the fix, 4 failing before it (same session, same commit history).
- Finding (confirmed, correctly out of scope for this PR): `orion/spark/concept_induction/falkor_materialization.py`'s `materialize_concept_profile_to_falkor()` calls `store.upsert_node()` directly with no `skip_metadata_keys` at all — and unlike this bug, it **nulls** (not just freezes) `prediction_error`/`contributing_turn_ids` on any node it touches, since `falkor_codec.py` always emits those params (default `None`) and `set_assignments()` never filters `None` out of the Cypher `SET` clause. Live default backend (`CONCEPT_PROFILE_GRAPH_BACKEND=falkor`), not opt-in.
  - Fix: not attempted here — separate service (`orion-spark-concept-induction`), separate call path, needs its own review given the null-vs-freeze distinction changes the blast radius.
  - Evidence: `orion/spark/concept_induction/falkor_materialization.py`, `orion/substrate/falkor_codec.py:205` (`_safe_float(row.get("prediction_error"), default=None)`), `orion/graph/falkor_client.py::set_assignments()`.
- Finding (confirmed, correctly out of scope): two lower-frequency, same-shape gaps — `services/orion-hub/scripts/api_routes.py`'s concept decay scheduler (~line 372) and `services/orion-recall/app/collectors/concept_region.py`'s reinforcement path (~line 248) — both read-modify-write a node's full metadata without `skip_metadata_keys` protection, event/scheduler-triggered rather than a tight loop so lower collision probability, but not zero.
  - Fix: not attempted here.
  - Evidence: file/line references above.
- Finding (confirmed, pre-existing, not introduced by this diff): `apply_record()`'s existing-node lookup has an asymmetry — when `identity_key` is non-`None` but has no identity-index entry yet, there's no fallback to a direct `get_node_by_id(node.node_id)` check, so a raw `node.node_id` collision with an already-existing node stored under a different identity would take the "created" branch (full wholesale overwrite, no merge, no protection at all) instead of merging. Unlikely to fire for `bus_synaptic` specifically in steady state.
  - Fix: not attempted here.
  - Evidence: `orion/substrate/materializer.py:50-56`.

## Restart required

```
bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Already run during this session's live verification; both are reflected in the currently-running containers. No further restart needed unless this branch is rebuilt from a fresh checkout.

## Risks / concerns

- Severity: High (live, currently unaddressed, separate from this PR)
- Concern: `orion/spark/concept_induction/falkor_materialization.py` nulls (not freezes) reducer-owned metadata on the live default backend.
- Mitigation: Documented above with exact file/line evidence; recommend a fast follow-up PR, not bundled here to keep this patch's blast radius scoped and reviewable.
- Severity: Medium
- Concern: Two more same-shape unprotected read-modify-write paths (`api_routes.py` decay scheduler, `concept_region.py` reinforcement).
- Mitigation: Documented above; recommend a tracking issue.
- Severity: Low
- Concern: `apply_record()`'s create/merge branch selection has a narrow identity-collision edge case predating this PR.
- Mitigation: Documented above; not urgent given low collision probability in steady state.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/bus-synaptic-materializer-clobber
