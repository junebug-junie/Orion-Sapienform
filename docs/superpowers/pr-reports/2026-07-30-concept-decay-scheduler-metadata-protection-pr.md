# Concept decay scheduler metadata-clobber protection

## Summary

- Fourth fix in a same-day incident chain (PR #1498, #1501, #1503) around `node:substrate.bus_synaptic`'s `prediction_error` getting durably frozen/nulled by unprotected writes.
- `decay_concept_activations()` (`services/orion-hub/scripts/api_routes.py`) reads a concept node's full metadata via `snapshot()`, recomputes only `signals.activation`, and re-persists the whole node — re-writing whatever stale copy of reducer-owned fields (`prediction_error`, `contributing_turn_ids`) happened to be in that read.
- Lower frequency than the materializer/concept-induction paths (scheduler-triggered, not a tight loop), so lower collision probability, but the same bug class.
- Fix: `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`, same reused protection as the earlier PRs.

## Outcome moved

Closes another live instance of the same-day incident's root bug class in the Hub's periodic concept-activation decay pass.

## Current architecture

`decay_concept_activations()` runs on a schedule (Hub's own scheduler) over `SUBSTRATE_SEMANTIC_STORE`'s concept nodes, decaying `signals.activation.activation` toward each node's configured floor based on elapsed time since `observed_at`, then re-persisting each node via `upsert_node()`.

## Architecture touched

`services/orion-hub` only. No contract/schema/bus changes.

## Files changed

- `services/orion-hub/scripts/api_routes.py`: `decay_concept_activations()`'s `upsert_node()` call now passes `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`.
- `services/orion-hub/tests/test_substrate_concept_decay_scheduler.py`: new test `test_decay_concept_activations_protects_reducer_owned_metadata` spies on the store's `upsert_node` and asserts the kwarg is passed correctly.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests/test_substrate_concept_decay_scheduler.py -q
9 passed (8 baseline + 1 new), zero regressions.
```

Red-before-green confirmed via `git show HEAD:<path>` (not `git stash` — see sibling PRs' reports for why).

## Evals run

Not applicable.

## Docker/build/smoke checks

```
$ bash scripts/safe_docker_build.sh orion-hub up -d --build
... rebuilt and restarted ...
```

Live-verified: clean startup, serving requests normally post-restart.

## Review findings fixed

Code review pass (subagent) found **no blocking issues**. Notes from that pass:

- Traced every concrete backend `build_substrate_store_from_env()` can return (`InMemorySubstrateGraphStore`, `FalkorSubstrateStore` — the live backend — `RoutedSubstrateGraphStore`, `GraphDBSubstrateStore`/`SparqlSubstrateStore`): all already declare and correctly handle `skip_metadata_keys` in their shared abstract interface, so this call could not have broken any real backend the way a mismatched test-double wrapper did on the sibling materializer PR.
- Confirmed this is the only `SUBSTRATE_SEMANTIC_STORE.upsert_node()` call site in `api_routes.py` — no other collision risk in this file.
- Flagged (informationally, not a defect in this diff): `services/orion-hub/scripts/concept_atlas_routes.py`'s `_CountingSubstrateStore.upsert_node()` still lacks `skip_metadata_keys` as of this branch's tip — already found and fixed on the sibling `fix/bus-synaptic-materializer-clobber` branch (commit `79acfd871`), unrelated to this diff's own call path.
- Minor, non-blocking observation: the new test verifies wiring (the kwarg is passed) via a spy, matching the rigor level of the sibling materializer PR's own test, but doesn't independently re-verify the *value* survives an actual race the way `orion/substrate/tests/test_dynamics.py`'s more elaborate race-simulation test does for the original `SubstrateDynamicsEngine.tick()` fix. Not fixed here — the underlying store-level merge behavior is already covered by that existing test suite; this PR's test only needed to prove correct wiring at this new call site.

## Restart required

```
bash scripts/safe_docker_build.sh orion-hub up -d --build
```

Already run during this session's live verification. No further restart needed unless this branch is rebuilt from a fresh checkout.

## Risks / concerns

- Severity: Low
- Concern: None material identified by review.
- Mitigation: N/A.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/concept-decay-scheduler-metadata-protection
