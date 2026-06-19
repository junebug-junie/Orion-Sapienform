# PR: Substrate Signal Bridge V1

**Branch:** `worktree-feat+substrate-signal-bridge-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.claude/worktrees/feat+substrate-signal-bridge-v1`  
**Head:** `1618043e` (5 commits)

## Summary

Bridges existing `OrionSignalV1` records produced by `CognitionTraceAdapter` (organ `cortex_exec`, signal kinds `cognition_run` and `cognition_step`) into the shared substrate grammar as `SubstrateMoleculeV1` of a new `organ_signal` molecule kind. The substrate can now consume real organ signals without a parallel signal schema.

```text
CognitionTraceAdapter (orion-cortex-exec)
  → OrionSignalV1 (organ_id=cortex_exec, signal_kind=cognition_run|cognition_step)
  → orion/substrate/signal_bridge.py :: signal_to_molecule()
  → SubstrateMoleculeV1(molecule_kind=organ_signal)
  → MoleculeJsonlStore / SubstrateExperimentHarness
  → daily rollup: organ_coverage["cortex_exec"]
```

**Non-goals respected:** no new atom kinds, predicate kinds, or gradient keys; no edits to `orion/signals/**`, `orion/substrate/molecules.py`, molecule_store, operators, or signal gateway.

## Architecture

| Layer | Component | Output |
|-------|-----------|--------|
| Schema | `orion/schema_kernel/registry.py` | `organ_signal` added to `default_registry()` molecule_kinds (one-line) |
| Bridge | `orion/substrate/signal_bridge.py` | Pure projection: `OrionSignalV1 → SubstrateMoleculeV1` |
| Worker | `orion/substrate/signal_bus_worker.py` | Optional bus-driver-agnostic seam (not wired to live bus yet) |

Gradient mapping from signal dimensions to canonical substrate gradients:

| Gradient | Source |
|----------|--------|
| `salience` | `max(latency_level, step_count, service_count, level, salience)` |
| `contradiction` | `max(error_present, 1-success if success present, contradiction)` |
| `coherence` | `max(success if present, coherence) × confidence` |
| `novelty` | `max(novelty, surprise)` |

## Files changed

| Path | Change |
|------|--------|
| `orion/schema_kernel/registry.py` | Add `organ_signal` to `default_registry()` molecule_kinds |
| `orion/substrate/signal_bridge.py` | New — pure `OrionSignalV1 → SubstrateMoleculeV1` projection |
| `orion/substrate/signal_bus_worker.py` | New — optional `handle_envelope()` seam for live bus bridging |
| `tests/test_substrate_signal_bridge.py` | 7 unit tests: registry, happy path, failure→contradiction, step error, unsupported skip/raise, grammar validation |
| `tests/test_substrate_signal_bridge_e2e.py` | 1 e2e test: bridge → store → harness → daily rollup shows `cortex_exec` coverage |

## Tests

```bash
cd .claude/worktrees/feat+substrate-signal-bridge-v1
/mnt/scripts/Orion-Sapienform/.orion_dev/bin/pytest tests/test_substrate_signal_bridge.py tests/test_substrate_signal_bridge_e2e.py -q
# 8 passed in 0.77s
```

## Code review

Reviewer flagged two worker issues (no bridge issues):

1. **Bridge/store exceptions unguarded** — `signal_to_molecule`/`store.add` exceptions propagated to bus driver loop. Wrapped in `try/except` with `logger.warning`. Fixed in `1618043e`.
2. **Payload type guard inconsistent with project convention** — missing `model_dump()` check for pre-deserialized Pydantic payloads (pattern from `ConceptWorker`). Fixed in `1618043e`.

Verdict: **approved after fixes**.

## Rollout

No migrations required. No env changes required.

To wire the live bus worker (follow-up, not in this PR):
1. Subscribe `SubstrateSignalBusWorker.handle_envelope` to `orion:signals:cortex_exec` via an `OrionBusAsync` driver
2. Verify: `cognition_run` signal → `organ_signal` molecule in store → daily rollup shows `organ_coverage["cortex_exec"] > 0`

## Test plan

- [x] `organ_signal` registered in `default_registry()`
- [x] `cognition_run` success → correct gradients (salience=latency, coherence=1, contradiction=0)
- [x] `cognition_run` failure → contradiction=1, coherence=0
- [x] `cognition_step` with `error_present` → contradiction=1
- [x] Batch helper skips unsupported; single raises `ValueError`
- [x] All bridged molecules validate against `default_registry()` (atoms, predicates, gradient keys)
- [x] End-to-end: bridge → store → harness → daily rollup `organ_coverage["cortex_exec"] == 3`
- [x] Worker fail-open: malformed payload silently dropped; bridge/store errors logged as warning
- [ ] Live: `cortex_exec` signal → `organ_signal` molecule observed in daily rollup (requires bus wiring follow-up)
