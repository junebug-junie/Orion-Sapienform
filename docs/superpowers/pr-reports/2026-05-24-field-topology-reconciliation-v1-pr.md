# PR: Field topology reconciliation + execution digestion hardening

**Branch:** `feat/field-topology-reconciliation-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-field-topology-reconciliation-v1`  
**Head:** `5487569f` (5 commits)

## Summary

Hardens the substrate digestion path after cortex-exec execution digestion (#618). Persisted `FieldStateV1` is reconciled against the current lattice every digestion tick; execution-run reduction merges partial trace batches monotonically. Prepares layer 5 (`AttentionFrameV1`) without implementing attention.

```text
ReductionReceiptV1 / StateDeltaV1
  → orion-field-digester
  → load FieldStateV1
  → reconcile with orion_field_topology.v1.yaml
  → perturb / decay / diffuse / suppress
  → persist FieldStateV1
```

## Stale-lattice bug (live)

After config gained execution channel maps (`execution_load → execution_pressure`, etc.), persisted field snapshots still carried old edges, e.g. `node:athena → capability:orchestration` with only `cpu_pressure → pressure`. Field-digester applied execution deltas but diffusion used stale `channel_map` until manual `substrate_field_state` reset.

## Reconciliation behavior

`reconcile_field_state_with_lattice()` on every digestion tick:

1. Ensures all lattice nodes/capabilities exist with configured channels (defaults for missing).
2. Preserves existing channel values and unknown extra channels.
3. Replaces `state.edges` from current lattice config.
4. Preserves `recent_perturbations`.
5. Sets optional `topology_id` / `topology_version` / `topology_loaded_from` on snapshots.

## Monotonic execution merge

`merge_execution_run_state()` before writing `projection.runs[trace_id]`:

- Step counts: `max` of existing/incoming.
- Flags (`final_text_present`, `reasoning_present`, `recall_observed`): OR — no downgrade from later partial batches.
- `status`: rank-based (`failed/error` > `partial` > `success` > `unknown`).
- `egress_confidence`: recomputed with `egress_emitted` true if either batch had full egress.
- `evidence_event_ids`: sorted union.

Fixes live downgrade: `egress_confidence: 1.0 → 0.25`, `status: success → unknown`, etc.

## Config

| File | Role |
|------|------|
| `config/field/orion_field_topology.v1.yaml` | Canonical topology (biometrics + execution) |
| `config/field/biometrics_lattice.yaml` | Compatibility alias (same body + comment) |
| `LATTICE_PATH` default | `/app/config/field/orion_field_topology.v1.yaml` |

## Files changed

| Path | Change |
|------|--------|
| `config/field/orion_field_topology.v1.yaml` | New canonical config |
| `config/field/biometrics_lattice.yaml` | Alias comment |
| `services/orion-field-digester/app/tensor/reconcile.py` | Reconciliation |
| `services/orion-field-digester/app/worker.py` | Reconcile + topology metadata each tick |
| `orion/substrate/execution_loop/merge.py` | Monotonic run merge |
| `orion/substrate/execution_loop/reducer.py` | Use merge |
| `orion/schemas/field_state.py` | Optional `topology_*` fields |
| `scripts/smoke_execution_field_digestion.sh` | Live SQL inspection |
| `tests/test_field_topology_reconciliation.py` | Reconcile tests |
| `tests/test_execution_substrate_reducer.py` | Merge tests |

## Tests

```bash
cd .worktrees/feat-field-topology-reconciliation-v1
# Field (docker image with digester deps)
docker run --rm -v "$PWD:/src" -w /src orion-field-digester-test:tmp \
  bash -c 'pip install -q pytest && PYTHONPATH=/src:/src/services/orion-field-digester \
  pytest tests/test_field_topology_config.py tests/test_field_topology_reconciliation.py \
  tests/test_field_execution_perturbations.py tests/test_field_state_schemas.py \
  tests/test_field_digestion_rules.py tests/test_field_deterministic_replay.py -q'
# 16 passed

docker run --rm -v "$PWD:/src" -w /src python:3.12-slim \
  bash -c 'pip install -q pytest pydantic pydantic-settings pyyaml requests && PYTHONPATH=/src \
  pytest tests/test_execution_substrate_reducer.py tests/test_execution_substrate_pipeline.py \
  tests/test_execution_projection_schemas.py tests/test_biometrics_pipeline.py \
  tests/test_node_pressure_reducer.py -q'
# 20 passed
```

## Manual live smoke

```bash
./scripts/smoke_execution_field_digestion.sh
# Optional dev reset (destructive):
RESET_FIELD_STATE=1 ./scripts/smoke_execution_field_digestion.sh
```

After one field-digester tick with stale persisted state, expect `node:athena.execution_load` and `capability:orchestration.execution_pressure` without manual field reset.

## Code review

- Reviewer: **APPROVED** — reconcile + merge correct; scope discipline maintained; no attention/self-state bleed.

## Test plan

- [x] Field topology reconciliation tests
- [x] Execution monotonic merge tests
- [x] Field + biometrics regressions
- [ ] Live: `scripts/smoke_execution_field_digestion.sh` on stack

## Non-goals

Does **not** implement `AttentionFrameV1`, `SelfStateV1`, mind, proposals, or bus field events. This PR only hardens substrate so layer 5 can consume reconciled `FieldStateV1`.
