# PR: Cortex exec substrate digestion v1

**Branch:** `feat/cortex-exec-substrate-digestion-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-cortex-exec-substrate-digestion-v1`  
**Head:** `0ed0d43a` (7 commits)

## Summary

Closes the digestion gap after cortex-exec grammar ingress (#617). Cortex-exec traces are reduced into committed `ReductionReceiptV1` / `StateDeltaV1(target_kind=execution_run)` by `orion-substrate-runtime`, then consumed by `orion-field-digester` to perturb node/capability lattice pressure — without teaching field-digester to read raw `grammar_events`.

```text
orion-cortex-exec (PUBLISH_CORTEX_EXEC_GRAMMAR=true)
  → grammar_events (source_service=orion-cortex-exec, trace_id cortex.exec:*)
  → orion-substrate-runtime execution_trajectory_reducer
  → substrate_reduction_receipts (execution_run deltas)
  → orion-field-digester
  → substrate_field_state (execution_load, execution_pressure, …)
```

**Non-goals respected:** no exec organ, no mind service, no cortex-exec runtime changes, biometrics loop unchanged.

## Architecture

| Stage | Component | Output |
|-------|-----------|--------|
| Ingress (prior PR) | `orion-cortex-exec` | `GrammarEventV1` on `orion:grammar:event` |
| Reduction (this PR) | `orion/substrate/execution_loop/` | `ExecutionTrajectoryProjectionV1`, `StateDeltaV1` |
| Persistence | `orion-substrate-runtime` | `substrate_reduction_receipts`, execution projection table |
| Field dynamics | `orion-field-digester` | Node `execution_load` / capability `execution_pressure` |

## Files changed

| Path | Change |
|------|--------|
| `orion/schemas/execution_projection.py` | `ExecutionRunStateV1`, `ExecutionTrajectoryProjectionV1` |
| `orion/schemas/registry.py` | Register execution projection schemas |
| `orion/substrate/execution_loop/*` | Extract, reducer, pipeline (no organ) |
| `services/orion-sql-db/manual_migration_execution_substrate_loop.sql` | Execution projection DDL |
| `services/orion-substrate-runtime/app/store.py` | Execution grammar fetch + cursor + projection I/O |
| `services/orion-substrate-runtime/app/worker.py` | Isolated execution tick |
| `services/orion-substrate-runtime/app/settings.py` | `ENABLE_EXECUTION_TRAJECTORY_REDUCER` (default false) |
| `services/orion-substrate-runtime/.env_example` | New flag |
| `services/orion-substrate-runtime/docker-compose.yml` | Env passthrough |
| `services/orion-substrate-runtime/README.md` | Execution loop docs |
| `services/orion-field-digester/app/ingest/state_deltas.py` | `execution_run` → perturbations |
| `services/orion-field-digester/app/tensor/channels.py` | Execution channel defaults |
| `config/field/biometrics_lattice.yaml` | Execution node/capability channels + edge maps |
| `services/orion-field-digester/README.md` | Lattice no longer biometrics-only |
| `tests/test_execution_projection_schemas.py` | Schema roundtrip |
| `tests/test_execution_substrate_reducer.py` | Extract, hints, reducer, stable ids |
| `tests/test_execution_substrate_pipeline.py` | Pipeline by trace |
| `tests/test_field_execution_perturbations.py` | Field ingest + diffusion |

## Tests

```bash
cd .worktrees/feat-cortex-exec-substrate-digestion-v1
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py tests/test_execution_substrate_pipeline.py tests/test_execution_projection_schemas.py tests/test_biometrics_pipeline.py tests/test_node_pressure_reducer.py -q
# 15 passed

PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_execution_perturbations.py tests/test_field_*.py -q
# 13 passed
```

## Rollout

1. Apply migration:
   ```bash
   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql
   ```
2. Enable cortex-exec publish: `PUBLISH_CORTEX_EXEC_GRAMMAR=true`
3. Enable reducer: `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true` on substrate-runtime
4. Verify: exec plan → `substrate_reduction_receipts` with `target_kind=execution_run` → Hub `GET /api/substrate/field/node/athena` shows `execution_load` > 0

## Code review

- Reviewer: **APPROVED** after worker tick isolation fix (`execution_substrate_tick_failed` vs biometrics).
- Field digester confirmed: no `grammar_events` reads.

## Test plan

- [x] Execution reducer + stable delta ids
- [x] Pipeline groups by trace_id
- [x] Field `execution_run` perturbations diffuse to orchestration
- [x] Biometrics regression (`test_biometrics_pipeline`, `test_node_pressure_reducer`)
- [x] Field regression (`test_field_*.py`)
- [ ] Live: `cortex.exec:* → receipt → field` observed in stack
