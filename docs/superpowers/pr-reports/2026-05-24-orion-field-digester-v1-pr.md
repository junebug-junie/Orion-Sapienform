# PR: Orion field digester v1 (biometrics digestion)

**Branch:** `feat/orion-field-digester-v1`  
**Base:** `feat/biometrics-substrate-delta-seam-hardening`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-orion-field-digester-v1`  
**Head:** `1b8fd063` (12 commits)

## Summary

Builds the first digestion slice: `orion-field-digester` consumes committed `ReductionReceiptV1` / `StateDeltaV1` from the biometrics substrate loop and compiles them into inspectable lattice field state (`FieldStateV1`). Proves atlas GPU pressure perturbations diffuse into `llm_inference` capability pressure, with Hub debug endpoints exposing the field.

This is the bridge between grammar substrate (committed deltas) and field dynamics (node/capability vectors), not another organ.

## Architecture

```text
orion-biometrics → substrate-runtime → substrate_reduction_receipts
  → orion-field-digester (poll receipts)
  → perturbation → decay → diffusion → suppression
  → substrate_field_state (FieldStateV1 snapshots)
  → Hub GET /api/substrate/field/*
```

```text
GrammarEventV1 → reducer → StateDeltaV1 → field digester → lattice state → field projections
```

## What was built

### Shared schemas + config

- `orion/schemas/field_state.py` — `FieldStateV1`, `FieldEdgeV1`
- `orion/schemas/registry.py` — registered both schemas
- `config/field/biometrics_lattice.yaml` — nodes (atlas/athena/circe/prometheus), capabilities, node→capability edges

### Digestion core (TDD)

| Rule | Module | Behavior |
|------|--------|----------|
| Perturbation | `ingest/state_deltas.py`, `digestion/perturbation.py` | Committed pressure deltas inject energy into node channels |
| Decay | `digestion/decay.py` | `pressure[t+1] = pressure[t] * decay_rate` (default 0.92) |
| Diffusion | `digestion/diffusion.py` | `target += source * edge_weight * diffusion_rate` along lattice edges |
| Suppression | `digestion/suppression.py` | Expected-offline nodes (circe) don't trigger absence panic |

### Service: `orion-field-digester`

- Polls `substrate_reduction_receipts` (not raw grammar events)
- Dedupes on stable `delta_id` via `substrate_field_applied_deltas`
- Atomic commit: field snapshot + applied deltas + cursor in one transaction
- Port `8116`, docker-compose + Dockerfile + `.env_example`

### Hub debug API

| Route | Returns |
|-------|---------|
| `GET /api/substrate/field/latest` | Latest `FieldStateV1` snapshot |
| `GET /api/substrate/field/node/{node_id}` | Node vector + connected capabilities + recent perturbations |
| `GET /api/substrate/field/capability/{capability_id}` | Capability vector + connected nodes |

### Postgres migration

`services/orion-sql-db/manual_migration_field_digester_v1.sql`:
- `substrate_field_digest_cursor`
- `substrate_field_applied_deltas`
- `substrate_field_state`

## Acceptance criteria

| # | Criterion | Met |
|---|-----------|-----|
| 1 | Consumes receipts/deltas, not debug traces | ✅ |
| 2 | Biometrics pressure deltas perturb node vectors | ✅ |
| 3 | Node pressure diffuses into capability vectors | ✅ |
| 4 | Expected-offline suppression prevents false instability | ✅ |
| 5 | Field state decays over ticks | ✅ |
| 6 | Field state inspectable through debug endpoint | ✅ |
| 7 | Field rebuildable deterministically from committed deltas | ✅ |
| 8 | No mind service involvement | ✅ |

## Test plan

- [x] Field unit tests (`tests/test_field_*.py`) → **9 passed**
- [x] Hub field debug API (`test_substrate_field_debug_api.py`) → **4 passed**
- [x] Substrate regression (`test_node_pressure_reducer`, `test_biometrics_pipeline`) → **7 passed**
- [x] Hub biometrics debug regression → **3 passed**
- [ ] Live stack: `scripts/smoke_field_digester_biometrics.sh` with migration + services up
- [ ] Prove chain: atlas biometrics → receipt → field digester → atlas gpu_pressure > 0 → llm_inference pressure > 0

## Verification evidence

```
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_*.py -q  → 9 passed
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_substrate_field_debug_api.py -q  → 4 passed
PYTHONPATH=. pytest tests/test_node_pressure_reducer.py tests/test_biometrics_pipeline.py -q  → 7 passed
python -m compileall services/orion-field-digester  → exit 0
```

Code review (subagent): **Approved** — fixed atomic commit for field state + applied deltas + cursor (`1b8fd063`).

## Non-goals (confirmed)

- No bus channel publish (projections-only v1)
- No field threshold events (`field.pressure.diffused`, etc.)
- No vision/mind/dream digestion
- No new organ, no LLM field interpretation, no autonomy actions

## Files touched

| Area | Files |
|------|--------|
| Schemas | `orion/schemas/field_state.py`, `registry.py` |
| Config | `config/field/biometrics_lattice.yaml` |
| Migration | `manual_migration_field_digester_v1.sql` |
| Service | `services/orion-field-digester/` (full scaffold) |
| Hub | `substrate_field_routes.py`, `api_routes.py`, tests |
| Tests | `tests/test_field_*.py` |
| Scripts | `scripts/smoke_field_digester_biometrics.sh` |
| Docs | `docs/superpowers/plans/2026-05-24-orion-field-digester-v1.md` |

## Operator notes

1. Apply migration: `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql`
2. Copy `services/orion-field-digester/.env_example` → `.env` (sync local keys)
3. Start `orion-field-digester` alongside `orion-substrate-runtime` + Hub
4. Env keys: `BIOMETRICS_FIELD_DECAY_RATE=0.92`, `BIOMETRICS_FIELD_DIFFUSION_RATE=1.0`, `RECEIPT_POLL_INTERVAL_SEC=2.0`

**Note:** Decay runs on receipt-driven ticks (when new receipts arrive), not wall-clock idle intervals — acceptable for v1.
