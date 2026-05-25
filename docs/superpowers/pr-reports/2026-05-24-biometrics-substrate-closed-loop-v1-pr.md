# PR: Biometrics substrate closed loop v1

**Branch:** `feat/biometrics-substrate-closed-loop-v1`  
**Base:** `feat/biometrics-node-grammar-ingress`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-biometrics-substrate-closed-loop-v1`

## Summary

- Closes the first event-native substrate loop: biometrics `GrammarEventV1` → `node_biometrics_projection` → `biometrics_pressure` organ candidates → node pressure reducer → `ReductionReceiptV1` → `active_node_pressure_projection`.
- Adds `orion-substrate-runtime` worker (polls `grammar_events`, no state authority in biometrics collector).
- Hub debug routes expose the full chain per node; accepted pressure candidates can publish back to `orion:grammar:event`.

## Architecture

```text
orion:grammar:event (orion-biometrics)
  → sql-writer grammar_events
  → orion-substrate-runtime poll
  → biometrics_node_reducer → node_biometrics_projection
  → biometrics_pressure → OrganEmissionV1 (candidates only)
  → node_pressure_reducer → ReductionReceiptV1 + active_node_pressure_projection
  → Hub GET /api/substrate/biometrics-node/{node_id}/latest
```

**Invariant:** orion-biometrics observes · biometrics_pressure interprets · substrate reducer commits · projections expose · traces explain.

## Changes by area

### `orion/schemas/`
- `organ_emission.py`, `reduction_receipt.py`, `state_delta.py`, `biometrics_projection.py`
- Registry entries for all new models

### `orion/biometrics/` + `orion/substrate/biometrics_loop/`
- Hoisted `NodeCatalog`; grammar extract, node reducer, pressure organ, pressure reducer, pipeline, candidate event builder

### `services/orion-sql-db/` + `orion-sql-writer`
- `manual_migration_biometrics_substrate_loop.sql`
- `app/models/biometrics_substrate.py`

### `services/orion-substrate-runtime/`
- Poll worker, Postgres store, optional bus publish for accepted candidates
- `.env_example` (+ operator `.env` sync)

### `services/orion-substrate-organs/`
- `biometrics_pressure` contract YAML + thin wrappers (logic in `orion/substrate/biometrics_loop/`)

### `services/orion-hub/`
- `substrate_biometrics_routes.py` — `/api/substrate/biometrics-node/{id}/latest`, `/biometrics/latest`, `/node-pressure/latest`

### `orion/bus/channels.yaml`
- `orion-substrate-runtime` producer on `orion:grammar:event`

### `scripts/smoke_biometrics_closed_loop.sh`

## Test plan

- [x] `PYTHONPATH=. pytest tests/test_biometrics_*.py tests/test_node_*.py orion/grammar/tests/test_biometrics_substrate_schemas.py services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q` → **36 passed**
- [x] `PYTHONPATH=services/orion-biometrics:. pytest services/orion-biometrics/tests/ -q` → **10 passed** (ingress regression)
- [x] Runtime import `BiometricsSubstrateWorker` → ok
- [ ] Apply `manual_migration_biometrics_substrate_loop.sql` + start stack → `scripts/smoke_biometrics_closed_loop.sh`
- [ ] `redis-cli SUBSCRIBE orion:grammar:event` — `biometrics.node:*` and `substrate.pressure:*` traces

## Verification evidence

```
36 passed in 1.49s (substrate loop + hub debug)
10 passed (orion-biometrics ingress, separate PYTHONPATH)
```

Live bus subscribe / smoke curl: **UNVERIFIED** in agent environment (no full compose stack).

## Acceptance criteria (spec §16)

| # | Criterion | Met |
|---|-----------|-----|
| 1 | Grammar persists durably | Yes (sql-writer unchanged; runtime consumes) |
| 2 | `node_biometrics_projection` from event log | Yes |
| 3 | Node-scoped, not cluster | Yes |
| 4 | `prometheous` → `prometheus` | Yes (catalog + tests) |
| 5 | Circe suppression | Yes (Rule A + test) |
| 6 | Organ consumes grammar events | Yes |
| 7 | Organ reads projections not blobs | Yes |
| 8 | `OrganEmissionV1` candidates | Yes |
| 9 | No state deltas in emission | Yes |
| 10 | `ReductionReceiptV1` | Yes |
| 11 | Pressure projection via reducer only | Yes |
| 12 | Debug endpoint chain | Yes |
| 13 | No `debug_trace` parsing | Yes |
| 14 | Biometrics telemetry if loop fails | Yes (ingress unchanged) |

## Non-goals (confirmed)

- No UI polish, autonomy actions, mesh scheduler, cluster aggregate grammar, biometrics collector rewrite.

## Operator notes

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
cp services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/.env
# Hub already uses POSTGRES_URI for debug routes
```
