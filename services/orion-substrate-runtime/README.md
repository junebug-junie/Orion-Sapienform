# orion-substrate-runtime

Event-native substrate worker for the biometrics closed loop:

```text
grammar_events (orion-biometrics) → node biometrics projection
  → biometrics_pressure organ → pressure reducer → active pressure projection

grammar_events (orion-cortex-exec, cortex.exec:*) → execution trajectory projection
  → execution_trajectory_reducer → StateDeltaV1(target_kind=execution_run)
  → substrate_reduction_receipts → orion-field-digester
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql
cp services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/.env
```

Set `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true` after cortex-exec grammar publish is enabled (`PUBLISH_CORTEX_EXEC_GRAMMAR=true` on orion-cortex-exec). Default is `false` for safe rollout.

## Run

```bash
cd services/orion-substrate-runtime
docker compose up --build
```

Health: `GET http://localhost:8115/health`

## Idempotent committed deltas

`ReductionReceiptV1.receipt_id` and `StateDeltaV1.delta_id` are derived from reducer inputs via `orion.substrate.ids` so replay/retry does not mint duplicate committed identities. Postgres inserts use `ON CONFLICT (receipt_id) DO NOTHING`.

Pressure receipts also include `emission_id` in the receipt preimage (organ still mints a fresh `emission_id` per invocation). Field digester dedupe should key on **`delta_id`** (event-scoped), not `receipt_id`, when comparing across emissions.

Hub debug (node-scoped lineage):

- `GET /api/substrate/biometrics-node/{node_id}/latest`
- `GET /api/substrate/receipts/{receipt_id}`
