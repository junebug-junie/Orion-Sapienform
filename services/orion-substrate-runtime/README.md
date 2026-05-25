# orion-substrate-runtime

Event-native substrate worker for the biometrics closed loop:

```text
grammar_events (orion-biometrics) → node biometrics projection
  → biometrics_pressure organ → pressure reducer → active pressure projection
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
cp services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/.env
```

## Run

```bash
cd services/orion-substrate-runtime
docker compose up --build
```

Health: `GET http://localhost:8115/health`

## Idempotent committed deltas

`ReductionReceiptV1.receipt_id` and `StateDeltaV1.delta_id` are derived from reducer inputs via `orion.substrate.ids` so replay/retry does not mint duplicate committed identities. Postgres inserts use `ON CONFLICT (receipt_id) DO NOTHING`.

Hub debug (node-scoped lineage):

- `GET /api/substrate/biometrics-node/{node_id}/latest`
- `GET /api/substrate/receipts/{receipt_id}`
