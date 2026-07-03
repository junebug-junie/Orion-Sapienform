# orion-substrate-runtime

Event-native substrate worker for the biometrics closed loop:

```text
grammar_events (orion-biometrics) → node biometrics projection
  → biometrics_pressure organ → pressure reducer → active pressure projection

grammar_events (orion-cortex-exec, cortex.exec:*) → execution trajectory projection
  → execution_trajectory_reducer → StateDeltaV1(target_kind=execution_run)
  → substrate_reduction_receipts → orion-field-digester

grammar_events (orion-bus, bus.transport:*) → transport bus projection
  → transport_bus_reducer → StateDeltaV1(target_kind=transport_bus)
  → substrate_reduction_receipts → orion-field-digester (when ENABLE_TRANSPORT_FIELD_DIGESTION=true)
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_transport_substrate_loop.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_reducer_quarantine_v1.sql
# Self-observability v2 (coalition dwell log + endogenous curiosity candidates):
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_coalition_dwell_v1.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_curiosity_candidates_v1.sql
cp services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/.env
```

Set `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true` after cortex-exec grammar publish is enabled (`PUBLISH_CORTEX_EXEC_GRAMMAR=true` on orion-cortex-exec). Default is `false` for safe rollout.

Set `ENABLE_TRANSPORT_BUS_REDUCER=true` after orion-bus transport traces are publishing (`PUBLISH_ORION_BUS_GRAMMAR=true`). Default is `false`.

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

## Grammar production observe

Truth: `GET http://localhost:8115/grammar/truth`

Includes per-reducer health (`reducer_health_by_name`), pending backlog, and stream lag
(cursor vs latest grammar event). Execution and transport reducers run on independent poll
loops with configurable batch limits so transport catch-up is not starved by biometrics.

Post-deploy smoke (from repo root):

```bash
./scripts/grammar_production_truth.sh
```

### Cursor recovery (internal operator endpoint)

**Not exposed via hub/Caddy.** Requires `SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN` and header `X-Orion-Operator-Token`.

See also: [docs/grammar_production_observe_deploy.md](../../docs/grammar_production_observe_deploy.md)

```bash
# Replay from earliest matching grammar event
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=earliest'

# Jump to newest row (operator acknowledges skip; marks /grammar/truth degraded)
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=tail'

# Set cursor to last event at/before a timestamp (timezone-aware ISO required)
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/cursor/reset?cursor_name=biometrics_grammar_consumer&mode=timestamp&at=2026-06-01T00:00:00Z'
```

Known cursors: `biometrics_grammar_consumer`, `execution_grammar_reducer`, `transport_grammar_reducer`.

### Poison quarantine acknowledgement (internal operator endpoint)

When a reducer quarantines a poison event, `/grammar/truth` stays **degraded** with
`reducer_quarantine_present:<cursor_name>` until an operator acknowledges the quarantine.
Quarantine state is durable in Postgres (`substrate_reducer_quarantine`); acknowledgement
sets `acknowledged_at` but preserves the audit row.

```bash
# Acknowledge a single quarantined event
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&event_id=gev_x'

# Acknowledge all unacked quarantine for a cursor
curl -X POST -H "X-Orion-Operator-Token: $SUBSTRATE_CURSOR_RESET_OPERATOR_TOKEN" \
  'http://127.0.0.1:8115/grammar/quarantine/ack?cursor_name=transport_grammar_reducer&ack_all=true'
```

Accepted-pressure reducer output publishes to `orion:grammar:accepted-pressure` (not canonical `orion:grammar:event`).

## Dynamics tick (rung-1 pacemaker)

`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` writes execution/transport prediction-error
values onto durable substrate graph nodes (`metadata['prediction_error']`), but by itself
nothing ever reads them back — the seeded surprise just sits inert on the node. Set
`SUBSTRATE_DYNAMICS_TICK_ENABLED=true` to run a periodic, bounded, fail-open
`SubstrateDynamicsEngine.tick()` against the same shared substrate graph store, which
seeds and propagates activation pressure from those `prediction_error` values.

- `SUBSTRATE_DYNAMICS_TICK_ENABLED` (default `false`): enable the tick loop.
- `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` (default `30.0`): tick cadence. Deliberately slower
  than `GRAMMAR_POLL_INTERVAL_SEC` because each tick issues a bounded but real query
  (`snapshot()`, `limit_nodes=500`) against the configured store backend, not an in-memory read.

Only meaningful once `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true` and
`SUBSTRATE_STORE_BACKEND=sparql` (Fuseki) are set — with the in-memory default store the
prediction-error nodes are process-local and this tick has nothing durable to consume.
