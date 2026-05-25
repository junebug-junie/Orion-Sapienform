# orion-consolidation-runtime

Layer 11 of the Orion cognition substrate: aggregates Layers 5–10 substrate history over a time window and persists `ConsolidationFrameV1` motif snapshots.

**Consolidation is pattern observation, not learning.** This service does not mutate policy, publish to the bus, invoke an LLM, or change runtime behavior.

## Inputs

- `substrate_self_state`
- `substrate_attention_frames`
- `substrate_proposal_frames`
- `substrate_policy_decision_frames`
- `substrate_execution_dispatch_frames`
- `substrate_feedback_frames`

## Outputs

- `substrate_consolidation_frames`

## Port

`8123` (`CONSOLIDATION_RUNTIME_PORT`)

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_consolidation_v1.sql
```

## Run

```bash
cd services/orion-consolidation-runtime
cp -n .env_example .env
docker compose up -d --build
```

## Smoke

```bash
./scripts/smoke_consolidation_v1.sh
```

## Idempotency

One frame per `(window_start, window_end, policy_id)` via stable `frame_id`. Re-runs use `INSERT ... ON CONFLICT (frame_id) DO NOTHING`.
