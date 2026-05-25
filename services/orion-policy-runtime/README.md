# orion-policy-runtime

Layer 8 substrate service: evaluates `ProposalFrameV1` candidates against `SubstratePolicyV1` and persists **governed decisions** (`PolicyDecisionFrameV1`). Policy is not execution.

## Data flow

```text
substrate_proposal_frames
+ substrate_self_state
  → orion-policy-runtime
  → PolicyDecisionFrameV1
  → substrate_policy_decision_frames
```

## Non-goals

- No cortex-exec, bus publish, operator notifications, settings mutation, or LLM calls.
- `execution_constraints` on decisions are instructions for Layer 9 only.

## Idempotency

One policy decision frame per `source_proposal_frame_id`. Re-running the worker for the same proposal frame is a no-op.

## Run

```bash
cp -n .env_example .env
docker compose up -d --build
curl -s http://localhost:8120/health
curl -s http://localhost:8120/latest | jq
```

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < ../../services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql
```

## Smoke

From repo root:

```bash
./scripts/smoke_policy_decision_frame_v1.sh
```
