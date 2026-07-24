# orion-policy-runtime

Layer 8 substrate service: evaluates `ProposalFrameV1` candidates against `SubstratePolicyV1` and persists **governed decisions** (`PolicyDecisionFrameV1`). Policy is not execution.

## Data flow

```text
substrate_proposal_frames
  → orion-policy-runtime
  → PolicyDecisionFrameV1
  → substrate_policy_decision_frames
```

2026-07-22 (SelfStateV1 burn): decisions are evaluated directly off
`ProposalFrameV1` (which already carries `source_field_tick_id`) -- no
separate self-state dependency or load.

## Non-goals

- No cortex-exec, bus publish, operator notifications, settings mutation, or LLM calls in the
  policy-decision data flow above -- `PolicyDecisionFrameV1` is Postgres-only, not a bus event.
- `execution_constraints` on decisions are instructions for Layer 9 only.

## Liveness telemetry

Separately from the policy-decision data flow (still Postgres-only, no change), this service
publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s) via its own independent Redis connection -- part of the
repo-wide service-heartbeat rollout
(docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md), not a policy
bus-publish.

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
