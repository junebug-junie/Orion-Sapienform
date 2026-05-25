# orion-execution-dispatch-runtime

Layer 9 of the Orion cognition substrate: converts `PolicyDecisionFrameV1` + `ProposalFrameV1` + `SelfStateV1` into `ExecutionDispatchFrameV1` envelopes.

## Safety (v1)

- Default mode: `EXECUTION_DISPATCH_MODE=dry_run`
- No bus publish and no cortex-exec calls in the default worker path
- Mutating dispatch is disabled in policy config (`allow_mutating_dispatch: false`)

## Prerequisites

1. `substrate_policy_decision_frames` populated (`orion-policy-runtime`, port 8120)
2. `substrate_proposal_frames` and `substrate_self_state` from Layers 7–6
3. Apply migration:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql
```

## Run

```bash
cd services/orion-execution-dispatch-runtime
cp -n .env_example .env
docker compose up -d --build
```

## Debug

- `GET http://localhost:8121/health`
- `GET http://localhost:8121/latest`
- Hub: `GET http://localhost:8080/api/substrate/execution-dispatch/latest`

## Smoke

```bash
./scripts/smoke_execution_dispatch_v1.sh
```
