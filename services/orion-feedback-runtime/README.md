# orion-feedback-runtime

Layer 10 of the Orion cognition substrate: observes `ExecutionDispatchFrameV1` outcomes and persists `FeedbackFrameV1` consequence snapshots.

**Feedback is consequence capture, not learning.** This service does not consolidate patterns, mutate policy, approve actions, retry actions, or invoke an LLM.

## Inputs

- `substrate_execution_dispatch_frames`
- `substrate_policy_decision_frames` (optional linkage)
- `substrate_proposal_frames` (optional linkage)
- `substrate_self_state` (before + first state after dispatch timestamp)

## Outputs

- `substrate_feedback_frames`

## Port

`8122` (`FEEDBACK_RUNTIME_PORT`)

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v1.sql
```

## Run

```bash
cd services/orion-feedback-runtime
cp -n .env_example .env
docker compose up -d --build
```

## Smoke

```bash
./scripts/smoke_feedback_frame_v1.sh
```

## Hub (optional)

```bash
curl -s http://localhost:8080/api/substrate/feedback/latest | jq
```
