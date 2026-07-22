# orion-feedback-runtime

Layer 10 of the Orion cognition substrate: observes `ExecutionDispatchFrameV1` outcomes and persists `FeedbackFrameV1` consequence snapshots.

**Feedback is consequence capture, not learning.** This service does not consolidate patterns, mutate policy, approve actions, retry actions, or invoke an LLM.

## Inputs

- `substrate_execution_dispatch_frames`
- `substrate_dispatch_results` — real cortex-exec results for evidenced `dispatched`
  candidates (P1 of the motor-nerve spec; `load_cortex_result_evidence` was a stub
  returning `[]` before this, so `cortex_result`-kind observations were previously
  never produced from a live dispatch)
- `substrate_policy_decision_frames` (optional linkage)
- `substrate_proposal_frames` (optional linkage)
- `substrate_field_state` (before + first field tick after dispatch timestamp --
  2026-07-22, SelfStateV1 burn: replaces the old `substrate_self_state` read;
  `field_pressures()` (`orion/field/pressure.py`) provides the same real,
  non-hand-tuned channel comparison self-state used to pass through)

## Outputs

- `substrate_feedback_frames`

## Port

`8122` (`FEEDBACK_RUNTIME_PORT`)

## Bus / Redis

| Env | Default | Purpose |
|-----|---------|---------|
| `ORION_BUS_URL` | `redis://bus-core:6379/0` (compose) | Redis URL for Orion bus publish |
| `ORION_BUS_ENABLED` | `true` | Disable bus connect/publish when `false` |
| `FEEDBACK_BUS_CHANNEL` | `orion:feedback:frame` | Channel for `feedback.frame.v1` envelopes |

Bring up with root `.env` so `ORION_BUS_URL` matches the live stack:

```bash
docker compose --env-file ../../.env --env-file .env -f docker-compose.yml up -d --build
```

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v2_drop_self_state.sql
```

## Run

```bash
cd services/orion-feedback-runtime
cp -n .env_example .env
docker compose --env-file ../../.env --env-file .env up -d --build
```

## Smoke

```bash
./scripts/smoke_feedback_frame_v1.sh
```

## Hub (optional)

```bash
curl -s http://localhost:8080/api/substrate/feedback/latest | jq
```
