# orion-proposal-runtime

Layer 7 substrate service: converts `SelfStateV1` (+ optional attention/field context) into **possible actions** (`ProposalFrameV1`), not automatic actions.

## Data flow

```text
substrate_self_state
+ substrate_attention_frames
+ substrate_field_state
  → orion-proposal-runtime
  → ProposalFrameV1
  → substrate_proposal_frames
```

## Non-goals

- No policy approval, cortex-exec, bus publish, operator notifications, or LLM calls.
- `execution_intent` on candidates is descriptive only.

## Idempotency

One proposal frame per `source_self_state_id`. Re-running the worker for the same self-state snapshot is a no-op. Policy/template changes do not regenerate until a new self-state row exists (v1 semantics).

## Run

```bash
cp -n .env_example .env
docker compose up -d --build
curl -s http://localhost:8119/health
curl -s http://localhost:8119/latest | jq
```

## Migration

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < ../../services/orion-sql-db/manual_migration_proposal_frame_v1.sql
```

## Smoke

From repo root:

```bash
./scripts/smoke_proposal_frame_v1.sh
```
