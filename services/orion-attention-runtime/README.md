# orion-attention-runtime

Layer 5 substrate service: polls latest `FieldStateV1` from Postgres and builds deterministic `FieldAttentionFrameV1` snapshots.

## Behavior

- Polls `substrate_field_state` every `ATTENTION_POLL_INTERVAL_SEC` (default 2s)
- Skips if an attention frame already exists for the latest field `tick_id` (idempotent)
- Persists to `substrate_attention_frames`
- Does **not** publish bus events or mutate field state

## Prerequisites

Apply migration:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_attention_frame_v1.sql
```

Requires `orion-field-digester` (or equivalent) writing `substrate_field_state`.

## Run

```bash
cp .env_example .env
docker compose up -d --build
curl -s http://localhost:8117/health
curl -s http://localhost:8117/latest | jq .
```

## Smoke

From repo root:

```bash
./scripts/smoke_attention_frame_v1.sh
```
