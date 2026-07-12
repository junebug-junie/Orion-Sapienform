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

## Health monitor

A background health monitor (`ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC`, default 900s) watches `substrate_attention_frames`'s oldest row: if it exceeds `ATTENTION_FRAME_STALL_MULTIPLIER` (default `1.5`) x `ATTENTION_FRAME_RETENTION_HOURS`, the hourly pruner may have stopped running. Staleness is keyed on `created_at` -- the same column the prune SQL's cutoff filters on -- so the two can never disagree about what "age" means.

The check is edge-triggered: an alert (via `orion-notify`'s `POST /attention/request`, surfacing in Hub's existing Pending Attention panel) fires only on a healthy->unhealthy transition, plus a lower-severity recovery note on the way back, so a persisting condition does not spam a fresh attention item every check. On worker restart mid-incident, it first checks `orion-notify` for an already-open alert for this service+reason before firing a duplicate. If `orion-notify` is unreachable at the exact moment of a transition, the alert retries every subsequent tick until delivery is actually confirmed -- it is never silently dropped.

Mirrors the identical pattern in `orion-field-digester` (`app/health_monitor.py`), adapted to this service's single table.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC` | `900.0` | Health-monitor check cadence |
| `ATTENTION_FRAME_STALL_MULTIPLIER` | `1.5` | Alert if `substrate_attention_frames`'s oldest row exceeds this x retention hours |
| `NOTIFY_BASE_URL` | `http://orion-athena-notify:7140` | `orion-notify` base URL for health-monitor attention alerts |
| `NOTIFY_API_TOKEN` | (empty) | `orion-notify` auth token, if configured |
