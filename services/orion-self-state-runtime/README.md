# orion-self-state-runtime

Layer 6 substrate service: synthesizes `SelfStateV1` from the latest `FieldAttentionFrameV1` and matching `FieldStateV1`.

## Data flow

```text
substrate_attention_frames + substrate_field_state
  → orion-self-state-runtime
  → substrate_self_state
```

## Dependencies

- `orion-field-digester` writing `substrate_field_state`
- `orion-attention-runtime` writing `substrate_attention_frames`
- Apply migration: `services/orion-sql-db/manual_migration_self_state_v1.sql`

### Optional self-observability inputs (v2)

Best-effort, never blocking: when the rows exist and are fresh, `SelfStateV1`
carries populated `attention_schema_type` / `attention_dwell_ticks` /
`attention_node_count` (from `substrate_attention_broadcast_projection`,
≤300s old) and `hub_presence` (from `substrate_hub_presence`, ≤600s old;
written by orion-hub, migration
`services/orion-sql-db/manual_migration_hub_presence_v1.sql`). Absent or
stale rows degrade to schema defaults.

## Run locally

```bash
cp .env_example .env
docker compose up -d --build
curl -s http://localhost:8118/health
curl -s http://localhost:8118/latest | jq .
```

## Non-goals (v1)

No bus publish, proposals, policy gates, cortex-exec steering, or LLM interpretation.
