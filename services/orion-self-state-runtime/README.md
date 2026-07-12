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

### Attention target detail (Phase 1, 2026-07-12)

`SelfStateV1.dominant_attention_target_details` carries structured per-target
data (`target_kind`, `pressure_score`, top `dominant_channel`, top `reason`)
for the same top-5 targets already named on `dominant_attention_targets` —
additive, same target_ids/order, not a replacement. Previously this data
(computed for real by `orion-attention-runtime`'s
`weighted_pressure`/`urgency_score`/`confidence_from_vector`) was read in
full and discarded down to bare ID strings.

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

## Inner-state registry

`SelfStateV1` is the one schema every cognition-facing prompt-builder is
expected to read from (directly, or via phi's `InnerStateFeaturesV1`).
`orion/self_state/inner_state_registry.py` tracks every "what does Orion
currently feel/perceive" signal in the repo — producer, cadence, whether
it's composed into `SelfStateV1`, an unresolved duplicate of another signal,
or a declared, justified shadow — so a new one is a registry entry, not a
silent duplicate discovered by grep-archaeology (see
`docs/superpowers/specs/2026-07-12-inner-state-unification-design.md`).
`python scripts/check_inner_state_registry.py` (`make check-inner-state-registry`)
gates against the registry going stale or a new inner-state-shaped
schema/channel appearing unregistered.
