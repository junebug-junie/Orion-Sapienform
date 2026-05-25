# PR: Attention Frame v1 — FieldState → What Matters Now

**Branch:** `feat/attention-frame-v1`  
**Base:** `feat/field-topology-reconciliation-v1`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-attention-frame-v1`  
**Head:** `4d43b49e` (4 commits)

## Summary

Implements **Layer 5** of the Orion cognition substrate: deterministic, read-only attention over digested field state.

```text
substrate_field_state (FieldStateV1)
  → orion-attention-runtime
  → FieldAttentionFrameV1
  → substrate_attention_frames
  → GET /api/substrate/attention/latest (Hub debug)
```

**Core phrase:** *Attention is the substrate’s first act of selection* — not agency, not selfhood.

### 11-layer roadmap placement

| Layer | Name | This PR |
|-------|------|---------|
| 1–4 | Organs → Grammar → Reducers → Field digestion | Prerequisite (field-digester) |
| **5** | **Attention** | **Implemented** |
| 6 | Self-state | Deferred |
| 7–11 | Proposals → Policy → Execution → Feedback → Consolidation | Deferred |

## Naming: FieldAttention* vs conversational AttentionFrameV1

`orion/schemas/attention_frame.py` already defines **conversational** `AttentionFrameV1` (`open_loops`, `selected_action`) for chat/curiosity. Substrate Layer 5 uses:

- `FieldAttentionTargetV1` / `FieldAttentionFrameV1`
- `schema_version: field.attention.frame.v1`
- Package: `orion/attention/field_attention/`

Conversational attention is **unchanged**.

## Architecture

| Component | Role |
|-----------|------|
| `config/attention/field_attention_policy.v1.yaml` | Weights, thresholds, channel maps |
| `orion/attention/field_attention/{scoring,selectors,builder,policy}.py` | Pure deterministic frame build |
| `services/orion-attention-runtime` | Poll field → build → persist (no bus) |
| `substrate_attention_frames` | Postgres persistence |
| `scripts/smoke_attention_frame_v1.sh` | SQL inspection |

## Example synthetic `FieldAttentionFrameV1`

From `node:athena.execution_load=1.0`, `capability:orchestration.execution_pressure=1.0`:

```json
{
  "schema_version": "field.attention.frame.v1",
  "frame_id": "attention.frame:tick_exec_attention:field_attention_policy.v1",
  "overall_salience": 0.386,
  "dominant_targets": [
    {
      "target_id": "node:athena",
      "target_kind": "node",
      "salience_score": 0.386,
      "dominant_channels": {"execution_load": 0.7, "reasoning_load": 0.158},
      "reasons": ["node execution_load is elevated"],
      "suggested_observation_mode": "watch"
    },
    {
      "target_id": "capability:orchestration",
      "target_kind": "capability",
      "salience_score": 0.36,
      "dominant_channels": {"execution_pressure": 0.8},
      "reasons": ["capability execution_pressure is elevated"],
      "suggested_observation_mode": "watch"
    }
  ],
  "source_field_tick_id": "tick_exec_attention"
}
```

`suggested_observation_mode` is an **attention hint only** — not action.

## Non-goals (explicit)

This PR does **not** implement:

- `SelfStateV1` (Layer 6)
- Proposal/action generation
- Policy gates or cortex-exec steering
- Mind service / LLM interpretation
- Bus-published events
- Mutation of `FieldStateV1`

## Tests run

```bash
PYTHONPATH=. pytest tests/test_attention_*.py -q
# 25 passed

PYTHONPATH=.:services/orion-field-digester pytest \
  tests/test_field_topology_reconciliation.py \
  tests/test_field_execution_perturbations.py \
  tests/test_field_state_schemas.py \
  tests/test_field_digestion_rules.py -q
# 13 passed

PYTHONPATH=. pytest \
  tests/test_execution_substrate_reducer.py \
  tests/test_execution_substrate_pipeline.py \
  tests/test_execution_projection_schemas.py -q
# 13 passed

PYTHONPATH=.:services/orion-hub pytest \
  services/orion-hub/tests/test_substrate_attention_debug_api.py -q
# 2 passed

PYTHONPATH=. python -m compileall orion/attention orion/schemas/field_attention_frame.py services/orion-attention-runtime -q
# ok
```

## Live / operator steps

1. Apply migration (done on dev if smoke run):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_attention_frame_v1.sql
```

2. Start runtime:

```bash
cd services/orion-attention-runtime
cp .env_example .env
docker compose up -d --build
```

3. Smoke:

```bash
./scripts/smoke_attention_frame_v1.sh
```

## Known gaps → Layer 6

- No `SelfStateV1` synthesis from attention + field
- Worker processes **latest** field tick only (intermediate ticks may skip attention frames if runtime lags)
- `high_salience` threshold in policy YAML reserved for future use
- No bus channel for attention frames (intentional v1)

## Files changed (attention slice)

- `orion/schemas/field_attention_frame.py`, `registry.py`
- `orion/attention/field_attention/*`
- `config/attention/field_attention_policy.v1.yaml`
- `services/orion-attention-runtime/*`
- `services/orion-sql-db/manual_migration_attention_frame_v1.sql`
- `services/orion-hub/scripts/substrate_attention_routes.py`
- `tests/test_attention_*.py`, `scripts/smoke_attention_frame_v1.sh`
