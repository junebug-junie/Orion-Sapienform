# PR: Self-State v1 — Attention + Field → Orion Operating Condition

**Branch:** `feat/self-state-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-self-state-v1`

## Summary

Implements **Layer 6** of the Orion cognition substrate: deterministic, read-only synthesis of Orion’s **operating condition** from digested field state and the attention frame.

```text
substrate_field_state (FieldStateV1)
+ substrate_attention_frames (FieldAttentionFrameV1)
  → orion-self-state-runtime
  → SelfStateV1
  → substrate_self_state
  → GET /api/substrate/self-state/latest (Hub debug)
```

**Core phrase:** *Self-state is the substrate’s estimate of operating condition* — not agency, not action, not personality prose.

### 11-layer roadmap placement

| Layer | Name | This PR |
|-------|------|---------|
| 1–5 | Organs → … → Attention | Prerequisite (field-digester + attention-runtime) |
| **6** | **Self-state** | **Implemented** |
| 7 | Proposals | Deferred |
| 8–11 | Policy → Execution → Feedback → Consolidation | Deferred |

## Architecture

| Component | Role |
|-----------|------|
| `config/self_state/self_state_policy.v1.yaml` | Dimension weights, channel→dimension map, condition thresholds |
| `orion/self_state/{policy,scoring,builder}.py` | Pure deterministic self-state synthesis |
| `services/orion-self-state-runtime` | Poll latest attention → load field → build → persist (no bus) |
| `substrate_self_state` | Postgres persistence |
| `scripts/smoke_self_state_v1.sh` | SQL inspection |

## Example synthetic `SelfStateV1`

From live builder run (`node:athena.execution_load=1.0`, `capability:orchestration.execution_pressure=1.0`):

```json
{
  "overall_condition": "steady",
  "overall_intensity": 0.3194,
  "execution_pressure": 1.0,
  "agency_readiness": 0.205,
  "summary_labels": [
    "execution_loaded",
    "reliability_clear"
  ],
  "dominant_attention_targets": [
    "capability:orchestration",
    "node:athena",
    "field:recent_perturbations"
  ]
}
```

`agency_readiness` is computed but **does not authorize action** — Layer 7+ owns proposals and policy gates.

## Non-goals (explicit)

This PR does **not** implement:

- proposal/action generation (`ProposalFrameV1`)
- policy gates or cortex-exec steering
- mind service / LLM interpretation
- bus-published events (`orion/bus/channels.yaml` unchanged)
- mutation of `FieldStateV1` or `FieldAttentionFrameV1`
- operator notifications

## Tests run

```bash
PYTHONPATH=. pytest tests/test_self_state_*.py -q
# 21 passed

PYTHONPATH=. pytest tests/test_attention_frame_schemas.py tests/test_attention_frame_builder.py \
  tests/test_attention_field_scoring.py tests/test_attention_policy_loader.py -q
# 20 passed

PYTHONPATH=. pytest services/orion-hub/tests/test_substrate_self_state_debug_api.py -q
# 2 passed

PYTHONPATH=. pytest tests/test_field_state_schemas.py -q
# 2 passed

PYTHONPATH=. python -m compileall orion/self_state orion/schemas/self_state.py services/orion-self-state-runtime -q
# clean
```

## Operator checklist

1. Apply migration: `services/orion-sql-db/manual_migration_self_state_v1.sql`
2. Ensure `orion-field-digester` and `orion-attention-runtime` are running
3. Start runtime: `services/orion-self-state-runtime` (port **8118**)
4. Smoke: `./scripts/smoke_self_state_v1.sh`

## Known gaps (Layer 7+)

- `previous_self_state` parameter reserved; continuity deltas not yet applied
- Worker processes **latest** attention frame only (same idempotency pattern as attention-runtime)
- No `ProposalFrameV1` or action pressure export

## Files changed (high level)

- `orion/schemas/self_state.py`, `orion/schemas/registry.py`
- `orion/self_state/*`, `config/self_state/self_state_policy.v1.yaml`
- `services/orion-self-state-runtime/*`
- `services/orion-sql-db/manual_migration_self_state_v1.sql`
- `services/orion-hub/scripts/substrate_self_state_routes.py`
- `tests/test_self_state_*.py`, `scripts/smoke_self_state_v1.sh`
