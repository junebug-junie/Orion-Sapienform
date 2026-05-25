# PR: Self-State Semantic Hardening v1

**Branch:** `feat/self-state-semantic-hardening-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-self-state-semantic-hardening-v1`

## Summary

Hardens **Layer 6** self-state semantics after live verification. `SelfStateV1` shape is unchanged; this PR fixes how stabilizers, pressure channels, and dimension evidence are classified and surfaced.

**Semantic hardening only** — no proposals, policy gates, cortex-exec steering, bus publish, LLM interpretation, or new organs. **Layer 7 remains deferred.**

```text
FieldStateV1 + FieldAttentionFrameV1
  → orion-self-state-runtime
  → SelfStateV1 (semantics hardened)
  → substrate_self_state
  → Hub GET /api/substrate/self-state/latest
```

## Before / after (live-shaped fixture)

**Before (bug):** stabilizing channels appeared as unresolved pressures:

```text
unresolved_pressures:
  availability→coherence
  confidence→coherence
  available_capacity→coherence
  execution_load→execution_pressure
  ...
```

**After (this PR):**

```text
stabilizing_factors:
  availability=1.00
  available_capacity=1.00
  confidence=1.00

unresolved_pressures:
  cpu_pressure→resource_pressure
  execution_load→execution_pressure
  execution_pressure→execution_pressure
  pressure→resource_pressure

dimensions.execution_pressure.dominant_evidence:
  execution_load=1.00, execution_pressure=1.00

dimensions.coherence.dominant_evidence:
  availability=1.00, confidence=1.00, available_capacity=1.00

summary_labels includes: stabilized_but_loaded, execution_loaded, attention_saturated
```

## Changes

| Area | Change |
|------|--------|
| `config/self_state/self_state_policy.v1.yaml` | `pressure_channels`, `context_channels` lists |
| `orion/self_state/policy.py` | Load channel role lists |
| `orion/self_state/builder.py` | Gate `unresolved_pressures` to pressure channels; `evidence_for_dimension()`; `stabilized_but_loaded` label; stabilizer overlap guard |
| Tests | `test_policy_channel_role_lists`, `test_self_state_builder_hardening.py` (6 tests) |

## Non-goals (confirmed)

- No `ProposalFrameV1`, action candidates, policy gates, cortex-exec steering
- No bus publish, mind service, operator notifications
- No `orion/schemas/registry.py`, `orion/bus/channels`, SQL migrations
- No service env/docker/requirements changes (runtime picks up shared `orion/self_state` on redeploy)

## Test plan

- [x] `PYTHONPATH=. pytest tests/test_self_state_*.py -q` → **28 passed**
- [x] Attention regression: `test_attention_frame_*.py`, `test_attention_field_scoring.py`, `test_attention_policy_loader.py` → **20 passed**
- [x] `python -m compileall orion/self_state orion/schemas/self_state.py services/orion-self-state-runtime -q`
- [ ] Optional live: restart `orion-self-state-runtime`, then `curl -s http://localhost:8080/api/substrate/self-state/latest | jq '.unresolved_pressures, .stabilizing_factors'`

## Code review

Subagent review: **Approved**. Post-review: stabilizer overlap guard + `cpu_pressure→resource_pressure` assertion.

## Operator notes

Redeploy or restart `orion-self-state-runtime` so the container loads updated `orion/self_state` and `config/self_state/`. Hub debug route unchanged (reads persisted JSON).
