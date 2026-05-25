# PR: Consolidation Frame v1 — Layer 11 Pattern Memory

**Branch:** `feat/consolidation-frame-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-consolidation-frame-v1`

## Summary

Implements **Layer 11** of the Orion cognition substrate: deterministic consolidation over Layers 5–10 history. Repeated substrate frames become inspectable motifs, expectations, sparse tensor slices, and schema candidates — without mutating policy, proposals, execution, or habits.

```text
substrate_self_state
+ substrate_attention_frames
+ substrate_proposal_frames
+ substrate_policy_decision_frames
+ substrate_execution_dispatch_frames
+ substrate_feedback_frames
  → orion-consolidation-runtime (port 8123)
  → ConsolidationFrameV1
  → substrate_consolidation_frames (+ expectations, tensor_slices, schema_candidates)
  → GET /api/substrate/consolidation/* (Hub, read-only)
```

**Consolidation is repeated consequence becoming structure.** No LLM, no bus publish, no automatic promotion.

## Phases delivered

| Phase | Scope |
|-------|--------|
| **11a** | `MotifObservationV1`, `ConsolidationFrameV1`, 6 motif rules, runtime, smoke |
| **11b** | `ExpectationV1`, motif→outcome mapping, `substrate_expectations` upsert |
| **11c** | `SparseTensorSliceV1`, three tensor kinds, coordinate cap |
| **11d** | `SchemaCandidateV1`, `promotion_status=candidate_only` only |
| **11e** | `repository.py`, five GET Hub routes |

## Motif rules (deterministic)

| Motif | Source signal |
|-------|----------------|
| `loaded_but_reliable` | Self-state: loaded + high execution / low reliability pressure |
| `attention_saturated_execution` | Attention salience ≥ 0.7 on athena/orchestration |
| `read_only_policy_loop` | Policy approved_read_only, execution disallowed |
| `dry_run_feedback_loop` | Feedback `outcome_status=dry_run_only` |
| `blocked_review_loop` | Policy review / dispatch blocked / feedback blocked |
| `stable_after_dry_run` | Dry-run feedback + unchanged self-state delta |

## Idempotency

- Window bounds align to `lookback_minutes` UTC buckets (hourly when lookback=60).
- `frame_id` = `consolidation.frame:{window_start}:{window_end}:{policy_id}` — stable within bucket.
- Runtime `ON CONFLICT (frame_id) DO NOTHING` on consolidation frames.

## Proof: no forbidden side effects

- `orion/substrate/consolidation.py` unchanged (graph consolidation is separate)
- `orion/bus/channels.yaml` unchanged
- Worker only reads Layers 5–10 tables; writes consolidation artifact tables only
- Hub routes are GET-only; tests assert no POST on router
- Schema candidates always `promotion_status=candidate_only`

## Files changed (40 files, ~3900 lines)

| Path | Role |
|------|------|
| `orion/schemas/consolidation_frame.py` | All Layer 11 v1 schemas |
| `orion/schemas/registry.py` | Registry entries |
| `config/consolidation/consolidation_policy.v1.yaml` | Policy + motif + tensor config |
| `orion/consolidation/*.py` | Policy, windows, motifs, expectations, tensorize, schema candidates, builder, repository |
| `services/orion-sql-db/manual_migration_consolidation_*.sql` | 4 DDL migrations |
| `services/orion-consolidation-runtime/` | Polling runtime (8123) |
| `services/orion-hub/scripts/substrate_consolidation_routes.py` | Debug API |
| `scripts/smoke_consolidation_v1.sh` | Live SQL smoke |
| `tests/test_consolidation_*.py` | Unit + store + worker tests |

## Migrations (apply in order)

1. `manual_migration_consolidation_v1.sql`
2. `manual_migration_consolidation_expectations_v1.sql`
3. `manual_migration_consolidation_tensor_slices_v1.sql`
4. `manual_migration_consolidation_schema_candidates_v1.sql`

## Test plan

```bash
cd .worktrees/feat-consolidation-frame-v1
PYTHONPATH=. pytest tests/test_consolidation_*.py \
  services/orion-hub/tests/test_substrate_consolidation_debug_api.py -q
# Expected: 67 passed

# After migrations + runtime:
./scripts/smoke_consolidation_v1.sh
```

## Review fixes (post code-review)

- **Window bucketing:** `compute_consolidation_window` aligns `window_end` to lookback-minute UTC buckets so repeated polls share one `frame_id` (was inserting every tick).

## Non-goals verified

- No policy/proposal/attention weight mutation
- No habit execution or RDF writes
- No cortex-exec steering
- No bus publish
- No LLM calls
