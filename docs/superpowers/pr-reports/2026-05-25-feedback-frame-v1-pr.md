# PR: Feedback Frame v1 — Layer 10 Consequence Capture

**Branch:** `feat/feedback-frame-v1`  
**Base:** `feat/execution-dispatch-v1`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-feedback-frame-v1`  
**Head:** `bac714f9` (6 commits)

## Summary

Implements **Layer 10** of the Orion cognition substrate: deterministic feedback over Layer 9 execution dispatch. Converts `ExecutionDispatchFrameV1` + related policy/proposal/self-state (+ optional cortex result evidence) into `FeedbackFrameV1` snapshots persisted for inspection.

```text
substrate_execution_dispatch_frames
+ substrate_policy_decision_frames
+ substrate_proposal_frames
+ substrate_self_state
  → orion-feedback-runtime (port 8122)
  → FeedbackFrameV1
  → substrate_feedback_frames
  → GET /api/substrate/feedback/latest (Hub, read-only)
```

**Feedback is consequence capture, not learning.** No consolidation, policy mutation, retries, bus publish, or LLM interpretation.

**Layer 11 consolidation is explicitly deferred.**

## Roadmap position (11-layer substrate)

| Layer | This PR |
|-------|---------|
| 9 Execution | Consumes `ExecutionDispatchFrameV1` |
| **10 Feedback** | **Implemented** |
| 11 Consolidation | Deferred |

## Example `FeedbackFrameV1` (dry-run dispatch)

Synthetic loaded self-state + inspect proposal → dry-run dispatch → feedback:

| Field | Value |
|-------|-------|
| `frame_id` | `feedback.frame:execution.dispatch.frame:policy.frame:proposal.frame:demo:substrate_policy.v1:execution_dispatch_policy.v1:feedback_policy.v1` |
| `outcome_status` | `dry_run_only` |
| `outcome_score` | `0.50` |
| `dispatch_attempted` (source) | `false` |
| Observations | dispatch candidates as `dry_run`; policy decision `not_attempted` |

Core phrase: **Feedback is consequence made observable.**

## Proof: absence is represented

When `dispatch_read_only` is attempted without matching cortex results:

- `absence_evidence` includes `missing_cortex_result:{dispatch_id}`
- Observations include `outcome_kind=absent`
- If some results succeed and others are missing, `outcome_status=mixed` (not bare `completed`)

## Proof: no forbidden side effects

- `orion-feedback-runtime` worker only calls `build_feedback_frame` + SQL upsert
- No `bus.publish`, no policy/proposal/dispatch mutation, no retry loops
- Builder tests assert input frames unchanged after `build_feedback_frame`
- `load_cortex_result_evidence` returns `[]` in v1 (no substrate cortex-result table yet)

## Files changed

| Path | Role |
|------|------|
| `orion/schemas/feedback_frame.py` | `OutcomeObservationV1`, `FeedbackFrameV1` |
| `orion/schemas/registry.py` | Schema registry |
| `config/feedback/feedback_policy.v1.yaml` | Windows, scoring, pressure channels |
| `orion/feedback/{policy,extractors,scoring,builder}.py` | Pure feedback logic |
| `services/orion-sql-db/manual_migration_feedback_frame_v1.sql` | DDL |
| `services/orion-feedback-runtime/` | Polling runtime (8122) |
| `services/orion-hub/scripts/substrate_feedback_routes.py` | Debug API |
| `scripts/smoke_feedback_frame_v1.sh` | Live SQL smoke |
| `tests/test_feedback_*.py` | Unit + store tests |

## Review fixes (post code-review)

- `_aggregate_outcome_status`: evaluate `absent` before bare `completed`/`failed` → correct `mixed` on partial cortex results
- `load_latest_self_state_after`: bound by `field_after_window_sec` from feedback policy

## Tests run

```bash
cd .worktrees/feat-feedback-frame-v1
PYTHONPATH=. pytest \
  tests/test_feedback_frame_schemas.py \
  tests/test_feedback_policy_loader.py \
  tests/test_feedback_builder.py \
  tests/test_feedback_scoring.py \
  tests/test_feedback_extractors.py \
  tests/test_feedback_runtime_store.py \
  tests/test_execution_dispatch_*.py \
  tests/test_policy_decision_frame_schemas.py \
  tests/test_policy_decision_builder.py \
  tests/test_policy_evaluator.py \
  tests/test_policy_runtime_store.py \
  tests/test_policy_loader.py \
  services/orion-hub/tests/test_substrate_feedback_debug_api.py \
  -q
# 91 passed

python3 -m compileall orion/feedback orion/schemas/feedback_frame.py services/orion-feedback-runtime -q
# OK
```

## Live operator steps

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_feedback_frame_v1.sql

cd services/orion-feedback-runtime
cp -n .env_example .env
docker compose up -d --build

./scripts/smoke_feedback_frame_v1.sh
curl -s http://localhost:8080/api/substrate/feedback/latest | jq
```
