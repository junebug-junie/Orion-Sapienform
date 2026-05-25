# PR: Proposal Frame v1 — Self-State → Possible Actions, Not Automatic Actions

**Branch:** `feat/proposal-frame-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-proposal-frame-v1`

## Summary

Implements **Layer 7** of the Orion cognition substrate: deterministic conversion of `SelfStateV1` (+ optional attention/field context) into **inspectable possible actions** (`ProposalFrameV1`), without execution, policy approval, or bus publish.

```text
substrate_self_state (SelfStateV1)
+ substrate_attention_frames (FieldAttentionFrameV1)
+ substrate_field_state (FieldStateV1)
  → orion-proposal-runtime
  → ProposalFrameV1
  → substrate_proposal_frames
  → GET /api/substrate/proposals/latest (Hub debug)
```

**Core phrase:** *Proposal is action pressure made inspectable* — not approval, not execution, not autonomy.

### 11-layer roadmap placement

| Layer | Name | This PR |
|-------|------|---------|
| 1–6 | Organs → … → Self-state | Prerequisite (merged on `main`) |
| **7** | **Proposals** | **Implemented** |
| 8 | Policy | **Explicitly deferred** |
| 9–11 | Execution → Feedback → Consolidation | Deferred |

## Architecture

| Component | Role |
|-----------|------|
| `config/proposals/proposal_policy.v1.yaml` | Templates, thresholds, dimension weights |
| `orion/proposals/{policy,scoring,templates,builder}.py` | Pure deterministic proposal synthesis |
| `orion/schemas/proposal_frame.py` | `ProposalCandidateV1`, `ProposalFrameV1` |
| `services/orion-proposal-runtime` | Poll latest self-state → build → persist (port **8119**, no bus) |
| `substrate_proposal_frames` | Postgres persistence |
| `scripts/smoke_proposal_frame_v1.sh` | SQL inspection |

## Example `ProposalFrameV1` (live-shaped self-state)

Synthetic input matching operator live output (`execution_pressure=1.0`, `resource_pressure=1.0`, `overall_condition=loaded`):

```json
{
  "schema_version": "proposal.frame.v1",
  "frame_id": "proposal.frame:self.state:live:tick:frame:self_state_policy.v1:proposal_policy.v1",
  "overall_action_pressure": 0.93,
  "overall_risk": 0.45,
  "policy_required": true,
  "candidates": [
    {
      "proposal_kind": "inspect",
      "title": "Inspect orchestration execution pressure",
      "required_policy_gate": "read_only",
      "execution_intent": {
        "mode": "descriptive_only",
        "template": "inspect_execution_pressure",
        "policy_gate": "read_only"
      }
    },
    {
      "proposal_kind": "summarize",
      "title": "Summarize loaded operating condition",
      "required_policy_gate": "read_only",
      "execution_intent": { "mode": "descriptive_only" }
    },
    {
      "proposal_kind": "request_policy_review",
      "title": "Prepare policy review for possible action",
      "required_policy_gate": "operator_review",
      "execution_intent": {
        "mode": "descriptive_only",
        "note": "policy_review_not_execution"
      }
    }
  ]
}
```

Top candidates are **read-only inspect/summarize** under loaded execution pressure. `request_policy_review` is a *proposal to prepare policy evaluation later* — not approval and not cortex-exec.

## Proof: proposals only, not actions

| Check | Evidence |
|-------|----------|
| No cortex-exec / verb bus | No imports or calls; `orion/bus/channels.yaml` unchanged |
| No action execution | Worker only `INSERT` into `substrate_proposal_frames` |
| Descriptive `execution_intent` | Builder sets `mode: descriptive_only`; policy-review adds `policy_review_not_execution` |
| No approval API | Hub route is `GET /latest` only |
| Tests | `test_no_execution_in_candidates`, read-only gate assertions |
| Layer 8 deferred | No consent, blast-radius, or autonomy mutation |

## Tests run

```bash
PYTHONPATH=. pytest tests/test_proposal_frame_schemas.py \
  tests/test_proposal_policy_loader.py tests/test_proposal_scoring.py \
  tests/test_proposal_frame_builder.py tests/test_proposal_runtime_store.py \
  tests/test_proposal_runtime_worker.py -q
# 29 passed

PYTHONPATH=. pytest tests/test_self_state_*.py tests/test_attention_*.py -q
# 46 passed

PYTHONPATH=. pytest services/orion-hub/tests/test_substrate_proposal_debug_api.py -q
# 2 passed

PYTHONPATH=. python -m compileall orion/proposals orion/schemas/proposal_frame.py \
  services/orion-proposal-runtime -q
```

## Operator steps

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_proposal_frame_v1.sql

cd services/orion-proposal-runtime
cp -n .env_example .env
docker compose up -d --build

./scripts/smoke_proposal_frame_v1.sh
curl -s http://localhost:8080/api/substrate/proposals/latest | jq
curl -s http://localhost:8119/latest | jq
```

## Layer 8 policy (deferred)

This PR does **not** implement policy gates, consent checks, blast-radius enforcement, cortex-exec dispatch, or operator notifications. Downstream Layer 8 will evaluate `required_policy_gate` and `risk_score` on each candidate.

## Code review

Subagent review: **Ready** after fixes for dimension weight wiring, missing-field worker skip (aligned with self-state runtime), and worker idempotency tests.
