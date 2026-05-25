# PR: Policy Gate v1 — ProposalFrame → Governed Decisions

**Branch:** `feat/policy-gate-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-policy-gate-v1`  
**Head:** `b0a97538` (4 commits)

## Summary

Implements **Layer 8** of the Orion cognition substrate: deterministic policy gating over Layer 7 proposals. Converts `ProposalFrameV1` + `SelfStateV1` into `PolicyDecisionFrameV1` snapshots persisted for inspection. **Policy is not execution** — no cortex-exec, bus publish, operator notifications, or settings mutation.

```text
substrate_proposal_frames + substrate_self_state
  → orion-policy-runtime (port 8120)
  → PolicyDecisionFrameV1
  → substrate_policy_decision_frames
  → GET /api/substrate/policy/latest (Hub, read-only)
```

**Layer 9 execution is explicitly deferred.**

## Roadmap position (11-layer substrate)

| Layer | This PR |
|-------|---------|
| 7 Proposal | Consumes `ProposalFrameV1` |
| **8 Policy** | **Implemented** |
| 9 Execution | Deferred |

## Example `PolicyDecisionFrameV1` (synthetic loaded state)

Built from demo proposal with inspect, summarize, request_policy_review, prepare_action candidates:

| Field | Value |
|-------|-------|
| `frame_id` | `policy.frame:proposal.frame:demo:proposal_policy.v1:substrate_policy.v1` |
| `operator_review_required` | `true` |
| `execution_allowed` | `false` |
| `approved_decisions` | 2 (inspect, summarize → `approved_read_only`) |
| `review_required_decisions` | 2 (request_policy_review, prepare_action) |

Per-candidate decisions:

```text
proposal:inspect   → approved_read_only
proposal:summarize → approved_read_only
proposal:review    → requires_operator_review
proposal:prepare   → requires_operator_review
```

## Proof: no execution performed

- `orion-policy-runtime` worker only calls `build_policy_decision_frame` + SQL upsert.
- Grep over `services/orion-policy-runtime` and `orion/policy` finds no `cortex_exec`, `bus.publish`, or runtime side effects (only README/docs and hard-block token `cortex_exec_direct_call`).
- `execution_constraints` on decisions are tagged `layer: 9_deferred` for future execution runtime.
- `allow_execution_without_operator: false` in `config/policy/substrate_policy.v1.yaml` with tests asserting no `approved_for_execution`.

## Files changed

| Path | Role |
|------|------|
| `orion/schemas/policy_decision_frame.py` | `PolicyDecisionV1`, `PolicyDecisionFrameV1` |
| `orion/schemas/registry.py` | Schema registry |
| `config/policy/substrate_policy.v1.yaml` | Substrate policy v1 |
| `orion/policy/{policy,rules,evaluator,builder}.py` | Pure policy logic |
| `services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql` | DDL |
| `services/orion-policy-runtime/` | Polling runtime (8120) |
| `services/orion-hub/scripts/substrate_policy_routes.py` | Debug API |
| `scripts/smoke_policy_decision_frame_v1.sh` | Live SQL smoke |
| `tests/test_policy_*.py` | Unit + worker tests |

## Review fixes (post code-review)

- Enforce `autonomy_policy` / `execution_policy` gates → `requires_operator_review` (no kind-default bypass).
- Worker backfills **oldest** proposal without a policy frame (not only latest).
- Nested policy YAML models use `extra="forbid"`.

## Tests run

```bash
cd .worktrees/feat-policy-gate-v1
PYTHONPATH=. pytest tests/test_policy_*.py services/orion-hub/tests/test_substrate_policy_debug_api.py -q
# 34 passed

PYTHONPATH=. pytest tests/test_proposal_*.py tests/test_self_state_*.py -q
# 57 passed

PYTHONPATH=. python -m compileall orion/policy orion/schemas/policy_decision_frame.py services/orion-policy-runtime -q
```

## Live operator steps

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql

cd services/orion-policy-runtime
cp -n .env_example .env
docker compose up -d --build

./scripts/smoke_policy_decision_frame_v1.sh
curl -s http://localhost:8120/latest | jq
curl -s http://localhost:8080/api/substrate/policy/latest | jq
```

## Test plan

- [ ] Apply migration on target DB
- [ ] Start `orion-policy-runtime` with proposal + self-state data present
- [ ] Confirm idempotent noop on same proposal frame
- [ ] Smoke script shows latest policy frame with partitioned decisions
- [ ] Hub `/api/substrate/policy/latest` returns frame JSON
- [ ] Confirm `execution_allowed` is false for live frames
