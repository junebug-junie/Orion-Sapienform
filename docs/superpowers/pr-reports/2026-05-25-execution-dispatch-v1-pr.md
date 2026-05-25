# PR: Execution Dispatch v1 — PolicyDecision → Cortex-Exec Request Envelope

**Branch:** `feat/execution-dispatch-v1`  
**Base:** `feat/policy-gate-v1` (Layer 8)  
**Port:** `8121` (`orion-execution-dispatch-runtime`)

---

## Layer 9 in the 11-layer substrate roadmap

| Layer | Role | This PR |
|-------|------|---------|
| 1–7 | Organs → proposals | Upstream (unchanged) |
| 8 | Policy gate | **Dependency** — `PolicyDecisionFrameV1` |
| **9** | **Execution dispatch** | **Implemented** — dry-run envelopes only |
| 10 | Feedback | **Deferred** |
| 11 | Consolidation | **Deferred** |

**Mental model:** Execution dispatch is approved action made explicit before it becomes effect. In v1: mostly dry-run envelopes, not autonomous mutation.

---

## Summary

- `PolicyDecisionFrameV1` + `ProposalFrameV1` + `SelfStateV1` → `ExecutionDispatchFrameV1`
- `orion-execution-dispatch-runtime` polls oldest policy frame without dispatch, persists to `substrate_execution_dispatch_frames`
- Default `EXECUTION_DISPATCH_MODE=dry_run` — no bus publish, no cortex-exec calls
- Hub: `GET /api/substrate/execution-dispatch/latest` (read-only)

---

## Example `ExecutionDispatchFrameV1` (unit fixture)

```json
{
  "schema_version": "execution.dispatch.frame.v1",
  "frame_id": "execution.dispatch.frame:policy.frame:proposal.frame:test:substrate_policy.v1:execution_dispatch_policy.v1",
  "dispatch_mode": "dry_run",
  "dispatch_attempted": false,
  "dispatch_count": 0,
  "candidates": [
    {
      "dispatch_status": "dry_run",
      "dispatch_kind": "inspect",
      "cortex_verb": "substrate.inspect",
      "request_envelope": {
        "verb": "substrate.inspect",
        "dry_run": true,
        "constraints": { "read_only": true, "no_file_writes": true }
      }
    }
  ],
  "blocked_candidates": [
    { "source_proposal_id": "proposal:review:state", "dispatch_status": "blocked" },
    { "source_proposal_id": "proposal:blocked:state", "dispatch_status": "blocked" }
  ]
}
```

---

## Proof: default dry_run, no cortex-exec

| Check | Result |
|-------|--------|
| Policy YAML `default_dispatch_mode` | `dry_run` |
| `allow_dispatch_read_only` | `false` |
| `allow_mutating_dispatch` | `false` |
| Builder default | `dispatch_attempted=false`, `dispatch_count=0` |
| Worker | No `bus.publish`, no HTTP clients — grep clean |
| `router.py` | **Not implemented** (v1 envelopes only) |
| `orion/bus/channels.yaml` | **No changes** (registry only) |

```bash
rg -n "bus\.publish|redis\.publish|httpx\.|requests\.|aiohttp" \
  services/orion-execution-dispatch-runtime orion/execution_dispatch
# (no matches)
```

---

## Tests run

```bash
cd .worktrees/feat-execution-dispatch-v1
PYTHONPATH=. pytest tests/test_execution_dispatch_*.py \
  services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py -q
# 30 passed

PYTHONPATH=. pytest tests/test_policy_*.py -q
# 32 passed

PYTHONPATH=. pytest tests/test_proposal_frame_*.py tests/test_proposal_runtime_store.py \
  tests/test_proposal_policy_loader.py tests/test_proposal_scoring.py -q
# 27 passed

PYTHONPATH=. pytest tests/test_self_state_*.py -q
# 28 passed

PYTHONPATH=. python -m compileall orion/execution_dispatch \
  orion/schemas/execution_dispatch_frame.py services/orion-execution-dispatch-runtime -q
# OK
```

---

## Operator steps

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql

cd services/orion-execution-dispatch-runtime
cp -n .env_example .env
docker compose up -d --build

./scripts/smoke_execution_dispatch_v1.sh
curl -s http://localhost:8080/api/substrate/execution-dispatch/latest | jq
curl -s http://localhost:8121/latest | jq
```

Requires `orion-policy-runtime` (8120) producing policy frames.

---

## Files touched (high level)

| Area | Paths |
|------|-------|
| Schemas | `orion/schemas/execution_dispatch_frame.py`, `registry.py` |
| Logic | `orion/execution_dispatch/{policy,envelopes,builder}.py` |
| Config | `config/execution_dispatch/execution_dispatch_policy.v1.yaml` |
| Runtime | `services/orion-execution-dispatch-runtime/` |
| DB | `manual_migration_execution_dispatch_frame_v1.sql` |
| Hub | `substrate_execution_dispatch_routes.py` |
| Tests | `tests/test_execution_dispatch_*.py` |
| Smoke | `scripts/smoke_execution_dispatch_v1.sh` |

---

## Layer 10 explicitly deferred

No feedback scoring, outcome learning, consolidation, or habitual actions in this PR.

---

## Commits

1. `feat(execution-dispatch): add ExecutionDispatchFrameV1 schemas and registry`
2. `feat(execution-dispatch): add policy loader, envelopes, and frame builder`
3. `feat(execution-dispatch): add runtime service, migration, hub route, and smoke`
4. `docs(execution-dispatch): add Layer 9 implementation plan`
5. `fix(execution-dispatch): address review — settings validation, worker tests, unique index`
