# PR Report: context-exec investigation_v2 readiness hardening

**Branch:** `fix/context-exec-v2-readiness-hardening`  
**Base:** `feature/investigation-v2-pr4`  
**Scope:** Expose bus consumer readiness on `/health`; preflight LLM gateway before v2 synthesis; preserve deterministic reports when synthesis is skipped.

## Problem

After investigation_v2 PR4, recall preflight avoided long RPC waits when RecallService had no bus subscriber, and `health_probe()` captured readiness metadata inside investigation artifacts. Two gaps remained:

- `GET /health` could still look healthy while Redis bus consumers for Recall and LLM gateway were dead.
- `investigation_v2` could still call LLM gateway RPC and wait for full synthesis timeout when the gateway bus consumer was absent.

## Solution

1. **`collect_bus_dependencies_health()`** in `bus_dependency_preflight.py` — shared helper returning aggregate `bus_consumer_ready` plus structured `dependencies.recall` / `dependencies.llm_gateway` (intake channel, subscriber count, redis ping, status).
2. **`GET /health`** — merges bus dependency block from live app bus connection (same NUMSUB checks as investigation preflight).
3. **LLM gateway synthesis preflight** — `_run_investigation_v2` checks gateway readiness before `run_agent_synthesis()`; dead consumer returns `synthesis_unavailable_for_llm_gateway_readiness()` without calling `llm_chat_route`.
4. **Deterministic degradation** — `apply_synthesis_to_report()` accepts optional failure message; limitation `"LLM synthesis unavailable: LLM gateway bus consumer not ready"` when preflight fails.

## Tests (A–E)

| Test | Coverage |
|---|---|
| **A** | `/health` exposes dependency readiness + aggregate when subscribers are 0 |
| **B** | Dead LLM gateway skips synthesis RPC; limitation + deterministic sections preserved |
| **C** | Ready gateway still calls synthesis path |
| **D** | Dead Recall consumer isolated; repo probe still runs → `partial_grounding` |
| **E** | Agent v2 read-broad/write-none permissions unchanged |

## Files changed

- `services/orion-context-exec/app/bus_dependency_preflight.py`
- `services/orion-context-exec/app/api.py`
- `services/orion-context-exec/app/agent_synthesis.py`
- `services/orion-context-exec/app/runner.py`
- `services/orion-context-exec/app/investigation_v2_reducers.py`
- `services/orion-context-exec/tests/test_context_exec_v2_readiness_hardening.py` (NEW)
- `services/orion-context-exec/tests/test_health.py`
- `services/orion-context-exec/README.md`

## Verification

```bash
cd .worktrees/fix-context-exec-v2-readiness-hardening
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_context_exec_v2_readiness_hardening.py \
  services/orion-context-exec/tests/test_health.py \
  services/orion-context-exec/tests/test_investigation_v2_readiness.py \
  services/orion-context-exec/tests/test_investigation_v2.py \
  services/orion-context-exec/tests/test_investigation_v2_reducers.py \
  -q --tb=short
# exit 0 — 25 passed

/mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m compileall \
  services/orion-context-exec/app/bus_dependency_preflight.py \
  services/orion-context-exec/app/api.py \
  services/orion-context-exec/app/agent_synthesis.py \
  services/orion-context-exec/app/runner.py \
  services/orion-context-exec/app/investigation_v2_reducers.py
# exit 0
```

Manual checks (stack required):

```bash
curl -s http://localhost:8096/health | jq '.bus_consumer_ready, .dependencies'
redis-cli PUBSUB NUMSUB orion:exec:request:RecallService
redis-cli PUBSUB NUMSUB orion:exec:request:LLMGatewayService
```

## Non-goals (unchanged)

- Semantic/LLM routing, new modes, shell/network/repo mutation
- Full RPC smoke probes on `/health`

## Remaining risks

- Preflight uses NUMSUB only (`check_heartbeat=False`); subscriber present but unhealthy may still fail on RPC.
- Live-stack `/health` verification not run in this session.
- Base branch is `feature/investigation-v2-pr4`; merge order should land PR4 before this hardening PR.
