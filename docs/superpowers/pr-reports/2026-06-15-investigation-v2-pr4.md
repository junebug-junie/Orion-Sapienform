# PR Report: investigation_v2 bus readiness + router burn-down (PR4)

**Branch:** `feature/investigation-v2-pr4`  
**Base:** `feature/investigation-v2-pr3`  
**Scope:** Real bus-consumer readiness, investigation_v2 dependency preflight, prompt-router neutering for v2, synthesis degradation language.

## Problem

After PRs 1–3, `investigation_v2` had capability-profile permissions, evidence sweep, reducers, and composite reporting — but dead Redis bus consumers could still waste full RPC timeouts and collapse into `no_reliable_evidence`. Hub/agent paths could still infer repo access from prompt keywords when v2 was off, and HTTP `/health` alone could say OK while no bus subscriber was listening.

## Solution

1. **Bus consumer readiness:** `orion/bus/consumer_readiness.py` implements Redis `PUBSUB NUMSUB`, optional heartbeat listen, and structured `BusConsumerReadinessV1` results. Recall and LLM gateway `/ready` endpoints expose `bus_consumer_ready`, `subscriber_count`, `heartbeat_fresh`, and `dependency_status`.
2. **Dependency preflight:** `bus_dependency_preflight.py` runs fast NUMSUB checks before recall RPC in `run_investigation_v2`. Dead RecallService marks recall `unavailable` immediately with `subscriber_count=0` metadata — other probes continue.
3. **Router burn-down:** Hub `build_context_exec_request` and `agent_compat` already send neutral `mode=investigation_v2` with `context_exec_permissions_for_llm_profile` when v2 flag is on; tests assert `_infer_context_exec_mode` is not called and no magic phrases are required.
4. **Failure language:** Dead dependencies → `dependency_unavailable`; mixed hits → `partial_grounding`; synthesis failure → `synthesis_unavailable` limitation + `answer_evaluation.synthesis_status`, not total investigation failure.
5. **Health probe enrichment:** investigation_v2 `health` source metadata includes recall/LLM gateway readiness snapshots when bus is connected.

## New env keys (context-exec)

| Key | Default | Purpose |
|---|---|---|
| `CONTEXT_EXEC_BUS_READINESS_HEARTBEAT_TTL_SEC` | 30 | Heartbeat freshness TTL for service `/ready` checks |
| `CONTEXT_EXEC_BUS_READINESS_TIMEOUT_SEC` | 2 | Fast preflight timeout for investigation_v2 |

## Tests added/updated (A–F)

| Test | Coverage |
|---|---|
| **A** | Recall bus consumer dead → fast unavailable, no `recall_query`, repo still runs, `dependency_unavailable` or `partial_grounding` |
| **B** | LLM synthesis failure → deterministic report + `synthesis_unavailable` limitation preserved |
| **C** | Hub v2 bypasses `_infer_context_exec_mode` for cortex-change prompt |
| **D** | No magic phrase (`repo`/`impact`/`what breaks`) required for `read_repo=True` |
| **E** | Legacy keyword routing when v2 flag disabled (Hub + agent_compat) |
| **F** | Agent v2 `write_repo=False`, `mutate_runtime=False`, no mutation regression |

## Files changed

- `orion/bus/__init__.py`
- `orion/bus/consumer_readiness.py` (NEW)
- `orion/bus/tests/test_consumer_readiness.py` (NEW)
- `orion/schemas/telemetry/system_health.py`
- `orion/schemas/telemetry/__init__.py`
- `orion/schemas/registry.py`
- `services/orion-context-exec/.env_example`
- `services/orion-context-exec/app/bus_dependency_preflight.py` (NEW)
- `services/orion-context-exec/app/investigation_v2.py`
- `services/orion-context-exec/app/investigation_v2_reducers.py`
- `services/orion-context-exec/app/runner.py`
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/tests/test_investigation_v2_readiness.py` (NEW)
- `services/orion-context-exec/README.md`
- `services/orion-hub/tests/test_investigation_v2_request.py`
- `services/orion-hub/README.md`
- `services/orion-recall/app/main.py`
- `services/orion-llm-gateway/app/main.py`

## Verification

```bash
cd .worktrees/feature/investigation-v2-pr4
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-context-exec/tests/test_investigation_v2_readiness.py \
  services/orion-context-exec/tests/test_investigation_v2.py \
  services/orion-context-exec/tests/test_investigation_v2_reducers.py \
  orion/schemas/tests/test_context_exec_investigation_v2.py \
  services/orion-hub/tests/test_investigation_v2_request.py \
  orion/bus/tests/test_consumer_readiness.py -q --tb=short
# exit 0 — 31 passed

/mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m compileall \
  orion/bus/consumer_readiness.py \
  services/orion-context-exec/app/bus_dependency_preflight.py \
  services/orion-recall/app/main.py \
  services/orion-llm-gateway/app/main.py
# exit 0
```

Local `.env` sync: added `CONTEXT_EXEC_BUS_READINESS_*` keys to `services/orion-context-exec/.env` (gitignored).

## Non-goals (unchanged)

- Semantic/LLM routing, spaCy/NLP, shell execution, repo mutation
- Parallel probe fanout (PR2 debt)
- Full runtime log probe wiring

## Remaining risks

- Preflight uses NUMSUB only (`check_heartbeat=False`) for speed; subscriber present but unhealthy may still fail on RPC.
- Live-stack verification of `/ready` against production Redis not run in this session.
- v2 flag must stay consistent on Hub **and** context-exec.
