# PR Report: investigation_v2 plumbing (PR1)

**Branch:** `feature/investigation-v2-pr1`  
**Scope:** Hub/context-exec request plumbing only — no evidence sweep, no removal of legacy modes.

## Problem

Hub inferred context-exec mode (and repo access) from prompt keywords (`repo`, `impact`, `what breaks`). Repo read permissions were unlocked by phrasing, not by the selected Agent compute profile. A duplicate keyword router also existed in context-exec `agent_compat.py`.

## Solution

1. **Schema:** Added `investigation_v2` to `ContextExecMode`; shared helper `context_exec_permissions_for_llm_profile()`.
2. **Hub:** `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED` (default `false`). When on, Agent lane sends `mode=investigation_v2` + profile-derived permissions; user text and correlation fields unchanged.
3. **Agent compat:** When flag on, skips keyword inference; forwards neutral `investigation_v2` + profile permissions.
4. **Context-exec:** Early skeleton handler returns `InvestigationV2SkeletonV1` without organ/RLM sweeps.

## Agent profile permissions (read-broad / write-none)

| Permission | Agent profile |
|---|---|
| read_memory, read_graph, read_recall, read_repo, read_runtime_logs, read_redis_traces | `true` |
| write_memory, write_graph, write_repo, mutate_runtime, network_enabled, shell_enabled | `false` |

Quick/other profiles keep narrow defaults (`read_repo=false`, all writes false).

## Feature flag behavior

| Flag | Agent lane + agent compute | Prompt without repo keywords |
|---|---|---|
| `false` | Legacy keyword inference | `general_investigation`, `read_repo=false` |
| `true` | `investigation_v2` | `read_repo=true` (agent profile) |

## Files changed

- `orion/schemas/context_exec.py`
- `orion/schemas/tests/test_context_exec_investigation_v2.py`
- `services/orion-hub/app/settings.py`
- `services/orion-hub/scripts/context_exec_agent_bridge.py`
- `services/orion-hub/tests/test_investigation_v2_request.py`
- `services/orion-hub/tests/conftest.py`
- `services/orion-hub/.env_example`, `docker-compose.yml`, `README.md`
- `services/orion-context-exec/app/investigation_v2.py`
- `services/orion-context-exec/app/runner.py`
- `services/orion-context-exec/app/agent_compat.py`
- `services/orion-context-exec/app/artifact_builder.py`
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/tests/test_investigation_v2.py`
- `services/orion-context-exec/.env_example`, `docker-compose.yml`, `README.md`

## Verification

```bash
cd .worktrees/feature/investigation-v2-pr1
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-hub/tests/test_investigation_v2_request.py \
  services/orion-context-exec/tests/test_investigation_v2.py \
  orion/schemas/tests/test_context_exec_investigation_v2.py -q
# exit 0 — 10 passed
```

Local `.env` synced (gitignored): `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=false` added to `services/orion-hub/.env` and `services/orion-context-exec/.env`.

## Non-goals (deferred)

- Full evidence sweep for `investigation_v2`
- Removing `_infer_context_exec_mode` or legacy modes
- Semantic/LLM/keyword routers

## Remaining risks

- Flag must be set consistently on Hub **and** context-exec for agent-compat bus/API paths.
- Skeleton response is not a grounded investigation — operators should treat as plumbing proof only until PR2+.
