## Summary

- Deleted `orion-planner-react` and `orion-agent-chain` services (~10.7k lines) and removed mesh/bus/signals wiring for both organs.
- Agent/council depth routing now uses a single `ContextExecService` plan step in cortex-orch; cortex-exec supervisor delegates via `_context_exec_escalation()`.
- Operational bound-capability verbs execute in-process via new `bound_capability_exec.py` (nested `orion:cortex:request` RPC); capability bridge + actions skill registry moved into cortex-exec.
- Ported `autonomy.goal.execute.v1` into cortex-exec (`autonomy_goal_execute.py`) so promoted goal execution and `autonomy.goal.planned` bus publish survive planner-react removal.
- Updated agent-trace normalizer, live-proof scripts, orch/exec tests, and README for the context-exec + bound-capability path.

## Outcome moved

- Dead planner-react / agent-chain containers and bus channels are gone; agent depth work has one delegate organ (`ContextExecService`).
- Bound operational verbs (mesh/storage/PR/housekeep) no longer require a separate agent-chain hop.
- Autonomy goal execute path remains callable when `AUTONOMY_GOAL_EXECUTION_ENABLED=true` (flag still defaults off).

## Current architecture

Before: Hub → gateway → orch → cortex-exec → planner-react → agent-chain → nested cortex skills.

After: Hub → gateway → orch → cortex-exec → context-exec (agent/council depth) **or** in-process bound-capability nested cortex RPC (operational verbs).

## Architecture touched

- **Deleted:** `services/orion-planner-react/`, `services/orion-agent-chain/`
- **Contracts:** `orion/bus/channels.yaml`, `orion/signals/registry.py`, `orion/normalizers/agent_trace.py`
- **Routing:** `services/orion-cortex-orch/app/orchestrator.py`, `services/orion-cortex-exec/app/supervisor.py`
- **New seams:** `bound_capability_exec.py`, `capability_bridge.py`, `actions_skill_registry.py`, `autonomy_goal_execute.py`
- **Hub:** removed `AGENT_CHAIN_*` env/compose surfaces
- **Scripts:** `locate-bound-capability-live-path.sh`, `verify-bound-capability-live.sh`, `run_answer_depth_proof_suite.py`, `run_answer_depth_live_proof.py`

## Files changed

- `services/orion-cortex-exec/app/supervisor.py`: removed planner/agent-chain escalation; context-exec + bound-capability + autonomy goal dispatch
- `services/orion-cortex-exec/app/bound_capability_exec.py`: operational verb execution via nested cortex RPC
- `services/orion-cortex-exec/app/autonomy_goal_execute.py`: ported autonomy goal execute + planned event publish
- `services/orion-cortex-orch/app/orchestrator.py`: `build_agent_plan()` → single `context_exec` step
- `orion/bus/channels.yaml`: removed planner/agent-chain channels; repointed autonomy goal planned producer to cortex-exec
- Tests/scripts/docs updated across cortex-exec, cortex-orch, and root `tests/`

## Schema / bus / API changes

- **Removed channels:** planner-react intake/result, agent-chain intake/result/capability reply patterns
- **Removed signal organs:** `planner`, `agent_chain`
- **Behavior changed:** `AgentChainService` / `PlannerReactService` verb plan steps fail closed with `removed_service:*`
- **Producer changed:** `orion:autonomy:goal:planned` → `orion-cortex-exec` (now has real publisher in `autonomy_goal_execute.py`)
- **Compatibility:** context-exec HTTP `/agent/chain/run` compat shim intentionally retained

## Env/config changes

- **Removed keys:** `CHANNEL_PLANNER_INTAKE`, `CHANNEL_AGENT_CHAIN_INTAKE`, hub `AGENT_CHAIN_*`
- **Added keys:** none required (uses existing `CONTEXT_EXEC_*` + `ORION_BUS_URL`)
- `.env_example` updated: hub, cortex-exec
- local `.env` synced: UNVERIFIED in agent session (operator should run `python scripts/sync_local_env_from_example.py`)

## Tests run

```text
# cortex-exec + contracts (25 passed)
PYTHONPATH=.:services/orion-cortex-exec pytest -q \
  services/orion-cortex-exec/tests/test_autonomy_goal_execute.py \
  services/orion-cortex-exec/tests/test_autonomy_goal_execution_mode.py \
  services/orion-cortex-exec/tests/test_operational_semantic_harness.py \
  services/orion-cortex-exec/tests/test_bound_capability_full_path.py \
  services/orion-cortex-exec/tests/test_context_exec_depth2_routing.py \
  tests/test_autonomy_goals_bus_catalog.py \
  tests/test_exec_result_channel_catalog_specificity.py \
  tests/test_agent_no_recall_live_proof.py

# cortex-orch (28 passed)
PYTHONPATH=.:services/orion-cortex-orch pytest -q \
  services/orion-cortex-orch/tests/test_auto_router.py \
  services/orion-cortex-orch/tests/test_verb_runtime_rpc.py \
  services/orion-cortex-orch/tests/test_agent_trace_bound_failure_summary.py
```

## Evals run

```text
Not run — no periodic eval harness for this cross-service removal. Gate tests above cover routing contracts.
```

## Docker/build/smoke checks

```text
Not run in agent environment. Compose config should be validated after merge:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml config
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml config
```

## Review findings fixed

- Finding: `autonomy.goal.execute.v1` deleted with planner but supervisor still dispatched it
  - Fix: ported `autonomy_goal_execute.py` + `_execute_autonomy_goal_action()` in supervisor
  - Evidence: `test_autonomy_goal_execute.py`, updated `test_autonomy_goal_execution_mode.py`

- Finding: `autonomy.goal.planned` catalog producer lied (no publisher)
  - Fix: `_publish_goal_planned_supervisor_event()` in cortex-exec
  - Evidence: `test_execute_autonomy_goal_v1_publishes_planned_event`

- Finding: orch `build_agent_plan` metadata validation error (`metadata.options` dict in string map)
  - Fix: metadata reverted to `{"mode": "agent"}` only
  - Evidence: `test_verb_runtime_rpc.py` passes

- Finding: agent-trace / live-proof scripts still assumed agent-chain path
  - Fix: `ContextExecService` support in `agent_trace.py`, script updates, test rewrites
  - Evidence: `test_agent_trace_bound_failure_summary.py`, `test_agent_no_recall_live_proof.py`

- Finding: broad `"execute"` autonomy trigger
  - Fix: require `"goal"` in user text (or explicit `autonomy.goal.execute`)
  - Evidence: `test_supervisor_blocks_unpromoted_goal_execute` still passes

## Restart required

```bash
# Stop removed services (if still running)
docker stop orion-athena-planner-react orion-athena-agent-chain 2>/dev/null || true

# Rebuild routing services
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: `AUTONOMY_GOAL_EXECUTION_ENABLED=true` requires graph DB configured; without it execute returns `graph_not_configured`
- Mitigation: flag defaults false in compose; operator promotes via hub API as before

- Severity: low
- Concern: `force_agent_chain` option still propagated from hub but ignored by supervisor
- Mitigation: follow-up cleanup PR

- Severity: low
- Concern: legacy tests (`test_answer_depth_pass3_golden_path_discord.py`) skipped — golden path needs context-exec rewrite
- Mitigation: `run_answer_depth_proof_suite.py` updated to cortex-exec harness tests

## PR link

Branch pushed: `chore/kill-planner-agent-chain` — open PR with:
`gh pr create --base main --head chore/kill-planner-agent-chain`
