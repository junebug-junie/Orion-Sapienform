# PR: Cortex exec grammar ingress (execution trajectory)

**Branch:** `feat/cortex-exec-grammar-ingress`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-cortex-exec-grammar-ingress`  
**Head:** `ed5b9ad9` (4 commits)

## Summary

Adds shadow observability to `orion-cortex-exec`: valid `GrammarEventV1` traces on `orion:grammar:event` describing plan/step/result lifecycle. Execution behavior is unchanged; publishing is behind `PUBLISH_CORTEX_EXEC_GRAMMAR` (default `false`) and fail-open.

This is ingress only — no field digester, reducer, or projection for exec traces yet.

## Architecture

```text
PlanExecutionRequest
  → PlanRunner.run_plan()  (collector: plan, recall gate, steps, assembled)
  → main.handle() / LegacyPlanVerb  (egress atom + flush)
  → orion:grammar:event (grammar.event.v1)
  → sql-writer / substrate-runtime (existing pipeline)
```

## Files changed

| Path | Change |
|------|--------|
| `services/orion-cortex-exec/app/grammar_emit.py` | Pure builder + `CortexExecGrammarCollector` |
| `services/orion-cortex-exec/app/grammar_publish.py` | Fail-open flush helper |
| `services/orion-cortex-exec/app/router.py` | Plan lifecycle instrumentation |
| `services/orion-cortex-exec/app/main.py` | Intake validation trace, egress flush in `finally` |
| `services/orion-cortex-exec/app/verb_adapters.py` | `legacy.plan` flush (no double trace with `handle`) |
| `services/orion-cortex-exec/app/settings.py` | `PUBLISH_CORTEX_EXEC_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL` |
| `services/orion-cortex-exec/.env_example` | New env keys |
| `services/orion-cortex-exec/docker-compose.yml` | Env passthrough |
| `services/orion-cortex-exec/README.md` | Grammar channel docs |
| `orion/grammar/publish.py` | Optional `channel` override |
| `orion/bus/channels.yaml` | `orion-cortex-exec` producer on `orion:grammar:event` |
| `services/orion-cortex-exec/tests/test_exec_grammar_emit.py` | Builder/schema tests |
| `services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py` | Non-fatal publish |
| `scripts/smoke_cortex_exec_grammar.sh` | Unit test + bus tap instructions |

## Emitted semantic roles

| `semantic_role` | `atom_type` | Layer |
|-----------------|-------------|-------|
| `exec_request_received` | `observation` | intake |
| `exec_request_invalid` | `uncertainty_marker` | intake (validation failure only) |
| `exec_plan_started` | `action_candidate` | plan |
| `exec_recall_gate_observed` | `signal` | memory_gate |
| `exec_step_started` | `action_candidate` | step |
| `exec_step_completed` | `reasoning_step` | step |
| `exec_step_failed` | `uncertainty_marker` | step |
| `exec_result_assembled` | `spoken_output` | result |
| `exec_result_emitted` | `signal` | egress |

`GrammarEventKind` values used: `trace_started`, `atom_emitted`, `edge_emitted`, `trace_ended` only.

`RelationType` values used: `contains`, `derived_from`, `temporal_successor`, `rendered_as`.

## Example trace id

```text
cortex.exec:athena:<correlation_id>
```

## Test results

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
# 7 passed

PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
# 1 passed

./scripts/smoke_cortex_exec_grammar.sh
# 8 passed (grammar tests)
```

Full `services/orion-cortex-exec/tests/` suite: some modules error on collection when run together (pre-existing verb-registry import ordering); grammar and router subset tests pass.

## Live bus status

**Unverified** — no `redis-cli SUBSCRIBE orion:grammar:event` tap during a live exec run in this session. Enable locally:

```bash
PUBLISH_CORTEX_EXEC_GRAMMAR=true
```

## Test plan

- [x] Grammar builder validates against closed `GrammarEventKind` / `RelationType`
- [x] Publish failure is non-fatal
- [x] No raw prompts / LLM text in atoms (`payload_ref` only)
- [x] Code review fixes: step edge direction, grammar flush on cognition failure
- [ ] Live bus tap with `PUBLISH_CORTEX_EXEC_GRAMMAR=true` (operator)

## Non-goals (unchanged)

- No `executor.py` service-level atoms
- No field digester / exec projection
- No new `GrammarEventKind` literals
