# PR: Agent grounding preflight — block fake-engine false investigation success

**Branch:** `feat/agent-grounding-preflight`  
**Worktree:** `.worktrees/feat/agent-grounding-preflight`

## Summary

Fixes a real bug where the Hub Agent lane reported **"Investigation complete"** / `status ok` for recall questions (e.g. *"do you recall where my mom lives?"*) even when:

- `engine_selected` was `fake`
- `model_synthesis_used` was `false`
- no recall/trace/repo evidence was captured
- `tool_call_count` was `1` while `tools: []` (from counting `rlm_engine.run` as a user tool)

The patch **separates runtime success from answer success** and blocks ungrounded investigations from presenting as complete.

## Behavior changes

### Answer evaluation (`answer_evaluation` in `runtime_debug`)

New fields (also surfaced in Hub `routing_debug` and agent trace `raw`):

| Field | Values |
|-------|--------|
| `runtime_status` | `ok` \| `failed` |
| `answer_status` | `answered_grounded` \| `partial_or_weak_evidence` \| `no_reliable_evidence` \| `failed_grounding_preflight` \| `failed_fake_engine_selected` |
| `grounding_status` | `not_required` \| `attempted` \| `skipped` \| `unavailable` \| `failed` |
| `synthesis_status` | `completed` \| `skipped` \| `deterministic_no_memory` \| `blocked` |
| `evidence_count` | int |
| `grounding_required` | bool |

### Grounding preflight rules

1. **Fake engine block** — If `engine_selected` is `fake`/`mock`/`smoke`/`stub` and the user did not explicitly request a smoke/fake run, set `answer_status=failed_fake_engine_selected` and replace summary with an explicit blocked message.
2. **No grounding attempted** — For grounding-required requests with zero evidence and no recall/trace/repo attempt, set `failed_grounding_preflight`.
3. **Placeholder summaries** — `"Investigation complete for: …"` is never shown as a successful answer when grounding failed.

### Tool count fix

- `rlm_engine.run` remains in the timeline (`steps`) as a runtime step.
- `tool_call_count`, `unique_tool_count`, and grouped tool usage only count **user-visible** tools (excludes `rlm_engine.run`).
- `raw.runtime_step_count` preserves total step count.

### UI messaging

- Hub inline agent response headline becomes **"Runtime completed (not a grounded investigation)"** when answer failed preflight.
- Agent trace overview shows **Answer** status and separates runtime steps from tool calls.

## Files changed

| Area | Path |
|------|------|
| Grounding eval (new) | `services/orion-context-exec/app/grounding_eval.py` |
| Runner wiring | `services/orion-context-exec/app/runner.py` |
| Hub bridge | `services/orion-hub/scripts/context_exec_agent_bridge.py` |
| Hub UI | `services/orion-hub/static/js/app.js` |
| Tests | `services/orion-context-exec/tests/test_grounding_eval.py` |
| Tests | `services/orion-context-exec/tests/test_runner_grounding_preflight.py` |
| Tests | `services/orion-hub/tests/test_context_exec_agent_grounding.py` |

No `.env_example` changes (no new env keys).

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_grounding_eval.py services/orion-context-exec/tests/test_runner_grounding_preflight.py -q` | 0 | 7 passed |
| `cd services/orion-hub && orion_dev/bin/python -m pytest tests/test_context_exec_agent_grounding.py tests/test_llm_route_selector.py -q` | 0 | 19 passed |
| `python -m compileall services/orion-context-exec/app/grounding_eval.py …` | 0 | OK |

## Acceptance mapping

| Criterion | Status |
|-----------|--------|
| Runtime can be `ok` while answer is not grounded | ✅ |
| Fake engine → `failed_fake_engine_selected` | ✅ |
| Summary not `"Investigation complete for: …"` on failure | ✅ |
| UI does not present as completed investigation | ✅ |
| `tool_call_count` agrees with `tools: []` when only `rlm_engine.run` ran | ✅ |
| Grounded path with evidence still succeeds | ✅ (`answered_grounded` for general_investigation + evidence) |

## Remaining risks

- `grounding_required` heuristics are broad (any `?` triggers grounding). Intentional for recall-style safety; may over-classify casual prompts.
- Fake-engine blocking applies to all non-`general_investigation` modes when engine is fake — existing smoke tests may need `scopes.smoke_run` if they rely on fake engine for structured modes.
