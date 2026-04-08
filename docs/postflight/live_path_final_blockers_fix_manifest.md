# Live Path Final Blockers Fix Manifest

## Scope
This pass fixes only four live-path blockers observed in production logs:
1. Supervisor pack propagation dropping `delivery_pack`.
2. PlannerReact parse salvage selecting inner tool content instead of the outer planner JSON object.
3. PlannerReact repair validation rejecting valid `finish=true` repair payloads without an action.
4. Agent-chain continuing after a successful `finalize_response`.

## Blocker 1: Supervisor pack propagation drops `delivery_pack`
- **Root cause:** Orch correctly merged `delivery_pack` into the request context, but `args.extra["packs"]` still carried the original pack list. Exec merges `context` and `args.extra`, so the stale extra payload overwrote the corrected context packs before supervisor tool resolution.
- **Fix:** `build_plan_request()` now writes the effective merged pack list back into `args.extra["packs"]`. Supervisor also computes an effective pack list from `ctx`, `ctx.metadata`, and `req.metadata` so delivery packs survive live-path merges and metadata fallbacks.
- **Files changed:**
  - `services/orion-cortex-orch/app/orchestrator.py`
  - `services/orion-cortex-exec/app/supervisor.py`
- **Tests:**
  - `services/orion-cortex-orch/tests/test_auto_router.py::test_build_plan_request_preserves_delivery_pack_in_args_extra`
  - `services/orion-cortex-exec/tests/test_supervisor_planner_delegate.py::test_supervisor_prefers_effective_packs_from_context_metadata`
- **Expected log difference after fix:** `supervisor_wiring` should log `packs=['executive_pack', 'delivery_pack', ...]` for delivery-oriented asks, and `tool_ids` should include delivery verbs such as `write_guide` and `finalize_response` when those packs are available.

## Blocker 2: PlannerReact salvage targets inner tool content instead of outer planner JSON
- **Root cause:** Salvage candidate ordering allowed wrapper stripping / fallback extraction paths to grab JSON-looking inner tool payloads before the valid outer planner response object.
- **Fix:** PlannerReact now attempts the outer balanced JSON object extracted from the raw response before any wrapper-stripping fallbacks, then parses that object first. Fallback salvage paths still exist, but only after the raw outer-object attempt.
- **Files changed:**
  - `services/orion-planner-react/app/api.py`
- **Tests:**
  - `tests/test_planner_react_contract.py::test_salvage_prefers_outer_balanced_json_over_inner_tool_content`
  - Existing salvage regression cases in `tests/test_planner_react_contract.py`
- **Expected log difference after fix:** valid outer planner JSON containing embedded code/text should no longer emit `Planner LLM returned non-JSON` parse failures for inner content snippets.

## Blocker 3: Repair validation rejects valid `finish=true` repair outputs
- **Root cause:** Repair responses needed to be accepted whenever `finish=true` and a usable `final_answer` exists, even when `action=null`. The live-path contract also needed explicit normalization for object-shaped final answers such as `{ "content": "..." }`.
- **Fix:** PlannerReact validation keeps the `finish=true` path authoritative, normalizes object/list final-answer forms, and preserves `action=null` as valid when a usable user-facing answer is present.
- **Files changed:**
  - `services/orion-planner-react/app/api.py`
- **Tests:**
  - `tests/test_planner_react_contract.py::test_repair_path_accepts_finish_true_without_action`
  - `tests/test_planner_react_contract.py::test_repair_path_accepts_finish_true_content_object_without_action`
- **Expected log difference after fix:** repair-path logs should stop collapsing valid `finish=true` repair payloads into `finish=false requires an action` schema failures.

## Blocker 4: Agent-chain continues after successful `finalize_response`
- **Root cause:** Agent-chain executed `finalize_response` like any other tool, appended the observation to trace, and continued the planner loop even when the finalizer already produced a usable user-facing answer.
- **Fix:** Agent-chain now inspects the `finalize_response` observation immediately. If it contains usable text/content, the chain returns that answer and stops instead of invoking planner-react again.
- **Files changed:**
  - `services/orion-agent-chain/app/api.py`
- **Tests:**
  - `services/orion-agent-chain/tests/test_agent_chain_delegate_loop.py::test_agent_chain_stops_after_successful_finalize_response`
- **Expected log difference after fix:** after repeated-tool breaker / triage override / direct finalizer selection invokes `finalize_response`, there should be no subsequent planner step unless the finalizer produced no usable answer.

## Validation Summary
Focused regression coverage now checks:
- merged delivery packs survive Orch → Exec → Supervisor live-path handoff,
- outer planner JSON is parsed before inner tool content,
- repair outputs with `finish=true` and usable final answers are accepted with `action=null`,
- agent-chain exits immediately after successful `finalize_response`.
