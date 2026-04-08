# Answer Depth Holdouts Pass 6 Manifest

## Purpose of this pass

Fix the remaining answer-depth holdouts after Hub Agent routing was already proven:

- make PlannerReact survive malformed JSON-ish outputs without collapsing,
- stop reflective recall from contaminating delivery-oriented asks by default,
- improve delivery grounding so Orion-specific prompts stay anchored to Orion's actual runtime architecture,
- strengthen finalization so it prefers trace-grounded delivery output instead of generic rewrites,
- preserve the proven Hub → Orch → Exec supervised path.

## Root causes found

1. **PlannerReact was too brittle at the JSON boundary.**
   - It handled some wrappers, but still failed on mixed prose/JSON, code fences, and raw answer-shaped markdown/code blocks.
   - Semantically valid outputs could still die in schema validation even when `final_answer.content` was obviously usable.

2. **Planner schema normalization was contradictory.**
   - Valid `finish=true` payloads with `final_answer` could still degrade into repair loops that later complained `finish=false requires an action`.

3. **Delivery prompts lacked repo-architecture grounding.**
   - Delivery verbs (`write_guide`, `answer_direct`, `finalize_response`) did not strongly anchor the answer to Orion's real runtime path, so generic stack drift remained possible.

4. **Reflective recall remained active for delivery-oriented asks.**
   - Default recall behavior could still select `reflect.v1` for implementation-guide style prompts, increasing contamination risk.

5. **Finalization did not clearly prefer the best trace artifact.**
   - Final response synthesis could rewrite from scratch even when prior trace already contained the strongest delivery draft.

## Files changed

- `services/orion-planner-react/app/api.py`
- `services/orion-cortex-exec/app/recall_utils.py`
- `services/orion-cortex-exec/app/router.py`
- `services/orion-cortex-exec/app/supervisor.py`
- `services/orion-agent-chain/app/api.py`
- `services/orion-cortex-orch/app/orchestrator.py`
- `orion/cognition/delivery_grounding.py`
- `orion/cognition/finalize_payload.py`
- `orion/cognition/quality_evaluator.py`
- `orion/cognition/prompts/write_guide_prompt.j2`
- `orion/cognition/prompts/finalize_response_prompt.j2`
- `orion/cognition/prompts/answer_direct_prompt.j2`
- `tests/test_planner_react_contract.py`
- `services/orion-cortex-exec/tests/test_recall_delivery_gating.py`
- `orion/cognition/tests/test_finalize_payload.py`
- `orion/cognition/tests/test_delivery_grounding.py`
- `services/orion-agent-chain/tests/test_delivery_grounding_pass4.py`

## Parser / normalization changes

- Added planner parse classification modes:
  - `raw_parse_success`
  - `normalized_from_jsonish`
  - `salvaged_from_code_block`
  - `salvaged_final_answer_from_mixed_text`
  - `unrecoverable_parse_failure`
- Added salvage metadata fields:
  - `planner_normalization_mode`
  - `salvage_source`
  - `final_answer_salvaged`
  - `action_salvaged`
- PlannerReact now salvages:
  - fenced JSON and JSON with stray wrapper text,
  - raw fenced code blocks as final-answer content,
  - mixed markdown/prose final answers when planner intent is obvious,
  - string actions into action objects,
  - action dict aliases such as `tool`, `name`, `verb_name`, `args`, `arguments`, `payload`.
- Schema semantics were tightened so:
  - `finish=true` + usable `final_answer` + no action is valid,
  - `finish=false` + action is valid,
  - `finish=false` + no action still fails cleanly,
  - inferred/normalized states do not contradict each other during repair.

## Recall gating changes

- Added delivery-safe recall policy in `services/orion-cortex-exec/app/recall_utils.py`.
- For delivery-oriented `output_mode` values:
  - `implementation_guide`
  - `tutorial`
  - `code_delivery`
  - `comparative_analysis`
  - `decision_support`
- Default reflective recall is now **disabled by default** unless explicitly requested.
- If recall is **required** for those delivery asks and no explicit profile is supplied, policy switches to `assist.light.v1` instead of `reflect.v1`.
- Router and Supervisor now log and expose:
  - `recall_gating_reason`
  - selected recall profile/source

## Grounding / prompt changes

- Added shared delivery-grounding helpers in `orion/cognition/delivery_grounding.py`.
- Orion-specific delivery asks now get a grounding mode of `orion_repo_architecture` plus context that explicitly anchors responses to:
  - Hub / Client
  - Cortex-Orch
  - Cortex-Exec
  - PlannerReact / AgentChain
  - LLM Gateway
  - bus-mediated execution
- Updated prompt templates for:
  - `write_guide`
  - `finalize_response`
  - `answer_direct`
- Added anti-generic drift instructions:
  - do not silently substitute Flask / Ubuntu / Gunicorn / Nginx unless requested,
  - do not swap away from Orion's runtime architecture,
  - treat Discord as an integration/bridge around Orion's existing runtime.

## Finalization handoff changes

- `build_finalize_tool_input()` now includes:
  - `trace_preferred_output`
  - `finalization_source_trace_used`
  - delivery grounding fields
- Agent-chain now enriches delivery verb inputs with grounding metadata and trace context.
- `finalize_response` calls prefer the strongest prior trace artifact as the primary draft.
- Agent-chain now logs:
  - `finalization_source_trace_used`
  - `generic_drift_detected`

## Tests added / updated

- Updated PlannerReact contract tests for:
  - `finish=true` with `final_answer.content`
  - fenced / mixed JSON salvage
  - raw code-block final answer salvage
  - mixed text final answer salvage
  - clean failure for `finish=false` without action
- Added delivery recall gating tests for:
  - delivery default disable
  - required recall narrowing to `assist.light.v1`
  - reflective-mode preservation
- Updated finalize payload tests to prove:
  - trace-preferred output selection
  - Orion Discord grounding fields
- Added delivery grounding tests to prove:
  - Orion-on-Discord prompts select repo-architecture grounding
  - generic Flask deployment text is detected as drift
- Added agent-chain delivery grounding tests to prove:
  - `finalize_response` prefers trace-grounded output
  - generic drift final answers trigger rewrite/finalization
- Re-ran the existing answer-depth proof tests:
  - `tests/test_answer_depth_pass3_golden_path_discord.py`
  - `tests/test_answer_depth_pass3_supervisor_meta_plan_proof.py`

## Remaining limitations

- The live proof suite script still depends on optional runtime packages (`pydantic-settings` in this environment), so I could not run the one-command proof harness here.
- Prompt grounding is intentionally heuristic rather than hardcoded to a single Discord answer; it strongly biases toward Orion architecture, but final quality still depends on the underlying model.
- Delivery-safe recall currently defaults to **disable** rather than a richer technical profile for most delivery asks; that is deliberate to prevent reflective contamination first.

## Recommended live retest steps

### Hub Agent
Use:

> would you write up detailed developer instructions on how to build Orion (AI) into a Discord server?

Expected observations:

1. Hub still emits `mode=agent` with supervised execution.
2. Orch still classifies to `output_mode=implementation_guide` and injects `delivery_pack`.
3. Exec/Supervisor logs show delivery-safe recall gating with reflective recall disabled unless explicitly required.
4. PlannerReact logs prefer salvage/normalization over parse collapse for JSON-ish outputs.
5. Agent-chain runtime debug shows:
   - `delivery_grounding_mode=orion_repo_architecture`
   - `generic_drift_detected` only when rewrite was needed
   - `finalization_source_trace_used=true` when finalization synthesized from trace
6. Final answer stays grounded in Orion architecture and Discord integration, not generic Flask deployment.

### Hub Auto
Repeat the same prompt in Auto mode and verify:

1. Auto still routes through Orch/Exec rather than any shortcut.
2. The same delivery-safe recall behavior and grounding cues appear downstream.
3. Final answer remains architecture-grounded and delivery-oriented.

## Explicit verdict

**Verdict: the remaining holdouts are materially fixed in code and covered by targeted tests.**

- PlannerReact is substantially more resilient to malformed JSON-ish output.
- Delivery-oriented asks no longer default to reflective recall contamination.
- Finalization is more trace-grounded.
- Orion-on-Discord prompts now have explicit anti-drift architectural grounding.
- The existing answer-depth proof tests still pass.

The only remaining gap is the full optional-dependency live proof harness, which could not be executed in this environment because the required runtime package set is not installed.
