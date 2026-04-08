# Orion Answer Depth Overhaul — Post-Flight Manifest

## 1. Problem Addressed

**What was wrong before:** Orion's answers were chronically shallow. The system excelled at executive cognition (triage, plan_action, assess_risk, evaluate) but produced meta-work artifacts instead of concrete answers. Requests like "How do I deploy you onto Discord?" yielded responses such as "gather requirements, create a guide, test deployment, review and refine"—managerial scaffolding rather than the actual deployment guide.

**Why answers were shallow:** The architecture had no delivery lane. All verbs were executive; the planner biased toward triage first; plan_action output could leak as the final answer; and there was no finalization contract or quality evaluator to enforce artifact production.

## 2. Audit Findings

**What we found in the repo before editing:**
- Flow: Gateway → Orch (optional auto-routing) → cortex-exec (LegacyPlanVerb) → PlanRouter → PlanRunner (brain) or Supervisor (agent)
- Auto-routing: DecisionRouter heuristic/LLM routed depth 0–3; depth 2 → agent mode → Supervisor → PlannerReact + AgentChain
- Verb selection: PlannerReact chose from executive_pack (triage, plan_action, analyze_text, etc.) with FIRST ACTION BIAS forcing triage
- No output modes, response profiles, or artifact-type concepts
- No delivery verbs (answer_direct, write_guide, finalize_response, etc.)
- Finalization: Planner final_answer → final_text; plan_action output could become final via trace fallback
- Max steps: Agent-chain raised RuntimeError; no best-effort finalization
- analyze_text was generic; no narrowing
- No repeated-tool loop breaker; no triage step cap

**False/stale assumptions:**
- Verb catalog had delivery-oriented verbs — false; all were executive
- Triage was advisory — false; it acted as a sub-planner
- plan_action was internal — false; its output could leak as user-facing

## 3. Architecture Changes

- **Output-mode routing:** OutputModeClassifier in Orch classifies user text into output_mode (implementation_guide, comparative_analysis, decision_support, etc.) and response_profile. Result flows via options and context to downstream services.
- **Response profiles:** direct_answer, tutorial_stepwise, technical_delivery, architect, reflective_depth. Used to bias tool selection and prompt instructions.
- **Delivery lane:** New delivery_pack with 9 verbs (answer_direct, finalize_response, write_guide, write_tutorial, write_runbook, write_recommendation, compare_options, synthesize_patterns, generate_code_scaffold). Added to packs when output_mode is delivery-oriented.
- **Finalization contract:** When max steps reached, agent-chain invokes finalize_response instead of raising. When plan_action would leak, finalize_response is called. When triage is blocked post step 0, finalize_response is used.
- **Quality evaluator:** Rule-based meta-plan phrase detection; when output_mode is instructional and text matches meta-plan patterns, finalize_response rewrite is invoked.
- **Executive leakage controls:** Triage capped to step 0; plan_action leakage guard; repeated-tool loop breaker; triage prompt narrowed to advisory-only.

## 4. Files Changed

| Path | Change | Purpose |
|------|--------|---------|
| orion/schemas/cortex/contracts.py | modified | Add OutputMode, ResponseProfile, OutputModeDecisionV1 |
| services/orion-cortex-orch/app/output_mode_classifier.py | **new** | Heuristic classifier for output_mode and response_profile |
| services/orion-cortex-orch/app/decision_router.py | modified | Integrate classifier, INSTRUCTION_TERMS heuristic, output_mode in options |
| services/orion-cortex-orch/app/orchestrator.py | modified | Output mode in context, delivery_pack when delivery-oriented |
| services/orion-cortex-orch/app/main.py | modified | Merge output_mode_decision into route_meta |
| orion/cognition/packs/delivery_pack.yaml | **new** | Delivery pack definition |
| orion/cognition/verbs/answer_direct.yaml | **new** | Direct answer verb |
| orion/cognition/verbs/finalize_response.yaml | **new** | Finalization verb |
| orion/cognition/verbs/write_guide.yaml | **new** | Step-by-step guide verb |
| orion/cognition/verbs/write_tutorial.yaml | **new** | Tutorial verb |
| orion/cognition/verbs/write_runbook.yaml | **new** | Debug runbook verb |
| orion/cognition/verbs/write_recommendation.yaml | **new** | Decision recommendation verb |
| orion/cognition/verbs/compare_options.yaml | **new** | Comparative analysis verb |
| orion/cognition/verbs/synthesize_patterns.yaml | **new** | Pattern synthesis verb |
| orion/cognition/verbs/generate_code_scaffold.yaml | **new** | Code scaffold verb |
| orion/cognition/prompts/answer_direct_prompt.j2 | **new** | Direct answer prompt |
| orion/cognition/prompts/finalize_response_prompt.j2 | **new** | Finalization prompt |
| orion/cognition/prompts/write_guide_prompt.j2 | **new** | Guide prompt |
| orion/cognition/prompts/write_tutorial_prompt.j2 | **new** | Tutorial prompt |
| orion/cognition/prompts/write_runbook_prompt.j2 | **new** | Runbook prompt |
| orion/cognition/prompts/write_recommendation_prompt.j2 | **new** | Recommendation prompt |
| orion/cognition/prompts/compare_options_prompt.j2 | **new** | Compare prompt |
| orion/cognition/prompts/synthesize_patterns_prompt.j2 | **new** | Synthesize prompt |
| orion/cognition/prompts/generate_code_scaffold_prompt.j2 | **new** | Code scaffold prompt |
| orion/cognition/prompts/triage_prompt.j2 | modified | Narrow to advisory classification |
| orion/cognition/prompts/analyze_text_prompt.j2 | modified | Narrow scope, exclude how-to |
| services/orion-agent-chain/app/tool_executor.py | modified | fallback_map for delivery verbs |
| services/orion-agent-chain/app/api.py | modified | Triage cap, plan_action guard, repeated-tool breaker, step-cap finalization, quality evaluator |
| services/orion-planner-react/app/api.py | modified | Triage step-0 rule, delivery verb mapping, no-repeated-tools |
| services/orion-cortex-exec/app/supervisor.py | modified | Pass output_mode, response_profile to planner and agent |
| orion/cognition/verb_catalog.py | modified | Ranking bias for delivery verbs on instruction-like queries |
| orion/cognition/quality_evaluator.py | **new** | Meta-plan detection, should_rewrite_for_instructional |
| orion/schemas/agents/schemas.py | modified | AgentChainRequest + output_mode, response_profile |
| services/orion-cortex-orch/tests/test_output_mode_routing.py | **new** | Output mode routing tests |
| services/orion-agent-chain/tests/test_delivery_verbs.py | **new** | Delivery pack/verb existence tests |
| services/orion-agent-chain/tests/test_triage_gating.py | **new** | Triage block at step>0 test |
| services/orion-agent-chain/tests/test_step_cap_finalization.py | **new** | Step-cap finalization test |
| orion/cognition/tests/test_quality_evaluator.py | **new** | Quality evaluator tests |
| docs/postflight/orion_answer_depth_overhaul_manifest.md | **new** | This manifest |

## 5. New or Changed Verbs

| Verb | Purpose | Lane | Prompt |
|------|---------|------|--------|
| answer_direct | Produce direct user-facing answer | Delivery | answer_direct_prompt.j2 |
| finalize_response | Transform trace → final artifact | Delivery | finalize_response_prompt.j2 |
| write_guide | Step-by-step guide | Technical Delivery | write_guide_prompt.j2 |
| write_tutorial | Tutorial with examples | Technical Delivery | write_tutorial_prompt.j2 |
| write_runbook | Debug/ops runbook | Technical Delivery | write_runbook_prompt.j2 |
| write_recommendation | Decision recommendation | Synthesis | write_recommendation_prompt.j2 |
| compare_options | Comparative analysis | Synthesis | compare_options_prompt.j2 |
| synthesize_patterns | Pattern synthesis | Synthesis | synthesize_patterns_prompt.j2 |
| generate_code_scaffold | Code scaffold | Technical Delivery | generate_code_scaffold_prompt.j2 |

## 6. Prompt / Profile Inventory

| File | Purpose |
|------|---------|
| answer_direct_prompt.j2 | Direct answer with concreteness; anti meta-plan |
| finalize_response_prompt.j2 | Synthesize trace into final artifact |
| write_guide_prompt.j2 | Step-by-step guide; no filler |
| write_tutorial_prompt.j2 | Tutorial with examples |
| write_runbook_prompt.j2 | Debug/ops runbook |
| write_recommendation_prompt.j2 | Decision recommendation |
| compare_options_prompt.j2 | Comparative analysis |
| synthesize_patterns_prompt.j2 | Pattern synthesis |
| generate_code_scaffold_prompt.j2 | Code scaffold |
| triage_prompt.j2 | Advisory classification only |
| analyze_text_prompt.j2 | Narrowed to summarization/extraction |

## 7. Routing Behavior After Changes

- **Direct answer requests** ("how do I X", "explain Y"): Output mode implementation_guide or direct_answer; delivery_pack added; INSTRUCTION_TERMS heuristic routes to depth 2; delivery verbs preferred.
- **Tutorial requests**: Same as direct answer; write_tutorial, write_guide available.
- **Implementation guide requests**: implementation_guide; technical_delivery profile.
- **Code delivery requests**: code_delivery mode; generate_code_scaffold preferred.
- **Planning requests**: Unchanged for project_planning; triage still at step 0 only.
- **Reflective/depth requests**: reflective_depth; architect profile; synthesize_patterns, compare_options.

## 8. Triage / Executive Leakage Controls

- **Triage step-0 cap:** Triage allowed only when TRACE is empty. At step_idx > 0, planner returns triage → agent-chain overrides to finalize_response.
- **plan_action leakage prevention:** When last executed tool was plan_action and planner finishes with thought/observation, agent-chain invokes finalize_response instead of returning plan_action output.
- **Repeated-tool loop breaker:** If same tool called consecutively, override to finalize_response.
- **Direct-answer bypass:** INSTRUCTION_TERMS in heuristic router; output_mode classifier for "how", "deploy", "guide", etc. → implementation_guide + depth 2.

## 9. Finalization / Quality Evaluation

- **When finalization is invoked:** Step cap reached; triage blocked post step 0; repeated tool; plan_action leakage guard.
- **Quality checks:** looks_like_meta_plan() detects phrases like "gather requirements", "create a guide", "review and refine". For instructional output_mode, triggers finalize_response rewrite.
- **Meta-plan correction:** _maybe_rewrite_meta_plan() calls finalize_response with rewrite instruction when flagged.

## 10. Tests Added / Updated

| Test | Behavior |
|------|----------|
| test_discord_deploy_routes_to_implementation_guide | "Please provide instructions on how to deploy you onto Discord" → implementation_guide |
| test_compare_routes_to_comparative | "Compare Discord vs Slack" → comparative_analysis |
| test_decide_routes_to_decision_support | "Help me decide whether to build" → decision_support |
| test_delivery_pack_exists_and_has_verbs | delivery_pack lists all 9 verbs |
| test_delivery_verb_yamls_exist | All 9 verb YAML files exist |
| test_triage_blocked_after_step_0 | Triage at step 1 → finalize_response |
| test_step_cap_yields_best_effort | Max steps → finalize_response, prose not error |
| test_meta_plan_flagged | Meta-plan phrases detected |
| test_should_rewrite_for_instructional | implementation_guide + meta-plan → should_rewrite |

## 11. Smoke-Test Prompts

| Prompt | Expected Behavior |
|--------|-------------------|
| "Please provide instructions on how to deploy you onto Discord." | implementation_guide; concrete Discord app setup, intents, env vars, hosting steps |
| "Write the code scaffold for the Discord bot bridge." | code_delivery; generate_code_scaffold; actual code structure |
| "Explain the architecture tradeoffs for the Orion Cortex flow." | reflective_depth or architect; synthesis, tradeoffs |
| "Compare Discord vs Slack for Orion deployment." | comparative_analysis; pros/cons, criteria |
| "Help me decide whether to build the Discord bridge now or later." | decision_support; recommendation with reasoning |
| "How do I debug the agent-chain when it times out?" | debug_diagnosis; runbook-style steps |
| "Synthesize the patterns from our last three architecture discussions." | reflective_depth; pattern synthesis |
| "Deploy Orion to Discord, compare with Slack, and give me a code scaffold." | Mixed; delivery artifact with comparison and code |

## 12. Known Limitations / Follow-Ups

- **LLM dependency:** Delivery quality depends on LLM following prompts; meta-plan detection is heuristic.
- **Triage at step 0:** Planner may still choose triage when other verbs would be better; prompt steering only.
- **Supervisor path:** Supervisor's own loop does not yet invoke finalize_response when it has trace but no final_text before escalating; relies on agent-chain.
- **Test environment:** Some agent-chain tests require redis, yaml, etc.; delivery_verbs and quality_evaluator tests run standalone.
- **Debug metadata:** output_mode_decision is in route_meta and context; optional compact metadata in CortexClientResult not fully wired for all paths.

## 13. QA Checklist

- [ ] Output modes implemented (direct_answer, tutorial, implementation_guide, code_delivery, decision_support, comparative_analysis, debug_diagnosis, project_planning, reflective_depth)
- [ ] Response profiles implemented (direct_answer, tutorial_stepwise, technical_delivery, architect, reflective_depth)
- [ ] Delivery verbs added (9 verbs in delivery_pack)
- [ ] Direct-answer bypass exists (INSTRUCTION_TERMS, output_mode classifier)
- [ ] Triage capped to step 0
- [ ] Repeated plan_action no longer leaks
- [ ] Finalization exists (step cap, plan_action guard, triage override)
- [ ] Quality evaluator exists (meta-plan detection, rewrite)
- [ ] analyze_text demoted/narrowed
- [ ] Tests exist (output mode, delivery verbs, triage, step cap, quality)
- [ ] No legacy path introduced (all flows through Cortex)
