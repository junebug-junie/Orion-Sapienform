# Orion Answer Depth Overhaul — Post-Flight Manifest (Pass 3)

## 1. Purpose of Pass 3
- Why Pass 3 was needed: Pass 1+2 made architectural claims and added unit/integration-adjacent tests, but there was still no **end-to-end proof bundle** showing the “answer-depth overhaul” behavior with concrete runtime evidence and a stable “golden path” for delivery-oriented prompts.
- What “A+ proof” means: evidence must be *runtime-observable* and reproducible, with (a) delivery-oriented packing + delivery verb tool availability, (b) guard/escalation behavior (triage cap, repeated `plan_action`), (c) consistent finalization (`finalize_response`) and meta-plan rewrite, and (d) a clean one-command way to run everything.

## 2. Remaining Gaps from Pass 2
- No hard proof bundle with concrete “golden path” evidence excerpts for delivery-oriented prompts.
- Weak/default test discovery: important runtime-relevant tests were only reliable via explicit paths.
- No dedicated supervisor/e2e validation comparable to the agent-chain proof.
- No canonical “golden path” evidence demonstrating final answer *shape* for Discord delivery-oriented asks.

## 3. Proof Added in Pass 3
- Golden-path harness: `tests/test_answer_depth_pass3_golden_path_discord.py`
  - Runs Cortex exec `PlanRunner` → real `Supervisor` control flow
  - Escalates into **real agent-chain tool resolution** (pack merge + verb YAML resolution) while stubbing planner/LLM calls deterministically
  - Asserts delivery_pack activation, delivery verbs in `resolved_tool_ids`, triage hard-cap behavior, repeated `plan_action` escalation, `finalize_response` invocation, and that the final answer is a concrete Discord deployment guide (no meta-plan scaffolding)
  - Writes proof artifacts under `docs/postflight/proof/` when `ORION_PROOF_WRITE=1`
- Supervisor-path proof: `tests/test_answer_depth_pass3_supervisor_meta_plan_proof.py`
  - Runs Cortex exec `Supervisor` loop with a stubbed planner meta-plan
  - Asserts `supervisor_finalize_response_invoked=1`-equivalent behavior by checking that `finalize_response` is invoked and the final artifact contains no meta-plan scaffolding
  - Writes proof artifacts under `docs/postflight/proof/` when `ORION_PROOF_WRITE=1`
- Test discovery / execution improvements
  - Added canonical runner `scripts/run_answer_depth_proof_suite.py` that runs Pass 2 + Pass 3 tests with explicit paths and avoids `app` package collisions by isolating pytest phases with different `PYTHONPATH`
- Proof bundle
  - Added `docs/postflight/proof/README.md` plus expected-evidence templates
  - Generated concrete evidence files:
    - `discord_deploy_golden_path_evidence.json/.md`
    - `supervisor_meta_plan_finalize_evidence.json/.md`
- Final-answer quality assertions
  - Discord-specific heuristics verify token/env handling, intents/permissions, invite step, and troubleshooting content
  - Forbidden meta-plan phrases are asserted absent

## 4. Files Changed in Pass 3
- `tests/test_answer_depth_pass3_golden_path_discord.py` — new — Golden-path delivery proof (Cortex exec + real agent-chain tool resolution + guard assertions + Discord-specific answer-shape checks)
- `tests/test_answer_depth_pass3_supervisor_meta_plan_proof.py` — new — Supervisor meta-plan rewrite proof (`finalize_response` invoked; scaffolding not leaked) + evidence writing
- `scripts/run_answer_depth_proof_suite.py` — new — One-command proof suite runner with dependency checks and collision-free execution phases
- `docs/postflight/proof/README.md` — new — Proof bundle description and regeneration instructions
- `docs/postflight/proof/discord_deploy_golden_path_expected_evidence.md` — new — Expected golden-path evidence shape + representative log lines + answer-shape requirements
- `docs/postflight/proof/supervisor_meta_plan_expected_evidence.md` — new — Expected supervisor proof summary (tool invocation + forbidden phrase absence)
- `docs/postflight/proof/discord_deploy_golden_path_evidence.json` — new (generated) — Concrete golden-path evidence captured from the harness
- `docs/postflight/proof/discord_deploy_golden_path_evidence.md` — new (generated) — Human-readable golden-path evidence excerpt
- `docs/postflight/proof/supervisor_meta_plan_finalize_evidence.json` — new (generated) — Concrete supervisor proof evidence captured from the harness
- `docs/postflight/proof/supervisor_meta_plan_finalize_evidence.md` — new (generated) — Human-readable supervisor proof evidence excerpt

## 5. How to Run the Proof Suite
From repo root:
1. If deps are missing:
   ```bash
   python scripts/run_answer_depth_proof_suite.py --auto-install-deps
   ```
2. If deps are already present:
   ```bash
   python scripts/run_answer_depth_proof_suite.py
   ```

This runner sets `ORION_PROOF_WRITE=1` by default, so it overwrites concrete evidence under `docs/postflight/proof/`.

## 6. Golden-Path Expected Evidence
For prompt: `Please provide instructions on how to deploy you onto Discord.`
- `output_mode`
  - Expected: `implementation_guide`
- `response_profile`
  - Expected: `technical_delivery`
- `packs`
  - Expected: includes `delivery_pack` (in addition to base packs)
- `resolved tools`
  - Expected: includes delivery verbs such as:
    - `write_guide`
    - `finalize_response`
- finalization behavior
  - Expected: `finalize_response_invoked == true`
  - Expected: triage is overridden after step 0 / with prior trace (`triage_blocked_post_step0 == true`)
  - Expected: repeated `plan_action` escalation triggers a delivery verb (`repeated_plan_action_escalation == true`)
- answer shape requirements
  - Must include Discord deployment material:
    - Discord app/bot setup
    - token/env var handling
    - intents/permissions
    - hosting/process steps
    - testing/troubleshooting
  - Must NOT contain meta-plan scaffolding phrases such as:
    - `gather requirements`
    - `create a guide`
    - `review and refine`

## 7. Actual Evidence Captured
This environment was able to run the suite and produce concrete evidence.

### Test output
- Pass 2 core (cognition + root wiring):
  - `14 passed in 0.36s`
- Pass 2 agent-chain runtime guards:
  - `8 passed in 1.24s`
- Pass 3 proof tests:
  - `2 passed in 2.01s`

### Golden-path runtime evidence (key excerpts)
From `docs/postflight/proof/discord_deploy_golden_path_evidence.json`:

- Resolved output modes / packs / flags
  ```json
  {
    "output_mode": "implementation_guide",
    "response_profile": "technical_delivery",
    "packs": ["executive_pack", "memory_pack", "delivery_pack"],
    "runtime_debug_flags": {
      "triage_blocked_post_step0": true,
      "repeated_plan_action_escalation": true,
      "finalize_response_invoked": true,
      "quality_evaluator_rewrite": true
    }
  }
  ```

- Representative resolved delivery verbs excerpt
  - Includes: `finalize_response`, `write_guide` (plus other delivery verbs such as `write_tutorial`, `write_runbook`, `write_recommendation`, etc.)

- Representative log excerpt (agent-chain)
  ```text
  [agent-chain] wiring corr=corr-discord-deploy output_mode=implementation_guide profile=technical_delivery packs=['executive_pack', 'memory_pack', 'delivery_pack'] tools=['analyze_text', 'triage', 'plan_action', 'assess_risk', 'goal_formulate', 'evaluate', 'extract_facts', 'summarize_context', 'answer_direct', 'finalize_response', 'write_guide', 'write_tutorial', 'write_runbook', 'write_recommendation', 'compare_options', 'synthesize_patterns', 'generate_code_scaffold']
  [agent-chain] repeated_plan_action_escalation=1 -> tool_id=write_guide output_mode=implementation_guide
  [agent-chain] triage_blocked_post_step0=1 step=2 prior_trace_len=1 -> finalize_response
  [agent-chain] quality_evaluator_rewrite=1 output_mode=implementation_guide
  ```

- Final answer excerpt (shape)
  - Starts with `Discord Deployment Guide (Orion Discord Bot Bridge)`
  - Includes `DISCORD_BOT_TOKEN`, gateway `intents`, `permissions`, `Invite` / authorization steps, and a troubleshooting section

## 8. Supervisor-Path Evidence
From `docs/postflight/proof/supervisor_meta_plan_finalize_evidence.json` / `.md`:
- Evidence of meta-plan detection and finalize invocation:
  - `called_tool_ids`: `["finalize_response"]`
  - Log excerpt includes:
    ```text
    supervisor_finalize_response_invoked=1 corr=corr-supervisor-meta-plan output_mode=implementation_guide (planner meta-plan)
    ```
- Final artifact proof:
  - `final_answer_excerpt` is `Discord Deployment Guide (Concrete)` and contains no meta-plan scaffolding phrases

## 9. Quality Gate Evidence
Meta-plan rewrite is proven in two ways:
- Agent-chain (delivery prompt)
  - `quality_evaluator_rewrite == true`
  - Log line: `[agent-chain] quality_evaluator_rewrite=1 output_mode=implementation_guide`
  - Final `final_text` contains Discord deployment guide material and does NOT contain forbidden meta-plan phrases (`gather requirements`, `create a guide`, `review and refine`)
- Supervisor (instructional meta-plan)
  - Supervisor calls `finalize_response` when planner output is shallow meta-plan for `output_mode=implementation_guide`
  - Final artifact does not leak meta-plan scaffolding

## 10. Remaining Limitations
- LLM and PlannerReact network calls are stubbed in the harness for determinism. The proof focuses on **runtime wiring** and **guard/escalation/finalization behavior**, not on the variability of real model outputs.
- `pytest.mark.timeout` emits warnings in this environment (plugin not registered). This does not affect pass/fail of the proof suite.

## 11. Final QA Checklist
- [ ] delivery-oriented asks activate delivery_pack
- [ ] resolved tools include delivery verbs
- [ ] output_mode is behaviorally active
- [ ] response_profile is behaviorally active
- [ ] triage blocked after step 0
- [ ] repeated plan_action escalates
- [ ] finalization runs when needed
- [ ] supervisor path proven
- [ ] meta-plan rewrite proven
- [ ] proof suite runnable with one command
- [ ] proof bundle exists

