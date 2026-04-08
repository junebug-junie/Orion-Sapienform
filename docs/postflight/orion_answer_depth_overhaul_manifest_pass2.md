# Orion Answer Depth Overhaul — Post-Flight Manifest (Pass 2)

## 1. Purpose of Pass 2

**Concerns after Pass 1:** The first manifest described output-mode routing, `delivery_pack`, delivery verbs, triage caps, repeated-tool handling, finalization, quality evaluation, and tests—but several items risked being **spec-only**: packs not merged on the live request path, planner toolsets missing delivery verbs, `output_mode` / `response_profile` unused in decisions, soft triage limits, `plan_action` loops ending shallowly, supervisor exits with planner meta-text, tests that only asserted enums/files rather than runtime wiring.

**Focus of this pass:** Audit the repo, **wire** behavior end-to-end (Orch → Exec context → Agent-chain / Planner-react), add **compact observability** (`runtime_debug` + structured logs), and add **proof-oriented tests** that fail if wiring regresses. No Cortex bypass; bus-first paths preserved.

---

## 2. Audit of Prior Claims

| Claim | Status |
|--------|--------|
| **Output modes** | **Partially wired, fixed in this pass** — classifier lives in shared `orion.cognition`; Orch classifies and merges packs; agent-chain uses effective modes in planner payload, finalize payload, guards, and quality checks. |
| **Response profiles** | **Partially wired, fixed in this pass** — carried in `external_facts`, finalize input, supervisor finalize branch, planner-react prompt hints. |
| **`delivery_pack`** | **Partially wired, fixed in this pass** — `ensure_delivery_pack_in_packs` used from Orch and agent-chain `_resolve_tools` so merged pack lists include delivery YAML when mode/text is delivery-oriented. |
| **Delivery verbs** | **Partially wired, fixed in this pass** — pack expansion + `PackManager` / verb YAML resolution yield LLM tools including `write_guide`, `finalize_response`, etc., in live `toolset` passed to planner-react. |
| **Triage step-0 cap** | **Partially wired, fixed in this pass** — `triage_must_finalize` in `orion.cognition.agent_chain_guards` blocks `triage` when `step_idx > 0` or prior trace exists; override to `finalize_response` with `runtime_debug.triage_blocked_post_step0`. |
| **Repeated-tool loop breaker** | **Confirmed and already wired** (strengthened with debug flags) — consecutive same `tool_id` forces `finalize_response` with `repeated_tool_breaker` and logging. |
| **Finalization contract** | **Partially wired, fixed in this pass** — `build_finalize_tool_input` includes trace + modes; step-cap, triage override, repeated-tool, plan_action leakage guard, and supervisor meta-plan path call `finalize_response` with structured input. |
| **Quality evaluator** | **Partially wired, fixed in this pass** — broader instructional modes + meta-plan patterns; drives `_maybe_rewrite_meta_plan` in agent-chain and supervisor post-planner finalize. |
| **Tests** | **Not actually wired (proof-wise), implemented in this pass** — added/updated tests for pack merge, YAML-level delivery verbs, guards, finalize payload, agent-chain triage/plan_action/meta-plan/step-cap; cognition tests avoid importing full planner stack (no hard `jinja2` dependency for those files). |

---

## 3. Runtime Wiring Fixes

- **`delivery_pack` → live resolution:** `orion.cognition.runtime_pack_merge.ensure_delivery_pack_in_packs` appends `delivery_pack` when `output_mode` is in `DELIVERY_ORIENTED_OUTPUT_MODES` or when `classify_output_mode(user_text)` yields a delivery-oriented mode. Orch applies this to plan packs; agent-chain `_resolve_tools` applies it again from the effective request so delegated chains stay consistent.
- **Delivery verbs in toolsets:** `_resolve_tools` loads packs via `PackManager` + verb configs; merged packs include `delivery_pack`, so resolved `ToolDef` lists expose delivery LLM verbs to planner-react’s `toolset`.
- **`output_mode` / `response_profile` behavior:** Shape planner payload `context.external_facts`, finalize and guard overrides (`_delivery_override_for_plan_action_repeat`), quality rewrite eligibility, and planner-react system prompt rules for delivery vs executive preference.
- **Triage hard cap:** `triage_must_finalize` — no triage after step 0 / with non-empty prior trace; runtime replaces tool with `finalize_response` and sets `triage_blocked_post_step0`.
- **Repeated `plan_action`:** `repeated_plan_action_needs_delivery` detects a second `plan_action`; escalates to `write_guide` (default), `generate_code_scaffold` (`code_delivery`), `compare_options` / `write_recommendation` for comparative/decision modes; sets `repeated_plan_action_escalation`.
- **Finalization consistency:** Agent-chain: triage block, repeated-tool breaker, plan_action leakage path, step cap, and explicit `finalize_response` steps set `finalize_response_invoked`. Supervisor: if planner-only `final_text` is shallow for an instructional `output_mode`, `_execute_action` with `finalize_response` and trace serialization; `debug.supervisor_meta_plan_finalize`.

---

## 4. Files Changed in Pass 2

| Path | Change | Issue addressed |
|------|--------|------------------|
| [orion/cognition/output_mode_classifier.py](orion/cognition/output_mode_classifier.py) | modified | Shared classification for Orch + agent-chain |
| [orion/cognition/runtime_pack_merge.py](orion/cognition/runtime_pack_merge.py) | modified | `DELIVERY_ORIENTED_OUTPUT_MODES`, `ensure_delivery_pack_in_packs` |
| [orion/cognition/agent_chain_guards.py](orion/cognition/agent_chain_guards.py) | new/modified | Hard triage cap + repeated `plan_action` detection |
| [orion/cognition/finalize_payload.py](orion/cognition/finalize_payload.py) | new/modified | `build_finalize_tool_input` (trace + modes) |
| [orion/cognition/quality_evaluator.py](orion/cognition/quality_evaluator.py) | modified | Instructional modes + meta-plan heuristics |
| [services/orion-cortex-orch/app/orchestrator.py](services/orion-cortex-orch/app/orchestrator.py) | modified | Classify output mode, merge delivery pack, `orch_plan_wiring` log |
| [services/orion-agent-chain/app/api.py](services/orion-agent-chain/app/api.py) | modified | Effective modes, `_resolve_tools` merge, guards, finalize payloads, `runtime_debug`, logging |
| [orion/schemas/agents/schemas.py](orion/schemas/agents/schemas.py) | modified | `AgentChainResult.runtime_debug` |
| [services/orion-cortex-exec/app/supervisor.py](services/orion-cortex-exec/app/supervisor.py) | modified | Meta-plan finalize via `finalize_response`, `supervisor_wiring` / `supervisor_finalize_response_invoked` logs |
| [services/orion-planner-react/app/api.py](services/orion-planner-react/app/api.py) | modified | Prompt block binding `output_mode` / `response_profile` to delivery tool preference |
| [orion/cognition/tests/test_runtime_pack_merge.py](orion/cognition/tests/test_runtime_pack_merge.py) | new/modified | Pack merge for deploy/scaffold/comparative modes |
| [orion/cognition/tests/test_live_tool_resolution_pass2.py](orion/cognition/tests/test_live_tool_resolution_pass2.py) | new/modified | YAML-level proof delivery verbs appear when packs merged (no jinja import) |
| [orion/cognition/tests/test_agent_chain_guards.py](orion/cognition/tests/test_agent_chain_guards.py) | new | Triage + repeated plan_action unit tests |
| [orion/cognition/tests/test_finalize_payload.py](orion/cognition/tests/test_finalize_payload.py) | new | Finalize payload includes trace + modes |
| [tests/test_answer_depth_pass2_wiring.py](tests/test_answer_depth_pass2_wiring.py) | new | Root wiring: Discord deploy text merges pack; LLM verb IDs |
| [services/orion-agent-chain/tests/test_triage_gating.py](services/orion-agent-chain/tests/test_triage_gating.py) | modified | `_resolve_tools` tuple + triage blocked post step 0 |
| [services/orion-agent-chain/tests/test_step_cap_finalization.py](services/orion-agent-chain/tests/test_step_cap_finalization.py) | modified | Step cap finalize + `runtime_debug` |
| [services/orion-agent-chain/tests/test_agent_chain_delegate_loop.py](services/orion-agent-chain/tests/test_agent_chain_delegate_loop.py) | modified | Mock `_resolve_tools` signature |
| [services/orion-agent-chain/tests/test_meta_plan_rewrite_pass2.py](services/orion-agent-chain/tests/test_meta_plan_rewrite_pass2.py) | new | Meta-plan → `finalize_response` + `quality_evaluator_rewrite` |
| [services/orion-agent-chain/tests/test_repeated_plan_action_pass2.py](services/orion-agent-chain/tests/test_repeated_plan_action_pass2.py) | new | Second `plan_action` → `write_guide` + escalation flag |
| [pyproject.toml](pyproject.toml) | modified | Optional `test-runtime` extras (`jinja2`, `pydantic-settings`) for service test imports |
| [docs/postflight/orion_answer_depth_overhaul_manifest_pass2.md](docs/postflight/orion_answer_depth_overhaul_manifest_pass2.md) | new | This manifest |

---

## 5. Runtime Observability Added

**Logs**

- **Orch:** `orch_plan_wiring corr=… output_mode=… profile=… packs=…`
- **Agent-chain:** `[agent-chain] wiring corr=… output_mode=… profile=… packs=… tools=…` (truncated tool id list); per-step `planner_action tool_id=…`; `triage_blocked_post_step0=1`; `repeated_plan_action_escalation=1`; `repeated_tool_breaker=1`; `plan_action leakage guard finalize_response_invoked=1`; `step_cap finalize_response_invoked=1`
- **Supervisor:** `supervisor_wiring corr=… output_mode=… profile=… packs=… tool_ids=…`; `supervisor_finalize_response_invoked=1 … (planner meta-plan)`; `agent_runtime_stop … output_mode=… packs=…`

**`AgentChainResult.runtime_debug` (compact dict)**

- `output_mode`, `response_profile`, `packs`, `resolved_tool_ids`
- `triage_blocked_post_step0`, `repeated_tool_breaker`, `repeated_plan_action_escalation`
- `finalize_response_invoked`, `quality_evaluator_rewrite`

**Supervisor:** `ctx["debug"]["supervisor_meta_plan_finalize"]` when meta-plan finalize runs.

---

## 6. Proof-Oriented Test Inventory

| Test module | What it proves |
|-------------|----------------|
| `orion/cognition/tests/test_runtime_pack_merge.py` | Instructional / code / comparative texts merge `delivery_pack`; mode-driven merge |
| `orion/cognition/tests/test_live_tool_resolution_pass2.py` | Merged packs include LLM delivery verbs (`write_guide`, `finalize_response`, `answer_direct`) via pack + verb YAML |
| `orion/cognition/tests/test_agent_chain_guards.py` | Triage blocked when trace/step disallows; repeated `plan_action` detected |
| `orion/cognition/tests/test_finalize_payload.py` | Finalize tool input carries trace snapshot + `output_mode` / `response_profile` |
| `orion/cognition/tests/test_quality_evaluator.py` | Meta-plan detection + instructional rewrite gating |
| `tests/test_answer_depth_pass2_wiring.py` | Discord deploy prompt merges `delivery_pack`; implementation_guide resolution includes delivery + `plan_action` |
| `services/orion-agent-chain/tests/test_triage_gating.py` | Runtime triage override after step 0 |
| `services/orion-agent-chain/tests/test_repeated_plan_action_pass2.py` | Second `plan_action` becomes delivery verb; `repeated_plan_action_escalation` |
| `services/orion-agent-chain/tests/test_meta_plan_rewrite_pass2.py` | Shallow final answer triggers `finalize_response`; `quality_evaluator_rewrite` |
| `services/orion-agent-chain/tests/test_step_cap_finalization.py` | Step cap invokes finalize; `finalize_response_invoked` |
| `services/orion-agent-chain/tests/test_agent_chain_delegate_loop.py` | Delegate loop still executes tools and returns final text |

**How to run**

- Cognition + root wiring: `PYTHONPATH=. pytest orion/cognition/tests/test_runtime_pack_merge.py orion/cognition/tests/test_live_tool_resolution_pass2.py … tests/test_answer_depth_pass2_wiring.py`
- Agent-chain (install optional deps): `pip install -e ".[test-runtime]"` then  
  `PYTHONPATH=<repo>:<repo>/services/orion-agent-chain pytest services/orion-agent-chain/tests/test_*pass2*.py services/orion-agent-chain/tests/test_triage_gating.py …`

Default `pytest` from repo root only discovers `tests/` (see `pyproject.toml` `norecursedirs`); **Pass 2 cognition and agent-chain tests are intentionally run with explicit paths.**

---

## 7. Expected Runtime Evidence

### “Please provide instructions on how to deploy you onto Discord.”

| Field | Expected |
|--------|-----------|
| `output_mode` | `implementation_guide` (instruction / deploy terms) |
| `response_profile` | `technical_delivery` |
| `packs` | Base packs + `delivery_pack` |
| Representative resolved tools | `write_guide`, `write_tutorial`, `answer_direct`, `finalize_response`, `plan_action`, … |
| Finalization | On triage-after-step0, repeated same tool, step cap, plan_action leakage, or shallow meta answer—not necessarily every turn; logs / `runtime_debug` show when `finalize_response` runs |

### “Write the code scaffold for a Discord bot bridge.”

| Field | Expected |
|--------|-----------|
| `output_mode` | `code_delivery` |
| `response_profile` | `technical_delivery` |
| `packs` | Includes `delivery_pack` |
| Representative resolved tools | `generate_code_scaffold`, `finalize_response`, `write_guide`, … |
| Finalization | Same guard conditions; repeated `plan_action` escalates to **`generate_code_scaffold`** |

### “Compare Discord vs Slack deployment for Orion.”

| Field | Expected |
|--------|-----------|
| `output_mode` | `comparative_analysis` |
| `response_profile` | `reflective_depth` |
| `packs` | Includes `delivery_pack` (comparative mode is delivery-oriented in merge set) |
| Representative resolved tools | `compare_options`, `write_guide`, `finalize_response`, … |
| Finalization | As needed via guards; repeated `plan_action` → **`compare_options`** |

### “Help me decide whether to build the Discord bridge now or later.”

| Field | Expected |
|--------|-----------|
| `output_mode` | `decision_support` |
| `response_profile` | `reflective_depth` |
| `packs` | Includes `delivery_pack` |
| Representative resolved tools | `write_recommendation`, `compare_options`, `finalize_response`, … |
| Finalization | As needed; repeated `plan_action` → **`write_recommendation`** |

---

## 8. Remaining Gaps

- **No dedicated supervisor integration test** in-repo that drives a full bus + planner-react + LLM stack; supervisor behavior is covered by code path review + unit tests on shared cognition; a full e2e test would need fixtures/mocks for `OrionBusAsync` and planner.
- **`tool_registry` / executor JSON schema** for `finalize_response` may still be stricter than the enriched payload in some deployments—verify against production registry if validation errors appear.
- **Default `pytest` root config** does not recurse `orion/cognition/tests` or `services/orion-agent-chain/tests`; CI must invoke those paths explicitly or adjust `norecursedirs` / workflows.
- **Planner stochasticity:** Evidence for exact tool *choice* is probabilistic; proofs anchor on **availability** (`resolved_tool_ids`), **guards** (forced tools), and **finalize** invocation flags—not on the LLM always picking `write_guide` first.

---

## 9. QA Checklist (Pass 2)

- [ ] `delivery_pack` appears in logs / `runtime_debug.packs` for delivery-oriented asks
- [ ] `resolved_tool_ids` includes delivery verbs (`write_guide`, `finalize_response`, etc.)
- [ ] `output_mode` is logged (`orch_plan_wiring`, `[agent-chain] wiring`, `supervisor_wiring`)
- [ ] `response_profile` is logged in the same lines / `runtime_debug`
- [ ] Triage after step 0 is blocked (`triage_blocked_post_step0` / log line)
- [ ] Repeated `plan_action` escalates to a delivery verb or finalize (`repeated_plan_action_escalation`)
- [ ] `finalize_response` runs where guards fire (`finalize_response_invoked` / supervisor meta-plan log)
- [ ] Meta-plan shallow answers trigger rewrite path (`quality_evaluator_rewrite` / `supervisor_meta_plan_finalize`)
- [ ] Pass 2 tests run on CI with explicit paths + `pip install -e ".[test-runtime]"` (or service venv) for agent-chain
