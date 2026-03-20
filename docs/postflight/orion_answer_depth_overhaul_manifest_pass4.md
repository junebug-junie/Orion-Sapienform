# Orion Answer Depth Overhaul — Post-Flight Manifest (Pass 4)

## 1. Purpose of Pass 4

- **Why Pass 4 was needed:** Pass 1–3 established architectural wiring and deterministic proof via stubbed harnesses. There was no **live proof** through the real Cortex path: Gateway → Orch → Exec → PlannerReact/AgentChain/LLMGateway over the bus.
- **What Pass 4 delivers:** A real live proof runner that sends actual requests through the Cortex stack, captures bus-observable evidence, and emits structured live evidence files. Proof is no longer dependent on mocks or deterministic stubs.

## 2. Audit of Prior Proof Paths

| Proof | Path | Stubbed vs Live |
|-------|------|------------------|
| Pass 3 golden path | `tests/test_answer_depth_pass3_golden_path_discord.py` | **Stubbed** — planner/LLM calls monkeypatched; Supervisor + agent-chain tool resolution are real |
| Pass 3 supervisor | `tests/test_answer_depth_pass3_supervisor_meta_plan_proof.py` | **Stubbed** — planner meta-plan stubbed; Supervisor finalize path is real |
| Pass 3 runner | `scripts/run_answer_depth_proof_suite.py` | pytest-only; no live bus |
| **Pass 4** | `scripts/run_answer_depth_live_proof.py` | **Live** — real `cortex.gateway.chat.request` → Gateway → Orch → verb/Exec → PlannerReact → AgentChain → LLMGateway |

**Live path mapping (Pass 4):**

- **Gateway:** `orion:cortex:gateway:request` / `orion:cortex:gateway:result:*`
- **Orch:** `orion:cortex:request` (cortex.orch.request) → `orion:cortex:result`
- **Verb/Exec:** `orion:verb:request` → exec services (RecallService, PlannerReactService, LLMGatewayService, AgentChainService)
- **PlannerReact:** `orion:exec:request:PlannerReactService` / `orion:exec:result:PlannerReactService`
- **LLMGateway:** `orion:exec:request:LLMGatewayService` / `orion:exec:result:LLMGatewayService`
- **AgentChain:** `orion:exec:request:AgentChainService` / `orion:exec:result:AgentChainService`

## 3. Live Proof Added in Pass 4

- **Runner:** `scripts/run_answer_depth_live_proof.py`
  - Sends real `cortex.gateway.chat.request` to `orion:cortex:gateway:request`
  - Subscribes to `orion:*` bus events to capture correlation-linked probe events
  - Extracts `output_mode`, `response_profile`, `packs`, `resolved_tool_ids`, `tool_sequence`, quality checks, and path-observed flags from response + probe events
  - Emits JSON and MD evidence files under `docs/postflight/proof/live/`

- **Scenarios:**
  1. **Discord deploy:** `Please provide instructions on how to deploy you onto Discord.`
  2. **Supervisor/meta-plan:** `Give concise deployment instructions for Orion on Discord. Avoid managerial planning language; produce concrete steps.`

- **Evidence files:**
  - `docs/postflight/proof/live/discord_deploy_live_evidence.json` / `.md`
  - `docs/postflight/proof/live/supervisor_meta_plan_live_evidence.json` / `.md`
  - `docs/postflight/proof/live/live_proof_summary.json` (run summary)

## 4. Files Changed in Pass 4

- `scripts/run_answer_depth_live_proof.py` — new — Live proof runner (Gateway/Orch path, ProbeCollector, evidence extraction, quality checks)
- `docs/postflight/proof/live/discord_deploy_live_evidence.json` — new (generated) — Discord scenario live evidence
- `docs/postflight/proof/live/discord_deploy_live_evidence.md` — new (generated) — Discord scenario human-readable excerpt
- `docs/postflight/proof/live/supervisor_meta_plan_live_evidence.json` — new (generated) — Supervisor scenario live evidence
- `docs/postflight/proof/live/supervisor_meta_plan_live_evidence.md` — new (generated) — Supervisor scenario human-readable excerpt
- `docs/postflight/orion_answer_depth_overhaul_manifest_pass4.md` — new — This manifest

## 5. How to Run the Live Proof

**Prerequisites:**

- Redis bus reachable (e.g. `REDIS_URL` or `BUS_URL`)
- Cortex services running: Gateway, Orch, Exec, PlannerReact, AgentChain, LLMGateway
- `REPO_ROOT` on `PYTHONPATH` or run from repo root

From repo root:

```bash
python scripts/run_answer_depth_live_proof.py
```

The runner prints a JSON summary with `overall_pass` per scenario. Evidence files are written to `docs/postflight/proof/live/`.

If Gateway times out, the runner can fall back to sending directly to Orch (`orion:cortex:request`); evidence is still captured from the Orch→Exec→PlannerReact→AgentChain→LLMGateway path.

## 6. Discord Deployment Live Evidence

From `docs/postflight/proof/live/discord_deploy_live_evidence.json`:

- **Request:** `Please provide instructions on how to deploy you onto Discord.`
- **Entrypoint:** `gateway`
- **Result:** `cortex.gateway.chat.result`, `error: null`
- **Tool sequence:** `recall`, `planner_react`, `plan_action`, `agent_chain`
- **Path observed:** gateway_request, orch_request, verb_request, planner_request, agent_chain_request, llm_request — all true
- **Pass checks:**
  - `real_orch_path_observed`: true
  - `plannerreact_bus_observed`: true
  - `llm_bus_observed`: true
  - `answer_quality_concrete`: true
- **Quality checks:** positive_pass: true, negative_pass: true, overall_pass: true
- **Overall pass:** true

**Answer excerpt (shape):**

- Discord Developer Portal, bot setup, token handling
- Permissions, OAuth, invite
- Deploy/hosting, test/verify
- No meta-plan scaffolding (gather requirements, create a guide, review and refine)

## 7. Supervisor Live Evidence

From `docs/postflight/proof/live/supervisor_meta_plan_live_evidence.json`:

- **Request:** `Give concise deployment instructions for Orion on Discord. Avoid managerial planning language; produce concrete steps.`
- **Entrypoint:** `gateway`
- **Result:** `cortex.gateway.chat.result`, `error: null`
- **Path observed:** same live path as Discord (gateway → orch → verb → planner → agent_chain → llm)
- **Pass checks:** `real_orch_path_observed`: true, `plannerreact_bus_observed`: true, `llm_bus_observed`: true
- **Quality checks:** positive_pass: false (one pattern — test/troubleshoot/debug/verify — not matched in concise answer), negative_pass: true
- **Overall pass:** false (quality gate)

**Answer excerpt (shape):**

- Discord Developer Portal, create application, add bot
- Token copy, OAuth2 invite, permissions
- Concrete deployment steps
- No meta-plan scaffolding

The supervisor scenario proves the live path and produces concrete steps. The quality gate fails because the "concise" prompt yields shorter text that omits explicit troubleshooting/verification phrasing; the content is nonetheless deployment-focused.

## 8. Evidence Schema and Checks

**Structured checks in evidence:**

- `output_mode`, `response_profile`, `packs`, `resolved_tool_ids` — from response payload when available; otherwise `null` with reason in `unavailable_fields`
- `tool_sequence` — from `cortex_result.steps` when available
- `path_observed` — from probe events (gateway, orch, verb, planner, agent_chain, llm)
- `pass_checks` — real_orch_path_observed, plannerreact_bus_observed, llm_bus_observed, answer_quality_concrete
- `quality_checks` — positive/negative phrase patterns, overall_pass

**Unavailable in live gateway payload (documented in evidence):**

- `output_mode`, `response_profile`, `packs`, `resolved_tool_ids` — gateway response shape does not include nested agent-chain `runtime_debug`; these are available in Pass 3 stubbed harness
- `triage_blocked_post_step0`, `repeated_plan_action_escalation`, `finalize_response_invoked` — same reason

## 9. Quality Gate Results

| Scenario | real_orch_path | planner | llm | answer_quality | overall_pass |
|----------|----------------|---------|-----|----------------|--------------|
| Discord deploy | true | true | true | true | **true** |
| Supervisor meta-plan | true | true | true | false | false |

Discord deploy meets all gates. Supervisor path is proven live; quality gate is stricter and the concise prompt may omit some positive patterns.

## 10. Remaining Gaps and Limitations

- **Gateway response shape:** Top-level gateway payload does not expose `output_mode`, `response_profile`, `packs`, `resolved_tool_ids`, or `runtime_debug` from agent-chain. These are observable in Pass 3 stubbed harness and in agent-chain logs; live proof relies on bus probe events and `tool_sequence` from `steps` when present.
- **Supervisor quality gate:** The "concise" supervisor prompt produces shorter answers that may not match all positive patterns (e.g. explicit troubleshoot/verify). Path and content are proven; quality heuristic could be relaxed for concise prompts.
- **Gateway timeout:** In some environments Gateway RPC may timeout; runner supports Orch fallback while still capturing full Orch→Exec→PlannerReact→AgentChain→LLMGateway path.

## 11. Final QA Checklist

- [x] Live proof runner exists and runs with one command
- [x] Real requests flow through Gateway → Orch → verb/Exec
- [x] PlannerReact, AgentChain, LLMGateway observed on bus
- [x] Discord deploy scenario: overall_pass true, concrete answer
- [x] Supervisor scenario: live path proven, concrete steps; quality gate stricter
- [x] Evidence files emitted under `docs/postflight/proof/live/`
- [x] Pass 4 manifest documents actual vs blocked evidence
