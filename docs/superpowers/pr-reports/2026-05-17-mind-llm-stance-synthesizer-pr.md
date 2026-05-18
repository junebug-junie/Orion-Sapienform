# PR: Mind LLM stance synthesizer — Cortex governance hardening

**Branch:** `feat/mind-llm-stance-synthesizer`  
**Worktree:** `.worktrees/feat-mind-llm-stance-synthesizer`  
**Commits (governance pass):** `6b41e280`, `0bcfa4d4`  
**Base:** `origin/main`

## Summary

Resurrects Mind as the pre-speech cognition organ under explicit Cortex governance: Orch builds `MindRunRequestV1` and calls `POST /v1/mind/run`; Mind runs `semantic_synthesis` → `active_frontier_judge` → `stance_handoff` via bus → `LLMGatewayService`; Exec validates and either skips legacy `chat_stance` or falls back. `chat_general` remains verbal delivery only. Default stays safe (`MIND_LLM_SYNTHESIS_ENABLED=false`, fail-open legacy).

This pass adds **budget enforcement**, **request identity propagation** into gateway envelopes, **per-phase telemetry** on `machine_contract`, and **defense-in-depth** on the Exec stance shortcut.

## Postflight verdict

**A. Cortex-governed Mind organ — acceptable after hardening**

Direct Mind → Gateway is **not** a boundary violation. No invariant requires all turn-time LLM calls to run inside cortex-exec step machinery. Mind is invoked only by Orch over HTTP, gated by `MIND_LLM_SYNTHESIS_ENABLED`, bounded by `MindRunPolicyV1`, traced, and validated at Exec before speech-affecting stance merge.

**Why not B (Exec-mediated phases):** No hard rule that Mind cannot call the shared gateway over the bus (same channel Exec uses). Moving phases into Exec would duplicate orchestration and violate the “Mind is the cognition organ” constraint without fixing a proven invariant breach.

## Preflight audit (call path & governance)

### 1. Full call path

```
Hub/Gateway → cortex-orch (plan + build_mind_run_request)
  → POST orion-mind/v1/mind/run
  → run_mind() → run_mind_llm_synthesis() [if enabled] else run_mind_deterministic()
  → bus RPC orion:exec:request:LLMGatewayService (per phase)
  → merge_mind_brief_into_plan_metadata
  → cortex-exec call_step_services
  → _attempt_mind_handoff_chat_stance_shortcut OR legacy synthesize_chat_stance_brief
  → chat_general speech
```

### 2. `MindRunRequestV1` creation

`services/orion-cortex-orch/app/mind_runtime.py` — `build_mind_run_request()` (correlation_id, session_id, trace_id, policy, snapshot facets).

### 3. Mind LLM phase invocation

`services/orion-mind/app/engine.py` — `run_mind_llm_synthesis()` only from `run_mind()` → `/v1/mind/run`.

### 4. Identity / metadata flow

| Field | Orch request | LLM gateway envelope | Mind result |
|-------|----------------|----------------------|-------------|
| `correlation_id` | `build_mind_run_request` | `BaseEnvelope.correlation_id` (UUID) | logs + `MindLLMRequestContext` |
| `session_id` | client context | `ChatRequestPayload.session_id` + trace baggage | — |
| `trace_id` | client context | `envelope.trace` | — |
| `mind_run_id` | generated in Mind | trace + `options.mind_run_id` | `MindRunResultV1.mind_run_id` |
| phase | — | `options.mind_phase`, trace | `mind.phase_telemetry[]` |
| router profile | `policy.router_profile_id` | trace + options | `machine_contract` routes |

`causality_chain`: wired on envelope when provided on `MindLLMRequestContext`; Orch does not yet populate from client request (thin spot).

### 5. Budget enforcement

- `MindRunPolicyV1.wall_time_ms_max` → `MindRunBudget`
- Per-phase `timeout_sec = min(MIND_LLM_TIMEOUT_SEC, remaining_wall)`
- `can_run_phase()` skips appraisal/stance when wall budget insufficient
- `n_loops_max` applies to deterministic loop path; LLM path is three fixed phases (by design)

### 6. Can synthesis run outside Orch MindRun?

**No alternate production path.** Only `/v1/mind/run` + `MIND_LLM_SYNTHESIS_ENABLED=true`. Tests may call `run_mind()` directly with a constructed request (equivalent contract).

### 7. Can Mind bypass Exec or speak?

**No.** Mind outputs stance payload / handoff artifacts only. Exec shortcut requires `meaningful_synthesis` + authorization + valid `ChatStanceBrief`. `chat_general` is unchanged.

### 8. Route governance

Routes from settings (`MIND_*_MODEL_ROUTE`). Empty route → fail-open (no LLM). Effective routes recorded on `machine_contract` (`mind.semantic_route`, etc.) and `mind.phase_telemetry`.

### 9. Artifact inspectability

`machine_contract` now includes: phase telemetry (route, model, elapsed_ms, ok/error, token usage), semantic/frontier counts and labels, authorization reasons, `mind.fallback_reason`, wall elapsed, `mind.llm_synthesis_enabled`.

### 10. Tests vs invariants

`test_mind_governance.py` + existing pipeline/orch/exec tests (42 passed in scoped suite).

## Files changed (governance pass)

| Area | Paths |
|------|--------|
| Budget / identity / telemetry | `services/orion-mind/app/{budget,llm_context,phase_telemetry}.py` |
| Mind pipeline | `services/orion-mind/app/{engine,llm_client,synthesis,appraisal,stance_handoff,settings}.py` |
| Ops | `services/orion-mind/{.env_example,docker-compose.yml}` (+ local `.env` parity, gitignored) |
| Exec seam | `services/orion-cortex-exec/app/executor.py` |
| Tests | `services/orion-mind/tests/test_mind_governance.py`, `services/orion-cortex-exec/tests/test_mind_handoff_shortcut.py` |

## Invariants enforced (after hardening)

1. LLM synthesis only inside valid `MindRunRequestV1` on `/v1/mind/run` with `MIND_LLM_SYNTHESIS_ENABLED`.
2. Original `correlation_id` / `session_id` / `trace_id` / `mind_run_id` / phase propagated to gateway.
3. `MindRunPolicyV1.wall_time_ms_max` bounds total run; phases respect remaining budget.
4. Routes configured + recorded; invalid empty route fails open to legacy.
5. Mind does not produce user speech; stance handoff only.
6. Exec requires `meaningful_synthesis` + `mind_authorized_for_stance_skip` + valid brief.
7. LLM/guardrail failures → `authorized_for_stance_use=false`, legacy stance preserved.
8. Phase telemetry + fallback reasons on `machine_contract` / Orch metadata.

## Tests added / strengthened

- `test_mind_governance.py`: disabled=no LLM calls; identity propagation; wall budget skips phases; invalid route fail-open; `identity_yaml` cannot authorize; shadow never authorizes.
- `test_mind_handoff_shortcut.py`: requires `mind_authorized_for_stance_skip`; invalid payload with authorization still blocks skip.

## Verification

```bash
cd .worktrees/feat-mind-llm-stance-synthesizer
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest \
  services/orion-mind/tests/ \
  services/orion-cortex-orch/tests/test_mind_evidence_facets.py \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  services/orion-cortex-exec/tests/test_mind_handoff_shortcut.py -q
```

**Result:** 42 passed (exit 0).

## Remaining thin spots

- No live bus/gateway integration test (unit tests use `FakeMindLLMClient`).
- `causality_chain` not yet passed from Orch into Mind LLM context.
- Partial LLM result on mid-run budget skip returns `invalid_handoff` LLM artifact (not deterministic fail-open) — Exec still runs legacy stance (safe).
- `MIND_LLM_FAIL_OPEN_LEGACY=false` error paths lightly tested.
- Thread-per-call bus client in sync handler (acceptable behind feature flag).

## Enable Qwen3 lanes (env)

In `services/orion-mind/.env` (mirror `.env_example`):

| Variable | Suggested | Lane |
|----------|-----------|------|
| `MIND_LLM_SYNTHESIS_ENABLED` | `true` | — |
| `MIND_SEMANTIC_MODEL_ROUTE` | `quick` | Qwen3 8B non-thinking |
| `MIND_APPRAISAL_MODEL_ROUTE` | `metacog` | Qwen3 30B/32B thinking (`MIND_LLM_THINKING_APPRAISAL=true`) |
| `MIND_STANCE_MODEL_ROUTE` | `chat` | Qwen3 30B non-thinking |
| `MIND_LLM_FAIL_OPEN_LEGACY` | `true` | Deterministic Mind + Exec legacy stance on failure |
| `MIND_LLM_PHASE_SAFETY_MS` | `50` | Wall-clock reserve before refusing a phase |
| `ORION_BUS_URL` | `redis://…` | Bus to gateway |
| `MIND_LLM_INTAKE_CHANNEL` | `orion:exec:request:LLMGatewayService` | Same as Exec |

Gateway `LLM_GATEWAY_ROUTE_TABLE_JSON` must define `quick`, `metacog`, `chat`.

## Manual smoke checklist

- [ ] `MIND_LLM_SYNTHESIS_ENABLED=true`, bus + gateway up
- [ ] `chat_general` with `mind_enabled=true`
- [ ] Logs: three Mind phases share same `correlation_id`
- [ ] Authorized handoff: `exec <- mind_handoff stance_payload`
- [ ] Break semantic route → legacy `chat_stance` still speaks
- [ ] `identity_yaml` not in `mind.active_frontier_top_labels`

## Architectural decisions still open

- Whether partial LLM runs on budget exhaustion should fail-open to deterministic Mind (today: return degraded LLM result for inspectability).
- Whether Orch should attach `causality_chain` from hub/client envelope into `MindRunRequestV1` for full Conjourney lineage on Mind LLM calls.

## Test plan (reviewer)

- [ ] Confirm default `MIND_LLM_SYNTHESIS_ENABLED=false` in deployed mind service
- [ ] Staged enable with Qwen routes on gateway
- [ ] Hub Mind tab shows phase telemetry when synthesis runs
- [ ] Regression: shadow_synthesis / contract-only never skip Exec stance
