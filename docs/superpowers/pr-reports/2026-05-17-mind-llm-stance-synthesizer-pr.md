# PR: Mind LLM stance synthesizer (pre-speech organ)

**Branch:** `feat/mind-llm-stance-synthesizer`  
**Worktree:** `.worktrees/feat-mind-llm-stance-synthesizer`  
**Base:** `origin/main` (`d554e9fa`)

## Summary

Resurrects Mind as the real pre-speech cognition organ: Orch prepares a bounded evidence pack, Mind runs three LLM steps (semantic synthesis → active frontier judge → stance handoff), Exec validates and either skips legacy `chat_stance` or falls back. `chat_general` remains verbal delivery only. Default remains deterministic/legacy-safe (`MIND_LLM_SYNTHESIS_ENABLED=false`).

## Files changed

| Area | Paths |
|------|--------|
| Contracts | `orion/mind/synthesis_v1.py`, `orion/mind/v1.py` |
| Mind service | `services/orion-mind/app/{evidence,guardrails,llm_client,synthesis,appraisal,stance_handoff,engine,main,settings}.py` |
| Mind ops | `services/orion-mind/.env_example`, `docker-compose.yml`, `requirements.txt` |
| Orch | `services/orion-cortex-orch/app/mind_runtime.py` |
| Exec | `services/orion-cortex-exec/app/executor.py` |
| Tests | `services/orion-mind/tests/test_mind_llm_pipeline.py`, `services/orion-cortex-orch/tests/test_mind_evidence_facets.py` |

**Not modified:** `orion/spark/concept_induction/*` (read-only per constraint).

## Call path

### Before

```
Hub → Orch (projection + MindRunRequest) → Mind (deterministic contract / shadow_synthesis)
     → Exec merge → legacy chat_stance LLM (always unless manual skip metadata)
     → chat_general speech
```

### After (when `MIND_LLM_SYNTHESIS_ENABLED=true` and bus/LLM healthy)

```
Hub → Orch (projection + evidence facets: recall, autonomy V2 compact, social, situation, identity background)
     → Mind:
         1. build evidence pack
         2. semantic_synthesis (route: MIND_SEMANTIC_MODEL_ROUTE, e.g. quick / Qwen3 8B)
         3. active_frontier_judge (route: MIND_APPRAISAL_MODEL_ROUTE, e.g. metacog / Qwen3 30B thinking)
         4. stance_handoff (route: MIND_STANCE_MODEL_ROUTE, e.g. chat / Qwen3 30B)
         5. deterministic authorization gates
     → Exec merge:
         if mind_quality=meaningful_synthesis AND mind_authorized_for_stance_skip AND valid ChatStanceBrief
           → skip legacy synthesize_chat_stance_brief LLM
         else → legacy chat_stance
     → chat_general speech
```

## Example: mocked meaningful synthesis

```json
{
  "mind_quality": "meaningful_synthesis",
  "brief": {
    "mind_authorized_for_stance_skip": true,
    "stance_payload": {
      "conversation_frame": "playful_relational",
      "task_mode": "playful_exchange",
      "user_intent": "going to watch a show with Amanda",
      "stance_summary": "Respond to a casual relational turn about watching a show.",
      "response_priorities": ["receive warmly", "stay concise"],
      "response_hazards": ["do not invent show details"]
    },
    "machine_contract": {
      "mind.authorized_for_stance_use": true,
      "mind.semantic_claim_count": 1,
      "mind.active_frontier_selected_count": 1
    }
  }
}
```

## Example: invalid / source-tag rejection

Claim label `identity_yaml` with `evidence_refs: ["current_turn:0"]` is suppressed (`source_tag_not_semantic`). Authorization returns `authorized_for_stance_use=false`, `mind_quality=invalid_handoff`, `authorization_reasons` includes `no_evidence_backed_claims` or `source_tag_claim_present`. Exec keeps `mind_skip_stance_synthesis=false` → legacy `chat_stance` runs.

## Exec fallback behavior

| Mind outcome | `mind_skip_stance_synthesis` | Exec stance step |
|--------------|------------------------------|------------------|
| `meaningful_synthesis` + valid brief + authorized | `true` | Uses `mind_handoff.stance_payload`, skips LLM gateway stance step |
| `shadow_synthesis`, `fallback_contract_only`, `invalid_handoff`, LLM fail-open | `false` | Full legacy `synthesize_chat_stance_brief` |
| Invalid payload after authorize attempt | `false` + `mind_stance_payload_invalid` | Legacy stance |

Exec shortcut now also requires `metadata.mind_quality == "meaningful_synthesis"`.

## What remains thin

- Live bus/LLM integration tests (unit tests use `FakeMindLLMClient`).
- Orch does not run semantic/appraisal LLM work (by design).
- `curiosity_move` is prompt guidance only; not a `ChatStanceBrief` field.
- Thread-per-call bus client in sync FastAPI handler (acceptable behind feature flag).

## Enable Qwen3 lanes (env)

Set in `services/orion-mind/.env` (copy from `.env_example`):

| Variable | Suggested route | Typical model lane |
|----------|-----------------|-------------------|
| `MIND_LLM_SYNTHESIS_ENABLED` | `true` | — |
| `MIND_SEMANTIC_MODEL_ROUTE` | `quick` | Qwen3 8B non-thinking |
| `MIND_APPRAISAL_MODEL_ROUTE` | `metacog` | Qwen3 30B/32B thinking (`MIND_LLM_THINKING_APPRAISAL=true`) |
| `MIND_STANCE_MODEL_ROUTE` | `chat` | Qwen3 30B/32B non-thinking |
| `MIND_LLM_FAIL_OPEN_LEGACY` | `true` | Fall back to deterministic Mind + Exec legacy stance on failure |
| `ORION_BUS_URL` | `redis://…` | Bus to `LLMGatewayService` |
| `MIND_LLM_INTAKE_CHANNEL` | `orion:exec:request:LLMGatewayService` | Same as Exec |

Routes must exist in `LLM_GATEWAY_ROUTE_TABLE_JSON` on the gateway (e.g. `quick`, `metacog`, `chat`).

## Verification

```bash
cd .worktrees/feat-mind-llm-stance-synthesizer
PYTHONPATH=. ../venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py \
  services/orion-mind/tests/test_http_contract.py \
  services/orion-cortex-orch/tests/test_mind_evidence_facets.py \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  services/orion-cortex-exec/tests/test_mind_handoff_shortcut.py -q
```

**Result:** 31 passed (exit 0).

## Test plan (manual)

- [ ] Enable `MIND_LLM_SYNTHESIS_ENABLED=true` on orion-mind with bus + gateway reachable
- [ ] Run `chat_general` turn with `mind_enabled=true`; confirm Hub thought process shows semantic/frontier debug metadata
- [ ] Confirm authorized handoff skips Exec stance LLM (`exec <- mind_handoff stance_payload` in logs)
- [ ] Disable gateway route or set invalid synthesis → confirm legacy `chat_stance` still produces speech
- [ ] Confirm projection labels like `identity_yaml` never appear as selected frontier labels in Mind artifacts
