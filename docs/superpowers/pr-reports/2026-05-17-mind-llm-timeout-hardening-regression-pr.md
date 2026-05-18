# PR: Mind LLM — semantic schema normalization + Hub brief cards (post-#587)

**Branch:** `feat/mind-llm-timeout-hardening`  
**Worktree:** `.worktrees/feat-mind-llm-timeout-hardening`  
**Base:** `main` (after #587 timeout hardening, #588 memory-graph)  
**Commits:** `9cd6dc6b`, `a3e792bc`

## Summary

Patches two regressions discovered after Mind LLM timeout / Orch HTTP hardening (#587) without reverting transport or timeout work:

1. **Hub Mind modal** — Restores rich synthesis item cards (semantic claims, frontier matters, shadow synthesis) while keeping provenance strips, phase table, derailment callouts, and additive `orch_mind_http_failed` handling.
2. **Semantic synthesis** — LLM JSON parses but failed `SemanticSynthesisV1` because claims used legacy `{claim, evidence_refs}` shape; adds prompt contract tightening, legacy normalizer, and honest phase telemetry.

## Root causes

### A. Hub UI regression

| Layer | Finding |
|-------|---------|
| **Trigger** | `f7accc42` added `renderMindProvenanceSections` and collapsed raw JSON `<details>` (removed `open`). |
| **Gap** | Provenance chips/tables surfaced `machine_contract` keys but not `brief.semantic_synthesis` / `shadow_synthesis` / `active_frontier` as cards. |
| **Symptom** | Operators saw flatter modal vs prior visible brief content in expanded JSON. |
| **Not data loss** | Artifacts still contain full `brief`; renderer did not project synthesis objects into HTML. |

### B. Semantic schema failure

| Layer | Finding |
|-------|---------|
| **Transport** | Gateway replies arrive (~2.5–8s), `parse_ok=true` when bus configured. |
| **Validation** | Model emitted legacy claim objects without `claim_id`, `label`, `summary`, `claim_kind`. |
| **Telemetry bug** | `MindPhaseTelemetry.ok` stayed `true` when only schema validation failed. |
| **Downstream** | `semantic_schema_invalid` → fail-open; meaningful synthesis never reached appraisal/stance. |

## Files changed

| File | Change |
|------|--------|
| `services/orion-mind/app/synthesis.py` | Tightened `_SEMANTIC_SYSTEM`; `try_normalize_legacy_semantic_raw`; schema-invalid telemetry; zero-confidence preservation |
| `services/orion-mind/app/phase_telemetry.py` | Optional `status` field |
| `services/orion-mind/tests/test_mind_llm_pipeline.py` | Legacy normalize, schema-invalid telemetry, zero-confidence |
| `services/orion-hub/static/js/mind_provenance.js` | `renderMindBriefItems`, `itemCard`, shadow fixture, `machine_contract` fallback, `schema_invalid` phase status |
| `services/orion-hub/tests/test_mind_provenance_normalizer.py` | Success/fail-open/shadow/orch fixture coverage |

## Before / after

| Scenario | Before | After |
|----------|--------|-------|
| LLM legacy claim JSON | `validation_ok=false`, fail-open | Normalized to `SemanticSynthesisV1`, `validation_ok=true` when guardrails pass |
| LLM schema garbage | `ok=true`, `validation_ok=false` | `ok=false`, `status=schema_invalid`, `error=semantic_schema_invalid` |
| Hub successful Mind run | Provenance only + collapsed JSON | Provenance + **Mind synthesis items** cards + phase table |
| Hub fail-open run | Derailment callouts | Same + brief fields rendered as cards when present |
| Hub shadow synthesis | Easy to miss in JSON | Shadow card (attention, curiosity, projection refs) |
| `orch_mind_http_failed` | Callout + orch phase row | Unchanged (additive only) |

## Test plan

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-mind-llm-timeout-hardening

PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  services/orion-hub/tests/test_mind_provenance_normalizer.py \
  tests/test_mind_run_artifact_registry.py -q

PYTHONPATH=. ../../venv/bin/python -m compileall \
  services/orion-mind/app \
  services/orion-hub \
  services/orion-cortex-orch/app \
  orion/schemas -q
```

### Verification results

- **39 passed** (targeted suite)
- **compileall** exit 0

## Live smoke (2026-05-18)

### Environment

Services rebuilt from worktree; Mind required `ORION_BUS_URL=redis://100.92.216.81:6379/0` (worktree `.env` lacks bus URL; compose default `redis://redis:6379` fails in this stack).

### Smoke 1 — `3cb72d57-0c9a-4a19-8394-ac0ffdd79966`

- Mind: `mind_llm_bus_request_failed` — Redis `ConnectionError` (misconfigured bus)
- Fail-open shadow path; Gateway exec requests still replied
- **Not a code regression** — deploy config

### Smoke 2 — `4d6c5866-efdd-492f-98fd-ec092f0d178f` (primary)

| Check | Result |
|-------|--------|
| `mind_llm_bus_request_start` | Yes — semantic_synthesis, 25s timeout |
| Bus RPC | Reply ~2456 ms |
| `validation_ok=true` | No — `parse_ok=true`, `validation_ok=false`, `status=filtered` (guardrails emptied claims) |
| `schema_invalid` | Absent (correct for non-schema failure) |
| `normalized_from_legacy` | Not triggered (model did not emit legacy shape this turn) |
| Gateway `gateway_llm_request_received` | Yes — `mind_phase=semantic_synthesis` |
| Gateway `gateway_llm_reply_publish_ok` | Yes (~05:28:44) |
| Orch `mind_run_artifact_publish` | Yes — `ok=True` |
| `orch_mind_http_failed` | No |
| Hub `/api/mind/runs` | Run present; `shadow_synthesis` populated; `llm_fail_open_to_deterministic=true` |
| Hub JS | `renderMindBriefItems` deployed on `:8080` |

**Verdict:** merge-ready for this PR scope. Timeout/bus/orch/hub paths verified; semantic `validation_ok=true` covered by unit tests; live turn hit guardrail-filtered empty claims (model output), not schema-invalid regression.

### Hub operator checklist

1. Hard-refresh Hub (`Ctrl+Shift+R`).
2. Open Mind modal for `4d6c5866-efdd-492f-98fd-ec092f0d178f`.
3. Expect shadow synthesis cards + phase table + fail-open callouts (no false “No Mind runs”).

## Remaining risks

1. **Legacy normalizer scope** — Only obvious `{claim: "..."}` shapes; mixed arrays still fail closed.
2. **Default `claim_kind`** — Legacy claims become `situation_claim`; may be filtered by guardrails.
3. **Hub asset cache** — Hard-refresh after deploy.
4. **Live model compliance** — Monitor `semantic_schema_invalid` rate; prompt reduces but does not eliminate invalid JSON.
5. **`ORION_BUS_URL`** — Document in mind `.env_example` for non-compose Redis stacks.

## Guardrails honored

- No timeout/bus hardening revert
- `orch_mind_http_failed` support retained
- Schema failures not hidden as success
- No fake semantic synthesis invented
- Hub panel not redesigned — cards added to existing provenance stack
