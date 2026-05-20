# PR: Mind semantic synthesis survival (`feat/mind-semantic-synthesis-survival`)

**Base:** `feat/mind-llm-timeout-hardening`  
**Branch:** `feat/mind-semantic-synthesis-survival`  
**Worktree:** `.worktrees/feat-mind-semantic-synthesis-survival`

## Summary

Mind’s semantic LLM path was receiving Gateway replies, but **all claims were removed by guardrails** because `evidence_refs` from the model often did not match the evidence pack (e.g. `projection:0` vs pack refs like `cognitive_projection:1` after `current_turn:0`). The pipeline then failed open with `semantic_synthesis_empty` / `deterministic_shadow` with little explainability.

This PR adds **deterministic evidence-ref resolution**, **filter telemetry**, a distinct **`semantic_synthesis_empty`** error code, and Hub phase/derailment display for retained vs filtered counts—without weakening label, source-tag, or “all refs must resolve” guardrails.

## Root cause

| Question | Answer |
|----------|--------|
| Does raw LLM output contain claims before filtering? | **Yes** when Gateway returns valid JSON (`parse_ok=true`). Prior live smoke `4d6c5866-efdd-492f-98fd-ec092f0d178f` had claims then `validation_ok=false`, `status=filtered`. |
| Why `semantic_synthesis_empty`? | `engine.py` fails open when `synthesis.claims` is empty after `filter_semantic_claims`. Diagnostic token `semantic_synthesis_empty` when no LLM error string is present. |
| Dominant filter reason (historical) | **`unsupported_or_weak`** — `evidence_refs` not in `evidence_refs_in_pack(pack)` (wrong kind prefix, per-kind `:0` vs global index, or missing refs). |
| Authorization blocked because | `evaluate_stance_authorization` requires `synthesis.claims` and `frontier.selected`; empty claims → `no_evidence_backed_claims` → fail-open shadow. |
| `claim_kind` mismatch? | **No** — guardrails do not filter on `claim_kind`; `situation_claim` is schema-valid. |
| `evidence_refs` / projection refs? | **Yes** — pack uses `{source_kind}:{len(items)}` at append time; projection items are not `:0` when `current_turn:0` exists. |

## Exact conditions

**`semantic_synthesis_empty` (fail-open):**

```python
# engine.py — after run_semantic_synthesis
if synthesis is None: → sem_err / semantic_synthesis_failed / semantic_schema_invalid
if not synthesis.claims: → error_code="semantic_synthesis_empty"
```

**Claim removed in guardrails:**

- `source_tag_not_semantic` — label matches infrastructure tags  
- `unsupported_or_weak` — no refs, any original ref unresolvable after normalization, or normalized ref not in pack  
- `empty` — missing label/summary  

**Stance handoff authorized (`authorized_for_stance_use=true`):**

- Non-empty retained claims, non-empty `frontier.selected`, valid stance payload, no source-tag leakage, not boilerplate-dominated, no `llm_errors`

**Semantic advisory (phase telemetry only):**

- `authorized_for_stance_use=true` on phase telemetry when retained claims exist; `authorized_for_stance_skip=false` until full handoff authorizes

## Files changed

| File | Change |
|------|--------|
| `services/orion-mind/app/evidence.py` | `resolve_evidence_ref_for_pack`, alias map, `normalize_evidence_refs_for_pack` |
| `services/orion-mind/app/guardrails.py` | `SemanticClaimFilterStats`, strict unresolved-ref check, `evaluate_semantic_handoff_authorization` |
| `services/orion-mind/app/synthesis.py` | Prompt lists `evidence_refs_available`; filter telemetry; legacy default `current_turn_claim` |
| `services/orion-mind/app/phase_telemetry.py` | `raw_claim_count`, `retained_claim_count`, `filter_reasons_by_count`, etc. |
| `services/orion-mind/app/engine.py` | Split `semantic_synthesis_empty` vs transport/schema failures |
| `services/orion-mind/.env_example` | `ORION_BUS_URL` for host-network stacks |
| `services/orion-hub/static/js/mind_provenance.js` | Phase row filter counts; derailment dedupe for filter-empty |
| `services/orion-mind/tests/test_mind_llm_pipeline.py` | Survival, alias, mixed-ref suppression, empty telemetry |
| `services/orion-hub/tests/test_mind_provenance_normalizer.py` | Hub filter count + derailment tests |

## Before / after artifact (conceptual)

**Before (live fail-open):**

```json
{
  "mind_quality": "shadow_synthesis",
  "machine_contract": {
    "mind.llm_fail_open_to_deterministic": true,
    "mind.llm_synthesis_failed_phase": "semantic_synthesis",
    "mind.llm_synthesis_error_code": "semantic_synthesis_failed",
    "mind.semantic_claim_count": 0,
    "mind.phase_telemetry": [{
      "phase_name": "semantic_synthesis",
      "ok": true,
      "parse_ok": true,
      "validation_ok": false,
      "status": "filtered"
    }]
  },
  "diagnostics": ["semantic_synthesis_empty"]
}
```

**After (happy path — unit-tested; live model-dependent):**

```json
{
  "mind_quality": "meaningful_synthesis",
  "machine_contract": {
    "mind.llm_fail_open_to_deterministic": false,
    "mind.semantic_claim_count": 1,
    "mind.authorized_for_stance_use": true,
    "mind.phase_telemetry": [{
      "phase_name": "semantic_synthesis",
      "ok": true,
      "validation_ok": true,
      "status": "ok",
      "raw_claim_count": 1,
      "retained_claim_count": 1,
      "filtered_claim_count": 0,
      "authorization_reason": "semantic_claims_retained"
    }]
  },
  "brief": { "semantic_synthesis": { "claims": [{ "evidence_refs": ["current_turn:0"] }] } }
}
```

**After (filter-empty — explainable fail-open):**

```json
{
  "mind.llm_synthesis_error_code": "semantic_synthesis_empty",
  "mind.phase_telemetry": [{
    "raw_claim_count": 2,
    "retained_claim_count": 0,
    "filtered_claim_count": 2,
    "filter_reasons_by_count": { "unsupported_or_weak": 2 },
    "authorization_reason": "semantic_synthesis_empty",
    "error": "semantic_synthesis_empty"
  }],
  "diagnostics": [
    "claims_generated=2",
    "claims_filtered=2",
    "dominant_filter_reason=unsupported_or_weak"
  ]
}
```

## Tests

```bash
cd .worktrees/feat-mind-semantic-synthesis-survival
PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py \
  services/orion-mind/tests/test_mind_governance.py \
  services/orion-hub/tests/test_mind_provenance_normalizer.py -q
# 33 passed

PYTHONPATH=. ../../venv/bin/python -m compileall \
  services/orion-mind/app services/orion-hub orion/schemas -q
# exit 0
```

## Live smoke (2026-05-20)

| Step | Result |
|------|--------|
| Rebuild `orion-mind` from worktree | OK |
| Set `ORION_BUS_URL=redis://100.92.216.81:6379/0` in mind `.env` | OK — bus RPC ~1.6s |
| Hub `mind_provenance.js` synced to running static mount | OK |
| Direct `POST /v1/mind/run` (LLM enabled) | **Partial** — model returned non-object JSON (`semantic_schema_invalid:not_object`) in this session; not a filter regression |
| Filter-empty telemetry path | Covered by unit + Hub normalizer tests |

**Operator check when model returns valid JSON:** Mind modal should show `raw=N retained=M filtered=K` on semantic phase row; filter-empty runs use warning `semantic_filtered` only (not duplicate `semantic_failed`).

## Remaining risks

1. **Live `meaningful_synthesis`** still depends on Gateway returning object-shaped `SemanticSynthesisV1` JSON and appraisal/stance phases completing within wall budget.  
2. **Alias normalization** maps `projection:0` to first projection ref only when index 0 is wrong; fabricated refs still suppress claims.  
3. **Hub static** in main repo was copied for smoke; merge PR updates `services/orion-hub/static/js/mind_provenance.js` on branch.

## Test plan

- [ ] Merge after `feat/mind-llm-timeout-hardening`  
- [ ] Deploy Mind + Hub from branch; set `ORION_BUS_URL` in mind `.env`  
- [ ] One `chat_general` turn; confirm semantic phase `retained_claim_count > 0` OR explainable `semantic_synthesis_empty` with filter counts  
- [ ] Hard-refresh Hub; verify phase table and derailment cards
