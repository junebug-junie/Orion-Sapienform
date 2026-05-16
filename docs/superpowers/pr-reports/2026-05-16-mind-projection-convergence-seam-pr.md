# PR: Mind projection convergence seam (Orch preflight ↔ Exec parity)

## Branch

- **Head:** `fix/mind-projection-convergence-seam`
- **Base:** `main` (or your integration branch)

---

## Summary

Mind was receiving an empty `CognitiveProjectionV1` at Orch preflight while Exec later built a rich projection for ChatStanceDebug — same shared builder, different **context and ordering**. This PR closes that seam: non-empty inline shells no longer block warm/cold rebuild, starved Mind emits explicit degraded diagnostics, Hub no longer treats contract-only Mind as healthy, and Orch preflight now injects **Exec-parity producer inputs** (identity YAML + optional recall prefetch) with side-by-side projection diagnostics.

---

## Root cause

| Gap | Detail |
|-----|--------|
| **Ordering** | Orch invokes Mind **before** Exec (`orchestrator.py`); recall, social bridge, and stance situation are materialized later in Exec. |
| **Missing ctx** | `identity_yaml` reads `orion_identity_summary` / `juniper_relationship_summary` / `response_policy_summary` from ctx (no disk); Orch did not run Exec’s `_inject_identity_context`. |
| **Recall** | `recall` producer has `pull_on_cold=False`; needs `ctx["recall_bundle"]` from `run_recall_step`. |
| **Empty inline shell** | `item_count=0` inline projection previously blocked warm/cold (fixed in earlier commits on this branch). |
| **Gateway E2E** | `POST /v1/cortex/chat` returned 500 when `causality_chain=None` was passed to `BaseEnvelope`. |

---

## What changed

| Area | Change |
|------|--------|
| **`orion/cognition/projection_context.py`** | Shared identity injection + `summarize_projection_inputs()` for Orch vs Exec parity. |
| **`orion/cognition/recall_prefetch.py`** | Optional recall RPC before Mind; populates `recall_bundle` on plan ctx. |
| **`orion/cognition/projection_builder.py`** | `summarize_projection_build()` includes `input_summary`. |
| **`services/orion-cortex-orch/app/mind_runtime.py`** | `prepare_plan_context_for_mind_projection`, resolution diagnostics, `projection_parity_diagnostics` on Mind request. |
| **`services/orion-cortex-orch/app/orchestrator.py`** | Calls preflight enrichment before `build_mind_run_request`. |
| **`services/orion-cortex-orch/app/settings.py`** | `CHANNEL_RECALL_INTAKE`, `MIND_RECALL_PREFETCH_*`. |
| **`services/orion-cortex-exec/app/chat_stance_shared_spine.py`** | Records `exec_projection_parity` (input summary + producer outcomes). |
| **`services/orion-cortex-gateway/app/bus_client.py`** | `causality_chain=causality_chain or []` on orch RPC envelope. |
| **Mind / Hub** (prior commits) | `projection_starved`, degraded facets, Hub banner for starved-before-Exec. |

---

## Operator / deploy notes

Set on **cortex-orch** (see `.env_example` and `docker-compose.yml`):

- `ORION_BUS_URL` — required for bus intake
- `ORION_MIND_BASE_URL` — e.g. `http://orion-athena-mind:6611` on `app-net`
- `CHANNEL_RECALL_INTAKE`, `MIND_RECALL_PREFETCH_ENABLED`, `MIND_RECALL_PREFETCH_TIMEOUT_SEC`

Redeploy **orch, gateway, exec, mind, hub** from the same revision before comparing live parity.

---

## Verification

```bash
cd /mnt/scripts/Orion-Sapienform
.venv/bin/python -m pytest \
  orion/cognition/tests/test_projection_context.py \
  orion/cognition/tests/test_projection_starvation_diagnostics.py \
  services/orion-cortex-orch/tests/test_mind_projection_resolver.py \
  services/orion-cortex-orch/tests/test_mind_projection_context_enrichment.py \
  services/orion-mind/tests/test_projection_starvation.py \
  -q
```

Live (after redeploy):

- `POST /v1/cortex/chat` with `metadata.mind_enabled=true` → **200** (gateway causality fix)
- `metadata.chat_stance_debug.cognitive_projection.projection.mind_projection_resolution.resolution_path` → `warm_shared_spine` with non-zero `resolved_item_count` when identity YAML is configured
- Compare `orch_preflight_input_summary` vs `exec_chat_stance_input_summary` under `projection_parity_diagnostics`

---

## Test plan

- [x] Unit: projection resolver (inline empty vs non-empty, cold rebuild, starved diagnostics)
- [x] Unit: Orch preflight context enrichment (identity + recall merge)
- [x] Unit: projection input summary / starvation diagnostics
- [x] Gateway E2E: `/v1/cortex/chat` returns 200 with `mind_enabled`
- [ ] Live: Mind HTTP succeeds with correct `ORION_MIND_BASE_URL` (no DNS failure)
- [ ] Live: recall prefetch completes within timeout (tune `MIND_RECALL_PREFETCH_TIMEOUT_SEC` if needed)

---

## Out of scope

- Mind LLM loops, prompt tuning, new ontology, curiosity schema, or a second projection builder.
