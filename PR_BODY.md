## Summary

Follow-up to #587 (Mind LLM timeout hardening). Fixes two post-hardening regressions without reverting transport, timeout, or `orch_mind_http_failed` work:

1. **Hub Mind modal** — Restores rich synthesis item cards (semantic claims, frontier matters, shadow synthesis) while keeping provenance strips, phase table, derailment callouts, and additive `orch_mind_http_failed` handling.
2. **Semantic synthesis** — LLM JSON parsed but failed `SemanticSynthesisV1` on legacy `{claim, evidence_refs}` shapes; adds prompt contract tightening, legacy normalizer, and honest phase telemetry (`ok=false`, `status=schema_invalid` on real schema failure; `status=filtered` when guardrails empty claims).

## Root causes

| Area | Issue | Fix |
|------|-------|-----|
| Hub UI | `f7accc42` provenance stack did not project `brief.semantic_synthesis` / `shadow_synthesis` into HTML | `renderMindBriefItems` + `itemCard` in `mind_provenance.js` |
| Semantic | Legacy claim objects missing `claim_id`, `label`, `summary`, `claim_kind` | `try_normalize_legacy_semantic_raw()` + tightened `_SEMANTIC_SYSTEM` |
| Telemetry | `ok=true` when only schema validation failed | `ok=false`, `status=schema_invalid` on invalid schema |

## Live smoke (2026-05-18)

**Correlation:** `4d6c5866-efdd-492f-98fd-ec092f0d178f` (Mind-enabled `chat_general`, session `smoke2-4daa18431561`)

| Layer | Result |
|-------|--------|
| Mind bus | `mind_llm_bus_request_start`; RPC reply ~2.5s (within `MIND_LLM_TIMEOUT_SEC=25`) |
| Gateway | `gateway_llm_request_received` (`mind_phase=semantic_synthesis`); `gateway_llm_reply_publish_ok` |
| Orch | `mind_run_artifact_publish` `ok=True`; no `orch_mind_http_failed` |
| Hub API | Run listed; `shadow_synthesis` + fail-open flags; deployed `renderMindBriefItems` JS |
| Semantic happy path | `parse_ok=true`, `validation_ok=false`, `status=filtered` (guardrails emptied claims) — **no `schema_invalid`**; unit tests cover normalize + schema-invalid telemetry |

**Deploy note:** set `ORION_BUS_URL` in `services/orion-mind/.env` to stack Redis (not compose default `redis://redis:6379`) before `docker compose up mind`.

## Tests

```bash
cd .worktrees/feat-mind-llm-timeout-hardening
PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py \
  services/orion-hub/tests/test_mind_provenance_normalizer.py \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  tests/test_mind_run_artifact_registry.py -q
```

**39 passed** (targeted suite); `compileall` OK.

## Deploy

```bash
cd services/orion-mind && docker compose up -d --build mind
cd ../orion-hub && docker compose up -d --build hub-app
```

Hard-refresh Hub (`Ctrl+Shift+R`) so `mind_provenance.js` loads.
