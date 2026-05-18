# Mind LLM timeout hardening & fail-open observability

## Summary

Fixes the proven Orch→Mind abandonment chain where `MIND_LLM_TIMEOUT_SEC=90` exceeded `ORION_MIND_TIMEOUT_SEC=45`, leaving Hub with no Mind row. Also catalogs Mind LLM reply channels so gateway replies survive `ORION_BUS_ENFORCE_CATALOG=true`, adds structured diagnostics on Mind→Gateway→Mind, and publishes a synthetic `MindRunArtifactV1` when Orch HTTP to Mind fails.

## Before / after

| Scenario | Before | After |
|----------|--------|-------|
| Mind LLM wait vs Orch HTTP | Mind waits 90s; Orch gives up at 45s | Mind LLM phase 25s; Orch HTTP 45s |
| Gateway reply to `orion:mind:llm:reply:*` | Uncataloged → publish may fail under enforcement | Cataloged as `ChatResultPayload` |
| Orch `ReadTimeout` on `/v1/mind/run` | Log only; Hub “No Mind runs” | Synthetic artifact on `orion:mind:artifact` |
| Hub provenance | `llm_fail_open` only | + `orch_mind_http_failed` callout & phase row |

## Files changed

- `orion/bus/channels.yaml` — `orion:mind:llm:reply:*`; `orion-mind` LLM intake producer
- `orion/core/bus/bus_service_chassis.py` — gateway reply publish diagnostics
- `services/orion-mind/app/settings.py`, `main.py`, `llm_client.py`
- `services/orion-mind/.env`, `.env_example`, `docker-compose.yml`
- `services/orion-cortex-orch/app/mind_runtime.py`, `orchestrator.py`, `main.py`
- `services/orion-cortex-orch/.env_example`
- `services/orion-llm-gateway/app/main.py`
- `services/orion-hub/static/js/mind_provenance.js`
- `tests/test_mind_llm_bus_catalog.py`, `tests/test_mind_run_artifact_registry.py`
- `services/orion-cortex-orch/tests/test_mind_orch.py`
- `services/orion-hub/tests/test_mind_provenance_normalizer.py`

## Test plan

### Unit tests (from worktree root)

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-mind-llm-timeout-hardening
PYTHONPATH=. ../../venv/bin/python -m pytest \
  tests/test_mind_llm_bus_catalog.py \
  tests/test_mind_run_artifact_registry.py \
  tests/test_channel_prefix_guardrail.py \
  tests/test_exec_result_channel_catalog_specificity.py -q

PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py \
  services/orion-mind/tests/test_mind_governance.py \
  services/orion-mind/tests/test_http_contract.py -q

PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-cortex-orch/tests/test_mind_orch.py \
  services/orion-cortex-orch/tests/test_mind_evidence_facets.py -q

PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-hub/tests/test_mind_hub_tab.py \
  services/orion-hub/tests/test_mind_routes.py \
  services/orion-hub/tests/test_mind_provenance_normalizer.py -q
```

### Compile

```bash
PYTHONPATH=. ../../venv/bin/python -m compileall \
  services/orion-mind/app \
  services/orion-cortex-orch/app \
  services/orion-llm-gateway/app \
  orion/core/bus -q
```

### Docker smoke (Juniper)

```bash
# From each service dir after pulling branch / copying .env
cd services/orion-mind && docker compose up -d --build mind
cd ../orion-cortex-orch && docker compose up -d --build orch  # service name per compose
cd ../orion-llm-gateway && docker compose up -d --build
cd ../orion-hub && docker compose up -d --build  # if Hub JS changed
```

Trigger one `chat_general` turn with `mind_enabled: true`, then:

```bash
CORR=<correlation_id>
docker logs orion-mind 2>&1 | rg "mind_llm_bus_request_start|mind_llm_bus_request_failed|mind_llm_run_start" | rg "$CORR"
docker logs <llm-gateway-container> 2>&1 | rg "gateway_llm_request_received|gateway_llm_reply_publish" | rg "$CORR"
docker logs <orch-container> 2>&1 | rg "orch_mind_http_failed|mind_run_artifact_publish" | rg "$CORR"
```

Hub: open Mind runs modal for `$CORR` — expect either normal Mind row, `llm_fail_open` derailment, or **Orch timed out waiting for Mind** (not empty list).

## Verification results (agent)

- Catalog/registry/guardrail: **12 passed**
- Mind: **21 passed**
- Orch: **18 passed**
- Hub: **16 passed**
- compileall: **exit 0**
- Live docker smoke: **not run** (no stack in agent environment)

## Remaining risks

- Gateway upstream `quick`/`metacog`/`chat` workers can still be slow; 25s fail-open is bounded but may still yield deterministic fallback.
- SQL writer must consume `orion:mind:artifact` for Hub DB row; bus publish alone is insufficient if writer is down.
- Synthetic artifact does not prove Mind process received the HTTP request if failure is connect-level before Mind handles.

## Env contract

| Service | Key | Value |
|---------|-----|-------|
| orion-mind | `MIND_LLM_TIMEOUT_SEC` | `25` |
| orion-mind | `MIND_LLM_PHASE_SAFETY_MS` | `50` |
| orion-mind | `ORION_MIND_TIMEOUT_SEC` | `45` (optional, for hierarchy warning) |
| orion-cortex-orch | `ORION_MIND_TIMEOUT_SEC` | `45` |
