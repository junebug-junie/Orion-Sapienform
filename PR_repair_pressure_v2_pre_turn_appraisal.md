## Summary

- Add pre-turn appraisal bus schemas, channels, and registry entries for request/result RPC between hub and cortex-exec.
- Implement repair_pressure_v2 paradigm: paired turn window, logprob probe scorer, kind-aware contract delta assembly, and weights YAML.
- Wire cortex-exec Rabbit listener + hub RPC client into chat handlers (REST + WebSocket); skip legacy phrase_match pipeline when v2 enabled.
- Add eval harness with transcript fixtures and regression tests across substrate, cortex-exec, and hub.
- Apply code review fixes: shared wiring helper, fail-closed on missing weights/bus, no legacy fallback when v2 enabled.

## Outcome moved

Orion can appraise repair pressure **before** the speech turn using logprob evidence instead of post-hoc phrase matching. Hub operators can enable the rail with `ENABLE_PRE_TURN_APPRAISAL=true` without breaking the default (off) path.

## Current architecture

Before this patch, repair pressure was inferred post-turn via phrase_match in the substrate effect pipeline and speech wiring. No pre-turn RPC, no logprob probe paradigm, and no kind-gated contract assembly for v2.

## Architecture touched

- **Contracts:** `orion/schemas/pre_turn_appraisal.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`
- **Substrate:** `orion/substrate/appraisal/` (turn_window, probe/logprob_runner, contract delta, repair_pressure_v2 paradigm, eval harness)
- **Cortex-exec:** second Rabbit listener, `pre_turn_appraisal.py` RPC handler, settings/env
- **Hub:** RPC client, wiring helper, chat handler integration (api_routes + websocket_handler), legacy pipeline guard

## Files changed

- `orion/schemas/pre_turn_appraisal.py`: request/result/evidence schemas for pre-turn appraisal RPC
- `orion/bus/channels.yaml` + `orion/schemas/registry.py`: bus channels and schema registration
- `orion/substrate/appraisal/turn_window.py`: paired turn window builder
- `orion/substrate/appraisal/probe/logprob_runner.py`: logprob probe scorer (pure functions)
- `orion/substrate/appraisal/contract.py`: `assemble_repair_contract_delta()` for kind-gated v2 rules
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: paradigm plugin rail
- `config/substrate/repair_pressure_weights.v2.yaml`: v2 weights config
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`: RPC handler
- `services/orion-hub/scripts/pre_turn_appraisal_client.py` + `pre_turn_appraisal_wiring.py`: hub client and wiring
- `services/orion-hub/scripts/api_routes.py` + `websocket_handler.py`: chat integration
- `services/orion-hub/scripts/substrate_effect_pipeline.py`: skip legacy when v2 enabled
- Tests and eval fixtures across substrate, cortex-exec, and hub

## Schema / bus / API changes

- Added: `PreTurnAppraisalRequestV1`, `PreTurnAppraisalResultV1`, `RepairEvidenceV1`, `KindProbeScoreV1` schemas
- Added: `orion:cortex:pre_turn_appraisal:request` and `orion:cortex:pre_turn_appraisal:result:{correlation_id}` channels
- Behavior changed: hub chat handlers call pre-turn appraisal RPC when `ENABLE_PRE_TURN_APPRAISAL=true`; legacy phrase_match skipped in that mode
- Compatibility notes: default off on hub (`ENABLE_PRE_TURN_APPRAISAL=false`); cortex-exec handler always serves RPC when hub sends requests

## Env/config changes

- Added keys (hub): `ENABLE_PRE_TURN_APPRAISAL`, `PRE_TURN_APPRAISAL_PARADIGMS`, `PRE_TURN_APPRAISAL_TIMEOUT_MS`, `CHANNEL_PRE_TURN_APPRAISAL_REQUEST`, `CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX`
- Added keys (cortex-exec): `REPAIR_PRESSURE_WEIGHTS_V2_PATH`, `REPAIR_PRESSURE_PROBE_ROUTE`, `CHANNEL_PRE_TURN_APPRAISAL_REQUEST`, `CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX`
- `.env_example` updated: yes (hub + cortex-exec)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not run in worktree (operator should sync after merge)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_pre_turn_appraisal_schemas.py tests/test_turn_window.py tests/test_logprob_probe_runner.py tests/test_repair_pressure_contract_v2.py tests/test_repair_pressure_v2_paradigm.py -q
pytest orion/substrate/evals/repair_pressure_v2_eval.py -q
pytest services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py -q
pytest services/orion-hub/tests/test_pre_turn_appraisal_wiring.py services/orion-hub/tests/test_handle_chat_request_substrate_effect.py -q
# Combined gate: 28 passed
```

## Evals run

```text
pytest orion/substrate/evals/repair_pressure_v2_eval.py -q
# 3 transcript fixtures: grounding_negative, neutral, ops_frustration_positive
```

## Docker/build/smoke checks

```text
Not run — live bus/Docker path UNVERIFIED in this session.
docker compose config validated implicitly via existing compose files (no compose changes).
```

## Review findings fixed

- Finding: duplicated wiring logic between REST and WebSocket handlers
  - Fix: extracted `run_pre_turn_appraisal_wiring()` and `repair_pressure_grammar_scalars()` in `pre_turn_appraisal_wiring.py`
  - Evidence: hub wiring tests pass
- Finding: WebSocket fell back to legacy when v2 enabled but bus missing
  - Fix: fail-closed path, log warning when bus is None
  - Evidence: `test_pre_turn_appraisal_wiring_v2_disabled_when_no_bus`
- Finding: missing weights YAML silently degraded
  - Fix: fail-closed with `weights_file_missing` note
  - Evidence: `test_repair_pressure_v2_paradigm_weights_file_missing`

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Enable on hub after smoke:

```bash
ENABLE_PRE_TURN_APPRAISAL=true
PRE_TURN_APPRAISAL_PARADIGMS=repair_pressure
PRE_TURN_APPRAISAL_TIMEOUT_MS=800
```

Also run from repo root after merge:

```bash
python scripts/sync_local_env_from_example.py
```

## Risks / concerns

- Severity: low
- Concern: live Docker/bus RPC path not smoke-tested in this PR
- Mitigation: run restart + manual chat smoke after deploy

## PR link

https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/repair-pressure-v2-pre-turn-appraisal?expand=1
