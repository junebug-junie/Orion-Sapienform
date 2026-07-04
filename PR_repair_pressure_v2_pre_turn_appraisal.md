## Summary

- Add pre-turn appraisal bus schemas, channels, and registry entries for Hub ↔ cortex-exec RPC.
- Implement `repair_pressure_v2`: paired turn window, logprob probe scorer, kind-aware contract delta, weights YAML, paradigm registry.
- Wire cortex-exec Rabbit listener + Hub RPC client into REST and WebSocket chat handlers; skip legacy `phrase_match` when v2 is on.
- Single operator gate: Hub `ENABLE_PRE_TURN_APPRAISAL` only (no cortex-exec kill switch).
- Handler resolves paradigms via `PARADIGM_REGISTRY` (not hardcoded).
- Move `RepairEvidenceV1` to `orion/schemas/repair_evidence.py` so registry imports do not eagerly load `orion.substrate` (fixes sql-writer CI collection).
- Env sync prefixes, service READMEs, eval fixtures, and regression tests.

## Outcome moved

Orion can appraise repair pressure **before** the speech turn using logprob evidence instead of post-hoc phrase matching. Operators enable with one Hub flag; legacy path remains default-off.

## Current architecture

Before this patch, repair pressure ran post-turn via `phrase_match` in the substrate effect pipeline. No pre-turn RPC, no logprob paradigm, no kind-gated v2 contract assembly.

## Architecture touched

- **Contracts:** `orion/schemas/pre_turn_appraisal.py`, `orion/schemas/repair_evidence.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`
- **Substrate:** `orion/substrate/appraisal/` (turn_window, logprob_runner, contract delta, `repair_pressure_v2`, eval harness, paradigm registry)
- **Cortex-exec:** second Rabbit listener, `pre_turn_appraisal.py` handler, settings/env
- **Hub:** RPC client, wiring helper, chat integration, legacy pipeline guard
- **Tooling:** `scripts/sync_local_env_from_example.py` — PRE_TURN / repair probe key prefixes

## Files changed

- `orion/schemas/pre_turn_appraisal.py`: request/bundle/slice schemas for pre-turn RPC
- `orion/schemas/repair_evidence.py`: `RepairEvidenceV1` + `EvidenceKind` (schema layer; avoids substrate import chain)
- `orion/substrate/appraisal/models.py`: re-exports evidence from schemas; keeps `RepairPressureAppraisalV1`
- `orion/bus/channels.yaml` + `orion/schemas/registry.py`: bus channels and schema registration
- `orion/substrate/appraisal/turn_window.py`, `probe/logprob_runner.py`, `contract.py`, `paradigms/repair_pressure_v2.py`, `paradigms/registry.py`
- `config/substrate/repair_pressure_weights.v2.yaml`
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`, `main.py`, `settings.py`, `.env_example`, README
- `services/orion-hub/scripts/pre_turn_appraisal_*.py`, `api_routes.py`, `websocket_handler.py`, `substrate_effect_pipeline.py`, settings, `.env_example`, README
- `scripts/sync_local_env_from_example.py`: sync prefixes for new env keys
- Tests + eval fixtures across substrate, cortex-exec, hub, and CI gate

## Schema / bus / API changes

- Added: `PreTurnAppraisalRequestV1`, `TurnAppraisalBundleV1`, `TurnAppraisalParadigmSliceV1`, `TurnWindowMessageV1`, `RepairEvidenceV1` (schema layer)
- Added channels: `orion:cortex:pre_turn_appraisal:request`, `orion:cortex:pre_turn_appraisal:result:{correlation_id}`
- Behavior changed: Hub calls pre-turn RPC when `ENABLE_PRE_TURN_APPRAISAL=true`; legacy `phrase_match` skipped in that mode
- Compatibility: default off on Hub; cortex-exec listener always registered

## Env/config changes

- Added keys (hub): `ENABLE_PRE_TURN_APPRAISAL`, `PRE_TURN_APPRAISAL_PARADIGMS`, `PRE_TURN_APPRAISAL_TIMEOUT_MS`, `CHANNEL_PRE_TURN_APPRAISAL_REQUEST`, `CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX`
- Added keys (cortex-exec): `REPAIR_PRESSURE_WEIGHTS_V2_PATH`, `REPAIR_PRESSURE_PROBE_ROUTE`, `CHANNEL_PRE_TURN_APPRAISAL_*`
- Removed keys: `ENABLE_REPAIR_PRESSURE_V2` (was redundant; Hub is sole operator gate)
- `.env_example` updated: yes (hub + cortex-exec)
- local `.env` synced: yes in worktree via `python scripts/sync_local_env_from_example.py orion-hub orion-cortex-exec`; run same on main repo after merge
- skipped keys requiring operator action: none

## Tests run

```text
# Substrate + eval
pytest tests/test_pre_turn_appraisal_schemas.py tests/test_turn_window.py \
  tests/test_logprob_probe_runner.py tests/test_repair_pressure_contract_v2.py \
  tests/test_repair_pressure_v2_paradigm.py tests/test_paradigm_registry.py \
  tests/test_schema_registry_import_light.py tests/test_repair_pressure_models.py \
  orion/substrate/evals/repair_pressure_v2_eval.py -q

# Cortex-exec
cd services/orion-cortex-exec && pytest tests/test_pre_turn_appraisal_rpc.py -q

# Hub
cd services/orion-hub && pytest tests/test_pre_turn_appraisal_wiring.py \
  tests/test_handle_chat_request_substrate_effect.py -q

# CI gate (sql-writer import chain)
PYTHONPATH=.:services/orion-sql-writer pytest -q \
  orion/grammar/tests/test_ledger.py \
  services/orion-sql-writer/tests/test_consumer_resilience.py \
  services/orion-sql-writer/tests/test_route_map_completeness.py \
  services/orion-sql-writer/tests/test_world_pulse_routing.py \
  services/orion-sql-writer/tests/test_grammar_event_routing.py \
  services/orion-sql-writer/tests/test_grammar_ledger_sql_shape.py \
  tests/test_schema_registry_import_light.py
# 29+ passed
```

## Evals run

```text
pytest orion/substrate/evals/repair_pressure_v2_eval.py -q
# 3 fixtures: grounding_negative, neutral, ops_frustration_positive
```

## Docker/build/smoke checks

```text
docker compose config validated for hub + cortex-exec (main repo root .env)
Live bus RPC smoke: UNVERIFIED
```

## Review findings fixed

- Finding: duplicated wiring between REST and WebSocket
  - Fix: `run_pre_turn_appraisal_wiring()` + `repair_pressure_grammar_scalars()` in `pre_turn_appraisal_wiring.py`
  - Evidence: hub wiring tests pass
- Finding: WebSocket fell back to legacy when v2 enabled but bus missing
  - Fix: fail-closed + log warning
  - Evidence: `test_run_pre_turn_wiring_skips_when_bus_missing`
- Finding: missing weights YAML silently degraded
  - Fix: fail-closed with `weights_file_missing` note
  - Evidence: `test_paradigm_fail_closed_on_missing_weights_file`
- Finding: dual flags (hub + cortex-exec) could desync
  - Fix: removed `ENABLE_REPAIR_PRESSURE_V2`; Hub `ENABLE_PRE_TURN_APPRAISAL` is sole gate
- Finding: handler hardcoded `repair_pressure`, registry unused
  - Fix: loop `req.paradigms_requested` through `PARADIGM_REGISTRY`
- Finding: `pre_turn_appraisal` schema imported substrate → broke sql-writer CI (no `requests`)
  - Fix: `RepairEvidenceV1` in `orion/schemas/repair_evidence.py` + import-light gate test

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

After merge, sync local env from repo root:

```bash
python scripts/sync_local_env_from_example.py orion-hub orion-cortex-exec
```

Enable on Hub:

```bash
ENABLE_PRE_TURN_APPRAISAL=true
PRE_TURN_APPRAISAL_PARADIGMS=repair_pressure
PRE_TURN_APPRAISAL_TIMEOUT_MS=800
```

Rollback: `ENABLE_PRE_TURN_APPRAISAL=false` — legacy phrase_match resumes.

Operator docs: `services/orion-hub/README.md` (Chat → Repair pressure v2), `services/orion-cortex-exec/README.md` (Pre-turn appraisal RPC).

## Risks / concerns

- Severity: low
- Concern: live Docker/bus RPC path not smoke-tested in this PR
- Mitigation: restart both services, send one chat turn with flag on, confirm hub log `substrate_effect_attached` and cortex log `pre_turn_appraisal`

## PR link

https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/repair-pressure-v2-pre-turn-appraisal?expand=1
