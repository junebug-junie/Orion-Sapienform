## Summary

- Gate pre-turn appraisal RPC handler to main `cortex-exec` only (`ENABLE_PRE_TURN_APPRAISAL_HANDLER`), eliminating multi-replica reply race on `orion:cortex:pre_turn_appraisal:request`.
- Harden repair-pressure v2 runtime: Docker weights copy, gateway payload normalization, logprob/text-classifier fallback, relaxed yes/no line parsing.
- Wire `repair_pressure_contract` and substrate effect fields through hub request metadata → cortex-exec cognition trace publish → hub trace cache/API redaction.
- Add mesh fleet helpers: explicit 4-lane `up --build` plus post-up verification (weights file + handler flags).

## Outcome moved

Pre-turn appraisal now reliably returns scored bundles from the weighted main lane; cognition trace API exposes `metadata.repair_pressure_contract` and related substrate fields for inspectable repair-pressure behavior (verified live on corr `fce75f9d` before trace hardening).

## Current architecture

All four cortex-exec compose services subscribed to the same pre-turn RPC channel; hub received whichever replica replied first (often a lane without weights). Trace publish carried routing/presence flags but dropped repair-pressure contract metadata.

## Architecture touched

- `services/orion-cortex-exec` — handler gate, pre-turn parsing, trace metadata builder, Dockerfile, compose
- `services/orion-hub` — pre-turn wiring attaches `substrate_effect_summary` to request metadata
- `orion/substrate/appraisal` — repair_pressure_v2 paradigm + logprob runner
- `mesh-utilities/common` — cortex-exec fleet bring-up + verification

## Files changed

- `services/orion-cortex-exec/app/main.py`: `build_cognition_trace_metadata()` reads repair-pressure fields from exec context; passes `ctx` from trace publish path
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`: gateway probe payload normalization
- `services/orion-cortex-exec/app/settings.py`, `docker-compose.yml`, `.env_example`, `README.md`: `ENABLE_PRE_TURN_APPRAISAL_HANDLER` gate
- `services/orion-cortex-exec/Dockerfile`: COPY substrate weights into image
- `services/orion-hub/scripts/pre_turn_appraisal_wiring.py`: attach `substrate_effect_summary` after bundle apply
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: text-classifier fallback when logprob alignment fails
- `orion/substrate/appraisal/probe/logprob_runner.py`: relaxed yes/no line parsing
- `mesh-utilities/common/cortex_exec_fleet_helpers.sh`: new fleet up + verify
- Tests across cortex-exec, hub, and substrate

## Schema / bus / API changes

- Added: none (metadata fields only on existing cognition trace payload / request metadata)
- Removed: none
- Renamed: none
- Behavior changed: only main `cortex-exec` consumes pre-turn RPC; trace API `metadata` now includes repair-pressure contract when attached
- Compatibility notes: lane containers must set `ENABLE_PRE_TURN_APPRAISAL_HANDLER=false` (compose defaults updated)

## Env/config changes

- Added keys: `ENABLE_PRE_TURN_APPRAISAL_HANDLER` (cortex-exec, default `true` on main service, `false` on chat/spark/background lanes)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_cognition_trace_metadata.py \
  services/orion-cortex-exec/tests/test_pre_turn_handler_gate.py \
  services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py \
  services/orion-hub/tests/test_cognition_trace_api.py \
  services/orion-hub/tests/test_pre_turn_appraisal_wiring.py \
  tests/test_logprob_probe_runner.py \
  tests/test_repair_pressure_v2_paradigm.py -q
23 passed
```

## Evals run

```text
None (unit/regression tests only for this patch)
```

## Docker/build/smoke checks

```text
Live smoke on Athena prior to trace hardening: corr fce75f9d — cortex level=0.698, hub level=MEDIUM changed=concrete_bias
Mesh verify_cortex_exec_fleet: all four lanes ✅ after rebuild
```

## Review findings fixed

- Finding: hub tests failed when run with cortex-exec suite due to `scripts` namespace shadowing
  - Fix: importlib load for cognition trace cache; deterministic unit test for substrate summary attach
  - Evidence: 23/23 focused tests pass in combined run

## Restart required

```bash
# Rebuild all four cortex-exec lanes (mesh helper or manual):
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build \
  cortex-exec cortex-exec-chat cortex-exec-spark cortex-exec-background

# Hub if not already running latest wiring:
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build hub
```

## Risks / concerns

- Severity: low
- Concern: trace metadata depends on hub attaching summary before exec publish; if hub restarts without this patch, traces revert to missing contract
- Mitigation: hub restart included above; trace fields are additive only

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/repair-pressure-v2-trace-and-fleet
