# PR: feat/repair-pressure-speech-wiring

**Title:** feat: wire repair pressure contract into chat TURN CONTRACT

**Branch:** `feat/repair-pressure-speech-wiring` → `main`

**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/repair-pressure-speech-wiring

---

## Summary

- Add shared metadata key `repair_pressure_contract` for substrate `contract_after` on Hub → cortex chat requests
- Hub `attach_repair_pressure_contract()` writes metadata when repair pressure changes contract mode (HTTP + WebSocket)
- Cortex-exec `compile_speech_contract()` merges repair overlay: `repair_concrete` overrides regime text, `concrete_bias` appends
- Executor reads metadata from `ctx` when `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=true` (default on both services)
- Regression tests cover benign turns, flag-off rollback, snapshot=None fail-open, and relational+concrete_bias blend

## Outcome moved

HIGH/MEDIUM repair pressure on a chat turn now changes the TURN CONTRACT on that same turn — substrate `contract_after` flows Hub metadata → cortex `compile_speech_contract` → `chat_general.j2`.

## Current architecture

Before this patch, Hub substrate appraisal could change `contract_after` and surface a Substrate Effect chip, but cortex-exec compiled speech contracts from stance brief alone. Repair pressure did not reach the TURN CONTRACT on the same turn.

## Architecture touched

- Shared: `orion/substrate/appraisal/contract.py` — metadata key constant
- Hub: `repair_pressure_wiring.py`, `api_routes.py`, `websocket_handler.py`, settings
- Cortex-exec: `chat_stance.py`, `executor.py`, settings

## Files changed

- `orion/substrate/appraisal/contract.py`: `REPAIR_PRESSURE_CONTRACT_METADATA_KEY` constant
- `orion/substrate/appraisal/__init__.py`: re-export metadata key
- `services/orion-hub/scripts/repair_pressure_wiring.py`: attach helper (new)
- `services/orion-hub/scripts/api_routes.py`: capture snapshot, attach before cortex call
- `services/orion-hub/scripts/websocket_handler.py`: same for WebSocket path
- `services/orion-hub/app/settings.py`, `.env_example`: `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING`
- `services/orion-cortex-exec/app/chat_stance.py`: `_compile_repair_speech_overlay` + merge in `compile_speech_contract`
- `services/orion-cortex-exec/app/executor.py`: read metadata, pass `repair_contract`
- `services/orion-cortex-exec/app/settings.py`, `.env_example`: feature flag
- Hub + cortex-exec tests: wiring, precedence, integration, review-gap regressions

## Schema / bus / API changes

- Added: metadata field `repair_pressure_contract` on `CortexChatRequest.metadata` (convention only, no schema registry change)
- Removed: none
- Renamed: none
- Behavior changed: repair pressure contract now affects TURN CONTRACT text when mode changes
- Compatibility notes: flag defaults on; set `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=false` on hub and/or cortex-exec to roll back

## Env/config changes

- Added keys: `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` (hub + cortex-exec)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (both services)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (parent repo; no new keys needed)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-hub && PYTHONPATH=.:../.. pytest tests/test_repair_pressure_wiring.py tests/test_handle_chat_request_substrate_effect.py -q
10 passed

PYTHONPATH=services/orion-cortex-exec:. pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  services/orion-cortex-exec/tests/test_repair_pressure_speech_wiring.py -q
40 passed
```

## Evals run

```text
No eval harness for this seam; covered by deterministic unit + integration tests per plan Task 7.
```

## Docker/build/smoke checks

```text
Not run in this session. Live smoke UNVERIFIED.
Recommended: high-pressure utterance → Substrate Effect chip changed_behavior=true AND cortex debug shows Repair turn or rule substring in speech_contract.
```

## Review findings fixed

- Finding: Benign HTTP turn did not assert metadata absence on cortex request
  - Fix: Extended `test_handle_chat_request_summary_marks_no_change_for_benign`
  - Evidence: 10/10 hub tests pass

- Finding: No flag-off regression tests
  - Fix: Added hub monkeypatch test + `test_metadata_disabled_keeps_instrumental_contract`
  - Evidence: tests pass

- Finding: No `snapshot=None` attach test
  - Fix: Added `test_attach_skips_when_snapshot_none`
  - Evidence: tests pass

- Finding: `concrete_bias` + relational blend untested
  - Fix: Added `test_compile_speech_contract_concrete_bias_appends_to_relational`
  - Evidence: 40/40 cortex-exec tests pass

- Finding: Inconsistent `getattr(settings, ...)` at call sites
  - Fix: Use `settings.ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` directly
  - Evidence: api_routes.py + websocket_handler.py updated

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build orion-cortex-exec
```

## Risks / concerns

- Severity: low
- Concern: Live end-to-end metadata → speech_contract propagation not smoke-tested with running containers
- Mitigation: Post-deploy spot-check with high-pressure fixture; flag-off rollback available on both services

## Test plan

- [ ] Merge and restart hub + cortex-exec
- [ ] Send high-pressure message via Hub HTTP chat
- [ ] Confirm Substrate Effect chip shows `changed_behavior=true`
- [ ] Inspect cortex debug/trace for `speech_contract` containing `Repair turn` or rule substring
- [ ] Send benign message; confirm no `repair_pressure_contract` in cortex metadata
- [ ] Set `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=false` on hub; confirm high-pressure message omits metadata

## Commits (7)

```
b819a89a feat(substrate): add repair_pressure_contract metadata key constant
c6568317 feat(hub): attach repair_pressure_contract to CortexChatRequest metadata
8ed460cb feat(cortex-exec): merge repair pressure into compile_speech_contract
e4997822 feat(cortex-exec): read repair_pressure_contract from ctx metadata
d58c74d4 feat(hub): wire repair contract into HTTP chat metadata
c789e6a7 feat(hub): wire repair contract into WebSocket chat metadata
6aefd8f5 test: address code review gaps for repair pressure speech wiring
```
