# PR: Gate vision-council metacog on host evidence transitions

**Branch:** `feat/vision-council-evidence-transition`  
**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/vision-council-evidence-transition

## Summary

- Replace fingerprint-based `evidence_preflight` skip with `evidence_transition` gate keyed on host `hard_labels` set and `person` presence — ignores `object_counts` drift that limited preflight to ~24% skip rate.
- Run atlas metacog only on `first_window`, `person_entered`, `person_exited`, `labels_changed`, or `refresh_ttl`; stable scenes noop with no bus publish.
- Add `interpret_in_flight` coalescing so concurrent windows on the same stream do not double-call metacog; `try/finally` abort clears in-flight on parse failure or unexpected errors.
- RPC path always interprets (`allow_transition_gate=False`); legacy `COUNCIL_EVIDENCE_SKIP_*` env keys alias to new transition settings.

## Outcome moved

Atlas V100 metacog load from `orion-vision-council` should drop sharply on static office scenes (5s window polling no longer triggers LLM on count jitter). Operator logs should show dominant `evidence_transition skip ... reason=stable_scene`.

## Current architecture

Vision windows arrive every ~5s on `orion:vision:windows`. Council previously fingerprinted full evidence material (including `object_counts`) and skipped ~24% of windows. Most stable scenes still hit atlas metacog.

## Architecture touched

- Service: `services/orion-vision-council`
- Choke point: `evidence_transition.py` → `_generate_interpretation()` / `_process_window()` in `main.py`
- Config: `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`
- Tests: `test_evidence_transition.py`, `test_transition_settings.py`

## Files changed

- `services/orion-vision-council/app/evidence_transition.py`: transition tracker with `interpret_in_flight`, begin/abort/record lifecycle
- `services/orion-vision-council/app/main.py`: gate wiring, in-flight begin under lock, try/finally abort
- `services/orion-vision-council/app/settings.py`: `COUNCIL_TRANSITION_*` with legacy aliases
- `services/orion-vision-council/docker-compose.yml`: explicit transition env passthrough
- `services/orion-vision-council/tests/test_evidence_transition.py`: transition + integration tests
- `services/orion-vision-council/tests/test_transition_settings.py`: legacy env alias tests
- Deleted: `evidence_preflight.py`, `test_evidence_preflight.py`

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: bus intake may skip LLM and publish nothing on stable scenes; RPC unchanged
- Compatibility notes: rename env `COUNCIL_EVIDENCE_SKIP_*` → `COUNCIL_TRANSITION_*` (legacy aliases still read)

## Env/config changes

- Added keys: `COUNCIL_TRANSITION_GATE_ENABLED`, `COUNCIL_TRANSITION_REFRESH_SEC`
- Removed keys: `COUNCIL_EVIDENCE_SKIP_ENABLED`, `COUNCIL_EVIDENCE_SKIP_MAX_SEC` (aliased, not removed from runtime)
- Renamed keys: see above
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest services/orion-vision-council/tests -q
55 passed
```

## Evals run

```text
No vision-council eval harness; gate behavior covered by unit/integration tests.
```

## Docker/build/smoke checks

```text
docker compose -f services/orion-vision-council/docker-compose.yml config  → OK
```

## Review findings fixed

- Finding: Concurrent windows could double-call metacog on same transition
  - Fix: `begin_interpretation` + `interpret_in_flight` short-circuit
  - Evidence: `test_tracker_skips_when_interpret_in_flight`
- Finding: Parse failure could wedge stream if in-flight not cleared
  - Fix: `abort_interpretation` on failure + `try/finally` guard
  - Evidence: `test_generate_interpretation_retries_after_parse_failure`
- Finding: Legacy env keys silently ignored after rename
  - Fix: `AliasChoices` in settings + compose passthrough
  - Evidence: `test_transition_settings.py`

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-vision-council/.env \
  -f services/orion-vision-council/docker-compose.yml \
  up -d --build orion-vision-council
```

## Risks / concerns

- Severity: low
- Concern: While interpretation is in flight, intermediate transitions on the same stream are coalesced (fast enter/exit during one LLM call may defer exit until next window or refresh TTL)
- Mitigation: `refresh_ttl` (default 120s) and subsequent windows catch up; suggestion #3 (cheap host_fallback heartbeat on stable scenes) deferred

## Test plan

- [ ] Rebuild and restart `orion-vision-council`
- [ ] Confirm logs show `evidence_transition skip ... reason=stable_scene` dominating on static cam0 scene
- [ ] Confirm `person_entered` / `person_exited` still trigger interpret lines and event publish
- [ ] Confirm RPC still returns on stable scene
- [ ] Monitor atlas GPU utilization drops vs preflight build
