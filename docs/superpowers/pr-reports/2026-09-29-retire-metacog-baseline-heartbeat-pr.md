## Summary

- **The scheduled metacog baseline heartbeat is retired** from orion-equilibrium-service: `_metacog_baseline_loop`, `_maybe_emit_baseline_metacog_trigger`, and their state and task wiring.
- **The substrate `dense`/`pulse` gate is retired** too (`app/substrate_metacog_gate.py`). It was structurally unable to fire.
- **8 equilibrium keys are removed** across settings, compose, `.env_example` and the README. The felt-state reader keys went with them, because only the dead gate used them in this service.
- **`orion/metacog/evidence_map.py`** drops the never-used substrate mapper. It keeps `map_baseline` so historical rows can still be replayed.
- **A guard test is added,** and the smoke/trace scripts now use a live trigger kind.

## Outcome moved

- **The baseline rows had nothing to say.** After #2393 let baseline rows publish again (checked live 2026-09-29: 4 of 4 triggers became rows), they said the same thing every hour: "Baseline check triggered with no active alerts, indicating stable system state." / "Stability is the foundation of progress." The trigger carries no evidence (`scheduled_check`, empty upstream), so it cost an LLM call per hour to report that nothing happened.
- **dense/pulse could never fire.** `compute_substrate_eventfulness()` maxes at 0.25, below both thresholds (0.30 and 0.55). There are zero dense or pulse rows in `metacog_trigger`.
- **`orion_metacog` now only receives rows where something happened.**

## Files changed

- `services/orion-equilibrium-service/app/service.py`: loop, method, state fields, task wiring and import removed.
- `services/orion-equilibrium-service/app/settings.py`, `docker-compose.yml`, `.env_example`, `README.md`: keys removed, and a "Retired" section documents why.
- `services/orion-equilibrium-service/app/substrate_metacog_gate.py`, plus tests `test_substrate_metacog_gate.py`, `test_baseline_hygiene.py` and `test_baseline_startup_emit.py`: deleted.
- `services/orion-equilibrium-service/tests/test_baseline_heartbeat_retired.py`: new guard test.
- `services/orion-equilibrium-service/tests/test_settings_empty_env.py`: rewritten on keys that still exist, to keep the empty-env behaviour covered.
- `services/orion-equilibrium-service/app/attention_self_model_reader.py`: docstring updated.
- `orion/metacog/evidence_map.py`: `map_substrate` removed; `map_baseline` documented as historical-only.
- `services/orion-cortex-exec/tests/test_metacog_publish_lane.py`: two fixtures moved from `dense` to `telemetry_anomaly`.
- `scripts/smoke_metacog.py`, `scripts/trace_metacog.py`, `scripts/smoke_metacog_source_service.py`, `services/orion-sql-writer/app/models/metacog_trigger.py`: now reference live kinds.
- `docs/metacognition_logging.md`: the key is marked retired.

## Schema / bus / API changes

- **Producers:** equilibrium no longer emits `trigger_kind` `baseline`, `dense` or `pulse`.
- **Schema:** `MetacogTriggerV1.trigger_kind` is a free-form `str`, so nothing validates against a fixed list and nothing breaks.
- **Consumers:** the review found no consumer that filters on these kinds.

## Env/config changes

- **Removed from equilibrium:**
  - `EQUILIBRIUM_METACOG_BASELINE_INTERVAL_SEC`
  - `EQUILIBRIUM_METACOG_BASELINE_MAX_SKIPS`
  - `EQUILIBRIUM_METACOG_SUBSTRATE_TRIGGER_ENABLE`
  - `EQUILIBRIUM_METACOG_SUBSTRATE_DENSE_THRESHOLD`
  - `EQUILIBRIUM_METACOG_SUBSTRATE_PULSE_THRESHOLD`
  - `ENABLE_SUBSTRATE_FELT_STATE_CTX`, `SUBSTRATE_FELT_STATE_DATABASE_URL`, `SUBSTRATE_FELT_STATE_MAX_AGE_SEC`
- **Other services:** orion-hub and orion-cortex-exec keep their own copies of the felt-state keys, unchanged.
- **Local `.env`:** the 8 keys were removed by hand from the primary checkout's `services/orion-equilibrium-service/.env`, since the sync script only adds keys. Backup is in the session scratchpad.

## Tests run

```text
services/orion-equilibrium-service/tests + evals + orion/metacog/tests -> 398 passed
services/orion-cortex-exec tests/test_metacog_*.py -> 72 passed
orion/metacog/evals/run_capture_eval.py -> PASS
scripts/smoke_metacog_source_service.py (PYTHONPATH=.:services/orion-cortex-exec) -> ok
check_definition_drift --gate PASS; check_env_template_parity PASS
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-equilibrium-service build -> image built
```

## Review findings fixed

- **Finding:** nothing material. The review confirmed no shared env or compose breakage, no remaining `hydrate_felt_state_ctx` use in equilibrium, and no consumer that validates trigger kinds.
- **Minor, fixed:** the smoke/trace scripts hand-built `baseline` triggers, and the sql-writer model comment listed retired kinds.
- **Minor, left as-is:** `orion/substrate/metacog_trigger_signals.py` still computes a dense/pulse *label* that its one remaining caller (the cortex-exec prompt cue) ignores. It is harmless; a follow-up can drop it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** until equilibrium is rebuilt, the running image keeps its baseline timer. If it restarts before then, the timer comes back at its code default of 3600 s, because the `.env` keys are gone.
  - **Mitigation:** rebuild after merge.
- **Severity: low.**
  - **Concern:** the cortex-exec baseline firebreak is now dead code.
  - **Mitigation:** it is harmless, and can be removed next time cortex-exec is touched.
