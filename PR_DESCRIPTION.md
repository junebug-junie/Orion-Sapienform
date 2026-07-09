## Summary

- Widen substrate execution trajectory reducer to accept `orion-harness-governor` lifecycle grammar alongside `orion-cortex-exec` (shared `cortex.exec:` trace IDs).
- Add harness lifecycle grammar collector and wire motor + finalize publish points (fail-open; finalize emits assembled+egress delta only).
- Spark emits `features_version=seed-v3`: `reliability_pressure` in infra only, 11-dim honest encoder trainable subset; saturated felt dims retained in corpus for audit.
- `fit_phi_encoder.py` versioned input (`--features-version seed-v3` default) with 9/11 variance gate @ 80%.
- Default `INNER_FEATURES_VERSION=seed-v3`; encoder stays off until operator promotes on seed-v3 corpus.
- **(uncommitted)** Isolate auxiliary cortex-exec verbs (`harness_finalize_reflect`, `orion_voice_finalize`) onto lane-suffixed trace IDs so they cannot pollute the primary unified-turn motor projection; add lifecycle publish INFO logging.

## Outcome moved

Unified Orion turns (`mode=orion` / harness-governor FCC motor) now populate `ExecutionTrajectoryProjectionV1` cognitive features (`reasoning_present`, step counters) that spark reads — unblocking honest φ encoder variance gates on an 11-dim trainable subset instead of structurally flat felt dims.

## Current architecture

- Only `orion-cortex-exec` grammar reached the execution trajectory reducer; harness `harness_fcc_step` events were no-oped.
- Spark `seed-v2` trained on 15 dims including saturated `field_intensity` / `resource_pressure` / flat `introspection_pressure` — variance gate stuck at 11/15.

## Architecture touched

- `orion/substrate/execution_loop/` — reducer widening, shared trace ID
- `orion/harness/grammar_emit.py` — new lifecycle collector + finalize delta builder
- `orion/harness/runner.py` + `services/orion-harness-governor/app/bus_listener.py` — publish seams
- `services/orion-spark-introspector/app/inner_state.py` — seed-v3 feature contract
- `scripts/fit_phi_encoder.py` — versioned trainable dims

## Files changed

- `orion/substrate/execution_loop/{constants,ids,grammar_extract,reducer}.py`: motor-agnostic source filter + fcc noop handling
- `orion/harness/grammar_emit.py`: lifecycle grammar producer + `build_harness_grammar_finalize_events`
- `orion/harness/runner.py`: motor lifecycle publish
- `services/orion-harness-governor/app/bus_listener.py`: finalize egress publish
- `services/orion-spark-introspector/app/inner_state.py`: seed-v3 felt/infra split
- `scripts/fit_phi_encoder.py`: versioned encoder input + corpus filter
- Tests across substrate, harness, spark, fit script
- `services/orion-cortex-exec/app/grammar_emit.py`: `trace_lane_for_verb` + lane-suffixed `trace_id` for isolated finalize verbs
- `orion/substrate/execution_loop/ids.py`: optional `lane` on `cortex_exec_trace_id`
- `orion/harness/{grammar_emit.py,runner.py}`: lifecycle publish INFO logs

## Schema / bus / API changes

- Added: harness lifecycle grammar roles (`exec_request_received` … `exec_result_emitted`) from `orion-harness-governor`
- Removed: nothing
- Renamed: nothing
- Behavior changed: reducer accepts both execution motors; `harness_fcc_step` ignored; spark default features_version seed-v3; auxiliary cortex finalize verbs publish to `cortex.exec:{node}:{corr}:{lane}` traces
- Compatibility: cortex-exec path unchanged; seed-v2 corpus replay via `--features-version seed-v2`

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: `INNER_FEATURES_VERSION=seed-v3` in `services/orion-spark-introspector/.env_example`
- local `.env` synced: operator `services/orion-spark-introspector/.env` set to `seed-v3` on this machine
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest \
  tests/test_execution_loop_ids.py \
  tests/test_execution_substrate_reducer.py \
  orion/harness/tests/test_harness_grammar_emit.py \
  orion/harness/tests/test_harness_runner.py \
  services/orion-cortex-exec/tests/test_exec_grammar_emit.py \
  services/orion-spark-introspector/tests/test_inner_state_seed_v3.py \
  tests/test_phi_encoder_fit_script.py \
  services/orion-spark-introspector/tests/test_compose_seed_v2_telemetry_mount.py -q
→ 40 passed (35 trace-isolation slice + prior 37 minus overlap)
```

## Evals run

```text
Not run — operator eval: strict fit on seed-v3 corpus post-deploy.
```

## Docker/build/smoke checks

```text
Compose default validated via test_compose_seed_v2_telemetry_mount.
Unified-turn projection smoke UNVERIFIED — operator step below.
```

## Review findings fixed

- Finding: Finalize replayed full harness lifecycle trace (duplicate bus events)
  - Fix: `build_harness_grammar_finalize_events()` emits only assembled+egress atoms
  - Evidence: `test_finalize_events_emit_only_assembled_and_egress`

- Finding: Empty-draft path double-published lifecycle
  - Fix: Removed redundant bus_listener publish when motor already emitted
  - Evidence: harness tests pass

- Finding: Mixed-batch `harness_fcc_step` noop IDs missing from reducer receipt
  - Fix: `noop_event_ids` includes filtered fcc step IDs
  - Evidence: `test_reducer_noops_harness_fcc_step_in_mixed_batch`

- Finding: Live turn `9ac4f7e5` — cortex finalize verbs shared correlation trace with harness motor, fragmenting projection (`verb=unknown`, `reasoning_present=false`)
  - Fix: `trace_lane_for_verb` → lane-suffixed `cortex.exec:` trace for `harness_finalize_reflect` / `orion_voice_finalize`
  - Evidence: `test_trace_lane_isolates_harness_finalize_verbs`, `test_isolated_lane_trace_does_not_merge_with_primary_motor_trace`

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Medium — Live unified-turn projection smoke not verified after trace-isolation patch
- Mitigation: Redeploy cortex-exec + harness-governor + substrate; re-run unified turn; check logs for `harness_lifecycle_grammar_published` and projection `verb=orion_unified`

- Severity: Low — Encoder off until seed-v3 corpus accrues
- Mitigation: Operator strict fit + promote gates

## Test plan

- [ ] Hub unified turn → projection `verb=orion_unified`, `reasoning_present=true` when FCC steps > 0
- [ ] Classic cortex-exec regression
- [ ] Spark corpus `features_version=seed-v3`, reliability in infra
- [ ] Strict fit ≥9/11 variance on seed-v3 corpus
