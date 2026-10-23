## Summary

- Every Hub chat turn waits for a repair-pressure probe before the stance runs. The probe now rides a new route, `quick_turn`. It is the same fast model as `quick`, at interactive pool priority, so the 34 C heat reflex no longer refuses it.
- The probe's request now carries the turn's correlation id. The gateway leases the GPU with that id, so the probe's pool events join to the turn.
- A refused, empty, or unparseable probe now reports a failure with a reason (for example `shed:cabinet_hot`). Before, it reported a level of 0.0 and the cockpit said `ok`. That made a missing reading look like a calm turn.
- Every reader of the repair-pressure level was checked. Each one now treats "no reading" as unknown, not as 0.0 (see the consumer audit below).

## Outcome moved

Failure mode removed: an empty result passed off as a measurement (CLAUDE.md "No empty-shell cognition"). On 2026-10-20..21 the probe was refused on 19 of 19 of Juniper's turns. 7 refusals came from the pool (`unavailable:shed:cabinet_hot`). 12 came from the gateway's local shed backoff, in about 15 ms each. Each one reached the cockpit as `appraisal · ok` with level 0.0. After this patch, two things change:

1. Heat does not refuse the probe, because `quick_turn` is interactive.
2. If the probe fails for any other reason, the cockpit reads `appraisal · FAILED failed_paradigms=repair_pressure:<reason>`. The level stays unknown.

## Current architecture

- Hub `orion/hub/turn_orchestrator.py::_run_pre_turn_appraisal` sends an RPC to cortex-exec and waits for the reply (timeout 180 s).
- cortex-exec `app/pre_turn_appraisal.py` runs `RepairPressureV2Paradigm`. The paradigm calls the LLM gateway on `REPAIR_PRESSURE_PROBE_ROUTE=quick`. That route is the fast class at `system` priority, which heat sheds. The request used a fresh `uuid4` as its correlation id.
- When the gateway refused the probe, the reply had empty text and set `raw.error`. The paradigm returned `level=0.0, confidence=0.0, notes=["no_repair_evidence"]`. The bundle listed no failure, so the Hub status read `ok`. The level then travelled on as a measured 0.0, in three places:
  - the `repair_pressure_reading` grammar atom
  - the `orion:repair_pressure:appraisal` publish
  - the chat projection

## Architecture touched

- GPU pool and route registry: new route `quick_turn`.
- cortex-exec pre-turn appraisal handler and the repair-pressure v2 paradigm.
- `TurnAppraisalBundleV1`: new additive field `failure_reasons`.
- Hub: turn orchestrator cockpit hop, grammar scalars, grammar emit.
- Thought: the stance prompt's view of the repair bundle.
- Chat projection: `ChatTurnStateV1` and `compute_chat_pressure_hints`.

## Files changed

- `config/gpu_pool.yaml`: added `quick_turn: {class: fast, priority: interactive}`. Updated the comment that said it was not added yet.
- `orion/llm/routes.py`: `quick_turn` is now in `ACCEPTED_LLM_ROUTES`, `LLM_ROUTE_DISPLAY_ORDER`, and `SYSTEM_LLM_ROUTES`. That hides it from the human picker and refuses it as an override. It is not metacog-pinned, because it is fast class. `quick` has no pin set.
- `orion/gpu_pool/tests/test_route_view.py`: golden allowance plus a pin test (`quick_turn` must look exactly like `quick` except for `priority`).
- `orion/gpu_pool/tests/test_route_priorities_explicit.py`, `orion/llm/tests/test_routes.py`, `services/orion-llm-gateway/tests/test_anthropic_passthrough.py`: route registration.
- `services/orion-cortex-exec/app/settings.py`, `.env_example`, `README.md`: `REPAIR_PRESSURE_PROBE_ROUTE` now defaults to `quick_turn`. The compose file passes env through `env_file` and has no explicit entry, so it did not change.
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`: three changes.
  - The envelope correlation id is now the turn's id. A fresh one is used only if the turn's id is not a UUID. The reply channel stays unique.
  - An undecodable reply is now a stated error.
  - The handler fills `failure_reasons`.
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: new `RepairProbeUnavailable` and `probe_failure_reason`. A shed, empty, unparseable, or weights-missing probe now raises instead of returning 0.0.
- `orion/schemas/pre_turn_appraisal.py`: `TurnAppraisalBundleV1.failure_reasons`. It is left out of the serialized payload when empty.
- `orion/hub/turn_orchestrator.py`: the cockpit texture and hop carry `failure_reasons`. The error names each reason. `has_repair_pressure_reading` is now true only when a level exists.
- `services/orion-hub/scripts/pre_turn_appraisal_wiring.py`: `repair_pressure_grammar_scalars` returns `(None, None)` when there is no reading.
- `services/orion-hub/scripts/grammar_emit.py`: a level of `None` emits no repair atom.
- `orion/thought/stance_react.py`: the stance prompt sees `unmeasured: {repair_pressure: "level unknown: <reason>"}`.
- `orion/schemas/chat_projection.py`, `orion/substrate/chat_loop/grammar_extract.py`: `repair_pressure_level` and `repair_pressure_confidence` default to `None` (no reading). The `repair_pressure` hint is left out when the level is `None`.
- `scripts/analysis/replay_route_chat_prediction_error_definitions.py`: the frozen v1 replay reads `None` as 0.0, the same way it read the old rows.
- Tests: `tests/test_repair_pressure_v2_paradigm.py`, `tests/test_pre_turn_appraisal_schemas.py`, `tests/test_chat_substrate_reducer.py`, `services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py`, and three files under `services/orion-hub/tests/`: `test_pre_turn_appraisal_wiring.py`, `test_repair_signal_threshold.py`, `test_turn_orchestrator_ws_frames.py`.

## Schema / bus / API changes

- Added: route `quick_turn`. Field `TurnAppraisalBundleV1.failure_reasons: dict[str, str]`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - A failed probe is now listed in `failed_paradigms` with no slice. Before, it was a 0.0 slice.
  - `ChatTurnStateV1.repair_pressure_level` is `None` when there is no reading.
- Compatibility notes:
  - `failure_reasons` is left out of the payload when empty, so a successful bundle is byte-identical to before. Both consumers validate with `extra="forbid"`: the Hub, and orion-thought through `StanceReactRequestV1`. They must be rebuilt before cortex-exec (see the deploy order).
  - An old Hub that receives a failed bundle from a new cortex-exec rejects it and reports `appraisal_unavailable`. That is still an honest failure.
  - Persisted chat-projection rows that hold `0.0` still validate.
  - `orion/schemas/registry.py`, `orion/bus/channels.yaml`, and sql-writer were not touched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Changed default: `REPAIR_PRESSURE_PROBE_ROUTE` `quick` -> `quick_turn` (cortex-exec).
- `.env_example` updated: yes, `services/orion-cortex-exec/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, run. It reports `orion-cortex-exec: REPAIR_PRESSURE_PROBE_ROUTE local='quick' example='quick_turn'` as diverged and does not change it, as designed.
- skipped keys requiring operator action: `REPAIR_PRESSURE_PROBE_ROUTE` in the live `services/orion-cortex-exec/.env`. This is deliberate. The gateway must know `quick_turn` first. See Restart required, step 4.

## Consumer audit: unknown repair pressure

| Consumer | Before | After |
| --- | --- | --- |
| Cockpit `pre_turn_appraisal` hop (`turn_orchestrator.py`) | `ok` | `failed`, error `failed_paradigms=repair_pressure:shed:cabinet_hot`, `failure_reasons` in summary and raw |
| `orion:repair_pressure:appraisal` publish, which feeds equilibrium's metacog gate | published level 0.0 | not published (no slice) |
| Hub chat grammar, unified turn and websocket paths | `repair_pressure_reading` atom at 0.0 | no repair atom |
| Chat projection `ChatTurnStateV1` | 0.0 | `None` |
| `compute_chat_pressure_hints`, which feeds the field digester `repair_pressure` channel | wrote 0.0 | key left out; the digester already skips a missing key (`state_deltas.py` `if key in hints`), so nothing is written and the channel decays as normal |
| `chat_prediction_error` / `chat_raw_reading` | 0.0 | missing hint read as 0.0. The number is unchanged on purpose, because this is a drift-locked definition. Pinned by a test. See Risks. |
| `endogenous_curiosity` repair candidate | `<= 0` skipped | `None or 0.0` skipped. Same result, and correct: no reading means no pull. |
| Stance prompt (`slim_repair_bundle_for_prompt`) | empty `paradigms` | `unmeasured: {repair_pressure: "level unknown: <reason>"}` |
| `ThoughtEventV1.repair_pressure_level`, finalize `sensitive_turn_reason` | already `float or None`, `None`-safe | unchanged |
| Repair contract (`_repair_pressure_contract`) | default mode | default mode (no slice, no contract). Unchanged. |
| cortex-exec `_substrate_trace_fields` | `None` when no summary | unchanged |

## Tests run

```text
orion/llm/tests + orion/gpu_pool/tests                                   525 passed
services/orion-llm-gateway/tests                                          459 passed (one run hit a known flake in
                                                                          test_gateway_cancel_not_upstream_failure; 3 reruns green, also green on main)
tests/test_repair_pressure_v2_paradigm.py, test_repair_pressure_appraisal.py   20 passed
services/orion-cortex-exec tests/test_pre_turn_appraisal_rpc.py (+speech wiring) 10 passed
root + orion tests touching chat_loop / ChatTurnStateV1 / appraisal       178 passed, 3 failed (test_felt_state_self_definition_lane.py, fails identically on main)
tests/test_pre_turn_appraisal_schemas.py, test_chat_substrate_*, replay   23 passed
services/orion-hub grammar/appraisal/turn_orchestrator/repair/stance      214 passed, 5 failed (all 5 fail identically on main:
                                                                          test_substrate_effect_endpoint x3, test_substrate_effect_pipeline,
                                                                          test_turn_orchestrator_utterance_origin)
services/orion-hub test_turn_orchestrator_ws_frames + appraisal wiring    64 passed
services/orion-thought/tests                                              745 passed
services/orion-substrate-runtime, field-digester, sql-writer (touching files) 75 / 8 / 67 passed
CI workflows covering config/gpu_pool.yaml and cortex-exec: gpu-pool svc 197, hub gpu routes 11, thought route 6,
exec lease 37, lane controller 101, shed-not-failure 67, hub unified placement 10, mind turn route 10 -- all passed
```

Regression tests that would have caught the bug:
- `test_shed_probe_fails_with_the_pool_reason_not_a_calm_zero`
- `test_shed_probe_is_a_failed_paradigm_with_its_reason_not_a_zero_level`
- `test_run_pre_turn_appraisal_shed_probe_is_failed_not_ok`
- `test_failed_appraisal_emits_neither_atom_and_projection_level_is_unknown`

## Evals run

```text
python -m pytest -q orion/substrate/evals/repair_pressure_v2_eval.py      4 passed
python -m orion.substrate.evals.repair_pressure_v2_eval                   PASS grounding_negative level=0.187
                                                                          PASS neutral level=0.050
                                                                          PASS ops_frustration_positive level=0.876 repair_concrete
                                                                          PASS probe_shed unavailable reason=shed:cabinet_hot
```

The `neutral` fixture had no probe lines (`mock_probe_lines: {}`). The empty reply that produced passed as a calm
turn. In other words, the eval itself encoded this bug, and it failed after the fix. `neutral` now has real
all-NO probe answers, which makes it a measured calm turn. A new `probe_shed` fixture checks that the gateway's
heat refusal fails with its reason.

## Static gates (from .github/workflows/orion-static-gates.yml)

```text
git diff --check                                   ok
check_metric_lineage.py --gate / --prompt-semantics PASS / PASS
check_definition_drift.py --gate                    PASS (chat_session definition unchanged)
check_inner_state_registry.py                       OK
check_chat_route_poachers.py                        PASS
check_env_template_parity.py                        PASS
check_sentience_instruments.py --static-only        All claims hold
test_no_personal_terms, metric_prompt_semantics, metric_rest_range, check_settings_defaults  98 passed
```

## Docker/build/smoke checks

```text
No Docker build was run. The changes are Python plus config that the existing images already copy, and no
dependency changed. Live path: UNVERIFIED until deploy. Smoke after deploy: send one Hub turn while
cabinet_hot is active. Then check three things:
  1. The cockpit pre_turn_appraisal hop reads ok with a real level, or FAILED with a reason. It must never read ok with 0.0.
  2. The gpu_pool lease events for the probe carry turn_correlation_id equal to the turn's id.
  3. The cortex-exec log shows the probe on quick_turn, for example on a failure:
     docker logs orion-athena-cortex-exec 2>&1 | grep -E "pre_turn_appraisal_paradigm_(result|failed)" | tail
     A failure line prints route=quick_turn. If it prints route=quick, the .env sed has not been applied.
```

## Review findings fixed

The review ran in an orion-repo-agent subagent (read-only).

- Finding (material): the repair-pressure eval (`orion/substrate/evals/repair_pressure_v2_eval.py`) failed on `neutral`. That fixture's empty probe reply had been passing as a calm 0.0, so it encoded the bug. This report also claimed no eval existed.
  - Fix: `neutral` now has real all-NO probe lines. Added a `probe_shed` fixture that expects `RepairProbeUnavailable("shed:cabinet_hot")`. The CLI `main()` now handles an expected failure. Corrected the Evals section.
  - Evidence: 4 passed, and the CLI prints PASS for all four fixtures.
- Finding (material): the live `.env` keeps `quick` until an operator runs the sed.
  - Fix: no code change, because the live `.env` is deliberately not edited by this patch. The Restart section gives the sed in order, and the smoke now checks the actual route in the cortex-exec log.
  - Evidence: Restart and Docker/smoke sections.
- Finding (minor): the field-lattice decay behavior change was not reported. Also, rolling back substrate-runtime would hit `null` in the persisted projection.
  - Fix: both added to Risks.
- Finding (nit): `build_chat_turn_grammar_events` defaulted `repair_pressure_level` to 0.0.
  - Fix: the default is now `None`, so an omitted reading means no reading.
  - Evidence: Hub grammar tests 26 passed. Both real callers pass the value explicitly.
- Finding (minor, accepted): the Hub cockpit test only fails on old code because `failure_reasons` is missing. The real regression is guarded elsewhere: the paradigm, the cortex-exec handler, and the `test_repair_signal_threshold` tests. Those tests fail on old code for the right reason. The cockpit test stays as a contract test for the reason text.
- Finding (minor, accepted): `failure_reasons` is also set for unknown paradigms. The deploy order (thought and hub before cortex-exec) covers this.
- Finding (minor, accepted): the probe text becomes an embedding under the turn's id, and later calls with the same id overwrite it. orion-mind and orion-thought already share that key, so this is not a new class of collision. Recorded in Risks.

## Restart required

Deploy from the clean primary `main` checkout after the merge. Order matters:

1. The gateway learns the route.
2. The `extra="forbid"` consumers learn `failure_reasons`.
3. Then cortex-exec.

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
sed -i 's/^REPAIR_PRESSURE_PROBE_ROUTE=quick$/REPAIR_PRESSURE_PROBE_ROUTE=quick_turn/' services/orion-cortex-exec/.env
scripts/safe_docker_build.sh orion-cortex-exec up -d --build --force-recreate
```

Notes on these commands:
- The gpu-pool service and circe's lane controller need no restart. The pool leases by class and priority, not by route name, and both already exist.
- If the post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) already rebuilt cortex-exec before the sed, cortex-exec runs the new code with the old `quick` route until the sed and recreate. It stays honest in that window: a shed probe reports `FAILED`.

## Risks / concerns

- Severity: low.
  - Concern: the probe's envelope now carries the turn's correlation id. The gateway keys its assistant-embedding publish (`doc_id`) on that id. orion-mind's turn calls already do the same (`MindLLMRequestContext.envelope_correlation_id`), so this adds one more call under an existing shared key. It is not a new class of collision.
  - Mitigation: none needed for this slice. If embedding doc collisions matter, key `doc_id` on the envelope id in the gateway.
- Severity: low.
  - Concern: `chat_prediction_error` still reads a missing repair reading as 0.0. It is a drift-locked metric definition (`chat_session` v3). Changing it needs the metric gate and a definition bump, so it was left out of this fix on purpose.
  - Mitigation: a test pins the unchanged number. Follow-up: skip unknown repair readings in `chat_raw_reading` (v4), with the metric gate.
- Severity: low.
  - Concern: the field-lattice `repair_pressure` channel behaves differently now. Before, every turn with no reading wrote a fabricated 0.0 (`mode="replace"`), which reset the node. Now nothing is written. The node keeps the last real reading, and the generic node decay (`decay.py` `NODE_DECAY_CHANNELS`) multiplies it by 0.92 each tick. During a long shed or skip streak, the node fades toward 0. That is still more honest than an instant fabricated 0.0, but a value that decayed to 0 looks like a calm 0.
  - Mitigation: with `quick_turn`, heat no longer causes shed streaks. Follow-up together with the `chat_raw_reading` v4 item above.
- Severity: medium (rollback only).
  - Concern: once the new orion-substrate-runtime writes `repair_pressure_level: null` into the singleton `substrate_chat_session_projection` row, the old code (`float = 0.0`) fails validation on every tick. That is the same crash-loop class `check_substrate_projection_schema_drift.py` guards against. Forward deploy is safe.
  - Mitigation: do not roll orion-substrate-runtime back past this PR without also reverting the schema change, or nulling those fields to 0.0.
- Severity: low.
  - Concern: `quick_turn` puts the probe at interactive priority on the fast class, beside `quick`'s background callers. The probe is one short call (`max_tokens=128`) per human turn.
  - Mitigation: none needed for this slice.

## PR link

Filled in after push.
