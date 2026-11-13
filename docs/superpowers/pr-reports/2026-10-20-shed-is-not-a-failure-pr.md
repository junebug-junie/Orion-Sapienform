# A deliberate GPU-pool shed is not a failure

The circe cabinet is hot, so the GPU pool turns away background LLM work on purpose (`shed:cabinet_hot`). Before this patch every refusal was recorded as an ordinary failed step. That made Orion read heat as "my work is failing": `failure_pressure` jumped, the anomaly gate fired about 68 times a day, and reverie put healthy themes into cooldown. After this patch a shed is recorded as a shed, and only real failures count as failures.

## Summary

- cortex-exec keeps the shed reason. A failed-step atom now carries `shed_reason=<name>` next to the unchanged `error_kind`. When the plan itself failed because of the shed, the result atom carries the same marker.
- The execution reducer counts shed steps separately (`shed_step_count`). `failure_pressure` and `execution_friction` now count only real failures. A pool deadline or outage, a validation failure, an inactive verb and empty output all still count.
- Reverie: when the pool refuses the narration call on a chain's first step, it is logged as a counted skip (`reverie_chain_skipped reason=llm_unavailable`). It is not counted as a hollow theme, so it does not add to the theme's `no_coalition` streak.
- Anomaly episode gate: when an episode's first window is only in the nominal band, the open waits one window. It then fires with the band of window 2, so the open carries the highest band of the first two windows. This removes the "open, then escalation one minute later" double fire.
- The glossary meanings and the metric definition lock are updated for `failure_pressure` and `execution_friction` (Juniper approved the change).

## Outcome moved

- Anomaly fires on 10-20: **68 to 4** in the live replay (scores 10-18 18:30Z to 10-20). This is an approximation, explained under Evals.
- Anomaly fires over the 15-day committed fixture (deferral only): **140 to 106**. Escalations within one window of a previous fire go from 18 to 12. Dream novelty is unchanged (10.0 to 10.0).
- `failure_pressure` on node:athena stops reading 1.0 for shed-only runs. Live evidence below: 485 of the 760 runs in the stored projection read `failure_pressure=1` today, and over 6h, 149 of 150 pool refusals were `shed:cabinet_hot`.
- Reverie themes no longer reach the 3-strike forced cooldown because of pool refusals.

## Current architecture

- `services/orion-cortex-exec/app/router.py` recorded a failed step with `error_kind=short_error_kind(step_res.error)`. That function cuts the error at the first `:`, so `gpu_pool_unavailable:shed:cabinet_hot` became `gpu_pool_unavailable` and the shed reason was lost.
- `orion/substrate/execution_loop/grammar_extract.py` set `failure_pressure = 1.0 if status_fail or failed > 0`. The digester wrote that value to node:athena (`services/orion-field-digester/app/ingest/state_deltas.py`).
- The telemetry anomaly gate (`orion/metacog/anomaly_episode.py` `step()`) opened a new episode on each short `failure_pressure` blip, and escalated on the next window when the band rose.
- In reverie (`services/orion-thought/app/chain.py`), any `None` from step 0 counted toward the `no_coalition` streak. A pool refusal came back as `None`.

## Architecture touched

- **Contract.** The `exec_step_failed` and `exec_result_assembled` atom summaries have a new optional `shed_reason=<name>` kv. It is additive: atom shape and `semantic_role` are unchanged.
- **Schema.** `ExecutionRunStateV1` (`extra="forbid"`) has two new fields: `shed_step_count: int = 0` and `result_shed: bool = False`. The model is already registered as part of `ExecutionTrajectoryProjectionV1` in `orion/schemas/registry.py`, so no registry entry was needed.
- **Shared helper.** `orion/gpu_pool/shed.py` adds `shed_reason_from_error()` and `is_gpu_pool_refusal()`. They use only the standard library.
- **Pure state machine.** `orion/metacog/anomaly_episode.py` has a new `defer_nominal_open` parameter (default True) and a new non-fire reason `open_deferred`. The persisted state shape is unchanged.

## Files changed

- `orion/gpu_pool/shed.py`: shed-reason parser and refusal check.
- `orion/schemas/execution_projection.py`: `shed_step_count`, `result_shed`.
- `orion/substrate/execution_loop/grammar_extract.py`: counts sheds separately; new `failure_pressure` rule.
- `orion/substrate/execution_loop/merge.py`: max-merge for `shed_step_count`, OR-merge for `result_shed`.
- `services/orion-cortex-exec/app/grammar_emit.py`, `router.py`: emit `shed_reason` on failed-step and result atoms.
- `services/orion-thought/app/reverie.py`, `chain.py`: `ReverieLlmUnavailable`, opt-in raise, skip at step 0.
- `orion/metacog/anomaly_episode.py`: nominal-open deferral.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: meaning updates.
- Tests: `tests/test_execution_substrate_reducer.py`, `orion/gpu_pool/tests/test_shed_reason_from_error.py`, `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`, `services/orion-cortex-exec/tests/test_router_shed_grammar.py` (new), `services/orion-thought/tests/test_reverie_chain.py`, `services/orion-thought/tests/test_reverie_spontaneous_thought.py`, `services/orion-equilibrium-service/tests/test_telemetry_anomaly_episode.py`, `services/orion-equilibrium-service/evals/test_telemetry_anomaly_episode_replay_eval.py`.
- `.github/workflows/shed-not-failure-tests.yml`: new CI gate.

## Schema / bus / API changes

- Added: `ExecutionRunStateV1.shed_step_count`, `ExecutionRunStateV1.result_shed`, and the atom summary kv `shed_reason`.
- Removed: none.
- Renamed: none.
- Behavior changed: `failed_step_count` now excludes shed steps. Every consumer is listed below.
- Compatibility: rows persisted before this change load with the new fields at their defaults (tested). The model is `extra="forbid"`, so a **reader still running old code fails validation** once a rebuilt substrate-runtime writes the new fields. The rollout must therefore be consumer-first: the readers first, substrate-runtime last. The deploy command below does this. The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh --list-only`) builds alphabetically, so the readers (orion-cortex-exec, orion-equilibrium-service, orion-field-digester, orion-hub) also come before orion-substrate-runtime. I checked the order. I did not watch a live rollout: UNVERIFIED.

### Every consumer of failed-step data / `error_kind` / `failed_step_count`

| Consumer | Effect |
|---|---|
| `grammar_extract.compute_pressure_hints` (`failure_pressure`, `execution_friction`) | Changed as intended: sheds excluded. |
| `services/orion-field-digester/app/ingest/state_deltas.py` | Reads the hints. Node channels change as above. |
| `orion/substrate/metacog_trigger_signals.py` (`_execution_has_failures`, `failed_runs=` cue, eventfulness) | A shed-only run no longer counts as a failed run. Intended. |
| `orion/grammar/atom_signals.py` (uncertainty for `exec_step_failed`) | Unchanged. It keys on `semantic_role`, which is unchanged. |
| `services/orion-hub/static/js/thought-process.js` (`failed_step_count` on Harness Work) | Unchanged in practice. That panel shows FCC-motor runs, and the harness runner is never pool-shed. |
| `services/orion-hub/scripts/chat_turn_trace_routes.py` | Validates the model. Needs the consumer-first rollout. |
| `orion/harness/grammar_emit.py` / `runner.py` (own `short_error_kind`) | Untouched. It is a separate collector with no GPU pool. |
| SQL / dashboards grouping by `error_kind` | `error_kind` is unchanged (`gpu_pool_unavailable`), so grouping is unchanged. |

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no env template changed.
- Skipped keys requiring operator action: none.
- Feature flags: the new behavior ships ON. `defer_nominal_open` defaults to True. It has no env kill switch: rolling it back means changing code (see Risks).

## Metric gate (failure_pressure / execution_friction definition change)

1. **Provenance.** The value is produced by `grammar_extract.compute_pressure_hints`, from `exec_step_failed` atoms and the `exec_result_assembled` status. cortex-exec writes those atoms in `router.py` (`record_step_failed` and `record_assembled_grammar`). The step error comes from `executor.gateway_error_step_failure`, which builds it from the gateway's `raw.error` and `raw.details.reason`.
2. **Independence.** This is not a new metric. It narrows an existing one. `execution_friction` and `failure_pressure` still share an upstream (failed steps), as they did before.
3. **Theory anchor.** A shed is the pool's own deliberate protective policy (`orion/gpu_pool/shed.py`: precedence-0 reflex for heat). The work was refused before it ran, so it was not executed and did not fail. "Execution failure" means the execution went wrong. A deadline or outage is still a capacity failure.
4. **Live data.**
   - The stored projection loads under the new model.
   - 760 runs are stored. 485 of them have `failed_step_count > 0` and `failure_pressure = 1`.
   - Over the last 6h, `exec_step_failed` atoms with `error_kind=gpu_pool_unavailable` ran at 8 to 25 per hour.
   - The cortex-orch journal for the same 6h shows 149 `gpu_pool_unavailable:shed:cabinet_hot` refusals and 1 `pool_unreachable`.
   - Example: correlation `79a7bc06-6694-490e-a761-4663ad2a0ea2` is a `draft_journal_entry` atom, and orch logged it as `metacog_draft_fallback:gpu_pool_unavailable:shed:cabinet_hot`.
   - Rest state: with sheds excluded, a shed-only run reads 0.0, and the channel decays to 0 when nothing really fails. The deadline and outage paths still read 1.0 (tested).
5. **Existing mechanism.** `orion/world_pulse_read/retry.py` and `orion/core/bus/rpc_degraded.py` already treat pool refusals as "not a reading" or "degraded" in their own lanes. Nothing in the execution reducer did. The new parser sits in `orion/gpu_pool/shed.py`, next to the shed reasons it parses.
6. **Reversibility.** This is cheap to undo. The two schema fields default to 0/False, and the kv is additive. Removing them is a forbid-model migration in reverse (writer first, then readers).

## Tests run

```text
tests/test_execution_substrate_reducer.py + tests/test_execution_projection_schemas.py
  + orion/gpu_pool/tests + tests/test_chat_substrate_reducer.py         530 passed
orion-cortex-exec: tests/test_exec_grammar_emit.py tests/test_router_shed_grammar.py
  tests/test_gpu_lease_forwarding.py                                     passed (23 + 3 + lease)
orion-thought: tests + evals                                             all passed except
  tests/test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off
  (pre-existing: fails identically on main, env-sensitive)
orion-equilibrium-service: tests + evals                                 267 passed
orion-field-digester: -k "state_delta or execution"                      12 passed
Static gates: check_definition_drift --gate, check_metric_lineage --gate / --prompt-semantics,
  check_sentience_instruments --static-only, check_inner_state_registry,
  test_metric_prompt_semantics, test_metric_rest_range, test_grammar_event_producer_catalog,
  test_no_private_eval_artifacts, test_no_personal_terms                 all pass
git diff --check                                                         clean
check_substrate_projection_schema_drift.py                               SKIPPED (no POSTGRES_URI in this shell);
  the live projection row was instead validated by hand against the new model (loads OK)
```

## Evals run

```text
services/orion-equilibrium-service/evals/test_telemetry_anomaly_episode_replay_eval.py
  15-day fixture: fires 140 -> 106, escalations within one window of a fire 18 -> 12,
  dream novelty 10.0 -> 10.0; the 10-08 outage still fires before 14:55Z (critical);
  the 10-03 availability outage still fires.

Live replay (scores 10-18 18:30Z .. 10-20, 9,597 windows; scratch replay3/replay4):
                                          10-18  10-19  10-20
  before (live rules)                        20     41     68
  nominal-open deferral only                 13     33     61
  shed excluded (approx.)                     7      5      9
  shed excluded + deferral (after)            4      0      4
```

How to read the replay:

- "Shed excluded" drops every window whose lead channel was `failure_pressure`. That is an upper bound on the effect. The few real failures (about 7 on 10-20) would still raise `failure_pressure`, so the true count after deploy is somewhat above 4 per day. The exact number is UNVERIFIED until it runs live.
- On 10-19 all 5 remaining fires were lone one-window nominal blips, which deferral now drops.

## Docker/build/smoke checks

```text
Not run. This change is not deployed (task scope: no deploy, no production writes).
No Docker/compose/env surfaces changed.
```

## Review findings fixed

- Finding: a plan that failed for its own reason after an earlier shed step (inactive verb, empty output) read `failure_pressure=0`.
  - Fix: a result-level marker. The `exec_result_assembled` atom carries `shed_reason` only when the plan's returned error is itself a shed (`ExecutionRunStateV1.result_shed`). The `failure_pressure` status term is zeroed only on that marker.
  - Evidence: `test_plan_failure_after_an_earlier_shed_step_is_still_a_failure`; `test_run_plan_marks_shed_step_and_shed_result` (real PlanRunner path).
- Finding: reverie could drop a real thought when a side step (recall) was shed but the plan succeeded.
  - Fix: a refusal counts only when the plan status is not `success`.
  - Evidence: `test_successful_plan_with_a_shed_side_step_still_yields_a_thought`.
- Finding: deferral was keyed on the raw window count, so a turn-explained first window allowed the double fire. Deferral also applied under the sustained override.
  - Fix: count from the first window that no turn explains, and never defer when the sustained override holds.
  - Evidence: `test_deferral_counts_from_the_first_window_no_turn_explains`, `test_nominal_open_is_not_deferred_under_the_sustained_override`.
- Finding: the shed reason would vanish silently if the gateway ever set `details.stage`.
  - Fix: a pin test that runs the gateway's `_pool_unavailable_result` shape through `gateway_error_step_failure`.
  - Evidence: `test_gateway_shed_reply_names_the_shed_in_the_step_error`.
- Finding: the two matchers used inconsistent match rules, and Unicode names were accepted.
  - Fix: the prefix-vs-substring difference is now documented, and shed names are ASCII-only.
- Finding: the forbid-model rollout order was not stated.
  - Fix: the consumer-first order is stated above, and the deploy command follows it.
- Finding: CI push paths were incomplete.
  - Fix: completed. The router test was added to the workflow.
- Finding: there was no PR report.
  - Fix: this file.

Choice made for item 4 of the brief ("hold escalation vs. open carries the max band"):

- I implemented and replayed both options.
- Absorbing the escalation for 2 windows cut fires to 122, but dream novelty fell from 10.0 to 7.0. The degraded band the onset reached was never reported, and dream weights only degraded and critical keys.
- Deferring a nominal open by one window keeps the novelty unchanged and cuts more fires. The open then carries the band of the first two windows, because nominal is the lowest flagged band.
- Degraded and critical opens still fire at once, so real-band onsets get no extra latency.

## Restart required

Deploy from the primary checkout, consumer-first (the readers before substrate-runtime):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-cortex-exec orion-cortex-orch orion-hub orion-equilibrium-service orion-field-digester orion-thought orion-substrate-runtime; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

## Risks / concerns

- Severity: medium.
  - Concern: during the rollout window, a reader that has not been rebuilt (old forbid model) fails to validate the projection once substrate-runtime writes the new fields. In cortex-exec this makes the metacog substrate cue go silently dark. In hub it logs a warning.
  - Mitigation: consumer-first deploy order; the post-pull rebuild order is alphabetical and also consumer-first.
- Severity: low.
  - Concern: a lone one-window nominal anomaly no longer fires.
  - Mitigation: it is threshold flicker with zero dream weight; the sustained override and every degraded/critical open are unaffected.
- Severity: low.
  - Concern: `defer_nominal_open` has no env kill switch.
  - Mitigation: it is a one-line code revert (the parameter default).
- Severity: low.
  - Concern: a pool refusal on a later chain step still ends the chain as `pressure_discharged` and arms the refractory cooldown, as before.
  - Mitigation: earlier thoughts were already published. Left unchanged to keep scope.
- Severity: low.
  - Concern: the standalone reverie tick now logs a pool refusal at info level and returns None before the hollow-narration path. Any hollow-tick counting that used to see refusals no longer does.
  - Mitigation: intended.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2819

🤖 Generated with [Claude Code](https://claude.com/claude-code)
