# Vision focus experiment: Orion's first real switchback

## Summary

- Orion can now run a real randomised experiment on itself: "if I look harder at the walkway camera, does the cabinet get hotter, and by how much?"
- The vision frame router honours a new Redis key, `orion:vision:experiment:<stream_id> = on`. The key must carry a TTL. While it is set, the router forces that camera onto its "triggered" tier (more frames to the GPU, plus captions when the detector sees something). If the key is absent, set to anything else, has no TTL, or Redis is down, the router behaves exactly as it does today.
- orion-prediction gets a runner, `POST /experiments/run`. It designs a randomised on/off schedule with the existing design code and refuses to start unless the overheat guard passes. It sets and clears the key block by block, logs every block through the existing intervention path, and aborts if the heat rises. At the end it runs the existing analysis.
- The runner also does a manipulation check: did the ON blocks actually change anything? It reads the router's dispatch tier counts per block, and GPU/chassis watts ON minus OFF. So "no measurable effect" can be told apart from "the switch did nothing".
- Nothing auto-starts. This PR does not deploy anything and does not start the experiment.

## Outcome moved

- Before: `prediction_experiment` and `prediction_intervention` were empty. The switchback design and analysis code existed, but nothing could perform an action. The only evidence was observational and confounded by the daily rhythm (+8.6 W GPU and +56 W chassis inside walkway expectation windows).
- After: one POST runs the whole experiment end to end and stores an `experimental_effect` estimate. `GET /effects/vision.focus:walkway?series_id=athena.cabinet_temp_c` then returns it.

## Current architecture

- Router (`services/orion-vision-frame-router/app/policy.py` `_tier_config`): the triggered tier is chosen by recent trigger labels, or by `orion:vision:expect:<stream_id>`. A background `ExpectationCache` reads that key every `ROUTER_EXPECTATION_REFRESH_SEC` (`app/expectation.py`), and `decide()` stays sync with no network calls.
- Prediction (`services/orion-prediction/app/effects.py`): `design_experiment`, `log_intervention`, `analyze_experiment`, `guard`. The old README said "No dispatcher exists".

## Architecture touched

- Redis key contract: `orion:vision:experiment:<stream_id>`. Its only writer is orion-prediction and its only reader is the router. It is not a bus channel and not a schema, so `channels.yaml` and the registry do not change.
- Runner lock: `orion:prediction:experiment:runner` (SET NX EX, refreshed every tick).
- HTTP: `POST /experiments/run`, `GET /experiments/runner`, `POST /experiments/runner/abort` on orion-prediction. The router's `/healthz` gains `experiment_override_*` and `dispatch_tier_counts`.
- orion-prediction calls the router's `/healthz` over app-net (`http://orion-vision-frame-router:8010`). This works live from the prediction container today: checked 2026-10-29 with urllib from inside `orion-athena-prediction`.
- The runner lives in orion-prediction, not orion-self-experiments. This service already owns the design, the guard, the log, and the analysis, and it already has the bus Redis URL. Putting it elsewhere would add a second service and another config surface for the same loop.

## Files changed

- `services/orion-vision-frame-router/app/expectation.py`: reads the experiment key in the same pipeline. `on` plus a TTL forces the tier. A key with no TTL is refused and counted.
- `services/orion-vision-frame-router/app/policy.py`: the experiment check comes first, so `triggered_by="experiment"`.
- `services/orion-vision-frame-router/app/metrics.py`, `app/dispatcher.py`, `app/main.py`: per-stream dispatch tier counts. The override is shown on `/healthz` and in the heartbeat. One cache runs when either flag is on.
- `services/orion-vision-frame-router/app/settings.py`, `.env_example`, `docker-compose.yml`: `ROUTER_EXPERIMENT_OVERRIDE_ENABLED=true`.
- `services/orion-vision-frame-router/tests/test_experiment_override.py` (new), `tests/test_walkway_policy_and_expectation.py` (the fake pipeline now answers get/pttl).
- `services/orion-prediction/app/experiment_runner.py` (new): runner, router probe, manipulation check, boot recovery.
- `services/orion-prediction/app/main.py`: endpoints, lifespan wiring (Redis client, `recover()` at boot, `shutdown()` on stop), and `/health.experiment_runner`.
- `services/orion-prediction/app/settings.py`, `.env_example`, `docker-compose.yml`: five `PREDICTION_EXPERIMENT_*` / `PREDICTION_VISION_ROUTER_URL` keys.
- `services/orion-prediction/tests/test_experiment_runner.py` (new).
- `services/orion-prediction/evals/run_causal_calibration_eval.py`: adds a trial with the runner's own design (12 x 20 min, 10 min burn-in, thermal-lag ramp).
- Both READMEs and `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`.
- Not touched: `services/orion-prediction/app/series.py`, `app/store.py`, `config/prediction/series.v1.yaml`. Another agent is editing those. GPU watts is read with an ad hoc spec inside the runner instead of a registry entry.

## Schema / bus / API changes

- Added: Redis key `orion:vision:experiment:<stream_id>`, Redis lock `orion:prediction:experiment:runner`, three HTTP routes on orion-prediction, and new `/healthz` fields on the router.
- Removed: none.
- Renamed: none.
- Behavior changed: `triggered_by` can now also be `"experiment"` in host task meta. `RouterMetrics.record_dispatch()` takes optional stream/tier args; the old no-arg call still works.
- Compatibility notes: no DB migration. An abort uses the existing status `cancelled` and an intervention row with `arm` NULL. The analysis already ignores rows with no arm, so an abort row never breaks the design match.

## Env/config changes

- Added keys: `ROUTER_EXPERIMENT_OVERRIDE_ENABLED`, `PREDICTION_EXPERIMENT_RUNNER_ENABLED`, `PREDICTION_VISION_ROUTER_URL`, `PREDICTION_EXPERIMENT_OVERRIDE_TTL_SEC`, `PREDICTION_EXPERIMENT_TICK_SEC`, `PREDICTION_EXPERIMENT_GUARD_EVERY_SEC`. All flags ship ON.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (both services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`--all-keys orion-vision-frame-router`, `--all-keys orion-prediction`). All six keys were added.
- skipped keys requiring operator action: none. Redis uses the existing `ORION_BUS_URL` (`redis://100.92.216.81:6379/0`); no new URL.

## Proposal mode

- **Capability change:** Orion switches one of its own perception settings (walkway camera focus) on and off on a random schedule, so it can learn what that costs in heat.
- **Data touched:** reads biometrics (`cabinet_temp_c`, `gpu_watts_total`, chassis watts) and the router's `/healthz`. Writes `prediction_experiment`, `prediction_intervention`, `prediction_effect_estimate`, and two Redis keys.
- **Privacy boundary:** none new. The ON arm makes the walkway camera look more often, which means more frames, crops, and captions of passers-by than normal operation for those blocks. It uses the same pipeline and the same storage rules that already apply whenever someone walks past.
- **Trace that proves it worked:** 12 `prediction_intervention` rows with `source=orion-prediction.experiment-runner`, an arm, and an `experiment_id`. Each row's `notes` holds the router dispatch delta for that block. ON rows should show `triggered:experiment` counts. There will also be one `experimental_effect` row for `athena.cabinet_temp_c`, and the runner result's `manipulation.verdict`.
- **Dangerous failure:** focus stays on during a heat event. Three things stop it. (1) The runner checks the cabinet every tick and aborts at the guard line (32 °C). (2) The key has a 90 s TTL, so a dead runner leaves focus on for at most 90 s, and the router ignores an `on` key with no TTL. (3) The 34 °C hardware-watch reflex is unchanged. The ON arm only adds load that already happens every day: the walkway is on this tier for most of the daylight hours.
- **Disable / roll back:** `POST /experiments/runner/abort`; or `ROUTER_EXPERIMENT_OVERRIDE_ENABLED=false` on the router; or `PREDICTION_EXPERIMENT_RUNNER_ENABLED=false`; or `redis-cli -u $ORION_BUS_URL DEL orion:vision:experiment:walkway`.

## What effect size can this detect? (plain English)

- I measured this on 17 days of real cabinet data (2026-10-12..29). I replayed the exact 12 x 20 min design, with a 10 min burn-in, at 45 random start times, with no effect planted. The uncertainty on the ON-minus-OFF difference came out around ±0.29 °C (one standard error). That means **only an effect of about 0.8 °C or more would reliably show up** (80 % power). False alarms stayed at 4 %, close to the nominal 5 %.
- More blocks barely help. 24 blocks (8 hours) only brings this to about 0.7 °C, because the cabinet drifts with the AC cycle and that drift does not average out quickly. So 12 blocks (4 hours) is kept.
- Expected effect: tiny. On the GPU, the triggered tier only drops the gap between walkway tasks from 3 s to 2 s. Live logs show about 1,435 walkway tasks per hour on the triggered tier against about 600-700 per hour on baseline. Observationally that is about +9 W of GPU, out of a cabinet holding about 770 W on athena alone. **The honest prediction is "no measurable cabinet effect", with an upper bound of roughly ±0.6 °C.** That is still a real result. It tells Orion that walkway focus is thermally cheap.
- The manipulation check on GPU watts can see about 10-12 W at 12 blocks, so a +9 W change is borderline there. The router's dispatch counts are the strong check: about twice as many tasks per hour.
- **When to run it:** at night. From about 16:00 to 23:00 UTC (10:00-17:00 MDT), the walkway is already on the triggered tier almost all the time. The patio skeleton reads as a person. In those hours the OFF arm is already "focused", so the ON arm changes nothing. The runner reports this as `no_lever_off_arm_already_triggered`. On 2026-10-28, the walkway was mostly on baseline from 09:00 to 14:00 UTC (03:00-08:00 MDT). Start the run around 09:00 UTC, when the cabinet is also coolest and the guard passes easily.

## Metric gate (manipulation-check signals)

- `athena.gpu_watts_total`: produced by the biometrics pipeline (`orion/telemetry/biometrics_pipeline.py`) and written to `orion_biometrics_summary.measurements` (also flattened at `services/orion-sql-writer/app/worker.py:1750`). It is live, at a 30 s cadence: mean 104 W, SD 13.5 W over 17 days. It is not flat, and it varies with GPU work. It is not independent of `athena.chassis_watts`: chassis watts already includes the GPU draw. Both are reported as checks of the switch, and neither is a predictor in any model. This is cheap to remove: it is an ad hoc spec in the runner, not a registry entry.
- Router `dispatch_tier_counts`: a direct count of the thing being switched. It is the primary check.

## Tests run

```text
cd services/orion-vision-frame-router && python -m pytest tests -q -p no:cacheprovider
  123 passed
cd services/orion-prediction && PYTHONPATH=../..:. python -m pytest tests -q -p no:cacheprovider
  187 passed (23 new in tests/test_experiment_runner.py)
python scripts/check_env_template_parity.py  -> PASS
```

## Evals run

```text
cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_causal_calibration_eval.py
switchback: coverage95=0.97 (>=0.88)  null FPR=0.05 (<=0.12)
runner design 12x20: coverage95=0.93 (>=0.85)  power=1.00 (>=0.80)  null FPR=0.05 (<=0.12)
ITS: coverage95=0.95 over 20/20 trials with an interval (>=0.80)
RESULT: PASS
```

The power figure is for a planted 0.6 °C on synthetic data with low noise. The live-noise power for the real cabinet is the 0.8 °C figure above.

## Docker/build/smoke checks

```text
No Docker build run (worktree; not deployed by design). Live checks done read-only:
- prediction container -> http://orion-vision-frame-router:8010/healthz reachable (200)
- GET /experiments/guard?series_id=athena.cabinet_temp_c -> {"safe": true, "probability": 0.0}
Live path of the runner: UNVERIFIED until deploy + first run.
```

## Review findings fixed

A code review subagent reviewed `origin/main...HEAD`. It found no must-fix bugs and six should-fix items. All six and three of the nits are fixed.

- Finding: the lock refresh did not check who owned the lock. After a stall longer than 180 s, a second run could take the lock, and the first run would then overwrite it.
  - Fix: refresh and release are now Lua compare-and-set operations (`LUA_REFRESH` / `LUA_RELEASE`). "Not mine" aborts the run.
  - Evidence: `test_another_run_taking_the_lock_aborts_this_one_and_is_not_overwritten`.
- Finding: a hard crash before block 0 was logged could leave an experiment in `designed` status forever once the lock had expired.
  - Fix: a no-TTL Redis record, `orion:prediction:experiment:active`, holds the run's id. It is deleted only once the run's status is final. `recover()` reads it. If anything fails during `start()` after the design is stored, the experiment is marked cancelled.
  - Evidence: `test_crash_before_the_first_block_is_still_recovered_after_the_lock_expired`.
- Finding: a failure after all blocks had run could mark a complete experiment `cancelled`.
  - Fix: the end-of-run analysis is now outside the abort path. On failure the error goes into the result and the DB status stays `designed`, so `POST /experiments/{id}/analyze` still works.
  - Evidence: `test_analysis_failure_after_all_blocks_keeps_the_experiment_analyzable`.
- Finding: an ON block could be logged as performed even though focus had lapsed (a stall longer than the TTL) or the block had started late.
  - Fix: the runner aborts if the key goes a full TTL without a refresh, and aborts if an ON block would start more than 2 min late.
  - Evidence: `test_a_stall_longer_than_the_ttl_inside_an_on_block_aborts_instead_of_logging_it`.
- Finding: a mistyped stream id, or a stream with no triggered tier, would run 4 hours as a no-op.
  - Fix: the runner refuses at start if the router has never dispatched for that stream. It aborts after the first ON block if that block produced no `triggered:experiment` dispatches.
  - Evidence: `test_unknown_stream_is_refused_before_anything_is_designed`, `test_override_that_never_takes_effect_aborts_after_the_first_on_block`.
- Finding: a TTL shorter than two ticks raised an error inside lifespan, so the whole service failed to boot.
  - Fix: a settings validator now rejects that config up front, with a clear message.
  - Evidence: `test_settings_refuse_a_ttl_shorter_than_two_ticks`.
- Nits fixed:
  - The guard no longer runs twice back to back before an ON block.
  - Cleanup now survives a second cancel.
  - The router replaces the outer `dispatch_tier_counts` dict, so `/healthz` cannot hit a "dict changed size" error.
  - The fake-only TTL test half was replaced by `test_every_key_write_carries_the_ttl`.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
./scripts/safe_docker_build.sh orion-prediction up -d --build
```

Deploy from the clean primary main checkout after merge. Then start the experiment at about 09:00 UTC:

```bash
curl -fsS -X POST http://localhost:8132/experiments/run -H 'Content-Type: application/json' -d '{"action":"vision.focus:walkway"}'
```

## Risks / concerns

- Severity: medium. Concern: the cabinet effect is probably below what 12 blocks can detect (about 0.8 °C). Mitigation: the result is reported with its interval. "No measurable effect" is an expected and useful outcome.
- Severity: medium. Concern: a daytime run has no lever, because the OFF arm is already triggered. Mitigation: start at night. The manipulation verdict reports it if this happens.
- Severity: low. Concern: a hard kill leaves focus on for up to 90 s. Mitigation: the TTL, plus boot recovery that marks the run cancelled.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2962
