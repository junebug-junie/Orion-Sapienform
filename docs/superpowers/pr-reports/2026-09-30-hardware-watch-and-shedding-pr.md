## Summary

Urgent-curiosity plans 4 and 5: a dead or lying cabinet AC now raises a critical alert within one
30 s tick of the rule tripping, starts an urgent investigation, and, while the incident is open and
the cabinet is warm or warming, stops the GPU pool from starting new background/system work.

- New service `orion-hardware-watch` (athena, port 8131): AC rule, CPU heat rule, GPU heat rule,
  incidents persisted one-open-per-subject, operator resolve endpoint, simulate hook (off).
- GPU pool rule U4: one shed lever of named reasons with precedence (`orion/gpu_pool/shed.py`).
  The reflex `cooling_incident` is precedence 0 and always wins. Visible in pool `/health` and
  `GpuPoolStateV1.shed`.
- New contract `hardware.watch.incident.v1` on `orion:hardware:watch:incident` (watcher -> pool).
- GPU die temperature is now collected (`gpu{N}_temp_c`, `gpu_temp_c_max`).
- The cabinet rise computation is a pure shared function, `cabinet_rise_c`, in
  `orion/hardware_watch/rules.py`, so Orion's later learned shed action imports it (attend-to-act
  "Amendment 2026-09-29", PR #2417). Orion's learned action is NOT in this PR.
- A replay eval runs the real watcher over the real Postgres history (committed fixture).

## Outcome moved

Before: nothing alerted on an AC failure. The 1 h 38 m stretch on 2026-09-26 where the AC plug
read a constant 33.11 W (the kWh-counter bug) went unnoticed by any rule. After: the replay opens
one `low_power` incident 3 minutes into that stretch, sends one critical alert, one urgent
investigation request, sheds the pool from the first warm tick, and resolves 10 minutes after real
watts return. Nothing else fires in 3.9 days of AC history.

## Current architecture

- AC power: `orion-zwave` -> `home_cooling_sample` (5 s). Staleness flag since #2382.
- Cabinet temperature: `orion-biometrics` on athena -> `orion_biometrics_summary.measurements->>'cabinet_temp_c'` (~30 s).
- CPU temperature: `measurements->>'temp_c_max'` per node. GPU temperature: not collected.
- Heat protections that existed: the reverie render gate (`services/orion-thought/app/visual_chain.py`)
  and the pool's swap-load `thermal` guard (`services/orion-gpu-pool/app/guards.py`). Both read
  temperature only; neither knows about the AC.
- Urgent investigations (plans 1-3, merged): `CuriosityUrgentRequestV1` on
  `orion:curiosity:urgent:request`, already listing `orion-hardware-watch` as a producer.

## Architecture touched

- New service `services/orion-hardware-watch/` (reads Postgres, publishes on the bus, calls notify).
- `orion/hardware_watch/rules.py` (pure rules, shared).
- `orion/gpu_pool/shed.py` + scheduler U4 + pool runtime/main (Hunter on the incident channel).
- Contract: `orion/schemas/hardware_watch.py`, registry, `orion/bus/channels.yaml`.
- Telemetry: `orion/sensors/gpu_host_stats.sh`, `orion/telemetry/biometrics_pipeline.py`.
- DB: new table `hardware_watch_incident` (operator-applied migration).

## What fires when

| Rule | Subject | Opens | Resolves | On open |
|---|---|---|---|---|
| cooling | `cabinet_ac` | live watts < 150 for 3 min (`low_power`); no live reading 5 min (`no_samples` / `device_offline` / `controller_not_ready` / `no_fresh_sample`); identical live watts 60 min (`frozen`) | >= 500 W for 10 min and no opening arm holds | critical alert (in_app + email, `dedupe_key=cooling:<id>:alert`), then urgent request (trigger `cooling`), then incident event |
| cpu_heat | `athena`, `circe` | `temp_c_max` above own trailing-7-day p95 for 10 min (needs 1 day history) | below p75 | urgent request (trigger `heat`) |
| gpu_heat | `<node>/gpuN` | >= 85 C for 2 min (always armed); above own p95 for 10 min once 3 days of history exist | below p75 (armed) / below 80 C | urgent request (trigger `heat`) |

Shedding: only while a `cooling` incident is open. The watcher asks the pool to shed when the
cabinet is at/above the thermal gate's elevated 29.5 C, has risen >= 1.0 C in 15 min
(`cabinet_rise_c`), or is unreadable for 5 min. The request latches until the incident resolves.
Every open incident is re-published every 60 s with `shed.valid_until = now + 300 s`, so a dead
watcher stops shedding within 5 min. The pool then grants nothing new to `background`/`system`;
running work finishes, nothing is recalled, chat and urgent work are never shed.

## Three heat mechanisms

| Mechanism | Reads | Stops |
|---|---|---|
| Reverie render gate (`orion-thought/app/visual_chain.py`) | cabinet temp | new reverie renders while hot |
| Pool swap-load `thermal` guard (`orion-gpu-pool/app/guards.py`) | cabinet temp | loading an extra model onto a swap seat |
| Pool shed `cooling_incident` (U4, this PR) | watcher incident event | new background/system grants on every role |

They stack; none reads another's state. Only the shed depends on the AC. A hot cabinet with a
healthy AC trips the first two and never the third. With the AC down and the cabinet >= 32 C, all
three are active.

## Replay numbers (real history, cutoff 2026-09-30 02:00Z)

`services/orion-hardware-watch/evals/run_rules_replay_eval.py`:

- AC data: 66,404 rows since 2026-09-26 04:52. `stale=true` 0, offline 0, controller-not-ready 0,
  null watts 0, < 150 W 1,171 (the 33.11 W stretch), largest row gap 76 s. 37,523 rows predate
  #2382 (staleness unknown, counted as live). Longest identical-watts run outside the bug: 38 rows.
- AC replay: 1 incident (`low_power`, 04:55:37 -> 06:40:37, 105 min), 1 critical alert + 1
  recovery notice, 1 urgent request, shed engaged on 185 ticks (~92 min) with reason
  `cabinet_rising`. Nothing else.
- Fault injection at three real moments (54 h, 30 h, 6 h before cutoff): silence, offline and stale
  open in 300 s; 0 W opens `low_power` in 180 s; a frozen wattage opens `frozen` at 60 min. All 15 pass.
- CPU heat, 7 days: circe 5 incidents (0.71/day), athena 1 (0.14/day); an incident is open 1.8% of
  the time. Spec gate predicted ~1.3/day circe, ~0.1/day athena.
- GPU heat: no `gpu{N}_temp_c` history yet (collector ships here), so the p95 arm stays disarmed;
  only the 85 C ceiling can fire.
- Shed rule at any tick, AC incident NOT required (what the cabinet looks like to the rule on a
  normal week): requested 65.7% of ticks (elevated 65.1%, rising-but-not-elevated 0.6%),
  hot >= 32 C 6.0%. Rise >= 1.0 C/15 min is true on 1.5% of ticks, 17 episodes (~2.4/day);
  >= 0.5 C on 6.9%. This is why shedding is gated on an open AC incident: without that gate it
  would shed two thirds of the day.

## Metric quality gate (new signals)

- **Cabinet rise (`cabinet_rise_c`)**. 1. Provenance: `orion/telemetry/cabinet_sensors.py:109`
  (`cabinet_temp_c` from the cabinet Nano snapshot) -> `orion_biometrics_summary`. 2. Independence:
  a transform of cabinet temperature, which the elevated leg also reads, so the two legs are not
  independent votes; they are OR-ed, and only the rise leg is decisive on 0.6% of ticks. It is
  independent of AC watts (different device), which is the point: it corroborates a watts-based
  incident. 3. Theory: with the compressor off, the cabinet's heat balance is the compute load, so
  temperature rises; a rise over a fixed window is the direct measure. 4. Live data: 7 days, varies,
  returns to 0 (rise is 0.0 on flat or falling windows; 15-min SD ~0.42 C per PR #2417), not
  saturated (true 1.5% of ticks). 5. Existing mechanism: thermal_gate has level, not rise. 6.
  Reversibility: an env threshold and a pure function; nothing persisted except `shed_reason`.
- **GPU die temperature (`gpu{N}_temp_c`)**. Provenance: `nvidia-smi temperature.gpu` in
  `gpu_host_stats.sh`. No live data exists yet, so the p95 arm is disarmed until 3 days of history
  (enforced by `HARDWARE_WATCH_GPU_MIN_HISTORY_SEC`, tested). Re-run the gate on real data before
  relying on the p95 arm.
- **CPU `temp_c_max`**: gate in the spec (2026-09-28); the replay reproduces its rates.

## Files changed

- `orion/hardware_watch/rules.py`: pure rules; `cabinet_rise_c` shared function.
- `orion/gpu_pool/shed.py`, `orion/gpu_pool/scheduler.py`: shed board + rule U4.
- `services/orion-gpu-pool/app/{main,runtime,settings}.py`, `.env_example`, `docker-compose.yml`: incident consumer, shed state in health/state, `GPU_POOL_SHED_ENABLED`.
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`, `tests/test_runtime_shed.py`, `orion/gpu_pool/tests/test_scheduler_shed.py`: U4 coverage.
- `orion/schemas/hardware_watch.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: contract.
- `orion/schemas/gpu_pool.py`: optional `GpuPoolStateV1.shed` block.
- `orion/sensors/gpu_host_stats.sh`, `orion/telemetry/biometrics_pipeline.py` + tests: GPU temperature.
- `services/orion-hardware-watch/**`: the service, tests, replay eval + fixture.
- `services/orion-sql-db/manual_migration_hardware_watch_v1.sql`: incident table.
- `.github/workflows/orion-hardware-watch-tests.yml` (new), `orion-gpu-pool-tests.yml` (path).
- `config/metrics/metric_definitions.lock.json`: re-lock for the new channel.
- `docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md`: the plan.

## Schema / bus / API changes

- Added: `HardwareWatchIncidentV1`, `HardwareWatchShedV1` (`hardware.watch.incident.v1`) on
  `orion:hardware:watch:incident`; producer `orion-hardware-watch`, consumer `orion-gpu-pool`.
- Added: optional `GpuPoolStateV1.shed` (dict, default empty). Pool `/health` gains `shed`.
- Added: table `hardware_watch_incident`. HTTP on the watcher: `GET /health`, `GET /incidents`,
  `POST /incidents/{id}/resolve`, `POST /incidents/simulate` (test hook, off).
- Added: measurement keys `gpu_temp_c_max`, `gpu{N}_temp_c`; CSV column `temperature_gpu_c` (last).
- Removed / renamed: none.
- Compatibility: shed facts ride an existing `queued` pool event with `reason="shed:<name>"`; no
  new pool event name, no pool YAML key. `GpuPoolStateV1.shed` has no validating consumer outside
  the pool (Hub/harness-governor read it as a dict). Deploy the pool before the watcher.

## Env/config changes

- Added keys (orion-hardware-watch, new): see `services/orion-hardware-watch/.env_example`;
  `HARDWARE_WATCH_ENABLED`, `HARDWARE_WATCH_SHED_ENABLED`, `HARDWARE_WATCH_URGENT_ENABLED` false in
  code, true in `.env_example`; `HARDWARE_WATCH_TEST_HOOK_ENABLED` false everywhere.
- Added keys (orion-gpu-pool): `GPU_POOL_SHED_ENABLED` (false in code, true in `.env_example`).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes for orion-gpu-pool.
  orion-hardware-watch is a new service the sync script cannot bootstrap; its primary-checkout
  `.env` was created from `.env_example` with `POSTGRES_URI` copied from orion-gpu-pool's `.env`.
- Skipped keys requiring operator action: none. `NOTIFY_API_TOKEN` is empty, like every other
  service on athena.

## Tests run

```text
PYTHONPATH=. pytest orion/gpu_pool/tests -q                                   318 passed
cd services/orion-gpu-pool && pytest tests -q                                 104 passed, 9 skipped
cd services/orion-hardware-watch && pytest tests -q                           52 passed
cd services/orion-biometrics && pytest tests/test_gpu_collector.py -q        4 passed
pytest tests/test_biometrics_measurements.py -q                               41 passed, 1 failed (pre-existing on main, see Risks)
static gates (metric lineage, definition drift after re-lock, inner-state registry, hostname refs,
compose mounts, health producers, async routes, chat poachers, env template parity,
static-gates pytest set)                                                      all pass
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py                     VERDICT: PASS (incl. shed scenario)
cd services/orion-hardware-watch && python evals/run_rules_replay_eval.py     PASS (numbers above)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hardware-watch build                       Image built
docker run ... python -c "import app.main, app.watcher"                       import ok
migration applied inside BEGIN ... ROLLBACK on conjourney                     CREATE TABLE/INDEX/DO ok, 20 columns, rolled back
PostgresStore read queries against live Postgres (read-only)                  cooling 773 rows 0.02 s; cabinet 0.003 s;
                                                                              7-day circe baseline 19,293 rows 0.2 s (p75 59, p95 62);
                                                                              gpu_keys [] (not collected yet); check_schema refuses (no table yet)
Live deploy: NOT done (production deploys from the primary checkout on main). UNVERIFIED live.
```

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`.

- Finding: resolving a drill (`simulated`) snoozed the real AC rule for an hour; an open drill blocked a real incident.
  - Fix: no snooze for drills; a real AC verdict closes the drill (`superseded`) and opens the real incident.
  - Evidence: `test_resolving_a_drill_does_not_snooze_the_real_ac_rule`, `test_a_real_failure_supersedes_an_open_drill`.
- Finding: an operator resolve could interleave with a tick, double-resolve, or re-publish `requested=True` for a closed incident (pool re-sheds up to 300 s).
  - Fix: one asyncio lock around tick / resolve / simulate; resolve is a conditional `UPDATE ... WHERE status='open'` (`resolve_incident`), a second resolve is a no-op.
  - Evidence: `test_a_second_resolve_is_a_no_op`.
- Finding: a failed cabinet-temperature query blocked the critical AC alert.
  - Fix: a failed read counts as `cabinet_unreadable` (sheds), the alert still goes out.
  - Evidence: `test_cabinet_query_failure_still_sends_the_alert_and_sheds_as_unreadable`.
- Finding: `HARDWARE_WATCH_SHED_ENABLED=false` did not stop an already-latched request.
  - Fix: `_emit` requires the switch; reason reported as `disabled`.
  - Evidence: `test_turning_the_watcher_shed_switch_off_stops_a_latched_request`.
- Finding (material, kept by design): the elevated leg (29.5 C) is true 65% of the time, so in practice "AC incident open => shed".
  - Decision: this is the spec's Juniper-locked rule (Decisions locked + Part 4, and PR #2417 relies on it). Stated plainly in this report and the README; shedding still never engages without an open AC incident (eval asserts it). Raising it to 32 C would be a knob change for Juniper, not a finding.
- Minor fixed: failed `resolved` event is retried next tick (`test_a_failed_resolved_event_is_retried_next_tick`); open heat incidents are re-evaluated even if the card/node stops reporting (`test_open_gpu_incident_is_evaluated_after_its_card_stops_reporting`); settings refuse `refresh/tick >= shed_valid` (`test_refresh_must_be_shorter_than_shed_validity`); the event now carries `cabinet_temp_c`/`cabinet_rise_c` (`test_event_carries_the_cabinet_temperature_and_rise`); cooling query filters `role` (`HARDWARE_WATCH_AC_ROLE=cabinet_cooling`, confirmed the only live role); f-string nit.
- Minor not changed: duplicate urgent request after a DB write failure -- Hub already holds one open run per `incident_id` (`services/orion-hub/scripts/curiosity_urgent.py`). Shed backlogged leases still dead-letter at `backlog_max_age` -- deadlines apply under shed by design, now commented in the scheduler. Future-timestamp readings count as fresh (clock skew) -- left. No Postgres-backed test of the store: covered by the live read-only query check above and the rolled-back migration; the second-insert-returns-False check is in the live proof steps.

## Restart required

```bash
# on athena, primary checkout, after merge + git pull on main
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_hardware_watch_v1.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-pool up -d --build        # consumer first
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hardware-watch up -d --build  # then producer
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-biometrics up -d --build      # athena GPU temps
# on circe, its own checkout: git pull on main (gpu_host_stats.sh is bind-mounted from the host
# checkout), then rebuild biometrics so the pipeline publishes gpu{N}_temp_c
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-biometrics up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the notify service's `critical_default` rule has a throttle (10 per
  5 min, shared by all critical notices). A refused alert is retried once per tick up to
  `HARDWARE_WATCH_ALERT_MAX_ATTEMPTS=60`, so it delays rather than drops, but the spec says the
  alert is never throttled. Mitigation: retry loop; follow-up could add a notify rule for
  `hardware.watch.cooling.alert` without a throttle.
- Severity: medium. Concern: the cabinet is elevated on ~65% of ticks, so almost every AC incident
  sheds immediately, including a false-positive AC incident (e.g. Z-Wave down with the AC fine).
  Mitigation: that is the intended direction (a dead AC in a 30 C room is the dangerous case);
  Juniper can close a false alarm with `POST /incidents/{id}/resolve`, which clears the shed.
- Severity: low. Concern: GPU p95 arm has no data; it arms itself after 3 days of collection.
  Re-run the metric gate on real GPU temperatures before trusting it.
- Severity: low. Concern: `tests/test_biometrics_measurements.py::test_review_disk_and_net_rates_are_not_promoted_to_physical_quantities`
  fails on origin/main too (not touched here; not run by any CI workflow).
- Severity: low. Concern: replay fixture is 1.0 MB gzip committed to the repo.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2424

🤖 Generated with [Claude Code](https://claude.com/claude-code)
