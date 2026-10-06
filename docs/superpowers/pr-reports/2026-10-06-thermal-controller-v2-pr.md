## Summary

Orion's heat protection now looks at how hot the cabinet actually is, not at how much power the AC draws. Before this, it shed GPU work in a 26 °C cabinet on a cool night, and did nothing while the cabinet sat at 31–33.5 °C for most of two days.

- **The reflex follows the cabinet.** hardware-watch reads the cabinet once per tick. At 34 °C or above (released below 33 °C), or when the sensor goes silent past a 5-minute grace, it tells the GPU pool to hold back work. It re-sends that every tick and stops when the cabinet cools. Nothing latches any more. (D1, D2)
- **AC power is a diagnosis.** A low AC opens an alert only while the cabinet is also warm. Those incidents never shed. (D5)
- **No more email or investigation storms.** At most one email and one urgent investigation per rule+subject per 6 h. GPU/CPU heat opens only at a fixed ceiling; the old p95 trigger is now a note on the incident. (D6, D7)
- **Human turns never wait out a heat shed.** A one-shot background request is refused at once, so its caller falls back right away. orion-mind's Hub-turn calls use a new `metacog_turn` route at interactive priority, which is never shed. (D3, D4)
- **Orion gets the 29.5–34 °C band.** Orion's learned shed becomes reachable there; only the reflex's own signal blocks it now. (D8, approved)
- **Smaller fixes:** an operator resolve snoozes only the reason it closed (D9). The swap guard starts as "unknown" instead of "hot", and one failed read no longer counts as unknown (D10). A new report measures whether shedding cools anything (D11).

Spec: `docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md` (PR #2498).

## Outcome moved

The 7-day replay of real data (2026-09-29 → 2026-10-06 04:30) through the real Watcher, checked hour by hour.

**Before (main): 55 failure lines.**
- The 10-06 cool night had cooling incidents open on 319 ticks, and sent 2 emails.
- In 39 hours at 29.5–34 °C on 10-04/05, Orion's learned shed was never eligible. It was refused for "not rising", for "hot", or because some incident was open.
- At ≥ 34 °C there was no shed at all (checked on 10-05 shifted up by 1 °C).
- The 10-03 sensor outage shed background **and** system work, and kept shedding 100+ s after readings came back.
- There were 8 GPU and 8 CPU heat incidents from the p95 rule.
- Emails went out less than 6 h apart.

**After: PASS.**
- **10-06 00:00–04:00:** no shed, no cooling incident, no email.
- **10-04/05, every hour at 29.5–34 °C:** no reflex shed, and the learned shed was eligible on 100 % of the ticks where the cabinet read elevated or hot.
- **≥ 34 °C (10-05 shifted +1 °C):** `cabinet_hot` within one tick on all 95 hot ticks, and no shed below the 33 °C release point.
- **10-03 outage:** one alert. The shed was background-only (`cabinet_unknown`). It cleared within 3 ticks after readings resumed at 21:14:39 and at 22:23:48.
- **GPU/CPU:** 0 heat incidents below the ceiling.
- **Email:** 1 for the whole week.
- **Share of ticks:** 33.1 % at ≥ 29.5 °C, 3.0 % at ≥ 32 °C, 0 % at ≥ 34 °C. The reflex was active on 0.79 % of ticks, all of it the 10-03 outage. The learned shed was eligible on 40 %.
- **Synthetic AC death on real history:**
  - on a hot afternoon it opened `low_power` after 750 s;
  - on the cool night it opened nothing;
  - AC dead plus sensor dead gave `cabinet_unknown` at 300 s, then `cabinet_hot` at 750 s.

## Current architecture

Before this patch:
- The hardware-watch cooling rule opened on AC watts under 150 W for 180 s.
- The shed verdict was only computed inside an open cooling incident, and latched until the incident resolved. It fired on ≥ 29.5 °C, on a 1 °C rise, or on an unreadable sensor.
- The pool read the shed from incident events (`cooling_incident`), plus an `_open_ac_incidents` set that never expired.
- The learned shed was eligible only at `elevated` + rising, with zero open incidents of any rule.
- The swap guard started in the "hot" state.
- Shed requests waited out their deadlines.

## Architecture touched

- **Contract:** new `HardwareWatchReflexShedV1` on `orion:hardware:watch:reflex_shed`. hardware-watch produces it; gpu-pool consumes it.
- **Shed board:** reasons `cabinet_hot` and `cabinet_unknown`. `cooling_incident` stays only for the v1 rollback.
- **hardware-watch:** watcher v2 tick, store dedupe/snooze queries, settings, `/health` (`reflex_shed`, `heat_controller`).
- **gpu-pool:** runtime reflex consumer, swap guard, scheduler U4 (D3).
- **Routes:** `config/gpu_pool.yaml` and `orion/llm/routes.py` (`metacog_turn`).
- **orion-mind:** `engine.py` routes live-turn metacog calls to `metacog_turn`.
- **autonomy:** `cabinet_heat.py`, `thermal_gate.py`, `self_shed.py`.

## Files changed

- `orion/autonomy/thermal_gate.py`: `DEFAULT_CRITICAL_C=34.0`, release at 33.0.
- `orion/autonomy/cabinet_heat.py`: adds `minutes_to_hot`, `reading_age_sec`, `critical`, `ac_low`, `effective_state` and `reflex`; grace window; seeding from the previous reading.
- `orion/autonomy/self_shed.py`: D8 eligibility; `HardwareWatchView.reflex_reason`.
- `orion/hardware_watch/rules.py`: `cooling_verdict_v2`, `ac_mean_w`, `AcLowConfig`; heat `p95_opens=False` and `lost_sec`.
- `orion/hardware_watch/shed_effect.py`: D11 reducer with a matched no-shed control.
- `orion/schemas/hardware_watch.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: new schema and channel.
- `orion/gpu_pool/shed.py`: new reasons, `REFLEX_REASONS`.
- `orion/gpu_pool/orion_shed.py`: `preempt_by_reflex(source)`.
- `orion/gpu_pool/scheduler.py`: D3 one-shot refusal.
- `services/orion-gpu-pool/app/{runtime.py,main.py,guards.py}`: reflex consumer, subscription, D10.
- `services/orion-hardware-watch/app/{watcher.py,store.py,settings.py,main.py}`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-hardware-watch/evals/{export_thermal_v2_fixture.py,run_rules_replay_eval.py,fixtures/thermal_v2_replay.csv.gz}`: read-only export (1.9 MB, full resolution) and the 7-day gate.
- `config/gpu_pool.yaml`, `orion/llm/routes.py`, `orion/gpu_pool/tests/fixtures_routes_compat_golden.json`: `metacog_turn`; every route written in long form with an explicit priority.
- `services/orion-mind/{app/engine.py,app/settings.py,.env_example,docker-compose.yml}`: `MIND_TURN_MODEL_ROUTE`.
- `config/metrics/metric_definitions.lock.json`: locks the new channel.
- **Tests:**
  - `tests/test_cabinet_heat_v2.py`
  - `services/orion-hardware-watch/tests/{test_watcher_v2.py,test_shed_effect.py}`
  - `orion/gpu_pool/tests/test_route_priorities_explicit.py`
  - `services/orion-mind/tests/test_mind_turn_route.py`
  - updated pool, guard, scheduler, route, proposal-runtime and shed-pipeline tests

## Schema / bus / API changes

- **Added:**
  - `HardwareWatchReflexShedV1` (`hardware.watch.reflex_shed.v1`) on `orion:hardware:watch:reflex_shed`;
  - shed reasons `cabinet_hot` (background + system) and `cabinet_unknown` (background);
  - route `metacog_turn` (metacog class, interactive priority);
  - `/health` fields `reflex_shed` and `heat_controller`.
- **Removed:** the pool's `_open_ac_incidents` set.
- **Renamed:** none.
- **Behavior changed:**
  - v2 cooling incidents carry `shed=None`;
  - the learned shed's refusal is now `reflex_active:<reason>` instead of `hardware_watch_incident_open`;
  - one-shot shed requests return `unavailable` with reason `shed:<reason>`;
  - `routes.agent` and `routes.quick` are written in long form; still `system`, so no behavior change.
- **Compatibility notes:**
  - The new message is additive, on a new channel; `HardwareWatchIncidentV1` is unchanged.
  - Deploy the pool (consumer) before hardware-watch.
  - The pool must have `metacog_turn` before orion-mind sends it. Otherwise Mind gets `route_not_in_pool` and fails open.
- **No database migration.**
  - D9 keys the snooze on the resolved row's own `open_reason`, so the spec's proposed `snooze_reason` column is not needed.
  - The D6 dedupe is a query over the existing `alert_sent_at` and `urgent_requested_at` columns.

## Env/config changes

- **Added keys:**
  - orion-hardware-watch: `HARDWARE_WATCH_HEAT_CONTROLLER=v2`, `HARDWARE_WATCH_READING_GRACE_SEC=300`, `HARDWARE_WATCH_HEAT_LOOKAHEAD_MIN=20`, `HARDWARE_WATCH_AC_LOW_MEAN_W=140`, `HARDWARE_WATCH_AC_LOW_WINDOW_SEC=900`, `HARDWARE_WATCH_ALERT_DEDUPE_WINDOW_SEC=21600`, `HARDWARE_WATCH_CPU_CEILING_C=90`, `HARDWARE_WATCH_CPU_CEILING_REARM_C=85`, `HARDWARE_WATCH_HEAT_SENSOR_LOST_SEC=900`
  - orion-mind: `MIND_TURN_MODEL_ROUTE=metacog_turn`
- **Removed keys:** none. `HARDWARE_WATCH_SHED_RISE_C` and `_WINDOW_SEC` are kept: the v1 rollback path still needs them for one week. The spec's "retire" waits until v1 is deleted.
- **`.env_example` updated:** yes, plus `settings.py` and compose. Flags ship ON: v2 is the default in code, compose and `.env_example`.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py --all-keys orion-hardware-watch` (and orion-mind). This wrote to the primary checkout's `services/*/.env`.
- **Skipped keys requiring operator action:** none. One diverged key was reported and left alone: hardware-watch `POSTGRES_URI`, which is a local secret.

## Defaults that Juniper has not confirmed

- **Q3:** CPU ceiling 90 °C, released at 85 °C.
- **Q4:** split routes instead of making `agent` interactive. Only `metacog_turn` was added. `quick_turn` and `agent_turn` were dropped because nothing calls them (see Deviations).
- **Q6:** at most one urgent investigation per rule+subject per 6 h sliding window.

## Metric gate (D11: cabinet ΔT after a shed)

1. **Where the numbers come from.**
   - `cabinet_temp_c`: `orion/telemetry/cabinet_sensors.py:109`.
   - `gpu_watts_total`: `orion/telemetry/biometrics_pipeline.py:194`, both via `orion_biometrics_summary`.
   - Shed episodes: `hardware_watch_incident.shed_requested_at` → `resolved_at` (v1), and `gpu_pool_orion_shed.started_at` → `ended_at`.
2. **Independence.**
   - It reads only the outcome window (start to +30 min), never the trigger window.
   - The cabinet falls back toward normal on its own after any high reading. So the result is the change minus the same change at no-shed times that match on temperature (±0.5 °C) **and** on the 15-minute rise before the start (±0.3 °C, the trigger's own quantity).
   - The AC compressor cycle is still a confounder. The GPU-watt drop is reported next to every number so this stays visible.
3. **Why it should measure anything.** GPU electrical power ends up as heat inside the cabinet. Holding back GPU work lowers heat input, and should cool the cabinet compared with the matched control.
4. **Live data check.**
   - The control at 32 °C on 10-04/05 has n=152, ranges from −1.02 to +0.955 °C, has a median of 0.0, and is negative 49 % of the time. It is not stuck, and it can read a genuine rest of 0.
   - Real episodes: 2. The 10-03 one has no cabinet data, because it is the outage itself.
   - The 10-06 02:12 episode reads −0.96 °C against its control (n=7). But GPU draw fell only 6 W (93.6 → 87.6 W), which cannot explain a 1 °C drop. The likely cause is the AC compressor cycle.
   - The report says `enough_to_judge: false` until there are at least 3 measured episodes.
5. **Existing mechanism:** none; the spec calls this gap C14.
6. **Reversibility:** a pure function plus the eval report. No schema, nothing persisted.

## Tests run

```text
services/orion-hardware-watch tests (PYTHONPATH=../..:.)                 73 passed
tests/test_cabinet_heat_v2.py tests/test_shed_background_gpu_pipeline.py
  tests/test_thermal_gate.py tests/test_agent_trace_schema_registry.py orion/autonomy   290 passed
orion/gpu_pool + orion/llm                                                391+ passed
services/orion-gpu-pool tests                                             154 passed, 15 skipped
services/orion-llm-gateway tests                                          384 passed
services/orion-mind tests                                                 118 passed
services/orion-proposal-runtime tests                                     7 passed
services/orion-durable-runs tests                                         267 passed, 72 skipped
services/orion-feedback-runtime tests (from repo root)                    62 passed
services/orion-substrate-runtime test_cabinet_heat_tick.py                4 passed
  (full suite: 17 failures + 1 collection error, all needing a live Postgres on
   localhost:5432 -- cursor/grammar/quarantine tests, none touch thermal code)
static gates: metric lineage PASS, definition drift PASS (after re-lock), env template parity PASS,
  env/compose parity hardware-watch OK, chat route poachers PASS, single-consumer OK,
  system-health producers OK, async routes OK, hostname refs OK, inner-state registry OK
```

## Evals run

```text
PYTHONPATH=../..:. python services/orion-hardware-watch/evals/run_rules_replay_eval.py
  v1 legacy checks (2026-09-30 fixture, pinned to HEAT_CONTROLLER=v1): PASS
  thermal v2 7-day gate: PASS (before this branch: 55 failure lines)
python services/orion-gpu-pool/evals/run_pool_day_eval.py: VERDICT PASS
  (new target shed_one_shot_waited: 27/27 refused with 0 wait)
```

## Docker/build/smoke checks

```text
Not run: no deploy or restart in this task (instructions). Compose env passthrough checked statically
(check_service_env_compose_parity). Live checks are UNVERIFIED until deploy; see spec Acceptance check 3.
```

## Review findings fixed

A code-review subagent ran against `origin/main...HEAD`. It found 1 HIGH, 3 MEDIUM and 5 LOW. Seven are fixed in `be9a625a7`; the last two are recorded below.

- **Finding (HIGH): the email dedupe could hide a real AC failure.** If one incident emailed and then closed, a real failure starting within 6 h got no email, no investigation and no close notice. Juniper's last message would have said "closed".
  - **Fix:** the re-opened incident is announced in the app once, and its close is announced too. The window is re-checked every tick, so if it is still open when the 6 h end, the email and investigation go out.
  - **Evidence:**
    - `test_second_incident_within_six_hours_sends_no_email_and_no_investigation_but_is_announced`
    - `test_a_deduped_incident_still_open_when_the_window_ends_gets_its_email_and_investigation`
- **Finding (MEDIUM): a stale cabinet reading could drive the reflex indefinitely.** If judging the readings raised an error, the reflex kept acting on the last tick's reading.
  - **Fix:** that case now counts as "no reading", judged at the current time.
  - **Evidence:** `test_a_reading_that_cannot_be_judged_is_unknown_now_not_last_ticks_state`
- **Finding (MEDIUM): one dead plug could page again.** It flips between `device_offline`, `no_samples` and `no_fresh_sample`, so snoozing one reason did not stop the others.
  - **Fix:** the four plug-silence reasons now snooze as one family. Resolving `low_power` still never silences them.
  - **Evidence:** `test_resolving_device_offline_also_snoozes_the_other_plug_silence_reasons`, `test_operator_resolving_low_power_does_not_silence_device_offline`
- **Finding (MEDIUM): the swap guard does not block at `elevated`.**
  - **Fix:** not changed. The approved Decisions keep the guard on the 32 °C line; recorded under Deviations.
- **Finding (LOW): the 34 °C latch was lost after a restart.**
  - **Fix:** the first cabinet read after a start covers 2 h.
- **Finding (LOW): "AC low" could be carried over from an earlier tick.**
  - **Fix:** it now comes only from the current tick's verdict.
- **Finding (LOW): the `sensor_lost` timer depended on the query window instead of its own setting.**
  - **Fix:** the heat query window now covers `HARDWARE_WATCH_HEAT_SENSOR_LOST_SEC`.
- **Finding (LOW): `/health` could report a shed the bus never carried.**
  - **Fix:** `reflex_shed.active` is set only after a successful publish; `publish_error` is shown otherwise.
  - **Evidence:** `test_health_does_not_claim_a_shed_the_bus_never_carried`
- **Finding (LOW): the learned shed's "eligible when unknown" cannot fire while the reflex is shedding, because past the grace window the reflex always sends `cabinet_unknown`.**
  - **Fix:** kept as written. It still applies when `HARDWARE_WATCH_SHED_ENABLED=false`, or when the proposal runtime's own read is stale but the watcher's is not. The spec's D2 table and D8 bullet disagree on this point.

Re-run after the fixes and after merging main:
- hardware-watch: 77 passed
- gpu-pool: 154 passed, 15 skipped
- shared tests: 666 passed
- orion-mind: 118 passed
- llm-gateway: 384 passed
- replay eval: PASS

## Restart required

Deploy order, from the primary checkout on main **after merge** (routes must exist before callers):

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-hardware-watch up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

cortex-orch: no change, because the memory annotation stays `quick_background` (see Deviations). No restart is needed.

## Deviations from the spec

- **Memory annotation stays on `quick_background`.** The spec wanted cortex-orch's memory annotation on `quick_turn`. But `memory_extractor.py:171` is a listener on `orion:chat:history:turn`: the turn has already finished, so no human is waiting on it. Under a shed, D3 now refuses it at once and the regex fallback runs immediately, instead of after 700 s.
- **`quick_turn` and `agent_turn` were not added.** No live-turn caller sends `quick` or `agent`; Hub Agent mode runs through the FCC harness. Adding routes nothing calls would be unused registry entries.
- **D3's rule had to change to cover orion-mind.**
  - The spec says to refuse when the class's `on_unavailable` is not `backlog`. But `metacog` and `agent` are backlog classes, and the gateway never sends `retryable`, so that rule would never have refused orion-mind.
  - The implemented rule: refuse unless the class is backlog **and** the lease is retryable.
  - Durable holds and children of a granted hold are unchanged.
- **The swap guard still blocks only at 32 °C (and on unknown).** The spec's D2 table says the guard blocks swap loads at `elevated`. The approved Decisions say the swap guard keeps the 32 °C `hot` line, and this follows the Decisions. As a result, D6's "urgent cooling work cannot trigger a swap load while the cabinet is warm" holds only from 32 °C up, not from 29.5 °C.
- **No `snooze_reason` column** (see Schema / bus / API changes).

## Risks / concerns

- **Severity: medium.** D8 opened eligibility across 29.5–34 °C. But the learned shed only becomes a proposal when the cabinet node wins attention. It wins through `cabinet_warming_error`, which is non-zero only while the cabinet is `elevated` **and** rising by at least 0.5 °C. In `hot`, the cabinet still cannot win, so the action stays out of reach there. Changing that is a metric-definition change, so it is left for Juniper.
- **Severity: medium.** orion-mind's live-turn check uses `trigger == "user_turn"`. That is a default value, not a positive mark set by a producer. A future background Mind caller would be treated as a live turn and never shed.
- **Severity: low.** The swap guard keeps its readings in memory, so after a pool restart it starts `unknown` and blocks swap loads until the first read. This matches the old start-up behaviour (it blocked then too), but the reason now reads `unknown`.
- **Severity: low.** D11 is not yet a verdict: one measurable episode, confounded.
- **UNVERIFIED (live, after deploy):**
  - next cool night: no `low_power` row and metacog grants stay above 0;
  - first afternoon at ≥ 34 °C: a `cabinet_hot` signal, with orion-mind still granted;
  - first elevated afternoon: a first `gpu_pool_orion_shed` row, or a snapshot showing why not.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2507

🤖 Generated with [Claude Code](https://claude.com/claude-code)
