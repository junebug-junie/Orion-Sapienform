# Thermal controller redesign: protect on cabinet heat, not on AC power

Date: 2026-10-06. Status: **design APPROVED 2026-10-06** (Juniper: reflex line 34°C; D8 yes). Q3/Q4/Q6 small defaults unconfirmed. See "Decisions".

Supersedes the shed/open/resolve logic of `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 4) and its plan `docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md`. The alert, urgent-run and pool-shed plumbing they built stays. The rules deciding when to use it change.

## Decisions (Juniper, 2026-10-06)

- **The reflex sheds at 34°C, not 32°C.**
  - New constant `DEFAULT_CRITICAL_C = 34.0` in `orion/autonomy/thermal_gate.py`, re-arming at 33.0°C (1°C hysteresis).
  - `ThermalState` names and the existing 32°C `hot` line are **unchanged**. These consumers keep using `hot`:
    - orion-thought visual-chain pause (`ORION_THERMAL_HOT_C`);
    - `world_settlement.py`;
    - the gpu2 swap guard.
  - Where this spec says the reflex acts "at `hot`", read **"at ≥ critical (34°C)"**.
  - Orion's learned shed owns 29.5–34°C, covering `elevated` and `hot`.
  - Live 7-day calibration of the ≥ 34°C share is to be computed by the replay. Peak seen: 33.5°C, so the reflex would not have fired on 10-04/05. That is intended: Orion acts there.
- **D8 approved** (Orion's learned shed reachable in the band).
- Q3 (CPU ceiling 90°C), Q4 (`agent_turn` split) and Q6 (one urgent investigation per 6 h) are proposed defaults; Juniper has not confirmed them.

## Arsonist summary

Orion's heat protection is backwards. It **shed GPU work in a 26°C cabinet** on 10-06. It **did nothing while the cabinet ran 31–33.5°C through most of 10-04 and 10-05**.

This is the fifth patch to the same process. Every earlier patch kept the same stand-in, the AC's power draw, and adjusted thresholds around it:
- a resolve rule;
- a dip tolerance (`3c48059be`);
- a snooze.

The stand-in is wrong in both directions:
- A healthy AC on a cool night sits under 150 W most of the time. It runs about 60 s, then rests about 3:00–3:20 on fan only (about 104 W). Between 01:00 and 02:00 on 10-06, 558 of 714 readings were under 150 W.
- An AC that is running flat out and losing to a hot day draws full power. That reads as healthy.

The hazard is **cabinet temperature**. Rebuild the controller around it. Demote AC power to an alert hint, and fix the side mechanics that turned each false alarm into a chat outage:
- shed requests wait out their deadlines;
- human-turn work is classed as background;
- every incident spawns a fresh email and investigation;
- a missing reading means three different things.

**Gate for done:** replay the real last 7 days through the new rules, and assert the outcome hour by hour.

## Evidence (all live, read-only, 2026-10-06)

| What | Data |
|---|---|
| False shed, cool room | Incident `7ec027dd` opened 02:00 (`low_power`). The shed latched 02:12 (`cabinet_rising`: 25.6 → 26.6°C over 15 min). It resolved at 03:49. Cabinet range for the whole incident: **25.5–27.1°C**. |
| No protection, hot room | Cabinet hourly max ≥ 29.5°C in **19 separate hours on 10-04 and 19 on 10-05**, with a peak of **33.5°C** (10-05 19:00). Zero sheds. Zero cooling incidents. |
| Orion's learned shed never ran | `gpu_pool_orion_shed`: **0 rows ever**. `substrate_world_action_episodes`: 0 rows. |
| Chat cost of tonight's false shed | metacog lane granted **0** requests 02:09–03:49 (213 refused). orion-mind waited 60 s and cancelled on both human turns (`1e362242` 02:43, `a530c5b5` 03:18). The cortex-orch memory annotation (`quick_background`) waited 700 s and fell back to regex on both. 129 cortex-exec metacog requests waited 45–300 s (mean 142 s) before cancel. |
| Alert/investigation spam | 14 incidents in 5 days, every one spawning an urgent investigation (9–13 min each on the agent lane). 3 cooling "AC FAILURE" emails were all false: 10-02 23:46, 10-06 00:19 and 10-06 02:00. The 02:00 run hit `fcc_timeout` after 24 steps with no conclusion. |
| GPU-heat false positives | 6 `gpu_heat` incidents opened at 73–80°C, against the card's own p95 of 56–58°C. A V100 at 80°C is in its normal range; its slowdown point is about 85°C. |
| 10-03 outage | Incident `no_samples` → `cabinet_unreadable` shed latched about 4 h until Juniper resolved it by hand. Hub chat was down. Still reachable today: see C6. |

## Current architecture

```text
home_cooling_sample (Z-Wave AC plug, ~5 s)  ─┐
orion_biometrics_summary.cabinet_temp_c     ─┤
                                             ▼
orion-hardware-watch  (orion/hardware_watch/rules.py, services/orion-hardware-watch/app/watcher.py)
  cooling rule: OPEN on AC watts < 150 for 180 s | no sample 5 min | identical watts 60 min
                RESOLVE on ≥500 W for 10 min (dips < 180 s after a ≥500 W reading count as working)
  shed verdict (only inside an open cooling incident): cabinet ≥ 29.5 | rise ≥ 1.0°C/15 min | unreadable
                → LATCHED until the incident resolves
  cpu_heat / gpu_heat: OPEN above own trailing-7d p95 for 10 min (or above 85°C ceiling for GPU)
  every incident → critical alert (cooling only) + urgent curiosity request (all rules)
                                             ▼
orion-gpu-pool (orion/gpu_pool/scheduler.py)
  shed: priorities {system, background} leave placement, but keep their place in line and their deadline
  routes without explicit priority default to "system" (orion/gpu_pool/config.py:193): agent, quick
  swap guard (services/orion-gpu-pool/app/guards.py): blocks gpu2 model load at ≥ 32°C (hysteresis 30.5), unknown = hot

Orion's learned shed (orion/autonomy/self_shed.py → orion-proposal-runtime → pool)
  eligible only if thermal_state == "elevated" (not "hot"), rising ≥ 0.5°C, and NO incident of any rule open
```

There are three separate controllers (reflex shed, learned shed, swap guard). They have three meanings of "no reading", and none of them owns the question "is the cabinet too hot right now?"

## Defects (what the redesign must remove)

| # | Defect | Where | Seen live |
|---|---|---|---|
| C1 | AC power used as the hazard signal. A cool-night rest reads as "AC dead"; a losing-battle hot day reads as "fine". | `rules.py:81-82,144-147` | yes, both directions |
| C2 | The rise-based shed has no absolute floor: 1°C of thermostat drift sheds. | `rules.py:284` | yes, 10-06 |
| C3 | The shed latches until the incident resolves, not until the temperature recovers. The cabinet was back at 25.7°C by 02:20 and the shed held until 03:49. | `watcher.py:296-300` | yes |
| C4 | Open and resolve share the same 180 s line, with no hysteresis. Normal off-cycles straddle it, so the incident flaps, and each reopen gets a new ID, a new email and a new investigation. | `rules.py:96-103,144-147`; dedupe keyed per incident `watcher.py:322` | yes, 01:32 resolve → 02:00 reopen |
| C5 | Shed requests keep their deadline and wait it out instead of being refused at once. | `scheduler.py:391-394,450-452,498-502` | yes, 45–700 s waits |
| C6 | Human-turn work is shed as background or system: orion-mind (`metacog` → system), cortex-orch memory annotation (`quick_background`), and routes without an explicit priority (`agent`, `quick`). If Z-Wave or sql-writer goes down, AC and cabinet both go silent → `no_samples` + `cabinet_unreadable` → maximum latched shed. | `config.py:193`; `config/gpu_pool.yaml:114,117`; `memory_extractor.py:171` | yes (mind, orch); agent UNVERIFIED |
| C7 | Orion's learned shed is disabled exactly when needed. It refuses on `thermal_state == "hot"` and refuses on *any* open incident, including junk heat incidents. | `orion/autonomy/self_shed.py:130,156` | yes, 0 rows ever |
| C8 | Every incident spawns an urgent agent-lane investigation. It is exempt from shedding, can trigger a gpu2 model load during a suspected cooling failure, and is spawned again on every flap. | `watcher.py:241,353-373`; `scheduler.py:808-811` | yes, 14 in 5 days |
| C9 | The GPU/CPU heat rule fires on its own p95, so it is true about 5% of the time by construction. | `rules.py:203-224` | yes, 6 GPU + 4 CPU |
| C10 | Three meanings of "no reading": `thermal_gate` allows, the swap guard blocks, hardware-watch sheds. One failed Postgres query counts as unreadable, and through the latch that lasts the whole incident. The pool and the watcher read the cabinet from different paths. | `thermal_gate.py:94-109`; `guards.py:47-48`; `watcher.py:285-287` | yes, 10-03 |
| C11 | An operator resolve snoozes **every** open reason for the rule for 1 h, including `device_offline`. A real failure in that hour is silent. | `watcher.py:222-225,249-250`; `store.py:79-82` | no |
| C12 | The pool's `_open_ac_incidents` (removed by D2) never expires. A lost "resolved" event leaves the learned shed refused as `reflex_active` until the pool restarts. A heat incident whose sensor dies never resolves. | `runtime.py:660-678`; `watcher.py:99,259`; `rules.py:205,226` | no |
| C13 | The swap guard starts up in the "hot" state. | `guards.py:28` | no (cosmetic) |
| C14 | **Nobody has measured whether shedding lowers cabinet temperature at all.** The whole mechanism is untested against its own goal. | none | n/a |

## Design

### D1. One owner for "how hot is the cabinet": extend `orion/autonomy/cabinet_heat.py`

The owner already exists:
- `read_cabinet_heat()` / `CabinetHeatReading` in `orion/autonomy/cabinet_heat.py`;
- built on `hysteretic_thermal_state()` and the `ThermalState` names `normal | elevated | hot | unknown` (`orion/autonomy/thermal_gate.py:41`);
- plus the shared `cabinet_rise_c` (`orion/hardware_watch/rules.py`).

Do not add a parallel function. Extend `CabinetHeatReading` with:
- `minutes_to_hot`: linear projection from the 15-min rise; None if not rising;
- `reading_age_sec`.

Make every consumer read it:
- hardware-watch (it replaces `shed_verdict`);
- the pool swap guard (`services/orion-gpu-pool/app/guards.py`, today a private copy at 32/30.5°C);
- Orion's learned shed (already does).

The state names stay as they are, so these existing consumers keep working unchanged:
- `orion-thought` `visual_chain.py` / `visual_steps.py`;
- `orion/feedback/world_settlement.py`;
- `orion/schemas/reverie_visual.py`.

Thresholds stay where they live today:
- `elevated` at 29.5°C, from `DEFAULT_ELEVATED_C`;
- `hot` at 32°C, from `thermal_gate`'s hot line, which the swap guard then reads instead of its own copy;
- `hysteresis_c = 1.0`;
- `lookahead_min = 20`.

What "unknown" means is decided **once**, here:
- Hold the last known state for `grace_sec` (default 300 s). One failed query does **not** make it unknown.
- After grace, unknown counts as `elevated`. It becomes `hot` only if the AC also reads low (D5): "the sensor is dead and the AC looks dead" is the one case where we assume the worst.
- If *every* reading is gone (AC plug and cabinet both silent, as on 10-03, when Z-Wave or sql-writer is down), it is a monitoring outage. The response is an alert plus the `elevated` tier, not `hot`.

**Live calibration fact (7 days to 10-06, 19,197 readings):** the cabinet was at or above 29.5°C for **34%** of readings and at or above 32°C for **3%**. The module's own 09-29 docstring said about 65%; that figure is now stale. "Elevated" is a common daily state, not an emergency. That drives D2.

### D2. Two layers on the existing shed board: the reflex owns `hot`, Orion owns `elevated`

The pool already has a fail-open shed signal board (`orion/gpu_pool/shed.py`):
- signals are `(reason, source_id, detail, valid_until)`;
- a signal past `valid_until` counts as absent;
- blocks are the union of active reasons;
- `interactive`/`urgent` are never shed.

Reuse it. No new transport.

| Cabinet state | Reflex (hardware-watch) | Orion's learned shed (D8) |
|---|---|---|
| normal | no signal | not eligible |
| elevated | no shed; blocks gpu2 swap loads (guard) | **eligible**: Orion decides whether to hold back `background` |
| hot (32–34°C) | no shed; blocks gpu2 swap loads | **eligible** |
| critical (≥ 34°C) | signal `cabinet_hot` → blocks `background` + `system` | not needed (reflex covers it) |
| unknown, past grace | signal `cabinet_unknown` → blocks `background` (→ `cabinet_hot` if AC also low) | eligible |

- **Why split it this way.** "Elevated" holds 34% of the time. A reflex that sheds background a third of the week would starve Orion's background cognition, and it would make Orion's learned action a permanent no-op: a lower-precedence reason can only add blocks (`shed.py:14-16`), so it would never get the chance to act.
- **Division of labor:**
  - the reflex handles real danger (`hot`, sensor loss) and needs no judgment;
  - the elevated band, where shedding is a trade-off, belongs to Orion's learned action;
  - D11 measures whether shedding actually cools anything.

  This keeps the agency the attend-to-act design intended (`docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md`, A1) and removes the redundancy.
- **Re-sent every tick, no latch.** hardware-watch re-sends its signal every tick with `valid_until = now + 3 × tick`. When the state drops (with hysteresis), it stops sending and the signal expires. This replaces the incident-scoped latch (C2, C3).
  - If the watcher dies, the shed lapses within about 90 s (fail-open), matching the board's own design.
- **New reasons** in `shed.py`:
  - `cabinet_hot` (precedence 0, blocks background + system);
  - `cabinet_unknown` (precedence 0, blocks background).
- **Retire `cooling_incident` as a shed reason completely**, not as a partial exclusion (CLAUDE.md "retire the old one completely"). Re-key the logic that reads it from "AC incident open" to "reflex signal active":
  - `services/orion-gpu-pool/app/runtime.py:637-650` signal handling;
  - `runtime.py:660-678` `_open_ac_incidents` / `reflex_active`;
  - `preempt_by_reflex`.

  The `_open_ac_incidents` set and its no-expiry bug (C12) go away with it.
- Heat alone sheds; no open incident is needed. This fixes C1 in the hot-day direction.

### D3. One-shot requests are refused at once; durable work keeps waiting

Refusing everything at once would break durable runs:
- `shed:*` is not in `RUN_TERMINAL_PREFIXES` (`services/orion-durable-runs/app/pool_hold.py:65`), so a refused hold is re-requested immediately (`admission_runtime.py:428-446`). That is a tight loop that leaves a lease row per attempt.
- `on_unavailable: backlog` classes are *meant* to wait.

So the change is scoped:
- **One-shot requests** (`kind == "request"`, no `hold_lease_id`, and a class whose `on_unavailable` is not `backlog`) get `Unavailable("shed:<reason>")` as soon as they would be shed. Their caller's fallback runs at once:
  - orion-mind fails open;
  - memory annotation uses its regex fallback;
  - cortex-exec background metacog turns end and are re-driven by their own schedulers.

  This removes tonight's 45–700 s waits (C5).
- **Durable-run holds** and **backlog-class leases** keep today's queued-under-shed behavior. Children of an already granted hold keep being granted (`orion/gpu_pool/scheduler.py:392-394`, the `hold_lease_id is None` filter), so a run in progress is never cut off mid-run.
- A test covers each class: a one-shot request refused at once; a durable hold stays queued; a backlog lease stays backlogged; a granted hold's child is still granted under shed.

### D4. Human-turn work never sheds

- Add explicit routes for calls on a human's critical path, rather than overloading `system`:
  - `metacog_turn: {class: metacog, priority: interactive}`, used by orion-mind when its request carries a Hub turn;
  - `quick_turn: {class: fast, priority: interactive}`, used by cortex-orch memory annotation on a live turn;
  - `agent_turn` per Q4.
- **There are two route registries, and both must change together:**
  - `config/gpu_pool.yaml` routes (the pool);
  - `orion/llm/routes.py` `ACCEPTED_LLM_ROUTES` plus `LLM_ROUTE_DISPLAY_ORDER`. The import-time assert at `:138` requires every accepted route to be listed. This registry is used by the Hub client (`llm_gateway_client.py:35`), orion-actions, cortex-exec route override (`executor.py:2059,4296`) and `route_view`.

  Also update the golden fixture `orion/gpu_pool/tests/fixtures_routes_compat_golden.json` and `test_route_view.py`.
- Give `agent` and `quick` explicit priorities in the YAML (C6).
- **Do not** make `RouteSpec.priority` a required model field. The shorthand validator (`orion/gpu_pool/config.py:214-219`, `agent: agent` → `{"class": "agent"}`) and tests that build `RouteSpec` without a priority (`services/orion-llm-gateway/tests/test_lane_senders.py:18-20`) would break.
  - Instead, add a config-load **test** that every route in `config/gpu_pool.yaml` is written in long form with an explicit priority. The gate lives in CI, and the model stays compatible.

### D5. AC power becomes a diagnosis, not a trigger

- **When an incident opens.** The cooling rule opens an incident only when **the AC looks low AND the cabinet is `elevated` or `hot`, or projected to reach `hot` within the lookahead**. There is no bare "rising" arm, so the C2 problem does not come back on the alert path.
- **What "AC looks low" means.** Mean watts over a **15-minute** window below a duty-cycle floor, rather than 180 s under 150 W.
  - On 10-06 the healthy AC averaged 187–375 W per 10 minutes while the compressor cycled.
  - A dead AC is about 100 W flat.
  - Floor proposal: mean < 140 W over 15 min. Replay will validate it (Q2).
- **How "unknown" counts here.** An `unknown` cabinet counts as elevated for this test, so "AC low + sensor dead" still opens an incident and escalates the reflex to `cabinet_hot` (D1).
- **Dead plug.** `device_offline` / `no_samples` still open an incident (a dead plug is real), but they **alert only**. Shedding comes from D2's cabinet state alone.
- **No flapping.** Open and resolve get separate thresholds: open below the floor for 15 min; resolve at or above 1.5× the floor for 15 min. This ends the flapping (C4).

### D6. Alerts and investigations deduplicate per rule and subject across a window

- The email dedupe becomes a **sliding** window per `rule+subject` (default 6 h): no alert if one was sent for the same rule and subject in the last 6 h. It is no longer keyed per incident ID.
- The urgent investigation is requested at most once per rule and subject per window. It is **not** requested for `gpu_heat`/`cpu_heat` p95 outliers (C8, C9).
- Urgent work requested by a cooling incident cannot trigger a gpu2 swap load while the cabinet is warm or hot (C8).

### D7. GPU/CPU heat opens on real limits, p95 becomes context

- `gpu_heat` opens on the existing fixed ceiling only: 85°C sustained, re-arming at 80°C.
- `cpu_heat` opens on a fixed ceiling (Q3).
- The trailing p95 is kept as an *annotation* on the incident ("this card is 22°C above its usual"), not as a trigger (C9).
- A heat incident whose sensor has gone silent for more than 15 min resolves as `sensor_lost`, so it cannot block anything indefinitely (C12).

### D8. APPROVED 2026-10-06 (autonomy change): Orion's learned shed owns the elevated band

In `orion/autonomy/self_shed.py`:
- Eligible when the cabinet is `elevated`, `hot` (below 34°C critical) or `unknown`. At ≥ 34°C the reflex already blocks background, so Orion's signal would add nothing.
- Drop the `cabinet_not_rising` requirement. In the elevated band, the decision belongs to Orion's attend-to-act loop, not a fixed rise threshold.
- Blocked only while the reflex's own signal is active (`cabinet_hot`/`cabinet_unknown`), not by *any* incident (`self_shed.py:130`, C7). Junk heat incidents can no longer disable it.

Proposal-mode block (CLAUDE.md §0A):
- **Capability:** Orion gains a real, frequently reachable action: holding back its own background GPU work when the cabinet is elevated (34% of the week). Today that action is unreachable (0 rows ever).
- **Data touched:** `gpu_pool_orion_shed`, `substrate_world_action_episodes`. No private content.
- **Proof it works:** rows appear in both tables on elevated days, and the D11 report shows cabinet ΔT per Orion-initiated shed.
- **Dangerous failure:** Orion holds background work back for long stretches, starving reverie, journal and curiosity on warm days. Mitigation: the board's `valid_until` caps every signal, and the pool reports how long background work was blocked per day.
- **Rollback:** `GPU_POOL_ORION_SHED_ENABLED` (existing RPC gate) or a revert of the eligibility change.

### D9. Operator resolve snoozes only the reason that was resolved

- Snooze is keyed by `rule+subject+open_reason`.
- `device_offline`/`no_samples` are never snoozed by resolving a `low_power` incident (C11).

### D10. Swap guard starts "unknown", not "hot"

It reads `CabinetHeatReading` like everyone else, and starts as `unknown` with the D1 grace rule (C13).

Today the "unknown = hot" behavior comes from two things together: the start-up value `hot` (`guards.py:28`), and the state not updating while a reading is degraded (`guards.py:47-48` returns `degraded:`).

### D11. Measure whether shedding works (C14)

- Every shed start/end already emits an incident or pool event.
- Add a reducer that records `cabinet_temp_c` at shed start, +15 min and +30 min, alongside the GPU watts drawn (`gpu_w` in biometrics).
- Surface it in the existing hardware-watch eval report.

If shedding background work never measurably lowers cabinet temperature, the shed tier is protecting nothing. In that case the real lever is the AC or the room, and Orion should alert rather than self-throttle. This is a metric, so it goes through the CLAUDE.md metric gate in the implementation PR:
- provenance: biometrics summary;
- independence from the shed trigger: it reads the outcome window, not the trigger window;
- the anchor is the plain thermal fact that GPU watts heat the cabinet;
- live sanity on the existing 10-04/05 hot days.

## Missing questions (for Juniper)

1. **Thresholds and who acts where.**
   - The cabinet is at or above 29.5°C for **34%** of the week, and at or above 32°C for **3%**.
   - Proposed:
     - the reflex sheds only at `hot` (≥ 32°C) or on sensor loss;
     - the elevated band (29.5–32°C) is Orion's to act in (D2/D8).
   - Alternative: the reflex also sheds background at elevated. That holds background back about a third of the week, and Orion's learned action becomes dead code that should be retired.
   - Is 32°C the real "must act" line? V100 intake is rated to about 35°C. I recommend 32°C, and letting D11 show within a week whether shedding moves the temperature at all.
   - Lookahead 20 min, hysteresis 1°C: confirm, or let replay pick.
2. **AC-low floor.** Mean < 140 W over 15 min, validated by replay. OK to let replay set the exact number?
3. **CPU ceiling.** What fixed CPU temperature (`temp_c_max`) should alert? circe ran 41–62°C during its incidents. I propose 90°C, typical Xeon throttle territory, pending the actual CPU model on each host.
4. **`agent` and `quick` priority.** Hub Agent mode and the Hub default `quick` route both carry human turns, but also background callers. Split them like D4 (`agent_turn`), or make both `interactive`? I recommend splitting. It keeps background agent work sheddable.
5. **Approve D8** (proposal mode: Orion's learned shed becomes reachable in the elevated band)?
6. **Urgent investigations on cooling.** Keep at most one per 6 h window (D6), or drop them entirely and rely on the alert? The 10-06 one timed out without a conclusion. I recommend keeping one per window, because a real AC failure warrants a look.

## Proposed schema / API changes

- `orion/autonomy/cabinet_heat.py`:
  - `CabinetHeatReading` gains `minutes_to_hot` and `reading_age_sec`;
  - `read_cabinet_heat` gains the grace and "unknown + AC low → hot" rules;
  - `ThermalState` names are unchanged.
- `orion/hardware_watch/rules.py`:
  - remove the incident-scoped `shed_verdict`;
  - the cooling open/resolve rules per D5;
  - `cabinet_rise_c` stays.
- `orion/gpu_pool/shed.py`:
  - new reasons `cabinet_hot` and `cabinet_unknown`, both precedence 0;
  - retire `cooling_incident`.

  Check whether the shed-signal RPC/bus payload validates reason names against a schema in `orion/schemas/`. If so, update the schema and registry (CLAUDE.md §6). Also check consumers of shed events for `extra="forbid"` (`feedback_additive_schema_fields_are_a_consumer_first_migration_on_forbid_models`).
- `orion/gpu_pool/scheduler.py`: a one-shot shed request yields `Unavailable("shed:<reason>")`. Durable holds and backlog classes are unchanged (D3).
- Routes (D4):
  - `config/gpu_pool.yaml` gets `metacog_turn`, `quick_turn` (+ `agent_turn` per Q4) and long-form explicit priorities for every route;
  - `orion/llm/routes.py` `ACCEPTED_LLM_ROUTES` + `LLM_ROUTE_DISPLAY_ORDER`;
  - the golden fixture `orion/gpu_pool/tests/fixtures_routes_compat_golden.json`;
  - `RouteSpec` model unchanged.
- `hardware_watch_incident`: add `snooze_reason` (text). Additive migration. The alert dedupe is a query over `alert_sent_at` by rule+subject, so it needs no new column.
- Env:
  - new `HARDWARE_WATCH_READING_GRACE_SEC`, `HARDWARE_WATCH_HEAT_LOOKAHEAD_MIN`, `HARDWARE_WATCH_AC_LOW_MEAN_W`, `HARDWARE_WATCH_AC_LOW_WINDOW_SEC`, `HARDWARE_WATCH_ALERT_DEDUPE_WINDOW_SEC`, `HARDWARE_WATCH_HEAT_CONTROLLER`;
  - retire `HARDWARE_WATCH_SHED_RISE_C`;
  - temperature lines stay in `thermal_gate` (one copy), and the swap guard's private 32/30.5 constants are removed;
  - `.env_example` and local `.env` synced in the same PR (CLAUDE.md §7).

## Files likely to touch

- `orion/autonomy/{cabinet_heat.py,thermal_gate.py,self_shed.py}`
- `orion/hardware_watch/rules.py`, `services/orion-hardware-watch/app/{watcher.py,store.py,settings.py}`, `.env_example`, `README.md`, `evals/run_rules_replay_eval.py`
- `orion/gpu_pool/{shed.py,scheduler.py}`, `config/gpu_pool.yaml`, `services/orion-gpu-pool/app/{guards.py,runtime.py}`
- `orion/llm/routes.py`, `orion/gpu_pool/tests/fixtures_routes_compat_golden.json`, `test_route_view.py`
- `services/orion-mind/app/llm_client.py` (+ settings: turn route); cortex-orch `memory_extractor.py` (turn route)
- tests under each service; `services/orion-hardware-watch/evals/`

## Non-goals

- No change to the alert transport (in-app + email), the urgent-curiosity runner, or durable runs' refusal handling (D3 is scoped to avoid it).
- No AC or Z-Wave control (Orion does not switch the AC).
- No new service. The controller stays inside hardware-watch, and the pool remains the only thing that grants GPUs.
- No change to GPU-pool scheduling beyond the shed path and the route priorities.

## Acceptance checks

1. **7-day replay gate (the deciding test).** Feed the real `home_cooling_sample` and `orion_biometrics_summary` rows from 2026-09-29 → 2026-10-06 through the new rules. Assert, hour by hour:
   - **10-06 00:00–04:00:** no shed, no cooling incident, no email.
   - **Hours with cabinet ≥ 32°C on 10-04/10-05:** reflex `cabinet_hot` active. **Hours at 29.5–32°C:** no reflex shed, and Orion's learned shed reports *eligible* (whether it acts is its own decision).
   - **10-03 21:14 `no_samples`:** an alert fires, the shed is at most `cabinet_unknown` (background only), and it lapses within 3 ticks of readings resuming.
   - **gpu_heat:** zero incidents for 73–80°C readings.
   - **emails:** at most one per rule+subject per 6 h.

   The replay eval (`run_rules_replay_eval.py`) is extended rather than rewritten. Fixture rows are exported to the test dir by a script, so the run is deterministic.
2. **Unit:**
   - `read_cabinet_heat` boundaries, hysteresis, grace, and unknown + AC-low → hot;
   - D3: a one-shot request refused at once; a durable hold stays queued; a backlog lease stays backlogged; a granted hold's child is still granted;
   - D2: a signal lapses after `valid_until` when the watcher stops sending;
   - D4 every route has an explicit priority (a config-load test fails on a missing one);
   - D8: learned shed eligible at elevated/unknown, not at normal/hot, and not blocked by a gpu_heat incident.
3. **Live after deploy:**
   - next cool night: `hardware_watch_incident` has no `low_power` row and metacog grants never drop to 0;
   - next afternoon above 32°C: a `cabinet_hot` signal appears, and orion-mind on Hub turns is still granted (`gpu_pool_leases` holder `orion-mind`, `granted_at` not null);
   - first elevated afternoon: `gpu_pool_orion_shed` gets its first row, or the eligibility snapshot shows why not.
4. **D11 report** shows cabinet ΔT after shed for at least 3 shed episodes. If ΔT ≈ 0, the follow-up is to drop the shed tier, not to tune it.

## Rollback

- D2/D5 go behind `HARDWARE_WATCH_HEAT_CONTROLLER=v2`. Setting `v1` restores today's rules, including the `cooling_incident` reason, which stays in code until v2 has run clean for a week. Then delete it. The flag ships **on** (`feedback_always_ship_with_flags_on`).
- D3 and D4 are independent pool/config changes, each a plain revert.
- The schema additions are additive columns; leave them in place on rollback.

## Recommended next patch

Ship in one PR, in this order inside it:
1. Export the replay fixtures and write the failing 7-day replay assertions. They must fail on today's code for the cool-night and hot-day reasons above.
2. Extend `read_cabinet_heat` (D1) + per-tick shed signals (D2) + AC-as-diagnosis (D5) → the replay passes.
3. Scoped immediate refusal (D3), turn routes in both registries (D4).
4. Dedupe (D6), heat ceilings (D7), snooze (D9), guard init (D10).
5. D8 only after Juniper approves it (proposal mode).
6. D11 reducer + report.

Deploy hardware-watch, gpu-pool, llm-gateway (routes), orion-mind and cortex-orch together; the route names must exist before callers use them. Deploy from the primary checkout on main after merge (`feedback_worktree_deploys_can_pin_a_worktree_as_production`).
