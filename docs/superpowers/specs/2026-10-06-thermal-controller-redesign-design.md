# Thermal controller redesign: protect on cabinet heat, not on AC power

Date: 2026-10-06. Status: **design, awaiting Juniper's answers to "Missing questions".** Nothing implemented.

Supersedes the shed/open/resolve logic of `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 4) and its plan `docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md`. The alert, urgent-run and pool-shed plumbing they built stays. The rules deciding when to use it change.

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
| C12 | The pool's `_open_ac_incidents` never expires. A lost "resolved" event leaves the learned shed refused as `reflex_active` until the pool restarts. A heat incident whose sensor dies never resolves. | `runtime.py:660-678`; `watcher.py:99,259`; `rules.py:205,226` | no |
| C13 | The swap guard starts up in the "hot" state. | `guards.py:28` | no (cosmetic) |
| C14 | **Nobody has measured whether shedding lowers cabinet temperature at all.** The whole mechanism is untested against its own goal. | none | n/a |

## Design

### D1. One owner for "how hot is the cabinet": `cabinet_heat_state`

A single pure function in `orion/hardware_watch/rules.py`, the existing shared home of `cabinet_rise_c`. Every consumer imports it: hardware-watch, Orion's learned shed, the pool swap guard and `thermal_gate`.

```text
cabinet_heat_state(readings, now, cfg) -> CabinetHeat{
  state: "ok" | "warm" | "hot" | "unknown",
  temp_c, rise_c_per_15m, minutes_to_hot (projection, None if not rising),
  reading_age_sec
}
ok      temp < warm_c - hysteresis_c (on the way down) or < warm_c (on the way up)
warm    temp ≥ warm_c, OR rising AND minutes_to_hot ≤ lookahead_min
hot     temp ≥ hot_c
unknown no reading younger than grace_sec (default 300 s; one failed query does NOT make it unknown)
```

- `warm_c = 29.5` (today's `DEFAULT_ELEVATED_C`).
- `hot_c = 32.0` (today's swap-guard line).
- `hysteresis_c = 1.0`.
- `lookahead_min = 20`.

All four are Juniper's call (Q1). They are named and stored in one place instead of three.

What "unknown" means is decided **once**, here: hold the last known state for `grace_sec`, then treat it as `warm`, never as `hot`. A dead sensor in a cool room limits the damage to one shed tier. It does not cause a full shed.

### D2. Shed on cabinet heat, with tiers, re-evaluated every tick, no latch

| Cabinet state | Pool behavior |
|---|---|
| ok | nothing held back |
| warm | hold back `background` work; block gpu2 swap loads |
| hot | also hold back `system`; still never `interactive`/`urgent`/human-turn |
| unknown (past grace) | same as warm |

- The shed is a **level**, recomputed every watcher tick from `cabinet_heat_state`.
- Release uses the hysteresis in D1. It is not tied to any incident's lifecycle. This removes C2 and C3.
- The shed no longer requires an open cooling incident. Heat alone sheds (fixes C1, the hot-day direction).

### D3. Held-back requests are refused at once, not queued to their deadline

- When a request's priority is shed, the pool returns `Unavailable("shed:<state>")` right away.
- The caller's existing fallback path runs immediately:
  - orion-mind fails open with no mind coloring;
  - memory annotation falls back to regex;
  - background turns re-queue through durable runs per `feedback_retries_go_on_durable_runs`.
- This replaces the "keep place in line + keep deadline" behavior (C5).
- A durable-run hold already granted is not killed; it simply gets no new work placed on it.

### D4. Human-turn work never sheds

- Add explicit routes for human-critical-path calls, rather than overloading `system`.
- The route table is already the contract between caller and pool, and the gateway refuses unknown routes:
  - `metacog_turn: {class: metacog, priority: interactive}`, used by orion-mind when its request carries a Hub turn;
  - `quick_turn: {class: fast, priority: interactive}`, used by cortex-orch memory annotation on a live turn.
- Give `agent` and `quick` explicit priorities. They need a decision, not the `system` default (Q4).
- Change the `RouteSpec.priority` default from `system` to **required**, so every route states its priority (C6).

### D5. AC power becomes a diagnosis, not a trigger

- The cooling rule opens an incident only when **AC looks low AND the cabinet is warming** (`cabinet_heat_state` in warm/hot, or rising).
- "AC looks low" means mean watts over a **15-minute** window below a duty-cycle floor, rather than 180 s under 150 W. On 10-06 the healthy AC averaged 187–375 W per 10 minutes with the compressor cycling. A dead AC is about 100 W flat.
  - Floor proposal: mean < 140 W over 15 min. Replay will validate it (Q2).
- `device_offline` / `no_samples` still open an incident (a dead plug is real), but they **alert only**. Shedding comes solely from D2.
- Open and resolve get separate thresholds (hysteresis), which ends the flapping (C4).

### D6. Alerts and investigations deduplicate per rule and subject across a window

- The email dedupe key becomes `cooling:<subject>:<window-bucket>` (default 6 h), not per incident ID.
- The urgent investigation is requested at most once per rule and subject per window. It is **not** requested for `gpu_heat`/`cpu_heat` p95 outliers (C8, C9).
- Urgent work requested by a cooling incident cannot trigger a gpu2 swap load while the cabinet is warm or hot (C8).

### D7. GPU/CPU heat opens on real limits, p95 becomes context

- `gpu_heat` opens on the existing fixed ceiling only: 85°C sustained, re-arming at 80°C.
- `cpu_heat` opens on a fixed ceiling (Q3).
- The trailing p95 is kept as an *annotation* on the incident ("this card is 22°C above its usual"), not as a trigger (C9).
- A heat incident whose sensor has gone silent for more than 15 min resolves as `sensor_lost`, so it cannot block anything indefinitely (C12).

### D8. Orion's learned shed: fix the eligibility inversions

In `self_shed.py`:
- eligible when `cabinet_heat_state` is warm **or hot** (removes the `!= "elevated"` refusal);
- blocked only by an open **cooling** incident (the reflex owns the AC-failure case), as its own docstring already says (C7).

Its 0.5°C rise threshold becomes the D1 projection. It shares one definition with the reflex, per the existing intent in `cabinet_rise_c`'s docstring.

The pool's `_open_ac_incidents` gets a TTL refreshed by the watcher's existing per-tick publish of open incidents. A missed "resolved" event then self-heals within one refresh period (C12).

### D9. Operator resolve snoozes only the reason that was resolved

- Snooze is keyed by `rule+subject+open_reason`.
- `device_offline`/`no_samples` are never snoozed by resolving a `low_power` incident (C11).

### D10. Swap guard starts "unknown", not "hot"

It uses `cabinet_heat_state` like everyone else (C13).

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

1. **Thresholds.**
   - Is 29.5°C the right "start holding back background work" line, and 32°C the "also hold back system work" line? On 10-04/05 a 29.5°C warm line could have held back background GPU work for up to ~19 h each day (hourly max; time actually above the line is UNVERIFIED until replay). Is that the behavior you want, or is the real danger line higher (V100 intake is rated to about 35°C)? I recommend keeping 29.5/32 and letting D11 tell us within a week whether shedding even moves the needle.
   - Lookahead 20 min, hysteresis 1°C: confirm, or let replay pick.
2. **AC-low floor.** Mean < 140 W over 15 min, validated by replay. OK to let replay set the exact number?
3. **CPU ceiling.** What fixed CPU temperature (`temp_c_max`) should alert? circe ran 41–62°C during its incidents. I propose 90°C, typical Xeon throttle territory, pending the actual CPU model on each host.
4. **`agent` and `quick` priority.** Hub Agent mode and the Hub default `quick` route both carry human turns, but also background callers. Split them like D4 (`agent_turn`), or make both `interactive`? I recommend splitting. It keeps background agent work sheddable.
5. **Urgent investigations on cooling.** Keep at most one per 6 h window (D6), or drop them entirely and rely on the alert? The 10-06 one timed out without a conclusion. I recommend keeping one per window, because a real AC failure warrants a look.

## Proposed schema / API changes

- `orion/hardware_watch/rules.py`: new `CabinetHeat`, `cabinet_heat_state()`, `CabinetHeatConfig`.
  - Remove the incident-scoped `shed_verdict`.
  - `cabinet_rise_c` stays as the internal helper.
- `orion/gpu_pool/config.py`: `RouteSpec.priority` becomes required. No default.
- `config/gpu_pool.yaml`: new routes `metacog_turn`, `quick_turn` (+ `agent_turn` per Q4); explicit priorities on every route.
- Pool shed result: `Unavailable("shed:warm" | "shed:hot")` replaces the queued `Shed` decision. Check `orion/schemas/` for any `gpu_pool` event schema carrying shed reasons; additive reason values only.
- `hardware_watch_incident`: add `alert_dedupe_bucket` (text) and `snooze_reason` (text). Additive migration; no backfill needed.
- Bus: the existing `orion:hardware_watch:*` incident events gain `heat_state` in the payload. Additive.
  - Consumers are on `extra="ignore"` or `forbid`? Check per `feedback_additive_schema_fields_are_a_consumer_first_migration_on_forbid_models` before shipping.
- Env:
  - new `HARDWARE_WATCH_WARM_C`, `HARDWARE_WATCH_HOT_C`, `HARDWARE_WATCH_HEAT_HYSTERESIS_C`, `HARDWARE_WATCH_HEAT_LOOKAHEAD_MIN`, `HARDWARE_WATCH_READING_GRACE_SEC`, `HARDWARE_WATCH_AC_LOW_MEAN_W`, `HARDWARE_WATCH_AC_LOW_WINDOW_SEC`, `HARDWARE_WATCH_ALERT_DEDUPE_WINDOW_SEC`;
  - retire `HARDWARE_WATCH_SHED_RISE_C`;
  - `.env_example` and local `.env` synced in the same PR (CLAUDE.md §7);
  - the pool and `thermal_gate` read the same keys, or receive the state over the bus. Pick one at implementation and document it. No second copy of the thresholds.

## Files likely to touch

- `orion/hardware_watch/rules.py`, `services/orion-hardware-watch/app/{watcher.py,store.py,settings.py}`, `.env_example`, `README.md`, `evals/run_rules_replay_eval.py`
- `orion/gpu_pool/{scheduler.py,config.py}`, `config/gpu_pool.yaml`, `services/orion-gpu-pool/app/{guards.py,runtime.py}`
- `orion/autonomy/{self_shed.py,thermal_gate.py}`
- `services/orion-mind/app/llm_client.py` (+ settings: turn route); `services/orion-cortex-orch/.../memory_extractor.py` (turn route)
- tests under each service; `services/orion-hardware-watch/evals/`

## Non-goals

- No change to the alert transport (in-app + email), the urgent-curiosity runner, or durable runs.
- No AC or Z-Wave control (Orion does not switch the AC).
- No new service. The controller stays inside hardware-watch, and the pool remains the only thing that grants GPUs.
- No change to GPU-pool scheduling beyond the shed path and the route priorities.

## Acceptance checks

1. **7-day replay gate (the deciding test).** Feed the real `home_cooling_sample` and `orion_biometrics_summary` rows from 2026-09-29 → 2026-10-06 through the new rules. Assert, hour by hour:
   - **10-06 00:00–04:00:** no shed, no cooling incident, no email.
   - **10-04 14:00–17:00 and 10-05 14:00–19:00:** background shed active (cabinet ≥ 29.5°C).
   - **10-03 21:14 `no_samples`:** an alert fires, and the shed is at most the warm tier, released as soon as readings resume.
   - **gpu_heat:** zero incidents for 73–80°C readings.
   - **emails:** at most one per rule+subject per 6 h.

   The replay eval (`run_rules_replay_eval.py`) is extended rather than rewritten. Fixture rows are exported to the test dir by a script, so the run is deterministic.
2. **Unit:**
   - `cabinet_heat_state` boundaries, hysteresis and grace;
   - D3 immediate `Unavailable`;
   - D4 every route has an explicit priority (a config-load test fails on a missing one);
   - D8 learned-shed eligibility on hot and on a non-cooling incident.
3. **Live after deploy:**
   - next cool night: `hardware_watch_incident` has no `low_power` row and metacog grants never drop to 0;
   - next warm afternoon: a `shed:warm` decision appears, and orion-mind on Hub turns is still granted (`gpu_pool_leases` holder `orion-mind`, `granted_at` not null).
4. **D11 report** shows cabinet ΔT after shed for at least 3 shed episodes. If ΔT ≈ 0, the follow-up is to drop the shed tier, not to tune it.

## Rollback

- D2/D5 go behind `HARDWARE_WATCH_HEAT_CONTROLLER=v2`. Setting `v1` restores today's rules. The flag ships **on** (`feedback_always_ship_with_flags_on`).
- D3 and D4 are independent pool/config changes, each a plain revert.
- The schema additions are additive columns; leave them in place on rollback.

## Recommended next patch

Ship in one PR, in this order inside it:
1. Export the replay fixtures and write the failing 7-day replay assertions. They must fail on today's code for the cool-night and hot-day reasons above.
2. `cabinet_heat_state` (D1) + level shed (D2) + AC-as-diagnosis (D5) → the replay passes.
3. Immediate refusal (D3), turn routes (D4).
4. Dedupe (D6), heat ceilings (D7), learned-shed eligibility (D8), snooze (D9), guard init (D10).
5. D11 reducer + report.

Deploy hardware-watch, gpu-pool, llm-gateway (routes), orion-mind and cortex-orch together; the route names must exist before callers use them. Deploy from the primary checkout on main after merge (`feedback_worktree_deploys_can_pin_a_worktree_as_production`).
