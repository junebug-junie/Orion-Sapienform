# orion-hardware-watch

Watches the cabinet's temperature, the cabinet AC, and the machines' temperatures. It protects on
**cabinet heat, not AC power** (thermal controller v2, the default):

- When the cabinet reaches 34 C, or its sensor goes silent, it tells the GPU pool to stop starting
  new background (and, at 34 C, system) work. It re-sends that every tick and stops when the cabinet
  cools, so nothing latches.
- When the AC looks dead *while the cabinet is warm*, it sends a critical alert and starts one urgent
  investigation (at most one of each per 6 h).
- GPU/CPU heat opens an incident only at a fixed ceiling (GPU 85 C, CPU 90 C).
- Between 29.5 and 34 C it does nothing itself: that band belongs to Orion's learned shed.

- Spec (v2): `docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md`
- Spec (v1, superseded rules): `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 4)
- Rules (pure, replayable): `orion/hardware_watch/rules.py`; cabinet owner: `orion/autonomy/cabinet_heat.py`
- Replay gate: `evals/run_rules_replay_eval.py` (7 real days, hour by hour; fixture from
  `evals/export_thermal_v2_fixture.py`)
- Host: athena, port 8131

## What fires when (v2)

Every `HARDWARE_WATCH_TICK_SEC` (30 s) it reads Postgres and evaluates:

| Rule | Subject | Opens when | Resolves when | On open |
|---|---|---|---|---|
| `cooling` | `cabinet_ac` | the AC's 15-min mean draw is under 140 W (`HARDWARE_WATCH_AC_LOW_MEAN_W`) AND the cabinet is elevated/hot/unknown or projected to reach 32 C within 20 min (`low_power`); no live reading for 5 min (`no_fresh_sample` / `device_offline` / `controller_not_ready` / `no_samples`); identical live watts for 60 min (`frozen`) | `low_power`: 15-min mean >= 1.5x the floor; others: mean back at/above the floor; no opening arm holds | critical alert (in-app + email) unless one went out for `cooling:cabinet_ac` in the last 6 h, then one urgent run per 6 h, then the incident event. **Alert-only: never sheds.** |
| `cpu_heat` | `athena`, `circe` | `temp_c_max` >= 90 C for 10 min (Q3 default, unconfirmed) | below 85 C, or sensor silent 15 min (`sensor_lost`) | urgent run (at most one per 6 h) |
| `gpu_heat` | `circe/gpu3` … | `gpu{N}_temp_c` >= 85 C for 2 min | below 80 C, or sensor silent 15 min (`sensor_lost`) | urgent run (at most one per 6 h) |

The trailing 7-day p95 is shown on every heat verdict (`above_p95_c`) as context; it no longer opens
anything (it is true ~5 % of the time by construction). An operator resolve snoozes only the reason it
resolved: resolving `low_power` never silences `device_offline`.

## The reflex shed (v2, GPU pool rule U4)

One cabinet read per tick (`read_cabinet_heat`). A reading up to 300 s old holds its state; one
failed query re-uses the last readings. Every tick the watcher publishes `HardwareWatchReflexShedV1`
on `orion:hardware:watch:reflex_shed` while:

| Cabinet | Pool shed reason | Blocks |
|---|---|---|
| >= 34 C (held until < 33 C) | `cabinet_hot` | background + system |
| no reading for 5 min | `cabinet_unknown` | background |
| no reading AND the AC's 15-min mean is low | `cabinet_hot` | background + system |

`valid_until = now + 3 ticks`. When the state drops it sends one `active=false` clear and stops; if
the watcher dies, the pool's copy lapses within ~90 s. `interactive` and `urgent` work is never shed;
one-shot background/system requests are refused at once (`shed:<reason>`) rather than waiting out
their deadline; durable-run holds keep waiting. `GET /health` -> `reflex_shed` shows the current
claim (and `would_shed` when `HARDWARE_WATCH_SHED_ENABLED=false`). Orion's learned shed is refused only
while this claim is active.

`HARDWARE_WATCH_HEAT_CONTROLLER=v1` restores the 2026-09-29 rules (AC-power incident + latched
`cooling_incident` shed) for the one-week rollback window; delete v1 after v2 runs clean.

### Heat mechanisms, one room

| Mechanism | Where | Line | Stops |
|---|---|---|---|
| Reverie render gate | `services/orion-thought/app/visual_chain.py` | `hot` 32 C (thermal_gate) | new reverie renders |
| Pool swap-load `thermal` guard | `services/orion-gpu-pool/app/guards.py` | `hot` 32 C or unknown (`read_cabinet_heat`, starts unknown) | loading an extra model onto a swap seat |
| Orion's learned shed | `orion/autonomy/self_shed.py` -> pool `orion_self_shed` | 29.5-34 C, Orion decides | new background grants for a bounded TTL |
| Reflex shed | this watcher -> pool `cabinet_hot` / `cabinet_unknown` | >= 34 C or sensor loss | new background (+ system at 34 C) grants |

### Does shedding cool anything? (D11)

The replay report's `d11_shed_effect` takes every real shed episode and reports the cabinet's
minute mean at start, +15 and +30 min and the GPU draw before/during, minus the same delta at no-shed
times with a matching temperature and prior rise (`orion/hardware_watch/shed_effect.py`). Until it
has >= 3 measured episodes it says `enough_to_judge: false`. If the effect stays ~0, drop the shed
tier rather than tune it.

## HTTP

- `GET /health` — flags, controller, last tick, per-rule verdicts (incl. `cabinet`, `reflex`), open incidents, `reflex_shed`.
- `GET /incidents?limit=50` — newest incidents.
- `POST /incidents/{id}/resolve` `{"by": "juniper"}` — close by hand; the same rule+subject+reason
  does not re-open for `HARDWARE_WATCH_OPERATOR_SNOOZE_SEC` (1 h). In v2 an incident never sheds, so
  resolving one does not change the pool; the reflex follows the cabinet.
- `POST /incidents/simulate` — only with `HARDWARE_WATCH_TEST_HOOK_ENABLED=true`: a SIMULATED AC
  incident (real alert, real urgent run; in v1 also a real shed), closed only by the resolve call. A
  drill never suppresses a real alert's dedupe window.

## Kill switches

| Key | Off means |
|---|---|
| `HARDWARE_WATCH_ENABLED` | nothing is evaluated |
| `HARDWARE_WATCH_SHED_ENABLED` | the watcher never asks the pool to shed (`/health` still shows `would_shed`) |
| `HARDWARE_WATCH_HEAT_CONTROLLER=v1` | the 2026-09-29 rules (rollback, one week) |
| `HARDWARE_WATCH_URGENT_ENABLED` | no urgent runs (AC alerts still go out) |
| `GPU_POOL_SHED_ENABLED` (pool) | the pool ignores every shed reason |

Kill switches default false in code, true in `.env_example` (test hook stays false). The controller
defaults to `v2` in code, compose and `.env_example`.

## Deploy (consumer first)

v2 (thermal controller redesign): from the primary checkout on main after merge, in this order --
gpu-pool (subscribes to `orion:hardware:watch:reflex_shed`, new routes), llm-gateway (route table from
the pool YAML), then hardware-watch, then orion-mind (`metacog_turn`). No migration.

First install (v1):

1. `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_hardware_watch_v1.sql`
2. `scripts/safe_docker_build.sh orion-gpu-pool up -d --build` (the consumer, `GPU_POOL_SHED_ENABLED`)
3. `scripts/safe_docker_build.sh orion-hardware-watch up -d --build`
4. GPU temperature: `orion-biometrics` on circe must be rebuilt for `gpu_host_stats.sh` to report
   `temperature.gpu` (the GPU rule has no data until then; its p95 arm needs 3 days after that).

## Replay numbers (real history, 2026-09-30 cutoff)

`evals/run_rules_replay_eval.py` over the committed fixture:

- AC: one incident in 3.9 days: `low_power` on the 33.11 W kWh-counter stretch, opened 3 min after
  the first row, resolved 10 min after real watts returned (105 min). No stale, offline or
  not-ready rows exist; the largest row gap is 76 s; nothing else fires.
- Fault injection at three real moments: silence / offline / stale open in 300 s, 0 W in 180 s,
  a frozen reading in 60 min.
- CPU heat (7 days): circe 5 incidents (0.71/day), athena 1 (0.14/day); open 1.8% of the time.
- GPU heat: no `gpu{N}_temp_c` history yet; the p95 arm stays disarmed, only the 85 C ceiling is live.

## Tests and evals

```bash
cd services/orion-hardware-watch
PYTHONPATH=../..:. python -m pytest tests -q
python evals/run_rules_replay_eval.py            # replays the committed real-history fixture
python evals/run_rules_replay_eval.py --export   # refresh the fixture from live Postgres
```
