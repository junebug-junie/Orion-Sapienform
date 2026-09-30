# orion-hardware-watch

Watches the cabinet AC and the machines' temperatures. When the AC fails it sends a critical alert
at once, starts an urgent investigation, and (if the cabinet is warming) tells the GPU pool to stop
starting new background/system work. CPU and GPU heat outliers start urgent investigations too.

- Spec: `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md` (Part 4)
- Plan: `docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md`
- Rules (pure, replayable): `orion/hardware_watch/rules.py`
- Host: athena, port 8131

## What fires when

Every `HARDWARE_WATCH_TICK_SEC` (30 s) it reads Postgres and evaluates:

| Rule | Subject | Opens when | Resolves when | On open |
|---|---|---|---|---|
| `cooling` | `cabinet_ac` | live `cooling_watts < 150` for 3 min (`low_power`); no live reading for 5 min (`no_fresh_sample` / `device_offline` / `controller_not_ready` / `no_samples`); identical live watts for 60 min (`frozen`) | live `>= 500 W` for 10 min and no opening arm holds | critical alert (in-app + email, `dedupe_key=cooling:<id>:alert`), then urgent run (trigger `cooling`), then incident event |
| `cpu_heat` | `athena`, `circe` | `temp_c_max` above its own trailing-7-day p95 for 10 min (needs 1 day of history) | newest reading below p75 | urgent run (trigger `heat`) |
| `gpu_heat` | `circe/gpu3` … | `gpu{N}_temp_c` >= 85 C for 2 min (always armed), or above own p95 for 10 min once 3 days of history exist | below p75 (armed) or below 80 C | urgent run (trigger `heat`) |

"Live" = not `stale`, device online, controller ready, watts present. Rows from before #2382
(unknown staleness) count as live.

One open incident per `(rule, subject)`, enforced by a unique index. A heat spike caused by the
investigation itself lands on the incident already open. A restart reads the table: nothing re-fires.

## Shedding (GPU pool rule U4)

While a `cooling` incident is open the watcher checks the cabinet (`cabinet_temp_c` on athena):
rising >= 1 C in 15 min, at/above `thermal_gate`'s elevated 29.5 C, or no reading for 5 min ⇒ it
asks the pool to shed. The request latches until the incident resolves. Every open incident is
re-published every 60 s with `shed.valid_until = now + 300 s`; if the watcher dies the pool stops
shedding within 5 minutes.

The pool then grants nothing new to `background` or `system` leases (running work finishes; chat
and urgent work are unaffected). See `orion/gpu_pool/shed.py`.

Note: with the AC working the cabinet sits at/above 29.5 C on 65% of 30 s ticks (7 days to
2026-09-30 02:00Z, replay eval), so most AC incidents shed immediately. Shedding still only ever
engages while a `cooling` incident is open: a warm cabinet with a healthy AC sheds nothing. The
rise leg (>= 1.0 C over 15 min) is true on 1.5% of ticks (17 episodes, ~2.4/day) and is the
deciding leg (rising while still below 29.5 C) on only 0.6%.

The rise is computed by `cabinet_rise_c(cabinet, now, window_sec)` in `orion/hardware_watch/rules.py`
(pure, threshold applied by the caller). Orion's later learned `shed_background_gpu` action
(attend-to-act loop) imports the same function with its own 0.5 C threshold; it must not write a
second one.

### Three heat mechanisms, one room

| Mechanism | Where | Reads | Stops |
|---|---|---|---|
| Reverie render gate | `services/orion-thought/app/visual_chain.py` | cabinet temp (thermal_gate) | new reverie renders while hot |
| Pool swap-load `thermal` guard | `services/orion-gpu-pool/app/guards.py` | cabinet temp | loading an extra model onto a swap seat |
| Pool shed `cooling_incident` (U4) | `orion/gpu_pool/shed.py`, scheduler | this watcher's incident event | new background/system grants on every role |

They stack and none reads another's state. Only the shed depends on the AC; the other two read
temperature alone. A hot cabinet with a working AC closes the first two and never the third.

The shed lever is one board of named reasons (`orion/gpu_pool/shed.py`), lowest precedence number
wins: `cooling_incident` is 0 (the reflex, always wins). A later reason (Orion's own) gets a higher
number, may add blocks, and can never lift the reflex's. Pool `GET /health` -> `shed` shows every
reason with `active`/`effective`/`sources`, and `GpuPoolStateV1.shed` carries the same block.

## HTTP

- `GET /health` — flags, last tick, per-rule verdicts, open incidents.
- `GET /incidents?limit=50` — newest incidents.
- `POST /incidents/{id}/resolve` `{"by": "juniper"}` — close by hand; clears the pool's shed; the
  same rule+subject does not re-open for `HARDWARE_WATCH_OPERATOR_SNOOZE_SEC` (1 h).
- `POST /incidents/simulate` — only with `HARDWARE_WATCH_TEST_HOOK_ENABLED=true`: a SIMULATED AC
  incident (real alert, real urgent run, real shed), closed only by the resolve call.

## Kill switches

| Key | Off means |
|---|---|
| `HARDWARE_WATCH_ENABLED` | nothing is evaluated |
| `HARDWARE_WATCH_SHED_ENABLED` | the watcher never asks the pool to shed |
| `HARDWARE_WATCH_URGENT_ENABLED` | no urgent runs (AC alerts still go out) |
| `GPU_POOL_SHED_ENABLED` (pool) | the pool ignores every shed reason |

All default false in code, true in `.env_example` (test hook stays false).

## Deploy (consumer first)

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
