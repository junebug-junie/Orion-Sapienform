# Hardware watch + pool shedding (Plans 4 and 5 of 5)

**Goal:** A dead or lying cabinet AC raises a critical alert within one 30 s tick of the rule
tripping, starts an urgent investigation, and — when the cabinet is warming — stops the GPU pool
from starting new background/system work until the AC is back. CPU and GPU heat outliers start
urgent investigations too.

**Spec:** `docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md`
(Part 4, "Decisions locked", Part 1 note on U4). Plans 1–3 are merged (#2382, #2385, #2391).

**Branch / worktree:** `feat/hardware-watch-and-shedding`,
`/mnt/scripts/Orion-Sapienform-hardware-watch-and-shedding`.

**Test interpreter:** `/mnt/scripts/Orion-Sapienform/.venv/bin/python` (`PY` below). Service
tests run from the service dir with `PYTHONPATH=<worktree>:.`.

## Live data this plan is built on (pulled 2026-09-29)

| Signal | Source | What the data says |
|---|---|---|
| AC watts | `home_cooling_sample` 09-26 04:52 → 09-29 22:44, 64,082 rows | < 150 W only in the 1 h 38 m 33.11 W stretch (1,171 rows, the kWh-counter bug). Longest identical-value run outside it: 3 min 7 s. Largest gap between rows: 75 s. `stale=true` never. `sample_age_sec` max 9.3 s since #2382 (09-28 09:28). |
| CPU temp | `orion_biometrics_summary.measurements->>'temp_c_max'`, 7 days | athena p75 62 / p95 73 / max 85; circe p75 59 / p95 62 / max 71. |
| Cabinet temp | same table, `cabinet_temp_c`, athena, 7 days | p5 26.9 / p50 30.0 / p95 32.1 / max 33.3 °C. **Above thermal_gate's elevated 29.5 °C about half the time with the AC working.** |
| GPU temp | not collected | This PR starts collecting it. |

Consequence for the design: the shed rule's "cabinet ≥ elevated" arm is true most of the day, so
in practice almost any open AC incident sheds. That is the intended direction (a dead AC in a room
already at 30 °C is the dangerous case); it is recorded here so nobody reads the rise-rate arm as
the usual trigger.

## Decisions made in this plan (not in the spec)

1. **Shed latches per incident.** Once a cooling incident requests shedding it keeps requesting
   until the incident resolves or Juniper resolves it. A rise-rate test would flap (+1 °C, plateau,
   unshed, warm again).
2. **An unreadable cabinet sensor sheds too** (no `cabinet_temp_c` in 5 min, the thermal gate's
   `DEFAULT_MAX_READING_AGE_SEC`). Same stance as the pool's `thermal` swap guard: a dead AC in a
   room nobody can measure is treated as warming.
3. **The pool's copy of a shed reason expires.** The watcher re-publishes every open incident every
   `HARDWARE_WATCH_REFRESH_SEC=60`; the event carries `shed.valid_until = now + 300 s`. If the
   watcher dies the pool stops shedding after 5 min (fail-open for shedding; the alert and the
   investigation have already gone out). A restarted pool re-learns the reason within 60 s.
4. **Frozen-reading resolve needs the freeze to break.** Resolve = `≥ 500 W` fresh for 10 min AND
   the frozen condition no longer holds; otherwise a frozen 850 W would open and resolve on
   alternate ticks.
5. **Heat rules need minimum history before the p95 arm arms**: CPU 1 day, GPU 3 days (spec). The
   GPU absolute ceiling 85 °C is always armed, sustained 2 min, re-arms below 80 °C when the p95
   arm is not armed (below p75 when it is).
6. **Only the cooling rule sends its own alert.** Heat incidents start an urgent run; that run's
   must-deliver report (Part 3) is their notice. A cooling incident sends the alert first, then the
   urgent request, then (on recovery) an in-app "AC recovered" notice.
7. **Per-GPU temperature is stored as `measurements.gpu{index}_temp_c`** next to
   `gpu_temp_c_max` in `orion_biometrics_summary` (indexed on `(node, timestamp)`), and also rides
   the raw sample payload (`gpu.gpus[].temperature_gpu_c`). `orion_biometrics` has no timestamp
   index, so the watcher never reads it.
8. **No new pool event name and no YAML key.** `GpuPoolEventV1.event` and the pool YAML's
   `Defaults` are both `extra="forbid"` with consumers that bake their own copies (sql-writer,
   lane-controller). Shed facts ride existing shapes: a `queued` event with
   `reason="shed:<name>"`, and a new optional `GpuPoolStateV1.shed` block (the state has no
   validating consumer outside the pool; Hub and harness-governor read it as a dict).

## The shared shedding lever (extension point)

`orion/gpu_pool/shed.py` is the one place the pool learns "stop starting X":

```text
SHED_REASONS = {
  "cooling_incident": ShedReasonSpec(precedence=0, blocks=("background", "system")),   # reflex, this PR
  # later (attend-to-act loop, docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md):
  # "orion_shed_background": ShedReasonSpec(precedence=10, blocks=("background",)),
}
```

- A reason is **named**, has a **precedence** (lower wins) and a fixed set of priorities it may
  block. Only `background` and `system` are sheddable; the registry refuses a reason that names
  `interactive` or `urgent` at import time.
- Signals are `(reason, source_id, detail, valid_until)`. The board keeps one per
  `(reason, source_id)`; a signal past `valid_until` is ignored.
- The effective shed is the **union** of active reasons' blocked priorities. Each blocked priority
  is **attributed** to the highest-precedence active reason that blocks it, and that name is what
  the lease trace and the state show. A lower-precedence reason can add blocks but can never lift
  one: there is no "unshed" signal, only a reason's own clear.
- `GPU_POOL_SHED_ENABLED` (pool) is the kill switch for the whole lever, any reason.
  `HARDWARE_WATCH_SHED_ENABLED` (watcher) only stops the watcher asking.
- **Adding Orion's reason later** = one `SHED_REASONS` entry + a producer that calls
  `ShedBoard.set(...)` through a pool bus verb or event it owns. No scheduler change: the scheduler
  only sees `priority -> reason name`.

Scheduler rule **U4** (in `schedule()`'s docstring): a queued lease whose priority is shed is not
granted (no new grants), reported as `Shed(lease_id, reason)`, and does not count as demand
anywhere (owner-waiting recalls, swap loads, draining a swap seat). Running work is untouched: a
granted hold keeps making calls (children are still granted), nothing is recalled. Deadlines still
apply. `interactive` and `urgent` are never shed — including under the urgent rollback
(`urgent_max_concurrent: 0`), because shed membership is decided on the lease's original
priority.

## How the three heat mechanisms interact

| Mechanism | Where | Input | What it stops |
|---|---|---|---|
| Visual reverie render gate | `services/orion-thought/app/visual_chain.py` (`thermal_state`) | cabinet temp | New reverie renders while hot |
| Pool swap-load `thermal` guard | `services/orion-gpu-pool/app/guards.py` | cabinet temp via Hub | Loading an extra model onto gpu2 (`SwapBlocked guard:thermal`) |
| Pool shed `cooling_incident` (U4, this PR) | `orion/gpu_pool/shed.py`, scheduler | watcher incident event | New background/system grants on every role |

They stack; none reads another's state. U4 is the only one that depends on the AC (the other two
read temperature alone). With the AC down and the cabinet ≥ 32 °C all three are active.

## Tasks

- [x] 1. Contract: `orion/schemas/hardware_watch.py` (`HardwareWatchIncidentV1`, `HardwareWatchShedV1`),
  registry entries, channel `orion:hardware:watch:incident` (producer `orion-hardware-watch`,
  consumer `orion-gpu-pool`). Test: schema round trip, registry resolve.
- [x] 2. GPU temperature: `gpu_host_stats.sh` adds `temperature.gpu` as the last CSV column
  `temperature_gpu_c`; `extract_measurements` adds `gpu_temp_c_max` and `gpu{i}_temp_c`. Tests in
  `services/orion-biometrics/tests/test_gpu_collector.py` and the pipeline tests.
- [x] 3. Rules: `orion/hardware_watch/rules.py` (pure: cooling, heat, shed, percentile). Tests:
  every open/resolve arm, boundaries, the frozen-resolve flap, stale/offline/no-row.
- [x] 4. Pool: `orion/gpu_pool/shed.py`, scheduler U4 + `Shed` decision, runtime board + Hunter on
  the incident channel, `GpuPoolStateV1.shed`, `/health` shed block, `GPU_POOL_SHED_ENABLED`.
  Tests: U4 in `orion/gpu_pool/tests`, runtime shed tests, pool-day eval shed scenario.
- [x] 5. Service `services/orion-hardware-watch/` (README, `.env_example`, compose, Dockerfile,
  requirements, `app/{settings,store,watcher,main}.py`, tests, evals). Migration
  `services/orion-sql-db/manual_migration_hardware_watch_v1.sql`.
- [x] 6. Replay eval over real history (fixture exported from Postgres, committed gzip):
  AC fires once on the 33.11 W stretch and nowhere else; CPU p95 episode counts; cabinet rise
  distribution.
- [x] 7. Env parity (`sync_local_env_from_example.py`), static gates, review, PR report.

## Acceptance

1. `pytest orion/gpu_pool/tests` + pool service tests + pool-day eval green with U4.
2. `pytest services/orion-hardware-watch/tests` green; replay eval passes on the committed fixture.
3. Replay: AC incident opens within 3.5 min of 04:52:37 on 09-26 (low_power), stays one incident
   through the frozen arm, resolves ≥ 10 min after the plug started reading real watts; zero other
   AC incidents in 3.5 days.
4. Live (after deploy, UNVERIFIED here): simulated incident → notify row with
   `dedupe_key=cooling:<id>:alert` → urgent run → pool `/health` shows `cooling_incident` active →
   resolve clears it.

## Rollback

`HARDWARE_WATCH_ENABLED=false` (watcher does nothing), `HARDWARE_WATCH_SHED_ENABLED=false`
(watcher never asks to shed), `GPU_POOL_SHED_ENABLED=false` (pool ignores every shed reason).
Each needs only its own container restarted.
