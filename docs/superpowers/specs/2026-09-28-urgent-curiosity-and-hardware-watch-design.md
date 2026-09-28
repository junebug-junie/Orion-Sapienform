# Urgent curiosity runs + hardware watch — design

**Date:** 2026-09-28
**Status:** Design approved by Juniper in chat (2026-09-28). Part 1 implemented (Plan 2, branch
`feat/gpu-pool-urgent-class`); Parts 2–5 not implemented.
**Branch:** `docs/urgent-curiosity-hardware-watch`

## Arsonist summary

Orion has no way to say "look at this **now**." Every curiosity run enters the GPU pool at the
lowest priority (`background`), the Hub's "run now" only skips cooldowns, and the pool never takes a
slot from running work for priority reasons. Meanwhile the room's only cooling (portable AC on a
Shelly Wave plug) can die silently: `orion-zwave` republishes its last cached wattage as a fresh
sample when the plug stops answering, so a dead AC looks like a healthy 850 W.

This design adds:

1. an **`urgent`** GPU priority that jumps every queue except chat and **pauses** (recall → requeue
   in original place → resume) running background/system work to get a slot;
2. **seeded urgent curiosity runs** — one bus contract, fed by a Hub button (typed question) and by
   a hardware watcher;
3. a **must-deliver report** — every urgent run ends in a critical Hub + email notice, including
   failures;
4. **`orion-hardware-watch`** — a thin service that fires urgent runs on AC failure and on
   CPU/GPU heat outliers, and sheds background GPU load while the AC is down;
5. an **AC sensing fix** so a silent plug reads as stale/absent, never as live watts.

## Decisions locked (Juniper, 2026-09-28)

| Topic | Choice |
|---|---|
| Strength | Urgent jumps every queue; pauses running background work, then system work if no background is left |
| Chat | Untouched. Urgent never recalls or outranks work on the chat role; it takes the next non-chat slot |
| Pause semantics | Paused work is replayed and resumed on the next freed slot, ahead of newer same-priority work |
| Grace | `urgent_preempt_grace_sec: 5` |
| Concurrency | Urgent runs may run in parallel, capped at `urgent_max_concurrent: 3` |
| Manual trigger | Hub button with a typed question |
| Automatic triggers | AC failure (primary), CPU heat outlier, GPU heat outlier |
| Heat threshold | High outliers: own 7-day p95, sustained 10 min, re-arm below p75 (not p75/p25 — see gate below) |
| AC trigger | < 150 W for 3 min, OR silent/offline/stale 5 min, OR identical reading 60 min |
| AC order | Alert first (immediately), investigate second, report findings |
| Notification | Mandatory: critical severity, `in_app` + `email`, never gated by quiet hours/throttle |
| Load shedding | While an AC incident is open and the cabinet sensor is rising: pool stops granting background + system work; chat and urgent keep running |
| Watcher host | New thin service `orion-hardware-watch` on athena |

## Current architecture (grounded)

### GPU pool

- Single decision function `schedule()` in `orion/gpu_pool/scheduler.py`; rules in
  `config/gpu_pool.yaml`. `priorities: [interactive, system, background]` (yaml line 32).
- `Priority = Literal["interactive","system","background"]` in `orion/schemas/gpu_pool.py:39`.
- Queue order = priority, then oldest (`_order`, scheduler.py:288). Priority only decides the next
  grant. Recall happens only for `owner_waiting`, `card_unlent`, `max_hold`, `draining`
  (scheduler.py:460-517). `test_hold_waits_for_a_busy_slot_rather_than_preempting_it` locks in
  "no priority preemption".
- Rule H1: at most one hold per role (`free_for`, scheduler.py:218-220).
- Recalled hold grace `hold_clawback_grace_sec: 600`; request grace `clawback_grace_sec: 60`.
- Priority comes from the route table (`routes:`), not per request. Durable-run holds are forced to
  `background` (`ResourceRequirementV1.priority: Literal["background"]`,
  `orion/schemas/resource_admission.py:22`).

### Durable runs (curiosity path)

- Hub `CuriosityInvestigation.tick()` (`services/orion-hub/scripts/curiosity_investigation.py:1452`)
  → `_dispatch_durable_run` (:3367) → cortex-orch → `orion:durable:run:request` → durable-runs
  acquires a pool hold (`pool_hold.py`) → calls back Hub on `orion:curiosity:turn:request`.
- LangGraph-checkpointed; a recalled hold is released at the next node boundary (`guard`,
  `admission_runtime.py:357-386`); a hold the pool re-queued is kept "same lease_id, same place in
  line" (`release(keep_requeued=True)`, :388). **This is the existing resume seam.**
- `MAX_CONCURRENT_DRIVERS = 4` (`admission_runtime.py:69`), fair rotation by `updated_at`.
- Hub `_run_lock` serializes curiosity turns (`already_running`).
- `POST /curiosity/api/run-now` takes no body; the subject is always self-authored.

### Telemetry

- Host CPU/board temp: `orion-biometrics` `sensors -j` max → `orion_biometrics_summary.measurements
  ->>'temp_c_max'`, ~30 s cadence, per `node` (athena, circe).
- GPU temperature: **not collected continuously** (`orion/sensors/gpu_host_stats.sh:16` omits
  `temperature.gpu`).
- Cabinet room sensor (Nano): `CABINET_SENSORS_PATH`, also persisted as
  `measurements->>'cabinet_temp_c'` in `orion_biometrics_summary`; `orion/autonomy/thermal_gate.py`
  hot at 32.0 °C, elevated at 29.5 °C.
- AC: `orion-zwave` → `orion:home:cooling:sample` (`home.cooling.sample.v1`) → `home_cooling_sample`.

### Notification

- `orion-notify` `POST /notify` publishes in-app unconditionally and emails when
  `channels_requested` includes `email` or severity is `error|critical`. `/notify` does not run
  quiet-hours/throttle policy. SMTP configured locally; delivery **UNVERIFIED**.
- Curiosity's own outreach path can be blocked by an in-progress Hub turn and never emails.

## Metric quality gate (triggers)

### CPU heat — `temp_c_max`

1. **Provenance:** `services/orion-biometrics/app/metrics.py:438-467` (`sensors -j`, max of all
   `temp*_input`) → `orion/telemetry/biometrics_pipeline.py` → sql-writer.
2. **Independence:** independent of AC watts (different device). Correlated with GPU temp via load
   (same chassis on circe) — the GPU rule is a separate incident key, not a second vote.
3. **Theory:** die/board temperature relative to the machine's own recent distribution flags
   abnormal heat regardless of absolute sensor calibration (sensor offsets differ per board).
4. **Live data (7 days, pulled 2026-09-28):**

   | node | n | p25 | p75 | p95 | p99 | max |
   |---|---|---|---|---|---|---|
   | athena | 19,326 | 54 | 62 | 72 | 77 | 84 |
   | circe | 19,242 | 51 | 59 | 63 | 66 | 80 |

   Sustained ≥ 10 min episodes per rule: p75 → 4.7/day athena, 5.6/day circe; p75+5 → 2.3 / 0.7;
   **p95 → 0.1/day athena, 1.3/day circe.** p75 (and p25) are rejected: above-p75 is true ~25% of
   the time by construction. Not degenerate; returns below p75 routinely (re-arm reachable).
5. **Existing mechanism:** none (only fixed 50–85 °C thermal pressure scale, no baseline).
6. **Reversibility:** pure rule + config thresholds; no schema baked in beyond the incident event.

### GPU heat — `gpu_temp_c_max` (new)

1–3. Provenance will be `nvidia-smi --query-gpu=temperature.gpu` in `gpu_host_stats.sh`; same theory
as CPU. 4. **No live data exists yet** → the p95 rule stays unarmed until ≥ 3 days of history; only
an absolute 85 °C ceiling fires meanwhile. Re-run the gate on real data before arming p95.
5. On-demand `skills.gpu.nvidia_smi_snapshot.v1` already parses it (reuse its parser).

### AC failure — `cooling_watts` + freshness

1. **Provenance:** `services/orion-zwave/app/main.py:170-191` polls `Electric_W` each 5 s cycle.
2. **Independence:** independent device from compute temps; cabinet Nano temperature is the
   independent corroboration for load shedding.
3. **Theory:** the AC compressor draws 700–950 W when cooling; fan-only draw is unmeasured; off draws ~0.
   A cooling-capable reading below 150 W for minutes means the room is not being cooled.
4. **Live data (2026-09-26 → 09-28, 36,524 rows):** 0 nulls, 0 offline, 0 switch-off. Normal band
   700–950 W. One anomaly: the first 1 h 38 m after pairing read **exactly 33.1 W for 1,171 samples** — the
   since-fixed bug that read the kWh energy counter (propertyKey 65537) as watts
   (`zwave_client.py` comment on `METER_W_PROPERTY_KEY`). It is a real example of a lying reading
   (the frozen and low-power rules both catch it), not evidence that fan-only draws ~33 W.
   Only 3–9 distinct values per hour ⇒ most samples are republished cache. `sample_age_sec` is
   always null. **Degenerate-in-a-dangerous-way:** freshness is unknowable today (see Part 5).
5. **Existing mechanism:** none alerting.
6. **Reversibility:** rule thresholds in env; the stale flag is an additive optional schema field.

## Design

### Part 1 — `urgent` priority in the GPU pool (as shipped, Plan 2)

Plan: `docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-2-pool-urgent.md`. Rules live in the
`schedule()` docstring (`orion/gpu_pool/scheduler.py`, U1–U3).

**Changed from the first draft of this part:**

- **Only durable-run holds are paused.** One-shot requests (a single inference) are never recalled
  for urgent work: they finish in seconds and urgent is first in line for the freed slot. A request
  can only be killed and replayed, not paused, so the gateway replay and the `gpu_pool_preempted`
  error code were dropped.
- **No `curiosity_urgent` route.** A hold's priority comes from `ResourceRequirementV1.priority`,
  not from the route table, so the route would be an unused label.
- **Rule U4 (load shedding) moved to Plan 5**, with the hardware watcher that turns it on. Shipping
  it now would leave a guard with nothing setting it.
- **Added:** durable-runs drives urgent runs first and outside its 4-driver cap (below).

**What shipped:**

- `priorities: [urgent, interactive, system, background]`; the pool `Priority` Literal and
  `ResourceRequirementV1.priority` both accept `urgent`.
- **U1 (pause):** an urgent lease that gets no slot pauses one granted `background` (then `system`)
  durable-run hold on a role it could use: `Recall(victim, now + 5 s, "urgent_preempt")`. Most
  recently granted first. Never paused: interactive, urgent, operator, request or child leases, or
  the owner of a role the urgent lease only borrows (re-queued, that owner would win the role back).
  One pause per waiting urgent lease; a pause already under way on a slot it could take counts.
  That includes a paused hold already re-queued whose last call (child lease) is still on the slot:
  the call is still the pause, so urgent waits for it instead of pausing a second run, and
  durable-runs polls every second after an urgent recall so it cancels that call quickly. Only a
  role in the urgent lease's own class counts, and only urgent leases the cap still has room for
  are owed a pause or count as waiting owners.
- **Urgent owner reclaiming a borrowed role:** when an urgent lease owns a role (e.g. `agent`) and a
  background/system hold is borrowing it, the owner reclaim of that hold *is* the pause: reason
  `urgent_preempt`, 5 s grace, re-queued in place. It is the hold's one pause, not an extra one.
  The borrower is paused whatever its priority: a system borrower is paused even when a
  background hold elsewhere could have been paused instead.
- **Chat stays untouched.** Urgent is `agent` class, so it reaches the `chat` role only while gpu0 is
  lent. No chat or interactive lease is ever recalled by urgent. If an urgent hold borrows `chat`,
  chat's own requests may use that hold's gaps (between its calls), same as any owner.
- **U2 (resume in place):** the pool aborts a paused hold after the grace and puts it back in line
  as `queued` with the same `lease_id` and `created_at`, reason `urgent_preempt`, no attempt spent
  (`lease_graph.transition`). A caller that would not use a re-grant (`retryable=false`) ends as
  before.
- **U3 (stacking):** urgent holds are exempt from H1's one-hold-per-role, bounded by slots and
  `urgent_max_concurrent`. Urgent skips a swap seat's `after_wait_sec` when no pause can serve it;
  seat guards (thermal) still apply.
- **The pool says why.** Its `queued` and `recall` replies carry `reason` (the field already existed
  on `GpuLeaseReplyV1`; now filled in).
- **Durable runs:**
  - A turn preempted mid-node records `run.preempted`, spends no attempt, keeps its hold (same
    lease, same place in line) and replays the node when the hold is granted again. The Hub's run
    story shows it as "paused for urgent work", not as a failure.
  - A hold recalled for urgent work *before* its step starts is not released. Durable-runs waits
    for the pool to re-queue it in place: local clock, at most grace 5 s + 3 s margin, and each
    status call is time-bounded. Any other recall reason (or no re-queue in time) keeps the old
    path: release as `recalled_before_start`.
  - Reconcile drives urgent runs first, outside `MAX_CONCURRENT_DRIVERS` (4), capped at
    `urgent_max_concurrent` (3). `list_pending()` puts urgent rows first.
- Config: `defaults.urgent_preempt_grace_sec: 5`, `defaults.urgent_max_concurrent: 3`.
- **Rollback:** `urgent_max_concurrent: 0`, then redeploy **both** `orion-gpu-pool` and
  `orion-durable-runs` (each bakes its own copy of the yaml; durable-runs reads
  `urgent_max_concurrent` for its driver-cap bypass). The scheduler then treats urgent as
  background (no pause, no stacking) and durable-runs drives urgent runs inside the normal cap.
  `list_pending()`'s SQL still sorts urgent rows first; harmless, it only changes which rows are
  read first.
- **Rollout order:** `orion-gpu-pool` and circe's `orion-gpu-lane-controller` go together. The
  lane-controller re-reads `/repo/config/gpu_pool.yaml` live from circe's checkout, but parses it
  with the `orion` code baked into its image, whose `Defaults` refuse unknown keys: pulling circe's
  checkout without rebuilding the lane-controller makes every actuation fail `config_unloadable`.
  Pulling circe's checkout and rebuilding the lane-controller is one step. `orion-llamacpp-host`
  needs nothing. `orion-sql-writer` must be redeployed before Plan 3 sends urgent work (it validates
  `GpuPoolEventV1.priority` against the priority list).
- **Known limitation:** a finished run's hold kept for Hub outreach compose (Door-A) is an ordinary
  background hold, so urgent work can pause it. Durable-runs then ends that hold at its next
  outreach heartbeat (the pool may briefly re-grant it first), so an outreach compose still in progress loses its GPU
  hold. No code change in Plan 2.
- **Live status:** unit tests + the pool eval's urgent scenario cover the pause/resume path. A live
  preemption smoke is **UNVERIFIED**: it pauses a real background run, so it waits for Juniper's
  approval.

### Part 2 — seeded urgent curiosity runs

- New schema `CuriosityUrgentRequestV1` (`orion/schemas/curiosity_urgent.py`):
  `incident_id`, `question` (1–2000 chars), `trigger: Literal["manual","heat","cooling"]`,
  `subject` (e.g. `athena`, `circe/gpu2`, `cabinet_ac`), `evidence: dict` (snapshot of the readings
  that fired), `requested_at`, `requested_by`.
- New channel `orion:curiosity:urgent:request`; producers `orion-hub`, `orion-hardware-watch`;
  consumer `orion-hub`. Registered in `orion/bus/channels.yaml` + `orion/schemas/registry.py`.
- Hub: `POST /curiosity/api/urgent {question}` publishes the request (manual); a Hub consumer turns
  any request into a run. Urgent runs bypass `_run_lock`, cooldown, daily cap, waking window,
  energy hold. They keep `MIN_HARNESS_STEPS`. At most one open run per `incident_id`.
- `CuriosityRunBriefV1` gains optional `urgent: CuriosityUrgentSeedV1` (incident_id, question,
  trigger, subject, evidence). The kickoff prompt uses the seed question + evidence instead of the
  self-authored subject.
- Already shipped in Part 1: `ResourceRequirementV1.priority` accepts `urgent` (that is what makes
  the hold urgent; there is no route for it), and durable-runs drives urgent runs outside the
  4-driver cap. Part 2 only has to set `priority: urgent` on the run's admission.
- Grant `orion_readonly` `SELECT` on `orion_biometrics_summary` (includes the cabinet Nano's
  `measurements->>'cabinet_temp_c'`) and `home_cooling_sample` (new
  `scripts/sql/2026-09-28_grant_orion_readonly_hardware.sql`).
- UI: Curiosity panel gets a question box + "Run urgent" button and an urgent-runs list (status,
  incident, trigger).

### Part 2b — urgent run mechanics (inside the turn)

Today (grounded): one `claude -p` turn (`orion/harness/fcc_motor.py:933-962`) on route `agent`
(`llamacpp/agent` via llm-gateway), inside `harness-governor` on athena, Bash with
`bypassPermissions`. Hub path `_turn_result_for` → `execute_unified_turn` → Thought stance
`react()` (may defer/refuse, `turn_orchestrator.py:1090`) → `HarnessRunRequestV1`. Kickoff prompt
(`orion/curiosity/kickoff_prompt.py:954-1034`) is "your own time, no task" + self material, prose
output, "no verdict schema"; only structured output is the `:TurnOutcome` graph node. Validity:
final frame, ≥ `MIN_HARNESS_STEPS=3`, non-error, non-empty. Limits: 7200 s turn, 420 s stall,
3 attempts, 30 s·2^n backoff. Output → journal; optional reach-out = second, blockable turn.

Urgent changes (durable graph `curiosity.investigate` reused; behaviour keyed off `brief.urgent`):

| Step | Urgent behaviour |
|---|---|
| Stance | Skipped. Hub passes the urgent seed so `execute_unified_turn` bypasses `react()` defer/refuse; a stance error is a run failure, not a deferral |
| Prompt | New `build_urgent_prompt(seed)` in `orion/curiosity/urgent_prompt.py`: assignment header (question or fired rule), evidence bundle, checklist (confirm real vs sensor fault → cause → severity → recommended operator action), tool guide, clock. No random material, priors, peer briefs, dreams |
| Tools | Existing sandbox. `orion_readonly` gets SELECT on `orion_biometrics_summary`, `home_cooling_sample`, `hardware_watch_incident`; curl to pool `GET` lease/state endpoints; `docker ps/logs/stats` on athena (socket already mounted — live smoke required). No live shell on circe: circe facts come from the evidence bundle |
| Evidence bundle | Collected by hardware-watch at incident open (manual runs: Hub collects the same bundle): last 60 min of the triggering series, current per-host temps/power/fan, per-GPU temp/power/util (from biometrics sample incl. new `temperature.gpu`), cabinet temp trend, AC readings + freshness, pool active leases, `docker ps` on athena, container restarts in last hour. Stored on the incident row; injected into the prompt and readable via psql |
| Output | Required graph node `:IncidentReport{incident_id, is_real: real|sensor_fault|unclear, likely_cause, evidence (list of cited readings/queries), severity: low|high|critical, operator_action, confidence 0-1}` plus the prose. `read_turn_result` reads it. Missing/malformed ⇒ report still sent, prose attached, flagged `no_structured_verdict` |
| Validity | Keep ≥ 3 steps, non-empty, non-error. Add: an `IncidentReport` whose `evidence` is empty counts as `no_structured_verdict` (no empty-shell verdicts) |
| Limits | Turn `HUB_CURIOSITY_URGENT_TURN_TIMEOUT_SEC=900`, stall 180 s, `max_attempts=2`, backoff 10 s, overall `HUB_CURIOSITY_URGENT_TIMEOUT_SEC=1200` |
| Timeout | Partial draft sent, labelled INCOMPLETE |
| After | Must-deliver report (Part 3) replaces the reach-out turn. Journal entry kept; `:TurnOutcome.continue_line` set to the incident so ordinary curiosity can revisit it |
| No GPU | If no urgent hold is granted within `HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC=120` (circe down/hot, pool unavailable), send a no-LLM report: evidence bundle + fired rule, marked "not investigated" |

Defaults chosen by the agent (Juniper skipped the question; override freely): graph-node report with
prose fallback; evidence bundle instead of SSH to circe; no-LLM report instead of cloud fallback.

### Part 3 — must-deliver report

- On urgent run end (completed, failed, timed out at `HUB_CURIOSITY_URGENT_TIMEOUT_SEC=1200`, or
  empty output), Hub calls `NotifyClient.send` with `severity="critical"`,
  `channels_requested=["in_app","email"]`, `dedupe_key=f"urgent:{incident_id}:report"`. Body order:
  verdict line (`is_real` / severity / confidence), operator action, likely cause, cited evidence,
  then the prose. Flags shown at top when present: INCOMPLETE, `no_structured_verdict`,
  "not investigated". Failure ⇒ `investigation failed: <reason>` + the raw evidence bundle.
  No outcome is silent.
- Retries until orion-notify accepts (bounded backoff, logged with incident_id).

### Part 4 — `orion-hardware-watch`

- New service `services/orion-hardware-watch/` (athena). Every `HARDWARE_WATCH_TICK_SEC=30` it reads
  Postgres and evaluates pure rules in `orion/hardware_watch/rules.py` (no I/O; replayable).
- Incidents: one open per `(rule, subject)`, persisted (table `hardware_watch_incident`) so a restart
  does not re-fire. Transitions emit `hardware.watch.incident.v1` on
  `orion:hardware:watch:incident` (opened / resolved), which is also the pool's shed guard input.
- **AC rule (subject `cabinet_ac`):** open when any of: `cooling_watts < 150` for 3 min; no fresh
  sample (`stale` or no row) for 5 min, or `device_online=false` / `controller_ready=false` for 5
  min; identical `cooling_watts` for 60 min. On open: **immediately** send critical notice
  (`dedupe_key=f"cooling:{incident_id}:alert"`), then publish `CuriosityUrgentRequestV1`
  (trigger `cooling`). Resolve when `cooling_watts ≥ 500` and fresh for 10 min.
- **CPU heat (subjects `athena`, `circe`):** open when `temp_c_max` > own trailing-7-day p95 for
  10 min; publish urgent request (trigger `heat`); resolve below p75. Percentiles recomputed hourly.
- **GPU heat (subjects `circe/gpuN`):** p95 rule armed only with ≥ 3 days of history; absolute
  ceiling `HARDWARE_WATCH_GPU_CEILING_C=85` always armed.
- **Feedback guard:** a subject with an open heat incident cannot open another; urgent runs started
  by the watcher carry `incident_id`, so a heat spike caused by the investigation itself lands on the
  already-open incident.
- **Shedding:** hardware-watch evaluates open `cabinet_ac` incident AND rising cabinet
  temperature (`orion_biometrics_summary.measurements->>'cabinet_temp_c'` on athena up ≥ 1 °C over
  15 min, or ≥ `thermal_gate`'s elevated 29.5 °C) and publishes the result on the incident event;
  the pool consumes it as guard `cooling_incident` → U4 (built here, in Plan 5): while set, no new
  grants to `background` or `system` leases; running ones finish (no recall); `interactive` and
  `urgent` unaffected.
  Cleared when the incident resolves or Juniper cancels it (`POST /incidents/{id}/resolve`).
- GPU temp collection: add `temperature.gpu` to `gpu_host_stats.sh`; carry through
  `biometrics_pipeline.py` to `measurements.gpu_temp_c_max` (+ per-GPU in sample payload).

### Part 5 — AC sensing fix (`orion-zwave`)

- Root cause: `refresh_meter_watts` swallows poll failures at debug level
  (`zwave_client.py:228-230`); `main.py:170-191` then publishes the cached value with `ts=now` and
  no age.
- Track `last_fresh_at` per node (successful poll or `value updated` event for `Electric_W`);
  publish `provenance.sample_age_sec`.
- If `sample_age_sec > COOLING_STALE_AFTER_SEC` (120): omit `cooling_watts/volts/amps`, set new
  optional `CoolingObservedStateV1.stale: bool = True`. Never republish stale as live.
- Poll failures: warning log + counter. Heartbeat `details` carry `cooling_sensor`
  (`fresh|stale|unknown|error`), `cooling_sample_age_sec`, `zwave_connected`,
  `consecutive_poll_failures` (the chassis status field stays `ok` = process alive).
- A dead websocket is reconnected by the poll loop; stale samples keep publishing meanwhile, so
  an outage is visible as `stale=true` rows rather than silence.
- Live check 2026-09-28: `node.poll_value` Electric_W succeeded 3/3 in 0.04–0.44 s (885.7 W), so
  "fresh = successful poll" will not false-alarm.
- sql-writer also persists `sample_age_sec`; Hub `/api/cabinet/cooling/latest` adds
  `sensor_stale`, `sample_age_sec`, `last_fresh_at` (read from `payload_json`).
- Hub Cabinet panel: red "AC reading STALE since HH:MM" instead of a wattage.
- sql-writer: persist `stale` (new nullable column) so history distinguishes stale from live.

## Proposal-mode notes (autonomy + load shedding)

- **Capability change:** Orion may start urgent investigations and reduce its own GPU work without
  asking. It still cannot actuate the AC or any host.
- **Data touched:** read-only telemetry tables; notify outbox; pool lease state.
- **Privacy boundary:** evidence is hardware telemetry only; no chat/memory content in alerts.
- **Proof it worked:** incident event on `orion:hardware:watch:incident`, pool `urgent_preempt` /
  shed events, durable run row with `urgent` seed, notify rows with the incident dedupe key.
- **Dangerous failure:** runaway urgent runs starving Orion (bounded by cap 3 + one incident per
  subject); false AC alarm shedding work (bounded by resolve + manual clear); silent alarm (the
  whole point of Part 5 + report-on-failure).
- **Rollback:** `HARDWARE_WATCH_ENABLED=false`; pool treats `urgent` as background when
  `urgent_max_concurrent: 0`; shed guard off with
  `HARDWARE_WATCH_SHED_ENABLED=false`.

## Proposed schema / API changes

- Shipped (Part 1): `urgent` in pool `Priority` and `ResourceRequirementV1.priority`; recall /
  abort reason `urgent_preempt`; `GpuLeaseReplyV1.reason` now filled on `queued` and `recall`
  replies (existing field); durable-run event `run.preempted`; `defaults.urgent_preempt_grace_sec`,
  `defaults.urgent_max_concurrent`.
- Dropped: gateway error `gpu_pool_preempted` and route `curiosity_urgent` (see Part 1).
- Still to add: `CuriosityUrgentRequestV1`, `CuriosityUrgentSeedV1`, `CuriosityRunBriefV1.urgent`;
  `HardwareWatchIncidentV1`; `CoolingObservedStateV1.stale`; `measurements.gpu_temp_c_max`.
- Channels: `orion:curiosity:urgent:request`, `orion:hardware:watch:incident`.
- HTTP: Hub `POST /curiosity/api/urgent`; hardware-watch `GET /incidents`,
  `POST /incidents/{id}/resolve`, `GET /health`.
- Tables: `hardware_watch_incident`; `home_cooling_sample.stale`.

## Files likely to touch

- `config/gpu_pool.yaml`, `orion/gpu_pool/{scheduler,config}.py`, `orion/schemas/gpu_pool.py`,
  `orion/gpu_pool/tests/`, `services/orion-gpu-pool/{app/guards.py,evals/run_pool_day_eval.py}`
- `services/orion-gpu-pool/app/runtime.py`, `orion/gpu_pool/lease_graph.py` (Part 1; the gateway is
  not touched)
- `orion/schemas/{resource_admission,durable_run,curiosity_urgent}.py`,
  `services/orion-durable-runs/app/{admission_runtime,admitted_graph,pool_hold}.py`,
  `orion/durable_admission/store.py`
- `services/orion-hub/scripts/{curiosity_investigation,curiosity_routes}.py`,
  `orion/curiosity/urgent_prompt.py` (new), Hub `turn_orchestrator.py` (stance bypass),
  `services/orion-durable-runs/app/runner.py` (`read_turn_result` reads `:IncidentReport`),
  Hub curiosity + cabinet templates/static JS
- `services/orion-hardware-watch/` (new), `orion/hardware_watch/rules.py` (new)
- `services/orion-zwave/app/{main,zwave_client}.py`, `orion/schemas/telemetry/home_cooling.py`,
  `services/orion-sql-writer/app/models/home_cooling_sample.py`
- `orion/sensors/gpu_host_stats.sh`, `orion/telemetry/biometrics_pipeline.py`
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `scripts/sql/`, `.env_example`s

## Non-goals

- No AC control (read-only stays). No host power actions.
- No change to chat scheduling or chat priority.
- No new generic alerting framework; the watcher owns three rules.
- No self-escalation of ordinary curiosity runs to urgent by Orion (only the watcher + Juniper).
- No pausing or replaying one-shot request leases (streaming or not); only holds are paused.

## Acceptance checks

1. Scheduler unit tests: urgent pauses background before system; never pauses chat/interactive/
   urgent/request leases; victim resumes ahead of newer same-priority leases; ≤ 3 urgent; rollback
   at `urgent_max_concurrent: 0`. The pool eval (`run_pool_day_eval.py`) replays the same story
   through the real scheduler and lease table. (Shed guard: Plan 5.)
2. ~~Gateway test~~ **Dropped:** request leases are never paused, so there is no gateway replay to
   test (Part 1).
3. Durable-runs test: `urgent_preempt` mid-node replays the node without a failed attempt; a hold
   paused before its step starts keeps its place.
4. Rules replay eval over the real 7-day history reproduces the gate counts (CPU p95 ≈ 1/week
   athena, ≈ 9/week circe) and fires the AC rule on the real 33.1 W stretch.
5. zwave test: failed poll never yields a fresh-looking sample; stale ⇒ no watts + `stale=true`.
5a. Urgent-run tests: urgent seed bypasses stance defer; urgent prompt contains question + evidence
   and none of the self-material sections; `read_turn_result` maps a valid `:IncidentReport` into
   the report, and a missing/empty-evidence one into `no_structured_verdict`; timeout ⇒ INCOMPLETE
   report; no grant within 120 s ⇒ "not investigated" report with the evidence bundle.
6. Live smoke: Hub "Run urgent" → pool shows an `urgent` hold → durable run completes → critical
   notice in Hub and email (email delivery verified, not assumed).
7. Live smoke: simulated AC incident (test hook on hardware-watch) → alert within one tick → urgent
   run → report; pool shed guard visible while open, cleared on resolve.

## Recommended build order

1. Part 5 (AC sensing fix) — smallest, removes a live safety blind spot today.
2. Part 1 (pool urgent + pause/resume + shed guard).
3. Part 2 + 3 (seeded runs, Hub button, report).
4. Part 4 (hardware-watch; GPU temp collection starts collecting immediately so p95 can arm later).
