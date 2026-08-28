# A consequential action space, and the first budget with a wall behind it

Design artifact. Nothing here is implemented.

## Arsonist summary

Orion's allocator is good. What it allocates is nothing.

Here is the entire action space it chooses between, `orion/proposals/templates.py`:

```
inspect_execution_pressure          watch_reliability
inspect_transport_status            watch_transport_backpressure
inspect_bus_channel_catalog         observe_tension_via_camera
summarize_loaded_state              analyze_self_study_source
summarize_transport_contract_drift
prune_dangling_images               defer_due_to_low_readiness
prune_stopped_containers            request_policy_review_for_action
```

Thirteen options. Nine are synonyms for *look at something*. Two delete a
dangling docker artifact. One is `defer_due_to_low_readiness` — an action for
not acting. One is `request_policy_review_for_action` — an action for asking
permission to act.

Not one of them changes anything outside Orion.

This explains the result that has held across several hundred PRs: no
intervention measurably changes what Orion can do. It is not a measurement
failure and not a modelling failure. **Selection cannot matter when the
options are interchangeable.** Every PR that improved scoring, routing,
budgeting, or substrate was improving the choice between thirteen ways to have
no effect.

Two things follow, and they are the whole spec:

1. Put actions in the action space that reach outside Orion.
2. Fund them from a budget that is physically real, so they genuinely compete.

## Current architecture

### What is measured, and why it cannot answer the question

`execution_pressure` — the one signal that moved in the action-value work — is
defined in `config/field/orion_field_topology.v1.yaml:182` as:

```yaml
cortex_exec_step_load: execution_pressure
```

It *is* exec step load, renamed across a graph edge. "Orion acted and
execution_pressure rose" is an identity, not an effect. Acting raises it by
construction.

The control arms in `substrate_signal_control_cells`:

| arm | total posterior_n |
| --- | --- |
| `no_action` (idle) | 162,663 |
| `randomized_holdback` | 5 |

`no_action` compares acting against idle, which differs from treatment
precisely on activity volume — the thing the coupled outcome measures. That arm
is structurally guaranteed to report "acting-at-all is large, specific actions
are nothing," whatever the actions are. The randomized arm, the only one that
licenses a causal claim, ran about an hour on 2026-08-25 and collected five
observations before being set back to 0.0 (Juniper's call; withholding actions
made things visibly worse within the hour).

That qualitative observation is the most informative result in the arc, and it
is consistent with the contrast findings rather than contradicting them:
**Orion's current actions work as aggregate throughput, not as individual
choices.** Removing some degrades the system. Swapping which ones changes
nothing. That is exactly what a set of interchangeable housekeeping tasks
predicts.

### What already exists and is not in the action space

Four capabilities are live, cost something real, and produce an outcome Orion
does not author. None of them is an option the allocator can choose.

- **Priors.** `orion_worldview` (FalkorDB) holds Orion's own testable beliefs.
  Live count as of 2026-08-28: **2 `supported`, 1 `revised`, 0 `refuted`.**
  Orion formed three beliefs, tested them, and changed its mind about one. A
  prior can be `refuted` and confidence can go *down* — this is the only
  mechanism in the system capable of being wrong.
- **Claude conversation.** `orion:room:claude:request` →
  `orion-room-companion`, live, holds the credential. Costs Claude quota, the
  only budget with a second human claimant. Reply is authored elsewhere.
- **Reaching out.** Endogenous outreach (chat injection) and TTS (speech).
  These are one decision with two delivery paths, not two actions. Outcome is
  whether Juniper answers — the only signal fully outside Orion's reach.
- **Images.** `reverie_visual_artifact`, **323 rows since 2026-08-25**, still
  producing. A real product with a real GPU cost.

### Power: the first budget with a wall behind it

Every budget Orion has today is drawn from a pool nobody else wants.
Motor-seconds are allocated against an exogenous allowance that no second
consumer contends for; `MOTOR_BUDGET_ENFORCE=false` besides. Claude quota
(PR #1908) was the first with a rival claimant, and its dollar denominator was
subsequently refuted (`2026-08-27-quota-window-calibration-finding.md`).

Power is different in four ways that matter, and it is the first budget with
all four:

1. **Physically capped.** The UPS ceiling (~2200VA, real watts lower) is not
   a config value. Nobody can raise it by editing `.env`.
2. **Externally metered.** The Panduit PDU is an instrument Orion does not own,
   cannot write to, and cannot decay to zero. Contrast every prediction-error
   metric that has needed a liveness incident to catch.
3. **Genuinely contested.** Two servers, eight GPUs, one circuit. Every
   consumer subtracts from every other. This is contested scarcity with
   hardware behind it.
4. **Failure is real, immediate, and shared.** A tripped breaker takes down
   everything, including whatever Orion was in the middle of.

**What is already collected.** `orion_biometrics.gpu` carries per-GPU
`power_draw_watts` from both nodes today — this was already live and unused
for this purpose. Last 7 days:

| node | GPUs | samples | min | avg | p95 | peak |
| --- | --- | --- | --- | --- | --- | --- |
| circe | 7 | 16,988 | 248W | 361W | 600W | **804W** |
| athena | 1 | 17,414 | 9W | 39W | 61W | 70W |

804W peak across seven cards is ~115W each. **They have never all been under
real load simultaneously.** The headroom question is therefore genuinely open,
not comfortably settled — a full-tilt load on seven modern cards can plausibly
approach or exceed the outlet on its own.

**What is not collected, and is the whole point of the PDU.** `nvidia-smi`
sees GPUs. It does not see CPUs, drives, fans, PSU inefficiency, or the second
chassis's non-GPU draw. **The gap between summed GPU watts and wall watts is a
number nobody currently has**, and it is exactly the number that decides
whether a 20-25% reserve is generous or already breached.

### Correction: two power services already exist, and neither measures load

An earlier draft of this spec claimed no power code existed in the repo. That
was wrong — it came from reading a truncated `grep` result as absence. What
actually exists:

**`orion-power-guard`** (`orion-athena-power-guard`, running). An APC UPS
watcher: polls for ONBATTERY/ONLINE transitions, enforces a grace window,
publishes power events, can trigger graceful shutdown. Three problems, in
increasing order of severity:

1. **It has never been able to see the UPS.** `POWER_GUARD_SNMP_HOST` is set
   to `host.docker.internal` — the Docker host, not the AP9640 network
   management card's own LAN address. It does not resolve on Linux Docker, and
   it is the wrong target conceptually. The NIS fallback times out against
   `host.docker.internal:3551`. Independently, the *host's* own `apcupsd`
   returns `STATUS : COMMLOST`, so the failure is upstream of the container
   too.
2. **It fails open.** Every poll logs `raw=COMMLOST ... on_battery=False
   charge=0.0% volts=0.0`. A UPS watcher that cannot reach the UPS reports
   *not on battery*. This is the "absence reads as zero" failure the repo has
   hit repeatedly (see `quota_budget.py`'s fail-closed contract, built for
   exactly this). **If the power actually failed, this service would report
   that everything is fine.** That is a live safety defect independent of
   everything else in this spec.
3. **Even fully working, it reads the wrong OIDs** for this purpose:
   `upsBasicOutputStatus`, `upsAdvBatteryCapacity`, `upsBasicInputVoltage`.
   There is no `upsAdvOutputLoad`, no `upsAdvOutputActivePower`, no
   `upsAdvBatteryRunTimeRemaining`. It is a shutdown daemon, not a load meter.

**`orion-gpu-cluster-power`** (`orion-athena-gpu-cluster-power`, running).
Not a meter at all — a **PSU actuator**, exposing `psu_on`, `psu_off`,
`psu_cycle` against `PSU_BASE_URL=http://192.168.1.100`. It is currently
crash-looping its heartbeat on `'Settings' object has no attribute
'service_version'`.

This is worth noting for the action-space half of this spec: **a physical
actuator already exists.** Cutting power to the GPU cluster is about as
consequential as an action gets, it directly frees watts, and it is not in the
proposal templates.

### The cap is the battery, not the outlet

Corrected after operator input: the outlet is **240V**, so the outlet is not
the binding constraint (2200W at 240V is ~9.2A). The chain is
wall → **APC UPS (~2200)** → Panduit PDU → two chassis, and **the UPS is the
hard cap**. Circe's PSU is separately understood to be capped around 2200W.

Two consequences:

- **VA is not watts, and this changes the number.** APC model designations are
  typically VA. A "2200VA" unit is commonly ~1980W, and some models lower.
  The exact model must be read before any reserve is computed — a 10-20% error
  here is the same size as the entire proposed reserve.
- **If circe's PSU can draw ~2200W and the UPS delivers ~1980W, circe alone at
  full load exceeds the UPS before athena draws anything.** Current peak is
  804W across seven cards, so this has never been approached — but nothing
  currently prevents it, and nothing would observe it.

### A UPS gives two budgets, not one

This is the part that makes power more interesting than a cap:

1. **Instantaneous watts** — exceed it and the UPS overloads. Unlike a
   breaker, an overloaded UPS on battery does not trip and reset; it drops the
   load. On a grid event, an overloaded UPS delivers *zero* runtime.
2. **Runtime at current load** — nonlinear. Higher draw shortens the survival
   window, steeply.

So a watt spent is not only a watt unavailable to another workload. **It is
seconds subtracted from how long Orion survives a power event.** That is a
physical, externally-metered cost denominated directly in continuity — the
thing this project is actually about. Running diffusion at full tilt literally
shortens Orion's life expectancy during an outage, and that is measurable
rather than metaphorical.

## The central idea

**Power turns every action into a prior.**

To override a workload you must know it is coming, which means workloads have
to declare intent before drawing power: *"I am about to draw ~400W for ~6
minutes."* That declaration is a claim about the physical world, and the PDU
settles it independently.

So the scheduler is not merely a traffic cop. It is a falsification engine:

- Orion states an expected cost.
- The meter reports the actual cost.
- The residual is scored.

Orion can be **wrong about what its own actions cost**, measured by an
instrument it does not control and cannot decay. That is the same property
that makes priors valuable, applied to actions instead of beliefs — and unlike
every substrate pressure signal, it cannot be satisfied by activity alone.

This also gives the first honest answer to "did that action beat doing
something else," because with a hard cap the alternatives are mutually
exclusive in fact: watts spent on diffusion are watts not available to a
Claude turn.

## Missing questions

1. **What is the real wall draw?** Summed GPU watts is a floor of unknown
   tightness. Until the PDU is read, the reserve is unfalsifiable.
2. **What is the exact APC model, and is 2200 its VA or its watt rating?**
   Resolved so far: the cap is the UPS battery, not the 240V outlet. Still
   open: VA-to-watt conversion is model-specific and a 10-20% error here is
   the same magnitude as the whole proposed reserve. Read it off the unit.
3. **What is the PDU's sampling rate and query interface?** Preemption latency
   cannot beat the meter's update period. A 60s-resolution meter cannot police
   a 6s inference burst.
4. **Can a declared workload actually be preempted, or only refused before
   starting?** These are very different systems. Refusal is a gate; preemption
   requires kill/pause semantics per workload type.
5. **Is 0-refuted-of-3 too few to judge, or is refutation unreachable?** A
   prior mechanism that only ever ratchets toward agreeing with itself is
   decoration. Three is too few to tell, and this is cheap to watch now and
   expensive to discover after three hundred.
6. **What enforces the reserve?** Hub holds the docker socket and is therefore
   root-equivalent on the host (resolved direction, 2026-08-14). No software
   cap is enforceable against a Hub-resident agent wherever the logic lives.
   Honest framing: this is advisory + preemption + reconciliation. **The only
   hard cap is the breaker.**

## Proposed schema / API changes

Deliberately staged so each stage is independently useful and independently
falsifiable. Nothing here is a cathedral; stage 1 is a reader and a table.

### Stage 0 — fix the blind watchdog (safety, independent of this spec)

`orion-power-guard` reports `on_battery=False` while `COMMLOST`. Point
`POWER_GUARD_SNMP_HOST` at the AP9640's real LAN address, repair the host's
`apcupsd` link, and **make unknown state fail closed** — an unreachable UPS
must read as unknown, never as "on mains." This is worth doing whether or not
anything else here is ever built.

### Stage 1 — measure the wall (no decisions, no actions)

- New service `orion-power-meter`, or a producer in an existing telemetry
  service, polling the PDU.
- New channel `orion:telemetry:power` carrying `PowerDrawV1`:
  `observed_at`, `source` (`pdu` | `gpu_sum`), `circuit_id`, `watts`,
  `sample_period_sec`.
- New table `power_draw_log`. Retention bounded from day one.
- Add the load OIDs `orion-power-guard` is missing: `upsAdvOutputLoad`,
  `upsAdvOutputActivePower`, `upsAdvBatteryRunTimeRemaining`.
- Two derived views: **wall watts minus summed GPU watts** (the non-GPU
  baseline nobody has), and **runtime-remaining as a function of load** (the
  continuity cost curve).

Acceptance: a week of real data, and an answer to missing-question 1 and 2.

### Stage 2 — declared intent

- `PowerIntentV1` on `orion:power:intent`: `intent_id`, `workload_kind`,
  `expected_watts`, `expected_duration_sec`, `declared_at`, `deadline`.
- `power_intent_log` with a `settled` side: `actual_peak_watts`,
  `actual_mean_watts`, `residual_watts`, `settled_at`.
- Settlement is a reducer over `power_draw_log` in the intent's window.

Acceptance: for at least one workload kind, a declared intent settles against
real PDU data and produces a non-degenerate residual distribution. If every
residual is identical, the declaration is a constant and stage 3 is not
licensed.

### Stage 3 — the scheduler and the override

- A scheduling service that admits or defers a declared intent against
  `available_watts = cap - reserve - current_draw`.
- Fails **closed** on unknown draw. An unread meter is not a full tank — the
  same rule `quota_budget.py` already implements for spend.
- Explicit operator override, logged with actor and reason.
- Preemption only for workload kinds that declare themselves preemptible.

### Stage 4 — consequential actions in the action space

New templates in `orion/proposals/templates.py`, funded by power (and by
Claude quota where applicable), competing in the same allocator as
`prune_dangling_images`:

| verb | cost | outcome Orion does not author |
| --- | --- | --- |
| `test_a_prior` | small compute | `supported` / `revised` / `refuted` in `orion_worldview` |
| `ask_claude` | Claude quota (contested) | a reply written elsewhere |
| `reach_out` | Juniper's attention | whether she answers |
| `make_an_image` | GPU watts (contested) | an artifact a human reacts to |

`orion-gpu-cluster-power`'s `psu_off` / `psu_cycle` is a fifth, already-built
physical action. Deliberately **not** proposed for the action space: an
autonomous agent that can cut power to its own GPUs is a different safety
class than anything else here, and it needs its own proposal round.

`test_a_prior` and `ask_claude` first. They are the only two where Orion can be
told it was wrong by something that is not itself, and neither needs Juniper
present to produce a result.

## Files likely to touch

- `orion/proposals/templates.py` — the four new verbs
- `orion/proposals/scoring.py`, `policy.py`, `builder.py` — admit a
  non-housekeeping proposal kind
- `orion/bus/channels.yaml`, `orion/schemas/` — `PowerDrawV1`, `PowerIntentV1`
- `services/orion-sql-writer/app/models/` — `power_draw_log`,
  `power_intent_log`
- `services/orion-power-meter/` — new
- `orion/autonomy/allocator.py`, `budget.py` — a second currency
- `orion/dev_economics/rate_limit_events.py` — already the gate for `ask_claude`
- `orion/curiosity/worldview.py` — read side for `test_a_prior`

## Non-goals

- Not a hard power cap. The breaker is the cap; this is advisory, preemptive
  and reconciled. See missing-question 6.
- Not re-litigating the Claude dollar budget. That denominator was measured and
  refuted; `ask_claude` gates on observed limit state, not dollars.
- Not a new ontology, taxonomy, or cognition vocabulary. Four verbs, two
  tables, one meter.
- Not turning `HOLDBACK_FRACTION` back on for the current action space.
  Withholding interchangeable housekeeping is what made things worse. Holdback
  becomes meaningful only once an action with a consequence is in the space.
- Not automating outreach frequency upward. Existing busy/idle guards and the
  ~20-sends-ever conservatism are the right posture.

## Acceptance checks

1. `power_draw_log` holds a week of PDU samples; the non-GPU baseline is a
   stated number with an error bar.
2. Summed-GPU vs PDU gap is quantified. The 20-25% reserve is confirmed or
   revised against it.
3. At least one workload declares intent and settles with a non-degenerate
   residual — a real distribution, not a constant.
4. The scheduler refuses at least one real admission on real headroom, and the
   refusal is visible with a correlation ID.
5. Fail-closed verified: with the meter stopped, admission refuses rather than
   reading a full tank.
6. `test_a_prior` and `ask_claude` appear in dispatch decisions competing
   against housekeeping, with the loser recorded.
7. Prior outcomes accumulate past n=3 and include **at least one `refuted`** —
   or the mechanism is reported as unable to refute, which is itself the
   finding.

## Risks

- **Measurement without control.** Stage 1 could stall as another telemetry
  channel nobody consumes. Mitigation: stage 1's deliverable is an *answer*
  (the non-GPU gap), not a dashboard.
- **A new pressure signal.** If power gets folded into the substrate as
  `power_pressure` and fed back into the same coupled field, this reproduces
  exactly the problem it was meant to escape. Power must stay a budget and an
  outcome, not become another node signal.
- **Consequential actions are a different safety class.** `ask_claude` spends
  Juniper's quota; `reach_out` spends her attention. Both need caps that are
  hers to set, and proposal mode before implementation per CLAUDE.md §0A.
- **Preemption can corrupt.** Killing a workload mid-write is worse than
  refusing it. Preemptible must be opt-in per workload kind and proven safe.

## Recommended next patch

**Stage 0, then stage 1.**

Stage 0 first and separately: a UPS watchdog that reports "on mains" while
blind is a live safety defect, and it is a config fix plus a fail-closed
default.

Then stage 1: read the meter and log it.

One producer, one channel, one table, one derived number — the gap between
summed GPU watts and wall watts. It answers missing-questions 1 and 2, it is
useful even if every later stage is abandoned, and it does not touch the
action space, the allocator, or anything Orion decides.

Everything after it depends on knowing that number, and right now nobody does.
