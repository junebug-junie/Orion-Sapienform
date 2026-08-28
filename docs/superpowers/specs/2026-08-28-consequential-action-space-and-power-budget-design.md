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

### Correction: the meter already exists, and it already works — for the wrong box

An earlier draft of this spec claimed no power code existed. That was wrong; it
came from reading a truncated `grep` as absence. The truth is better and more
specific.

`orion/telemetry/biometrics_pipeline.py` already reads **per-outlet PDU power
at the wall**, publishes it as `pdu_watts`, and falls back to it for
`chassis_watts` on a node with no BMC. It was built carefully: the two meters
are deliberately never summed (they measure the same watts), iLO's 60s cadence
versus the PDU's instantaneous read is documented, and the pair was
cross-validated on atlas at 291W on both instruments at the same instant.

`orion_biometrics_summary.measurements`, last 7 days:

| node | rows | has `chassis_watts` | has `pdu_watts` | avg | peak |
| --- | --- | --- | --- | --- | --- |
| athena | 17,455 | 17,429 | 4,026 | 350W | 580W |
| **circe** | 17,028 | **0** | **0** | — | — |

**athena is metered at the wall. Circe is not measured at all.** That is
exactly backwards: circe is the box with seven GPUs and a ~2200W PSU, and it is
invisible. Athena, which cannot threaten the budget, is the one we can see.

Circe also has no BMC, so the PDU is the *only* chassis measurement it could
ever have. The pipeline already knows this and handles it. The gap is that
circe's Panduit outlets are not mapped through.

**So stage 1 is not "build a PDU reader."** The reader exists and is proven on
another node. Stage 1 is: point it at circe's outlets.

### ROOT CAUSE FOUND, and the numbers are no longer estimates

**Why circe reads zero.** `orion-biometrics` logs `pdu_poll_failed error=5
second timeout exceeded on UDP transport` every cycle, for both nodes. The PDU
is healthy and pingable. The cause is source-address selection:

```
$ ip route get 192.168.1.39
192.168.1.39 dev eno5 src 192.168.1.43

eno1  192.168.1.42/24     <- the address in the PDU's SNMP Manager whitelist
eno5  192.168.1.43/24     <- the address the route actually uses
```

Athena has two NICs. The route to the PDU leaves via `eno5` as `.43`, which is
not whitelisted, so the PDU silently drops the requests. Verified with a
counterfactual, live:

```
snmpget ... 192.168.1.39 <outlet-1-oid>
  -> Timeout: No Response from 192.168.1.39.

snmpget ... --clientaddr=192.168.1.42 192.168.1.39 <outlet-1-oid>
  -> INTEGER: 265
```

This is a recurrence in kind of the 2026-08-21 outage recorded in
`services/orion-biometrics/.env` — that time a stale DHCP lease was whitelisted
instead of athena's real address. Same failure mode, new cause: a second
interface now wins the route. The `.env` comment documents the fix but not the
fragility, which is why it came back.

Note the container is on the `app-net` bridge, so its packets are SNAT'd by the
host to the route-selected source. **Binding the source inside the container
cannot fix this** — the fix has to be on the host route or in the PDU
whitelist.

### Measured fleet power, 2026-08-28

Read directly off the PDU while diagnosing:

| | outlets | wall watts |
| --- | --- | --- |
| circe | 1, 7, 13 | 276 + 415 + 296 = **987W** |
| athena | 34, 35 | 246 + 257 = **503W** |
| **fleet** | | **1490W** |

At that instant circe's GPUs drew **337W**, so:

| node | non-GPU baseline | source |
| --- | --- | --- |
| athena | **311W** (201-493W range) | 9,061 paired minutes |
| **circe** | **~650W** | 987W wall − 337W GPU, measured |

Circe's non-GPU draw is **more than double athena's**, and materially above the
300-500W this spec previously guessed by analogy. Borrowing athena's constant
would have understated the fleet by ~340W — the exact borrowed-constant error
this repo has been burned by before.

### The headroom is thinner than assumed

```
circe non-GPU (measured)          ~650 W
circe GPU peak, 7d (measured)      804 W
                                 -------
circe at observed peak            ~1454 W
athena peak, 7d (measured)          580 W
                                 -------
fleet at coincident peaks         ~2034 W
UPS deliverable (~2200VA)         ~1980 W
```

**Coincident observed peaks already exceed the battery**, and current draw sits
at 1490W — roughly **75% of deliverable while the GPUs are near idle** (337W of
a seven-card box).

Two honest caveats. The two nodes' peaks are not known to have occurred
simultaneously, so 2034W is a worst-case composition of separately observed
maxima, not a measured instant. And the ~1980W figure still depends on the APC
model's VA-to-watt conversion, which has not been read off the unit.

Neither caveat is reassuring, because the GPU peak in that sum was recorded at
low utilisation. Seven cards under genuine load would add far more than the
~130W of margin.

### RESOLVED 2026-08-28: fleet wall power is live

Juniper whitelisted `192.168.1.43` in the PDU's SNMP Manager list. The poll
recovered on its own within two cycles — no restart, no code change. Confirmed
from the live `orion:biometrics:cluster` payload:

```
pdu_watts             1526        <- true fleet wall draw
chassis_watts         1415-1457
gpu_watts_total        448
measurements_proxied  {"circe": ["chassis_watts", "pdu_watts"]}
measurements_missing  {"fan_pct_max": ["circe"]}
nodes_absent          ["atlas"]
```

Circe has left `measurements_missing` for power and is correctly labelled as a
proxied reading rather than a self-report — the provenance guarantee in
`main.py:431` held.

**The fleet number nobody had:**

| quantity | watts |
| --- | --- |
| fleet wall (PDU) | **1526** |
| fleet GPU | 448 |
| **fleet non-GPU baseline** | **1078** |
| UPS deliverable (~2200VA) | ~1980 |
| **margin** | **~454** |

**The fleet sits at ~77% of the battery with eight GPUs drawing 448W.** The
non-GPU baseline — 1078W, more than twice the entire GPU draw — is the dominant
term and is essentially fixed cost. The remaining ~454W is what all GPU
headroom must fit inside, and eight cards under genuine load would exceed it
several times over.

This retires the earlier estimate-based section: circe's non-GPU draw was
guessed at 300-500W by analogy to athena, then measured at ~650W, and the fleet
baseline is 1078W. Every step of that borrowing was wrong in the same
direction.

**Stage 1 of this spec is therefore already complete**, achieved by a whitelist
entry rather than a build. What remains is a bounded retention table for
history, and then stage 2.

### The cap is the battery, not the outlet

Corrected after operator input. The outlet is **240V**, so the outlet is not
binding (2200W at 240V is ~9.2A). The chain is wall → **APC UPS** → Panduit
PDU → two chassis, and the UPS is the hard cap. Circe's PSU is separately
understood to be capped around 2200W — meaning **circe alone could exceed the
UPS before athena draws anything**, though at 804W observed peak it has never
come close.

APC model designations are typically VA, and a 2200VA unit is commonly ~1980W.
That 10-20% gap is the same size as the entire proposed reserve, so the model
has to be read off the unit rather than inferred.

**The APC's own network card is dead**, so the UPS cannot be polled directly —
no load, no runtime, no on-battery state from that path. **The Panduit PDU over
SNMP is the meter.** `orion-power-guard` is not working and is not a dependency
of anything here; it should be stopped rather than left running, because it
reports `on_battery=False` while `COMMLOST`, which is worse than reporting
nothing. `orion-gpu-cluster-power` is a stale PSU switch and is out of scope.

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

A second framing — that load also shortens how long Orion survives a grid
event, making the cost denominated in continuity — is deliberately parked in
the appendix. It is currently unmeasurable anyway: reading battery runtime
requires the APC's network card, which is dead.

## Missing questions

1. **What is circe's non-GPU draw?** This is now *the* question. Athena's is
   measured at 311W avg (201-493W). Circe's is unmeasured, plausibly higher,
   and it alone decides whether the fleet sits at 86% or over 100% of the
   battery.
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

### Stage 0 — stop the service that lies (safety, independent of this spec)

`orion-power-guard` reports `on_battery=False` while `COMMLOST`, and the APC
card it would need is dead, so it cannot be repaired in place. Stop it. A
watchdog that reports "on mains" while blind is worse than no watchdog. If
on-battery state is wanted later, take it from the Panduit, not the APC.

### Stage 1 — measure the wall (no decisions, no actions)

- **Map circe's Panduit outlets into the existing `pdu` sample path.** No new
  reader: `biometrics_pipeline` already consumes `pdu_watts` correctly and is
  proven on athena.
- New channel `orion:telemetry:power` carrying `PowerDrawV1`:
  `observed_at`, `source` (`pdu` | `gpu_sum`), `circuit_id`, `watts`,
  `sample_period_sec`.
- New table `power_draw_log`. Retention bounded from day one.
- One derived view: **wall watts minus summed GPU watts, per node** — already
  computable for athena (311W avg, 201-493W range, n=9,061 paired minutes),
  and the open question for circe.
- A fleet total across both chassis against the UPS ceiling.

Acceptance: a week of circe wall data, a stated non-GPU baseline for circe
with an error bar, and a fleet total that can be compared to the UPS rating.

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

(`orion-gpu-cluster-power` is a stale PSU switch and is not proposed for the
action space.)

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

---

## Appendix A — power as a continuity cost

Parked rather than dropped, because it is currently unmeasurable: reading
battery runtime needs the APC's network management card, which is dead.

A UPS is two budgets, not one:

1. **Instantaneous watts.** Exceed it and the UPS overloads. Unlike a breaker
   it does not trip and reset — it drops the load. Overloaded at the moment of
   a grid event, it delivers *zero* runtime.
2. **Runtime at current load.** Nonlinear: higher draw shortens the survival
   window steeply.

If runtime were readable, a watt spent would not only be a watt denied to
another workload — it would be **seconds subtracted from how long Orion
survives a power event**. That is a physical, externally metered cost
denominated directly in continuity, which is closer to what this project is
actually about than any substrate pressure signal.

It would also give the scheduler a second axis: a workload could be admissible
on instantaneous headroom and still be refused for cutting the survival window
below a floor.

To revive this: repair or replace the APC network card, then read
`upsAdvOutputLoad`, `upsAdvOutputActivePower`, and
`upsAdvBatteryRunTimeRemaining`. Until then it stays an appendix.
