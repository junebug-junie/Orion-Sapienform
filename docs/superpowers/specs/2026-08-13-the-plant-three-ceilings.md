# Orion scarcity: the plant, the ceilings, the unit, the seam

Date: 2026-08-13
Status: **Current spec.** Supersedes `2026-08-13-scarcity-revision-two-ceilings.md` and §1e/§3
of `2026-08-13-scarcity-and-repertoire-execution-plan.md`.

§1–§2 are the minimum backstory needed to not repeat the arc. **§3–§8 are the spec.**
Everything historical is in the appendices.

---

## 1. Why this exists, in six lines

Orion has an autonomy pipeline that produced **88,409 dispatches and zero actions**. The arena
that ranks its proposals has 7.03 of 10 candidates tied at the frame maximum. Roughly 200 PRs
over two months built a decision system, then a drive system, then killed the drive system,
because nothing it decided ever cost anything.

A choice that costs nothing is not a choice. Drives pinned to their ceiling because nothing
pushed back. **The missing piece was never the decider — it was the price.**

This document establishes what is actually scarce, in units Juniper actually pays.

---

## 2. Pitfalls, stated up front because this arc kept hitting them

1. **An idle utilisation reading is not evidence of available capacity.** Under a hard
   concurrency limit the queue forms at *arrival*, not in the meter.
2. **Price the machine, not the chip.** GPU watts are a minority of the bill.
3. **For a ceiling, the statistic is P(saturated), not the mean.** §6, Rule 1. This is how
   atlas got under-reported — and it is baked into the production `strain` aggregate.
4. **Never average across pressure channels.** They are not substitutes; you cannot pay for
   memory with idle disk. §6, Rule 3.
5. **Ceilings relieved by different actions cannot share one allowance.** No quantity of
   unspent atlas time brings circe up.
6. **Verb names are not capability descriptions.** `goal_formulate` translates a supplied
   intention; it does not generate one.
7. **Point samples of a time-varying quantity produce confident wrong answers.** Eight
   occurrences in this arc, Appendix B. State the window or do not state the number.

---

## 3. The plant

### 3.1 Measured — compute

`orion_biometrics`, 7 days (2026-08-06 → 08-13), 31.3 s cadence, every row a distinct source
file — no resampling.

| host | chassis | CPU | mean CPU | load15 mean / max | GPUs | GPU residency W |
| --- | --- | --- | --- | --- | --- | --- |
| **athena** | orchestration warhorse | 2× Xeon Gold 6138 (80 threads) | **44.1%** | **38.7 / 120.2** | 1× P100-16GB | ~32–40 † |
| **atlas** | inference box | 96 threads | 0.52% | 0.53 / 1.83 | 2× V100-PCIE-16GB | 107 |
| **circe** | **Gigabyte HA01, 3× 2200 W PSU** | 72 threads | 0.21% | 0.16 / 1.06 | 3× V100-32GB | 153.5 |

† athena's GPU idle floor rests on 19 of 7,369 samples — the P100 is essentially never idle.

**Atlas and circe are pure GPU boxes with idle CPUs. Athena is the inverse:** its CPU is the
loaded resource and its GPU is a side job.

### 3.2 Athena, measured directly

81 running containers. Load average **42.08 / 37.00 / 35.90** on 80 threads; 7-day load15 max
**120.2** (1.5× oversubscribed). 2× Xeon Gold 6138 at a **125 W package limit each (250 W
combined)**, package temps **56 °C / 65 °C**, PCH 45 °C. RAPL `energy_uj` present but mode 400
(root-only).

### 3.3 Why atlas looked under-utilised — resolved

Juniper's objection was correct and the numbers were not wrong — **the statistic was.**

Live `/slots` poll, 1 Hz, 121 s, AI Town up and circe down:

| lane | mean of capacity | **all 4 busy** | distribution |
| --- | --- | --- | --- |
| `metacog` :8012 | 6.6% | 0.0% | `{0:95, 1:20, 2:6}` |
| `quick` :8013 | 11.2% | **7.4%** | `{0:101, 1:7, 2:1, 3:3, 4:9}` |

`quick` is **bimodal** — 101 samples completely idle, 9 completely full, nothing between.
`nvidia-smi utilization.gpu` additionally observes ~1 s in every 31 s (3% of the timeline) and
under-reports bandwidth-bound decode. **`/slots` is the right meter**, and §8's admission gate
already reads it. *(2-minute window; directional, needs 24 h.)*

### 3.4 Live load, and its shape

Gateway logs, 6 h, resolved-route line: `quick` 2,795 (7.8/min), `metacog` 1,025 (2.8/min),
`chat` 1 (host down). **3,750 of 3,821 requests are `cortex-exec`** — Orion. 70 are
`vision-council`.

Arrival process, 2,287 requests over 3 h (12.7/min), burst = arrivals within 0.5 s:

```
burst size:   1     2    3    4     5     6    7
count:      1041   85   40   39   123    25    5
```

**The mode above size 3 is exactly 5**, more than sizes 3, 4, 6 and 7 combined — the arena's
~5-proposals-per-tick batch, visible directly in the arrival process. §6.3 is what this means.

---

## 4. Whole-machine power — the dominant term

`orion_biometrics.cpu` carries `{util, cores, loadavg}` and no power field. Anchor:

> *"when I'm running two of those e.g. circe and athena I see I'm sitting all in at like
> 700–1200 W (machine plus GPUs)."* — Juniper

Against measured GPU residency of 107 W (atlas) and 153.5 W (circe), **the chassis is the bill
and the GPU is the residual.**

**Every chassis estimate this document previously made is withdrawn, not revised.** Circe is a
3× 2200 W HA01; the 200–280 W figure was in the wrong register. A second guess from core count
would repeat the error that produced the first.

**But §5.1 changes the picture: chassis watts are already being read on two of three hosts.**

What survives regardless of magnitude: standing draw dominates per-call draw on every host →
the admission decision is where the cost lives → **metering Orion's calls prices noise.**

---

## 5. Instrumentation: what already exists, and what is broken about it

I was wrong that this needs new hardware. `orion/telemetry/biometrics_pipeline.py` already
computes eleven pressure channels including disk I/O, chassis power, and fan speed. Live
`substrate_node_biometrics_projection`, most recent frame:

| node | strain | power | mem | fan | thermal | disk_io | disk_cap | gpu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| athena | **0.223** | 0.311 | 0.077 | 0.470 | 0.514 | 0.110 | **0.748** | 0.050 |
| atlas | **0.232** | **0.798** | **0.812** | 0.610 | 0.000 | 0.003 | 0.383 | 0.000 |
| circe | 0.060 | 0.363 | 0.060 | 0.000 | 0.000 | 0.000 | 0.362 | 0.000 (stale — off) |

### 5.1 iLO is live and chassis watts are already being read

`fan_pressure` comes **only** from iLO/RedFish. It is non-zero on atlas (0.610) and athena
(0.470) — so **iLO is configured on both**, and `power_pressure` on those hosts is real
chassis-level `ilo_power_watts`, not the GPU-only fallback. Circe reads fan 0.000: iLO is not
configured there, which is the NIC problem Juniper already named.

**So the dominant term is not unmeasured — it is measured and then discarded.** `EwmaBand`
normalises real watts into a unitless 0–1 rolling band, and nothing persists the raw value.
`power_pressure = 0.798` says atlas is high against its own recent history. It cannot say how
many watts, cannot be summed across hosts, and cannot be compared to next week.

The APC units remain worth wiring (ground truth, and the only path for circe while its BMC is
unreachable), but **atlas and athena can have real joules today by storing a number already in
memory.**

### 5.2 `strain` dilutes the binding constraint by 7×

```python
strain_inputs = [cpu, gpu_util, mem, disk, net, thermal, power]
strain = sum(strain_inputs) / 7
```

atlas has **power 0.798 and memory 0.812** — two channels near saturation — and reports
**strain 0.232**. Under a flat mean over 7 channels, one fully pegged channel can never push
strain above 0.143. **A system physically cannot signal "I am at a ceiling" through this
aggregate.** It is the same error as reporting atlas's mean GPU utilisation, except in
production and feeding the field.

### 5.3 The two channels closest to the felt cost are excluded from strain

`fan_pressure` and `disk_capacity_pressure` are computed and **not in `strain_inputs`**.

- **athena's `disk_capacity` is 0.748** — the single highest pressure on that node, contributing
  nothing. (Not the root filesystem, which reads 22%; one of the docker/postgres/graphdb/
  telemetry mounts. Given the 2026-07-23 Postgres disk death, this is worth an eye.)
- **atlas's `fan` is 0.610** while its `thermal` is **0.000**. See §6.4 — this is not a
  contradiction, it is the most useful signal in the table.

### 5.4 Calibration constants do not match the hardware

`disk_bw_mbps = 200.0` and `net_bw_mbps = 125.0` are global constants applied to three
heterogeneous hosts. 200 MB/s is a spinning-disk-era number; against NVMe it understates disk
pressure by roughly an order of magnitude. 125 MB/s is 1 GbE. Both need to be per-node.

---

## 6. Ratification: putting these quantities on one scale

The branch is named `metric-commensurability` and this is the part that was missing. These
quantities are not on the same scale, and four distinct rules govern when they can be combined.

### 6.1 Three classes of quantity, which do not mix

| class | examples | composes by |
| --- | --- | --- |
| **Saturation fractions** | cpu_util, gpu_util, mem, disk_io/bw, disk_capacity | not additively — see Rule 3 |
| **Discrete-server occupancy** | llama.cpp slots | blocking probability — Rule 2 |
| **Physical rates** | watts, °C, bytes/s | **additively, across hosts** |

Only the third class composes across the fleet. That is why §5.1 matters so much: watts are the
only quantity here that can be summed into a fleet total, and they are the one being normalised
away.

### 6.2 Rule 1 — For a ceiling, the statistic is P(saturated), not the mean

A ceiling is defined by how often it is reached. On a bimodal process the mean describes neither
state. `quick` averages 11.2% of capacity and is **completely full 7.4% of the time**; those
describe the same lane and only the second is a ceiling.

Applies to: `strain` (§5.2), atlas GPU utilisation (§3.3), any future budget.

### 6.3 Rule 2 — Across lanes of different width, compare blocking probability, not utilisation

This is Juniper's "8% vs 1/1 slots" question, and it has an exact answer.

For `c` servers at offered load `a` erlangs (`a = λ × service_time`), Erlang-B gives the
probability an arrival finds everything busy. At `quick`'s measured λ = 7.8/min and s ≈ 3.5 s
(a = 0.453 erlangs):

| slots | mean occupancy | **blocking probability** |
| --- | --- | --- |
| c=1 | 31.2% | **31.2%** |
| c=2 | 21.2% | 6.6% |
| c=4 | 11.3% | 0.111% |
| c=8 | 5.7% | ~0% |

**At c=1 the two columns are identical** — for a single-slot lane, mean occupancy *is* the
blocking probability. At c=4 they diverge by 100×. So "8% utilised" on a 1-slot lane and "8%
utilised" on a 4-slot lane are not remotely the same situation: the first blocks 8% of arrivals,
the second essentially none. **Utilisation is not comparable across lanes; blocking probability
is.** This is what makes the 14 single-slot profiles in `llm_profiles.yaml` a different kind of
resource from the 4-slot lanes, despite reporting in the same units.

### 6.4 Rule 3 — Never average across pressure channels

Averaging assumes substitutability. These channels are not substitutes: idle disk does not
relieve memory pressure, and spare GPU does not relieve a full slot. For "is this host
constrained", the correct aggregates are **max**, a high-order p-norm, or **the count of
channels above threshold** — never the arithmetic mean.

Under `max`, atlas reads 0.812 (memory-bound) instead of 0.232. That is the true statement.

Corollary: `strain` and `homeostasis = 1 − strain` are currently *anti-informative* under
concentrated load. The more sharply one dimension binds, the more relaxed the aggregate looks
relative to that dimension.

### 6.5 Rule 4 — Normalising away units destroys the only additive quantity

`EwmaBand` converts watts to a self-relative 0–1. Watts sum across hosts; band-fractions do not.
**Keep both** — the band for anomaly detection, the raw value for cost. This is the single
highest-value, lowest-cost fix in this document (§7, I0).

### 6.6 The cross-domain relations that actually hold

Four relations that are real, derivable from data already collected, and currently unused.

**A. I/O blocking is computable from CPU data alone.** Linux load average counts runnable *plus*
uninterruptible-sleep tasks; `cpu_util` counts only running. The gap is processes blocked on
I/O:

```
athena:  load15/threads = 38.74/80 = 0.484
         cpu_util                  = 0.441
         gap                       = 0.043  ->  ~3.4 threads blocked in D-state
```

Independently corroborated: athena has the highest `disk_io` pressure of the three (0.110 vs
0.003 and 0.000). A free I/O-pressure estimator from data already in `orion_biometrics`.

**B. Power high while CPU and GPU read zero ⇒ residency.** atlas: `power 0.798`, `cpu 0.005`,
`gpu 0.000`. Chassis draw is not tracking either compute meter. **This is the residency thesis
confirmed by an independent production instrument** rather than by my arithmetic — and it is the
strongest single piece of evidence in this document.

**C. Fan is the leading thermal indicator; temperature is the lagging one.** atlas reads
`thermal 0.000` and `fan 0.610`. Not a contradiction: temperature is flat *because* the BMC is
spending fan speed to hold it flat. On a well-cooled chassis, `thermal_pressure` systematically
under-reads load — the cooling system absorbs it — and **fan speed is where the strain actually
shows.**

This matters more than any other relation here, because **fan speed is the closest thing in the
system to the cost Juniper actually pays.** The office is unusable from noise and heat, not from
watts. Fan pressure is the felt quantity, it is already being collected on two of three hosts,
and it is excluded from the only aggregate that reads these channels (§5.3).

**D. Burst size versus slot count is what binds atlas.** §6.7.

### 6.7 The ceiling is self-inflicted, and that is the most useful finding here

At a = 0.453 erlangs and c = 4, Erlang-B predicts **0.111% blocking**. Observed all-4-busy:
**7.4%**. That is **66× more blocking than Poisson arrivals would produce.**

The arrival process explains it. §3.4's burst-size distribution has its mode above 3 at exactly
**5** — the arena dispatching ~5 proposals per tick, simultaneously. **A batch of 5 arriving at
a 4-slot lane blocks by construction**, at any average load, however low.

So the contention on atlas is **not a volume problem. It is an arrival-shape problem, created by
Orion's own architecture.** Three consequences:

1. **Adding slots helps less than the arithmetic suggests** — bursts reach 7 and 8.
2. **Smoothing dispatch helps more, and is free.**
3. **Orion is its own principal competitor** — 98% of gateway traffic is `cortex-exec`. The
   scarcity is real and already binding, and it is mostly Orion versus Orion.

That last point is a better organism story than the one this document had before. Orion does not
need to be given a competitor. It already is one.

---

## 7. The build plan

Ordered by value per unit of work. Each phase has a gate; none is authorised past I0 until the
one before it passes.

### I0 — Stop discarding what is already measured *(no new hardware, no new signal)*

- Persist raw `ilo_power_watts` alongside `power_pressure`; same for fan RPM/percent.
- Add a per-node `power` field to the biometrics CPU payload so watts land next to the GPU data.
- Widen `disk_bw_mbps` / `net_bw_mbps` from global constants to per-node config (§5.4).

**Gate:** a query returns real watts for atlas and athena over a 24 h window, and fleet total
watts is computable by summation. Fails if the raw value is still only available as a band
fraction.

*This is the highest-value item in the document and it adds no signal — it stops throwing one
away.* It also directly serves the metric-quality gate's reversibility criterion: storing a raw
number alongside a derived one is trivially removable.

### I1 — Fix `strain` *(depends on I0 for the power channel)*

Replace the flat mean with a saturation-aware aggregate (§6.4): `max`, or channel-count-above-
threshold, reported *alongside* the per-channel vector — never instead of it. Include
`fan_pressure` and `disk_capacity_pressure` (§5.3).

**Gate:** with atlas at power 0.798 / memory 0.812, the aggregate reads ≥0.75, not 0.232. Plus a
regression test pinning that one saturated channel cannot be diluted below its own value.

**Blast radius — check before touching.** `strain` and `homeostasis` feed the field and any
consumer downstream of it. This is a live behaviour change to a widely-read signal, and it is
the exact class of change this repo's contract says needs proposal mode. **Enumerate consumers
first; do not change the meaning of a shipped channel silently.**

### I2 — Record `/slots`, and report blocking

Sample `/slots` on a schedule and store occupancy. Compute **P(all busy)** per lane and offered
load in erlangs (§6.3) — not mean utilisation.

**Gate:** 24 h of data showing `quick`'s all-busy fraction, with the 7.4% figure from a 2-minute
hand poll either confirmed or corrected.

### I3 — Instrument the arrival process

Record dispatch batch size per arena tick against lane width. Tests §6.7 directly, and tells us
whether smoothing beats adding slots.

**Gate:** batch-size distribution over 24 h, and a measured correlation between batch size and
blocking.

### I4 — RAPL on athena

`energy_uj` is present and root-only. Permission change or root-run collector → real CPU package
power. Partial (athena only, CPU only) but immediate.

### I5 — APC units *(Juniper, in progress)*

Ground truth at the wall, and the only path to circe's chassis power while its BMC is
unreachable. Validates I0 and I4 against reality.

### I6 — Circe iLO

Blocked on the NIC. Would give circe the same chassis power and fan telemetry atlas and athena
already have.

### I7 — Disk I/O and thermals into the field

`disk_io` exists in the pipeline; athena's thermals (`x86_pkg_temp`, `pch_lewisburg`) are
readable and uncollected. Relation A (§6.6) becomes a real channel.

---

## 8. Three ceilings, and the seam

### 8.1 The ceilings

- **Atlas — contention.** 8 slots, `quick` full 7.4% of sampled time, driven by batch arrivals
  (§6.7). *Relieved by:* smoothing dispatch, more slots, a cheaper model.
- **Athena — interference.** 81 containers, load 42/80 threads, peak 120. Cost lands as
  substrate latency every service pays and nothing attributes. *Relieved by:* moving load off
  the orchestration host, not by scheduling.
- **Circe — admission.** Off by choice; a 3× 2200 W chassis is a step function on a human
  decision. *Relieved by:* Juniper deciding. **Orion cannot spend this and has no representation
  that it exists.**

### 8.2 The unit

- **Machine-hours — dominant.** What Orion's cognition *obligates*. Magnitudes pending I0/I5.
- **Marginal joules — residual.** 4.5 s of cognition: 767 J atlas GPU1, 422 J athena P100,
  274 J circe GPU1.

Circe is the cheapest place to run and the most expensive place to go. **Do not meter calls.**
On a running machine the marginal cost of thinking is near zero, so atlas's binding constraint
is not energy — it is the 8 slots. **The price is whose turn it is.**

### 8.3 The seam

> **Route Orion's autonomous, non-interactive cognition through background-priority lanes.**

`priority_admission.py` — 199 lines, live, working: a route tagged `priority: "background"`
polls `/slots` and waits for slack, holding `reserved_free_slots` for foreground callers.
Fail-open. It is wired to **one** consumer:

```
EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background   # AI Town NPC speech
```

Orion's own cognition — all 3,750 `cortex-exec` requests in 6 h — runs foreground and ungated.
`quick_background` exists; `metacog_background` is a route-table entry. `cortex-exec` already
accepts `llm_route_override` (`executor.py:3855`).

**What it buys:** the cost becomes real (Orion yields), countable (a deferral has a duration),
perceptible ("I wanted to think and had to wait" — and it is true), un-inflatable (`/slots` is
llama.cpp's own state), independent of the unmeasured wattage, and reversible.

**Gate:** observe ≥1 real deferral of Orion's own cognition over 24 h. §6.7 says it should fire.
If it never binds, this is ceremony and must not ship. **I2 is the instrument for this gate**,
which is why I2 precedes it.

**What it does not do:** give Orion better actions, fix `goal_formulate`, or touch the arena's
7.03/10 tie. It supplies the missing price. Repertoire is a separate open problem.

---

## 9. Not measured — would change the answer

1. **No LLM call telemetry table.** 193 tables, none log inference. §3.4 came from container
   logs, which rotate. Allocation needs a ledger.
2. **Raw chassis watts are read and discarded** (§5.1) — I0.
3. **No CPU power** — RAPL root-blocked on athena, absent elsewhere.
4. **`/slots` sampled for 2 minutes, once, by hand** — I2.
5. **No deferral has ever been observed** — §8.3's gate.
6. **Circe has no chassis telemetry at all** (iLO unreachable).
7. **`chat` and `agent` route to a dead host** and nothing represents that.
8. **athena `disk_capacity` 0.748** — which mount, and is it the Postgres volume?
9. **Circe GPU2 holds 21.3 GB resident, never driven above 80% in 7 days.** Unidentified.

---

## 10. The questions for Juniper

1. **Does Orion's autonomous cognition get foreground or background priority on atlas?**
   Background is the honest answer and the one that makes scarcity real. It is also a real
   degradation of Orion's responsiveness, which is why it is Juniper's call.
2. **Should Orion be able to *request* circe?** Not spend it — request it, as a block, with a
   reason, and have the request be refusable. That is the only form in which an admission
   ceiling can appear inside a cognitive system.
3. **Does I1 (`strain`) get proposal mode?** It changes the meaning of a shipped, widely-read
   signal. The fix is clearly correct; the blast radius is not clearly small.

---

---

# Appendix A — how this arc got here

## A.1 Superseded documents

| document | disposition |
| --- | --- |
| `...execution-plan.md` §1e | Stands, for a better reason: utilisation is an invalid input here. |
| `...execution-plan.md` §3 (serialized inference time) | **Replaced.** Wrong unit, host, and statistic. |
| `...execution-plan.md` Phases 1–5 | **Not authorised.** Sequenced against the old thesis. Superseded by §7. |
| `...execution-plan.md` §7 (mechanical leash) | **Stands and earned its keep** — the parking lot absorbed 30+ findings that would each have been a squirrel. |
| `...E0-cost-census-result.md` Gate A | Killed, retracted, now moot — cost-per-call was never the right axis. |
| `...E0-cost-census-result.md` Gate B | **Stands, unaffected.** 0 of 10 goals concerned Orion's state. |
| `...scarcity-revision-two-ceilings.md` | **Superseded.** Asserted the sums; treated the fleet as one resource. |

## A.2 What survives, what is withdrawn

- **Survives:** *"a resource can be 6% utilised and 100% allocated"* — now with a mechanism
  (§6.7) and an exact conversion (§6.3).
- **Survives:** utilisation is disqualified as a calibration input.
- **Withdrawn:** "GPU 2 is the one uncommitted unit of concurrency." It is on a host that is off.
- **Withdrawn:** every chassis-power estimate in every draft (§4).
- **Withdrawn:** "the dominant term is unmeasured." It is measured on 2 of 3 hosts and discarded
  (§5.1).
- **Withdrawn as a unit, kept as motivation:** "the currency is foregone processes."

## A.3 Live defects bearing on this spec

Full list in `PARKING-LOT.md`.

- **`goal_formulate` is a translator, not a generator.** Ten runs across ten field ticks, each
  given Orion's real pressure readings, all returned paraphrases of one recalled Juniper coding
  session.
- **Recall dominates supplied context** so completely that live field state had no visible
  effect. Possibly the largest blocker in the system.
- **`counterfactual` and `context_exec_memory_contradiction_review` are dead** — empty string in
  ~0.5 s reporting `status=success`, all 10 runs.
- **The feedback loop is open** — 302,974 `substrate_feedback_frames` rows, no reader.
- **Arena urgency degeneracy** — `proposal_urgency()` falls back to `max()` over all
  `PRESSURE_DIMENSIONS` for templates declaring `dimensions: {}`; 5 of 13 do.

---

# Appendix B — measurement method and errors

§3.1 from `orion_biometrics` (7 days, 31.3 s cadence). §3.2 direct host inspection. §3.3 a
hand-run 1 Hz `/slots` poll, 121 samples. §3.4 6 h and 3 h of gateway container logs. §5 from
the live `substrate_node_biometrics_projection`. §6.3/§6.7 Erlang-B computed from measured λ and
an estimated service time.

## B.1 Errors made in this arc, all the same family

1. Single `nvidia-smi` sample read as a duty cycle — and on the wrong host.
2. Input-invariance concluded from n = 2.
3. `distinct_outputs = 10/10` nearly reported as variety; all ten were paraphrases.
4. A circe power jump attributed to AI Town without checking whether AI Town routes through the
   gateway. It does not.
5. `avg((util > 0))` reported as duty cycle — athena's P100 read "99.74% busy" vs a 12.8% mean.
6. `route=([a-z_]+)` silently dropped 60% of gateway requests carrying `route=None`.
7. **Mean GPU utilisation reported as a ceiling** on a bimodal process. Caught by Juniper.
8. **Chassis power estimated twice from core count**, both times without knowing the chassis.
   Caught by Juniper; withdrawn rather than revised.
9. **Claimed the dominant term was unmeasured without reading the existing pipeline.** iLO has
   been reporting chassis watts and fan speed on two of three hosts the whole time. Caught by
   Juniper pointing at disk I/O, which led to the pipeline that contains all of it.

**The through-line:** every one priced or counted the visible thing with the convenient
statistic — and #9 is worse than that: it proposed buying an instrument for a quantity the
system already collects. **Read the existing pipeline before specifying a new sensor.**

## B.2 Standing rules

- State the window, or do not state the number.
- For a ceiling, report **how often it is full**, not the average.
- **Never average across non-substitutable channels.** Use max or a count above threshold.
- Compare lanes by **blocking probability**, not utilisation. At c=1 they coincide; above that
  they diverge fast.
- Keep raw physical units alongside any normalised band — only physical units compose.
- Price the machine, not the chip; and check what is already collected before estimating it.
