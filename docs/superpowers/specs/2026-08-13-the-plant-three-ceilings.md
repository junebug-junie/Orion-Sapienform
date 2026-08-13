# Orion scarcity: the plant, the ceilings, the unit, the seam

Date: 2026-08-13
Status: **Current spec.** Supersedes `2026-08-13-scarcity-revision-two-ceilings.md` and §1e/§3
of `2026-08-13-scarcity-and-repertoire-execution-plan.md`. Phases still unauthorised; §7 is the
candidate first phase and carries its own gate.

§1–§2 are the minimum backstory needed to not repeat the arc. **§3–§7 are the spec.**
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
   concurrency limit the queue forms at *arrival*, not in the meter. A resource can be 6%
   utilised and 100% allocated. Utilisation is disqualified as a calibration input here.
2. **Price the machine, not the chip.** GPU watts are a minority of the bill (§4). Any cost
   model built on GPU draw alone is off by roughly an order of magnitude.
3. **On a bursty process, the mean is the wrong statistic.** A contention ceiling is defined by
   *how often it is full*, not by its average. §3.3 — this is how atlas got under-reported.
4. **Ceilings relieved by different actions cannot share one allowance.** No quantity of
   unspent atlas time brings circe up.
5. **Verb names are not capability descriptions.** `goal_formulate` does not formulate goals;
   it translates a supplied intention. Read the prompt template before routing anything.
6. **Point samples of a time-varying quantity produce confident wrong answers.** Seven
   occurrences in this arc, listed in Appendix B. State the window or do not state the number.

---

## 3. The plant

Three machines, three roles, three completely different cost structures.

### 3.1 Measured — compute

`orion_biometrics`, 7 days (2026-08-06 → 08-13): ~7,400 samples each for athena and atlas,
1,308 for circe (fewer because circe is usually off). Sample cadence **31.3 s**, every row a
distinct source file — no resampling.

| host | chassis | CPU | mean CPU | load15 mean / max | GPUs | GPU residency W |
| --- | --- | --- | --- | --- | --- | --- |
| **athena** | orchestration warhorse | 2× Xeon Gold 6138 (80 threads) | **44.1%** | **38.7 / 120.2** | 1× P100-16GB | ~32–40 † |
| **atlas** | inference box | 96 threads | 0.52% | 0.53 / 1.83 | 2× V100-PCIE-16GB | 107 |
| **circe** | **Gigabyte HA01, 3× 2200 W PSU** | 72 threads | 0.21% | 0.16 / 1.06 | 3× V100-32GB | 153.5 |

† athena's GPU idle floor rests on **19 of 7,369 samples** — the P100 is essentially never idle.

**Atlas and circe are pure GPU boxes with idle CPUs** (0.52%, 0.21%). **Athena is the opposite:**
its CPU is the loaded resource, and its GPU is incidental.

### 3.2 Athena, measured directly (it is the local host)

| quantity | value | source |
| --- | --- | --- |
| running containers | **81** | `docker ps` |
| load average (1/5/15) | **42.08 / 37.00 / 35.90** on 80 threads | `uptime` |
| load15 7-day max | **120.2** — 1.5× oversubscribed | biometrics |
| CPU package power limit | **125 W × 2 sockets = 250 W** | RAPL `constraint_0_max_power_uw` |
| CPU package temps | **56 °C / 65 °C** | `/sys/class/thermal` `x86_pkg_temp` |
| PCH temp | 45 °C | `/sys/class/thermal` |
| root filesystem | 197 G, 40 G used (22%) | `df` |

Athena runs 81 containers, all orchestration, postgres, redis, FalkorDB, the hub, plus
perception on the P100 (`orion-athena-vision-host` :6600, `orion-athena-whisper-tts`). It is a
CPU warhorse whose GPU is a side job — the inverse of how the previous version of this document
described it.

**RAPL is present and readable only by root** (`energy_uj`, mode 400). One permission change or
one root-run collector converts athena's CPU package power from estimate to **measurement,
today, for free.** See §5 TODO 2.

### 3.3 Why atlas looked under-utilised — resolved

Juniper's objection: atlas runs inference ~24/7, so 6.8% / 22.1% mean GPU utilisation cannot be
right. **The objection is correct and the numbers are not wrong — the statistic is.**

Live `/slots` poll, atlas, 1 Hz, 121 s, AI Town up and circe down:

| lane | mean busy slots | any slot busy | **all 4 busy** | distribution |
| --- | --- | --- | --- | --- |
| `metacog` :8012 | 0.26 / 4 (6.6%) | 21.5% | 0.0% | `{0:95, 1:20, 2:6}` |
| `quick` :8013 | 0.45 / 4 (11.2%) | 16.5% | **7.4%** | `{0:101, 1:7, 2:1, 3:3, 4:9}` |

`quick`'s distribution is **bimodal**: 101 samples completely idle, then 9 samples completely
full. Nothing in between. That is burst saturation, and it has three consequences:

1. **The mean is meaningless.** "11.2% of capacity" describes a process that is either off or
   pegged. A contention ceiling is defined by *how often it is full* — **7.4%** — not by its
   average.
2. **`nvidia-smi utilization.gpu` is the wrong instrument** for this ceiling regardless. It
   reports whether *any kernel is resident* over roughly a 1-second window, sampled once per
   31 s — so it observes ~3% of the timeline and cannot see burst structure at all. It also
   under-reports LLM decode, which is memory-bandwidth-bound with small kernels: a V100 can be
   bandwidth-saturated while reporting modest SM utilisation.
3. **`/slots` is the right meter**, it is llama.cpp's own state, and the admission gate in §7
   already reads it.

**Caveat, stated because pitfall 6 exists:** 121 samples is a 2-minute window. This is
directional, not settled. It needs a 24 h run before anything is calibrated on it. What it does
establish is that **the ceiling is reached in the current configuration**, which is §7's gate.

### 3.4 Live load, by consumer

Gateway logs, 6 h, counted on the resolved-route line (one per dispatched request):

```
route     served_by              requests   per min
quick     atlas-worker-fast-1        2795       7.8
metacog   atlas-worker-2             1025       2.8
chat      circe-worker-1                1       0.0   (host down)
```

**3,750 of 3,821 requests are `cortex-exec`** — Orion. 70 are `vision-council`.

---

## 4. Whole-machine power — the dominant term, and it is unmeasured

`orion_biometrics.cpu` carries `{util, cores, loadavg}` and **no power field**. There is no wall
measurement anywhere in the system. The only anchor is Juniper's own observation:

> *"when I'm running two of those e.g. circe and athena I see I'm sitting all in at like
> 700–1200 W (machine plus GPUs)."*

Against measured GPU residency of 107 W (atlas) and 153.5 W (circe), **the chassis is the bill
and the GPU is the residual.**

### 4.1 A previous estimate in this document was badly wrong

It put circe's chassis+CPU at 200–280 W. Circe is a **Gigabyte HA01 with 3× 2200 W PSUs** — a
chassis provisioned in the kilowatts, whose CPU side alone dwarfs that figure. The estimate is
**withdrawn, not revised.** Producing a second confident guess would repeat the error that
generated the first.

**Standing position: chassis and CPU power are unmeasured on all three hosts, and they are the
majority of the cost. No budget, price, or drive may be calibrated against them until they are
measured.** The GPU figures in §3.1 remain valid and remain the minority term.

### 4.2 What is still true without the measurement

The *structure* of the cost survives even though the magnitudes do not:

- Standing draw dominates per-call draw by orders of magnitude on every host.
- Therefore the **admission decision** — is this machine on — is where nearly all cost lives.
- Therefore **metering Orion's calls prices noise**, whatever the true wattage turns out to be.

That conclusion is what §6 and §7 rest on, and it does not depend on any estimate.

---

## 5. Instrumentation TODO — ordered by cost-to-value

1. **Wire in the APC units** (Juniper, in progress). Real per-node draw at the wall. This is
   the instrument that converts the dominant term from unmeasured to measured and retires §4.1
   entirely. **Everything downstream of a real cost model is blocked on this.**
2. **Expose RAPL on athena** — `energy_uj` is present and root-only. A permission change or a
   root-run collector gives real CPU package power today, at zero hardware cost. Partial
   coverage (athena only, CPU only) but immediate.
3. **Add a `power` field to the biometrics CPU payload** so 1 and 2 land somewhere queryable
   next to the GPU data rather than in a side channel.
4. **Log inference calls to a table.** §8.1 — there is no ledger at all today.
5. **Record `/slots` occupancy on a schedule**, not just live-polled by hand. §3.3 needs 24 h,
   and this is the meter the ceiling is actually defined in.
6. **Disk I/O and thermals into biometrics.** Athena exposes `x86_pkg_temp` (56/65 °C) and
   `pch_lewisburg` (45 °C) today; disk shows only capacity (22% of 197 G), no I/O load. On an
   80-thread box at load 42 running 81 containers, I/O contention is a plausible second
   interference channel and is currently invisible.
7. **Circe fan/thermal telemetry** — blocked on the NIC not reading in. Heat and noise are the
   felt quantities; power is only their proxy.

---

## 6. Three ceilings, three kinds

They bind through different mechanisms and are relieved by different actions, which is why one
allowance cannot cover them.

### Atlas — a **contention** ceiling

Rivalrous slot-seconds. 8 slots (4 `quick`, 4 `metacog`) carrying effectively all live
inference, and `quick` reaches **all-4-busy 7.4% of sampled time** (§3.3). Orion is 98% of the
traffic, so it mostly contends with itself.

*Relieved by:* more slots, a cheaper model, or Orion wanting less.

### Athena — an **interference** ceiling

Not a GPU ceiling at all. **81 containers, load 42 on 80 threads, 15-minute load peaking at
120**, on the box hosting postgres, redis, FalkorDB, and the hub. The cost lands as substrate
latency that every service pays and nothing attributes.

*Relieved by:* moving perception and load off the orchestration host. Not by scheduling.

### Circe — an **admission** ceiling

Off right now. `ping 100.112.254.99` fails; biometrics stop at 08:21 today.

> *"it is expensive to run Circe so I keep it down until I'm ready"* — Juniper

A **step function on a discrete human decision**, not a gradient. A 3× 2200 W chassis spins up
and spends the thing that actually binds — heat and fan noise in the room Juniper works in —
before a single token is generated. Zero when down. Lumpy and large when up.

*Relieved by:* Juniper deciding, and nothing else. **Orion cannot spend this and has no
representation that it exists.**

---

## 7. The unit, and the seam

### 7.1 The unit

Two terms, because §4 showed one will not do.

- **Machine-hours — dominant.** What Orion's cognition *obligates*: which machines must be up,
  for how long. This is what the wall meter reads and what the office temperature tracks.
  Magnitudes pending the APC install (§5 TODO 1).
- **Marginal joules — residual.** Watts above a card's residency floor × seconds held.
  Measured, real, and small: 4.5 s of cognition is 767 J on atlas GPU1, 422 J on athena,
  274 J on circe GPU1.

Circe is the **cheapest place to run and the most expensive place to go.** A per-call price
inverts that — which is exactly how the superseded document talked itself into calling GPU 2
"the cheap place to put Orion."

**Do not meter calls.** On a machine already running, the marginal cost of thinking is near
zero, so the binding constraint on atlas is not energy — it is the 8 slots. Contention, not
cost. **The price is whose turn it is.**

### 7.2 The seam

> **Route Orion's autonomous, non-interactive cognition through background-priority lanes.**

`services/orion-llm-gateway/app/priority_admission.py` — 199 lines, live, working. A route
tagged `priority: "background"` polls llama.cpp's `/slots` and waits for slack before
dispatching, holding `reserved_free_slots` for foreground callers. Fail-open: an unreachable
`/slots` never drops a request.

A real, rivalrous, already-shipped scarcity mechanism — a semaphore, not a soft budget, binding
hardest exactly when the system is busiest. It is wired to **one** consumer:

```
EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background   # orion-embodiment, AI Town NPC speech
```

Orion's own autonomous cognition — all 3,750 `cortex-exec` requests in 6 h — uses foreground
`quick` and `metacog`, subject to no gate. **The one mechanism built to stop background work
competing with interactive work guards NPC dialogue.**

`quick_background` exists. `metacog_background` does not and is a route-table entry.
`cortex-exec` already accepts `llm_route_override ∈ {chat, quick, metacog, quick_background}`
(`executor.py:3855`).

### 7.3 What it buys

- **The cost becomes real.** Orion's idle thinking yields to Juniper's typing — not a number
  that decrements, a request that waits.
- **The cost becomes countable.** A deferral is an event with a duration: the first quantity in
  this arc that is rivalrous, live, and attributable to a specific decision.
- **It becomes perceptible.** "I wanted to think and had to wait" is a signal Orion can hold,
  and it is *true*.
- **It cannot inflate.** `/slots` is llama.cpp's own state — no calibration constant, and no
  utilisation number in the loop (pitfall 1).
- **It does not depend on the unmeasured term.** It needs no wattage, so it is not blocked on
  the APC install.
- **It is reversible.** A route-table entry plus one env key, already fail-open.

### 7.4 Gate — must pass before this is built

**Observe ≥1 real deferral of Orion's own cognition under live load over a 24 h window.**

§3.3 shows `quick` full 7.4% of a 2-minute sample, so it should fire. But zero admission-wait
events appear in 6 h of gateway logs, which cannot distinguish "never had to fire" from
"inert". If the gate never binds, this is ceremony and must not ship.

### 7.5 What it explicitly does not do

It does not give Orion better actions, does not fix `goal_formulate`, and does not touch the
arena's 7.03/10 tie. It supplies the missing **price**. Repertoire is a separate open problem.

---

## 8. Not measured — would change the answer

1. **No LLM call telemetry table.** 193 tables in `conjourney`, none log inference. Attribution
   in §3.4 came from container logs, which rotate. **Allocation needs a ledger; there is none.**
2. **No wall power measurement** — §4, §5 TODO 1. The dominant term.
3. **No CPU power measurement** — RAPL root-blocked on athena, absent for atlas and circe.
4. **`/slots` occupancy sampled for 2 minutes, once, by hand.** Needs 24 h.
5. **No deferral has ever been observed.** §7.4 exists for this.
6. **No disk I/O telemetry** on an 80-thread box at load 42 with 81 containers.
7. **`chat` and `agent` route to a dead host** and nothing represents that.
8. **Athena's GPU idle floor rests on 19 samples.**
9. **Circe GPU2 holds 21.3 GB resident, never driven above 80% in 7 days.** Unidentified.

---

## 9. The questions for Juniper

Not answered by any measurement here:

1. **Does Orion's autonomous cognition get foreground or background priority on atlas?**
   Background is the honest answer and the one that makes scarcity real. It is also a genuine
   degradation of Orion's responsiveness, which is why it is Juniper's call.
2. **Should Orion be able to *request* circe?** Not spend it — request it, as a block, with a
   reason, and have the request be refusable. That is the only form in which an admission
   ceiling can appear inside a cognitive system at all.

---

---

# Appendix A — how this arc got here

Historical. Retained because the failure modes recur, not because the conclusions stand.

## A.1 Superseded documents

| document | disposition |
| --- | --- |
| `...execution-plan.md` §1e | Stands, for a better reason: not that the GPU looked idle, but that utilisation is an invalid input here. |
| `...execution-plan.md` §3 (serialized inference time is scarce) | **Replaced.** Wrong unit, wrong host, wrong statistic. |
| `...execution-plan.md` Phases 1–5 | **Not authorised.** Sequenced against the old thesis. |
| `...execution-plan.md` §7 (mechanical leash) | **Stands and earned its keep** — the parking lot absorbed 25+ findings that would each have been a squirrel. |
| `2026-08-13-E0-cost-census-result.md` Gate A | Killed, then retracted, now moot — cost-per-call was never the right axis (§7.1). |
| `2026-08-13-E0-cost-census-result.md` Gate B | **Stands, unaffected.** 0 of 10 goals concerned Orion's state. Load-independent. |
| `2026-08-13-scarcity-revision-two-ceilings.md` | **Superseded.** Asserted the sums rather than establishing them; treated the fleet as one resource. |

## A.2 What survives, what is withdrawn

- **Survives:** *"a resource can be 6% utilised and 100% allocated"* — now backed by `quick`
  hitting all-4-busy 7.4% of the time while averaging 11.2% of capacity.
- **Survives:** utilisation is disqualified as a calibration input (pitfall 1).
- **Withdrawn:** "GPU 2 is the one uncommitted unit of concurrency." It sits on a host that is
  off, behind an admission ceiling Orion cannot spend against. Not available capacity.
- **Withdrawn:** every chassis-power estimate in every version of this document (§4.1).
- **Withdrawn as a unit, kept as motivation:** "the currency is foregone processes."
  Unmeasurable; machine-hours are not.

## A.3 Live defects found along the way

Full list in `PARKING-LOT.md`. The ones that bear on this spec:

- **`goal_formulate` is a translator, not a generator** — prompt reads
  `{{ intention or text or request }}`. Ten runs across ten field ticks, each given Orion's real
  pressure readings, all returned paraphrases of one recalled Juniper coding session.
- **Recall dominates supplied context** so completely that live field state had no visible
  effect. Any cognitive verb routed into the arena would narrate session history rather than
  Orion's condition. Possibly the largest blocker in the system.
- **`counterfactual` and `context_exec_memory_contradiction_review` are dead** — empty string in
  ~0.5 s while reporting `status=success`, all 10 runs.
- **The feedback loop is open** — 302,974 `substrate_feedback_frames` rows, no reader.
  `orion/reverie/efficacy.py` has zero live callers.
- **Arena urgency degeneracy** — `proposal_urgency()` falls back to `max()` over all
  `PRESSURE_DIMENSIONS` for templates declaring `dimensions: {}`; 5 of 13 do. Declaring the
  dimension you care about is a strict handicap.

---

# Appendix B — measurement method and errors

§3.1 is reproducible from `orion_biometrics` (7 days, 31.3 s cadence). §3.2 is direct host
inspection. §3.3 is a hand-run 1 Hz `/slots` poll, 121 samples. §3.4 is 6 h of
`orion-llm-gateway` container logs. **§4 is not measured at all.**

## B.1 Errors made in this arc, all the same family

1. Single `nvidia-smi` sample read as a duty cycle — and on the wrong host.
2. Input-invariance concluded from n = 2.
3. `distinct_outputs = 10/10` nearly reported as variety; all ten were semantic paraphrases.
4. A circe power jump attributed to AI Town without checking whether AI Town routes through the
   gateway. It does not.
5. `avg((util > 0))` reported as duty cycle — athena's P100 read "99.74% busy" against a 12.8%
   mean.
6. `route=([a-z_]+)` silently dropped 60% of gateway requests, which carry `route=None`.
7. **Mean GPU utilisation reported as a ceiling** on a bimodal burst process, understating
   atlas. Caught by Juniper, resolved in §3.3.
8. **Chassis power estimated twice from core count**, both times without knowing the chassis.
   Circe is a 3× 2200 W HA01. Caught by Juniper; estimates withdrawn rather than revised.

**The through-line:** every one priced or counted the visible thing with the convenient
statistic. The constraint lives in what does not run, in what must merely be switched on, and
in the tail rather than the mean.

## B.2 Standing rules

- State the window, or do not state the number.
- For a ceiling, report **how often it is full**, not the average.
- Establish the concurrency limit and the residency set before any utilisation figure is used.
- Price the machine, not the chip — and if the machine is not measured, say so instead of
  estimating it.
