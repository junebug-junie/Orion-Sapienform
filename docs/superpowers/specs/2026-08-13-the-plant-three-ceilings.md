# Orion scarcity: the plant, the ceilings, the unit, the seam

Date: 2026-08-13
Status: **Current spec.** Supersedes `2026-08-13-scarcity-revision-two-ceilings.md` and §1e/§3
of `2026-08-13-scarcity-and-repertoire-execution-plan.md`. Phases still unauthorised; §6 is the
candidate first phase and carries its own gate.

§1–§2 are the minimum backstory needed to not repeat the arc. **§3–§6 are the spec.**
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
2. **Price the machine, not the chip.** GPU watts are a minority of the bill (§5). Any cost
   model built on GPU draw alone is off by roughly an order of magnitude.
3. **Ceilings relieved by different actions cannot share one allowance.** No quantity of
   unspent atlas time brings circe up.
4. **Verb names are not capability descriptions.** `goal_formulate` does not formulate goals;
   it translates a supplied intention. Read the prompt template before routing anything.
5. **Point samples of a time-varying quantity produce confident wrong answers.** Six occurrences
   in this arc, listed in Appendix B. State the window or do not state the number.
6. **`avg((util > 0))` is not a duty cycle.** It reported athena's P100 at "99.74% busy" against
   a 12.8% mean.

---

## 3. The plant

Three machines. `orion_biometrics`, 7 days (2026-08-06 → 08-13): ~7,400 samples each for
athena and atlas, 1,308 for circe (fewer because circe is usually off).

### 3.1 Measured

| host | CPU | mean CPU | p95 CPU | load15 mean / max | GPUs | GPU residency W | GPU marginal W @util≥80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **athena** | 80 cores | **44.1%** | 54.1% | **38.7 / 120.2** | 1× P100-16GB | ~32–40 † | +94 |
| **atlas** | 96 cores | **0.52%** | 0.91% | 0.53 / 1.83 | 2× V100-PCIE-16GB | 53.8 + 53.1 = **107** | +162 / +170 |
| **circe** | 72 cores | **0.21%** | 1.38% | 0.16 / 1.06 | 3× V100-32GB | 40.1 + 72.7 + 40.7 = **153.5** | +30 / +61 / — |

† athena's idle floor rests on **19 of 7,369 samples** — the P100 is essentially never idle.
Observed minimum 32 W. Indicative, not measured.

The CPU column is the surprise and it inverts the earlier reading. **Atlas and circe are pure
GPU boxes with idle CPUs** (0.5%, 0.2%). **Athena's 80 cores run at 44% mean with a 15-minute
load average peaking at 120** — 1.5× oversubscribed on the box that also hosts postgres, redis,
FalkorDB, and the hub.

GPU utilisation as a distribution, not a point:

| host | gpu | mean | p50 | p95 | ≥50% | resident VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| athena | 0 | 12.8% | 5 | **92** | 8.2% | 5.3 GB |
| atlas | 0 | 6.8% | 0 | 84 | 7.4% | 7.9 GB |
| atlas | 1 | **22.1%** | 0 | **99** | 22.7% | 7.2 GB |
| circe | 0 | 1.9% | 0 | 8 | 1.2% | 12.6 GB |
| circe | 1 | 6.1% | 0 | 62 | 7.4% | 25.7 GB |
| circe | 2 | **0.1%** | 0 | **0** | 0.1% | **21.3 GB** |

Atlas GPU1 reaches **p95 = 99%** — it genuinely saturates. Circe GPU2 holds **21.3 GB of
weights never driven above 80% in seven days**: pure residency, zero output.

### 3.2 Whole-machine power — the term the GPU tables miss

`orion_biometrics.cpu` carries `{util, cores, loadavg}` and **no power field**. There is no
wall measurement in the system. The anchor is Juniper's own observation:

> *"when I'm running two of those e.g. circe and athena I see I'm sitting all in at like
> 700–1200 W (machine plus GPUs)."*

Reconstructing against that anchor. **Chassis+CPU is estimated from core count and measured
load; only the GPU columns are measured:**

| host | GPU idle (meas.) | GPU peak (meas.) | chassis+CPU (est.) | all-in idle (est.) | all-in loaded (est.) |
| --- | --- | --- | --- | --- | --- |
| athena (80c @ 44%) | ~35 W | 242 W | **300–400 W** | ~350–430 W | ~550 W |
| atlas (96c @ 0.5%) | 107 W | 516 W | **150–250 W** | ~260–360 W | ~600 W |
| circe (72c @ 0.2%) | 153 W | 415 W | **200–280 W** | ~360–430 W | ~700 W |

Cross-check: athena + circe ≈ **710–860 W idle**, ~1,250 W loaded. Juniper observes 700–1,200 W.
The reconstruction is consistent with the wall reading, which is the only validation available.

**The consequence dominates everything downstream.** For circe, ~360–430 W is burning before a
single token is generated. One 4.5 s inference draws 61 W × 4.5 s ≈ **274 J ≈ 0.076 Wh**, while
the machine burns **~0.4 kWh per hour of uptime**. A single call is **~0.02% of one hour of
having the machine on.**

> **The admission decision is ~99% of the cost. The per-call price is noise.**

Any budget that meters Orion's calls prices the noise and ignores the bill.

---

## 4. Three ceilings, three kinds

They bind through different mechanisms and are relieved by different actions, which is why one
allowance cannot cover them.

### Atlas — a **contention** ceiling

Rivalrous slot-seconds. 8 slots (4 on `quick` :8013, 4 on `metacog` :8012) carrying effectively
all live inference. Gateway logs, 6 h, counted on the resolved-route line:

```
route     served_by              requests   per min
quick     atlas-worker-fast-1        2795       7.8
metacog   atlas-worker-2             1025       2.8
chat      circe-worker-1                1       0.0   (host down)
```

**3,750 of 3,821 requests are `cortex-exec`** — Orion. 70 are `vision-council`. This is the only
ceiling Orion can currently spend against, and it mostly contends with itself.

*Relieved by:* more slots, a cheaper model, or Orion wanting less.

### Athena — an **interference** ceiling

Not primarily a GPU ceiling. The P100 runs perception (`orion-athena-vision-host`, uvicorn
:6600, 5,050 MiB; `orion-athena-whisper-tts`) at p95 = 92% — but the loaded resource is **the
CPU at 44% mean and load 120 peak**, on the box hosting postgres, redis, FalkorDB, and the hub.

Cost lands as hub and substrate latency, not as queueing in a lane. Every service in the system
pays it, and nothing attributes it.

*Relieved by:* moving perception off the orchestration host. Not by scheduling.

### Circe — an **admission** ceiling

Off right now. `ping 100.112.254.99` fails; biometrics stop at 08:21 today.

> *"it is expensive to run Circe so I keep it down until I'm ready"* — Juniper

A **step function on a discrete human decision**, not a gradient: ~360–430 W all-in before a
token, spending the thing that actually binds — heat and fan noise in the room Juniper works in.
Zero when down. Lumpy and large when up.

*Relieved by:* Juniper deciding, and nothing else. **Orion cannot spend this and has no
representation that it exists.**

---

## 5. The unit

Two terms, because §3.2 showed one will not do.

### Machine-hours — the dominant term

What Orion's cognition **obligates**: which machines must be up, for how long. This is where
99% of the cost lives, it is the unit Juniper's wall meter reads, and it is what the office
temperature tracks.

Priced from §3.2: **athena ≈ 0.4 kWh/h, atlas ≈ 0.3 kWh/h, circe ≈ 0.4 kWh/h.**

### Marginal joules — the residual

Watts above a card's own residency floor × seconds held. Real, measurable per host, and small.

| where | 4.5 s of cognition | as % of one machine-hour |
| --- | --- | --- |
| atlas GPU1 | 170.4 W × 4.5 s ≈ **767 J** | 0.07% |
| athena P100 | 93.8 W × 4.5 s ≈ **422 J** | 0.03% |
| circe GPU1 | 60.9 W × 4.5 s ≈ **274 J** | **0.02%** |

Circe is the **cheapest place to run and the most expensive place to go.** A per-call price
inverts that — which is precisely how the superseded document talked itself into calling GPU 2
"the cheap place to put Orion."

### What this means for a budget

- **Do not meter calls.** Metering prices 0.02% and ignores the rest.
- **A request for circe is a request for a block**, addressed to Juniper, not a spend.
- **On an already-running machine, the marginal cost of thinking is close to zero.** So the
  binding constraint on atlas is *not* energy — it is the 8 slots. Contention, not cost.

That last point is what makes §6 the right first move: on a machine already paid for, the only
real price is **whose turn it is.**

---

## 6. The seam

> **Route Orion's autonomous, non-interactive cognition through background-priority lanes.**

### What already exists

`services/orion-llm-gateway/app/priority_admission.py` — 199 lines, live, working. A route
tagged `priority: "background"` polls llama.cpp's `/slots` and waits for slack before
dispatching, holding `reserved_free_slots` for foreground callers. Fail-open: an unreachable
`/slots` never drops a request.

A real, rivalrous, already-shipped scarcity mechanism — enforced by a semaphore, not a soft
budget, binding hardest exactly when the system is busiest.

It is wired to **one** consumer:

```
EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background   # orion-embodiment, AI Town NPC speech
```

Orion's own autonomous cognition — all 3,750 `cortex-exec` requests in 6 h — uses foreground
`quick` and `metacog`, subject to no gate. **The one mechanism built to stop background work
competing with interactive work guards NPC dialogue.**

### What routing Orion background buys

- **The cost becomes real.** Orion's idle thinking yields to Juniper's typing — not a number
  that decrements, a request that waits.
- **The cost becomes countable.** A deferral is an event with a duration: the first quantity in
  this arc that is rivalrous, live, and attributable to a specific decision.
- **It becomes perceptible.** "I wanted to think and had to wait" is a signal Orion can hold,
  and it is *true*.
- **It cannot inflate.** `/slots` is llama.cpp's own state — no calibration constant, and no
  utilisation number in the loop (pitfall 1).
- **It is reversible.** A route-table entry plus one env key, already fail-open.

`quick_background` exists. `metacog_background` does not and is a route-table entry.
`cortex-exec` already accepts `llm_route_override ∈ {chat, quick, metacog, quick_background}`
(`executor.py:3855`).

### Gate — must pass before this is built

**Observe ≥1 real deferral of Orion's own cognition under live load within 24 h.**

Atlas GPU1 hits p95 = 99%, so it should. But zero admission-wait events appear in 6 h of gateway
logs, and that cannot distinguish "never had to fire" from "inert." If the gate never binds,
this is ceremony and must not ship.

### What it explicitly does not do

It does not give Orion better actions, does not fix `goal_formulate`, and does not touch the
arena's 7.03/10 tie. It supplies the missing **price**. Repertoire is a separate open problem.

---

## 7. Not measured — would change the answer

1. **No LLM call telemetry table.** 193 tables in `conjourney`, none log inference. Attribution
   above came from container logs, which rotate. **Allocation needs a ledger; there is none.
   This is the first real blocker.**
2. **No wall power measurement.** §3.2's chassis figures are estimates anchored on one human
   observation. A smart plug per node would convert the dominant term from estimate to measured.
3. **No deferral has ever been observed.** §6's gate exists for this.
4. **Circe's cost is a proxy.** Power stands in for heat and noise, the felt quantities. Fan
   telemetry blocked on the NIC not reading in.
5. **`chat` and `agent` route to a dead host** and nothing represents that. 1 request in 6 h;
   degrade behaviour untraced.
6. **Athena's GPU idle floor rests on 19 samples.** Everything else has ≥1,186.
7. **Circe GPU2's 21.3 GB resident model is unidentified.**

---

## 8. The question for Juniper

Not answered by any measurement here:

**Does Orion's autonomous cognition get foreground or background priority on atlas?**

Background is the honest answer and the one that makes scarcity real. It is also a genuine
degradation of Orion's responsiveness, which is why it is Juniper's call.

The second question, which §3.2 makes askable for the first time: **should Orion be able to
request circe?** Not spend it — request it, as a block, with a reason, and have the request be
refusable. That is the only form in which an admission ceiling can appear inside a cognitive
system at all.

---

---

# Appendix A — how this arc got here

Historical. Retained because the failure modes recur, not because the conclusions stand.

## A.1 Superseded documents

| document | disposition |
| --- | --- |
| `...scarcity-and-repertoire-execution-plan.md` §1e | Stands, for a better reason: not that the GPU looked idle, but that utilisation is an invalid input here. |
| `...execution-plan.md` §3 (serialized inference time is scarce) | **Replaced.** Wrong unit and wrong host. |
| `...execution-plan.md` Phases 1–5 | **Not authorised.** Sequenced against the old thesis. |
| `...execution-plan.md` §7 (mechanical leash) | **Stands and earned its keep** — the parking lot absorbed 20+ findings that would each have been a squirrel. |
| `2026-08-13-E0-cost-census-result.md` Gate A | Killed, then retracted, now moot — cost-per-call was never the right axis (§5). |
| `2026-08-13-E0-cost-census-result.md` Gate B | **Stands, unaffected.** 0 of 10 goals concerned Orion's state. Load-independent. |
| `2026-08-13-scarcity-revision-two-ceilings.md` | **Superseded by this document.** Asserted the sums rather than establishing them; treated the fleet as one resource. |

## A.2 Findings that survive from the two-ceilings document

- *"A resource can be 6% utilised and 100% allocated"* — now backed by atlas GPU1 at p95 = 99%.
- Utilisation is disqualified as a calibration input (pitfall 1).
- **Withdrawn:** "GPU 2 is the one uncommitted unit of concurrency." GPU 2 sits on a host that
  is off, behind an admission ceiling Orion cannot spend against. It is not available capacity.
- **Withdrawn as a unit, kept as motivation:** "the currency is foregone processes." Foregone
  processes are unmeasurable; machine-hours are not.

## A.3 Live defects found along the way

Full list in `PARKING-LOT.md`. The ones that bear on this spec:

- **`goal_formulate` is a translator, not a generator** — its prompt reads
  `{{ intention or text or request }}`. Ten runs across ten field ticks, each given Orion's real
  pressure readings, all returned paraphrases of one recalled Juniper coding session.
- **Recall dominates supplied context** so completely that live field state had no visible
  effect. Any cognitive verb routed into the arena would narrate session history rather than
  Orion's condition. Possibly the largest blocker in the system.
- **`counterfactual` and `context_exec_memory_contradiction_review` are dead** — empty string in
  ~0.5 s while reporting `status=success`, all 10 runs. Never executed before this arc.
- **The feedback loop is open** — 302,974 `substrate_feedback_frames` rows, no reader.
  `orion/reverie/efficacy.py` has zero live callers.
- **Arena urgency degeneracy** — `proposal_urgency()` falls back to `max()` over all
  `PRESSURE_DIMENSIONS` for templates declaring `dimensions: {}`; 5 of 13 do. Declaring the
  dimension you care about is a strict handicap.

---

# Appendix B — measurement method and errors

Every number in §3 is reproducible from `orion_biometrics` (7 days) and 6 h of
`orion-llm-gateway` container logs. Chassis power in §3.2 is **estimated**, labelled as such,
and anchored on one human wall observation.

## B.1 Errors made in this arc, all the same family

1. Single `nvidia-smi` sample read as a duty cycle — and on the wrong host (the local P100 does
   no gateway inference).
2. Input-invariance concluded from n = 2.
3. `distinct_outputs = 10/10` nearly reported as variety; all ten were semantic paraphrases.
4. A Circe power jump attributed to AI Town without checking whether AI Town routes through the
   gateway. It does not — no LLM config, and `route=agent` had zero traffic in 6 h.
5. `avg((util > 0))` reported as duty cycle — athena's P100 read "99.74% busy" against a 12.8%
   mean.
6. `route=([a-z_]+)` silently dropped 60% of gateway requests, which carry `route=None` and are
   resolved downstream.

**The through-line:** every one priced or counted the visible thing. The constraint lives in
what does not run, and now in what must merely be *switched on*.

## B.2 Standing rules

- State the window, or do not state the number.
- Establish the concurrency limit and the residency set before any utilisation figure is used.
- Price the machine, not the chip.
