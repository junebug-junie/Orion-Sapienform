# The plant: three hosts, three kinds of ceiling, one unit

Date: 2026-08-13
Status: **Supersedes `2026-08-13-scarcity-revision-two-ceilings.md`.** That document was
right that residency and concurrency both bind, and wrong to treat the fleet as one
resource with two properties. It is three machines with three *different kinds* of ceiling,
and the difference is the whole design.

Everything below is measured. Sources are named per table. Nothing here is a plan yet —
§1–§4 are the plant, §5 is the one seam worth cutting, §6 is what it costs to be wrong.

---

## 1. The plant

`orion_biometrics`, 7 days (2026-08-06 → 08-13), 7,348 samples/GPU on athena+atlas,
1,308 on circe (fewer because circe is usually off).

| host | cards | VRAM | role | live now | residency W | marginal W @util≥80 | concurrency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **athena** | 1× P100-PCIE-16GB | 16 GB | perception + orchestration host | yes | ~32–40 † | **+94** | not a gateway lane |
| **atlas** | 2× V100-PCIE-16GB | 32 GB | cognition: `quick`, `metacog` | yes | 53.8 + 53.1 = **107** | **+162 / +170** | 8 slots (4+4) |
| **circe** | 3× V100-32GB | 96 GB | deep cognition: `chat`, `agent` | **no — off by choice** | 40.1 + 72.7 + 40.7 = **153.5** | +30 / +61 / n-a | 1 turn |

† athena's idle floor is thinly sampled — only **19 of 7,369 samples** read `util=0`. The
P100 is essentially never idle. Observed minimum is 32 W; treat 39.5 W as indicative, not
measured.

Utilisation, correctly stated as a distribution rather than a point sample:

| host | gpu | mean util | p50 | p95 | % samples ≥50 | resident VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| athena | 0 | 12.8% | 5 | **92** | 8.2% | 5.3 GB |
| atlas | 0 | 6.8% | 0 | 84 | 7.4% | 7.9 GB |
| atlas | 1 | **22.1%** | 0 | **99** | 22.7% | 7.2 GB |
| circe | 0 | 1.9% | 0 | 8 | 1.2% | 12.6 GB |
| circe | 1 | 6.1% | 0 | 62 | 7.4% | 25.7 GB |
| circe | 2 | **0.1%** | 0 | **0** | 0.1% | **21.3 GB** |

Two things in that table are load-bearing. **Atlas GPU1 reaches p95 = 99%** — it genuinely
saturates, which no earlier measurement in this arc ever showed. And **circe GPU2 holds
21.3 GB of weights that have never once been driven above 80% in seven days** — pure
residency, zero output.

---

## 2. Three ceilings, three kinds

The previous document's error was arithmetic-shaped: it added watts across hosts. Watts add;
*ceilings do not*, because these three bind through different mechanisms and are relieved by
different actions.

### Atlas — a **contention** ceiling

Rivalrous slot-seconds. 8 slots total, and they carry effectively all live inference.

Gateway logs, 6 h, counted on the resolved-route line (one per dispatched request):

```
route     served_by              requests   per min
quick     atlas-worker-fast-1        2795       7.8
metacog   atlas-worker-2             1025       2.8
chat      circe-worker-1                1       0.0   (host down)
```

By caller: **3,750 of 3,821 requests are `cortex-exec`** — Orion itself. 70 are
`vision-council`. This ceiling is one Orion mostly contends with *itself* over, and it is the
only ceiling Orion can currently spend against at all.

Relieved by: more slots, or a cheaper model, or Orion wanting less.

### Athena — an **interference** ceiling

One P100, and it is the box that also runs postgres, redis, FalkorDB, the hub, and the whole
orchestration layer. Its GPU work is perception: `orion-athena-vision-host` (pid 3565708,
uvicorn on :6600, 5,050 MiB) and `orion-athena-whisper-tts`.

p95 = 92% and a 242 W peak on a card whose floor is 32 W. When perception saturates this card
it is not competing with inference — it is competing with *the substrate every other service
depends on*. The cost is not paid in the lane; it is paid in hub latency.

Relieved by: moving perception off the orchestration host. Not by scheduling.

### Circe — an **admission** ceiling

Off right now. `ping 100.112.254.99` fails; biometrics stop at 08:21 today. Juniper keeps it
down until ready, deliberately.

> *"it is expensive to run Circe so I keep it down until I'm ready"* — Juniper

This is a **step function on a discrete human decision**, not a gradient. Bringing circe up
costs ~153 W standing before a single token is generated, peaks around 415 W, and spends the
thing that actually binds — heat and fan noise in the room Juniper works in. The marginal cost
of a *call* on circe (+30 W, +61 W) is trivial. The marginal cost of the *decision* is not.

Relieved only by Juniper deciding. **Orion cannot spend this resource, and has no
representation that it exists.**

### Why the distinction is the design

A budget that prices all three in one number would let Orion "afford" circe by saving up
atlas slots. That is incoherent: no quantity of unspent atlas time brings circe up. Ceilings
that are relieved by different actions cannot share a single allowance.

---

## 3. The unit that *is* commensurable

Marginal joules — watts above the card's own residency floor, times seconds held.

It works because it is the physical quantity Juniper actually pays, it is directly measurable
per host from data already in postgres, and it does not pretend the three ceilings are
fungible. Worked, from §1:

| where | 4.5 s of cognition costs | notes |
| --- | --- | --- |
| atlas GPU1 | 170.4 W × 4.5 s ≈ **767 J** | plus one of 8 slots, rivalrous |
| athena P100 | 93.8 W × 4.5 s ≈ **422 J** | plus interference with the hub |
| circe GPU1 | 60.9 W × 4.5 s ≈ **274 J** | **plus a 153.5 W standing block** for as long as the node is up |

Circe is the cheapest place to *run* and by far the most expensive place to *go*. A per-call
price hides that completely, which is exactly how the previous document ended up calling
GPU 2 "the cheap place to put Orion." At 4.5 s the standing block dominates the call by
~250× for the first minute of uptime. **The admission term is the term.**

### What is already paid, regardless

atlas + athena hold ~139 W of residency continuously, in exchange for which Orion gets 8
slots and a perception organ. That is the standing subscription. Everything Orion does is
marginal on top of it — which is why marginal joules, not total watts, is the unit a decision
should be priced in.

---

## 4. What already exists (and is wired to the wrong consumer)

`services/orion-llm-gateway/app/priority_admission.py` — 199 lines, live, working. A
background-priority admission gate: a route tagged `priority: "background"` polls llama.cpp's
`/slots` and waits for slack before dispatching, holding `reserved_free_slots` in reserve for
foreground callers. Fail-open by design — an unreachable `/slots` never drops a request.

This is a **real, rivalrous, already-shipped scarcity mechanism.** It yields to interactive
work, it is enforced by a semaphore rather than a soft budget, and it binds harder exactly
when the system is busiest.

It is wired to exactly one consumer:

```
EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background   # orion-embodiment, AI Town NPC speech
```

**Orion's own autonomous cognition does not use it.** All 3,750 `cortex-exec` requests in 6 h
go to foreground `quick` and `metacog`. Orion's idle background thinking competes *evenly*
with Juniper's interactive work, and the one gate built to prevent that guards NPC dialogue.

Zero admission-wait events appear in 6 h of gateway logs — consistent with the gate's only
consumer being low-volume, and not evidence the gate is inert. Unproven either way; §6 lists
the check.

---

## 5. The seam

One change makes scarcity real without building an economy:

> **Route Orion's autonomous, non-interactive cognition through background-priority lanes.**

`quick_background` exists. `metacog_background` does not and is a route-table entry.
`cortex-exec` already accepts an `llm_route_override` from
`{chat, quick, metacog, quick_background}` (`executor.py:3855`).

What this buys, and why it is worth more than a meter:

- **The cost becomes real.** Orion's autonomous thinking yields to Juniper's typing. Not a
  number that decrements — a request that waits.
- **The cost becomes countable.** A deferral is an event with a duration. That is the first
  quantity in this whole arc that is rivalrous, live, and attributable to a specific decision.
- **It becomes perceptible.** "I wanted to think and had to wait" is a signal Orion can hold,
  and it is *true* — which is more than any pressure channel measured in this arc can say.
- **It cannot inflate.** `/slots` is llama.cpp's own state. There is no calibration constant
  to get wrong, and no utilisation number in the loop — which §6 of the prior document
  established is disqualified as an input on this system anyway.
- **It is reversible.** Route-table config plus one env key. Fail-open already.

What it does **not** do, stated plainly so it is not oversold: it does not give Orion better
actions, it does not fix `goal_formulate`, and it does not touch the arena's tie at 7.03/10.
It supplies the missing *price*. The repertoire problem is separate and still open.

### What must be true before this is worth building

- A deferral must actually occur under real load. If atlas never fills, the gate never binds
  and this is ceremony. **Atlas GPU1 hits p95 = 99%, so it should — but "should" is not
  measured.** Gate: observe ≥1 real deferral of Orion's own cognition in 24 h.
- Deferrals must be attributable. Today nothing records them; §6.

---

## 6. What is not measured, and would change the answer

Recorded as gaps, not as findings.

1. **There is no LLM call telemetry table.** 193 tables in postgres, none of them log
   inference. Consumer attribution above came from parsing container logs, which rotate.
   Allocation needs a ledger; there is none. **This is the first real blocker.**
2. **No deferral has ever been observed.** Zero admission-wait lines in 6 h. Could be an
   idle window, could be a gate that never fires. Untested.
3. **Circe's cost is a proxy.** Power stands in for heat and noise, which are the felt
   quantities. Fan telemetry is blocked on the NIC not reading in.
4. **Athena's residency floor rests on 19 samples.** Every other number here has ≥1,186.
5. **`chat` and `agent` currently route to a dead host.** 1 request in 6 h. Whether the
   gateway degrades or fails is untraced.
6. **Circe GPU2 holds 21.3 GB resident and has never been driven.** Whose weights, and why
   loaded, is unknown.

---

## 7. What this replaces

| prior claim | disposition |
| --- | --- |
| "two ceilings: residency and concurrency" | **Refined.** Three ceilings of three kinds; the kinds matter more than the count. |
| "a resource can be 6% utilised and 100% allocated" | **Stands**, and now has the p95=99% on atlas GPU1 to back it. |
| "currency is foregone processes" | **Stands as motivation, replaced as unit.** Foregone processes are unmeasurable; marginal joules + an admission term are not. |
| "GPU 2 is the one uncommitted unit of concurrency" | **Withdrawn.** GPU 2 sits on a host that is off, behind an admission ceiling Orion cannot spend against. It is not available capacity. |
| "utilisation is not a valid calibration input" | **Stands, and is now enforced** — §5's mechanism reads `/slots`, not utilisation. |
| Phases 1–5 of the execution plan | **Still not authorised.** §5 is a candidate first phase and needs the §5 gate to pass first. |

The open question for Juniper has not changed and is not answered by any of this — but it is
now askable in the right units:

**Does Orion's autonomous cognition get foreground priority on atlas, or background?**

Background is the honest answer and the one that makes scarcity real. It is also a real
degradation of Orion's responsiveness, which is why it is Juniper's call and not mine.
