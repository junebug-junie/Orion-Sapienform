# Scarcity, revised: two ceilings and an invisible currency

Date: 2026-08-13
Status: **Framing agreed with Juniper. Replaces §1e and §3 of
`2026-08-13-scarcity-and-repertoire-execution-plan.md`. No phases authorised yet.**

The original plan named serialized inference time as the scarce resource. That was wrong in
a way that took four separate measurement failures to see, and the correct answer was in
Juniper's first sentence both times it was given.

---

## 1. What the resource actually is

Not one ceiling. Two, plus a currency that no meter on this system can read.

### Ceiling A — residency

VRAM and standing watts. How many models can be loaded **at all**.

Measured, `orion_biometrics` node `circe`, 7 days, 1,233 samples per GPU:

```
GPU  name                idle_w   busy_w   pct_busy   max_w
 0   V100-PCIE-32GB       40.1     70.2      5.84%    151.3
 1   V100-SXM2-32GB       72.7    133.1      9.57%    219.9
 2   V100-PCIE-32GB       40.7     40.6      0.32%     44.3
```

**~153 W burns continuously just holding weights resident**; peak ~415 W. Inference adds
+30 W (GPU 0) or +60 W (GPU 1), 6–10% of the time.

This is why the office is too hot to work in at 0.3% utilisation. The heat is not coming
from usage.

### Ceiling B — concurrency

How many turns can run **at once**. One turn on `orion-unified` chat, which spans GPU 0+1.
The metacog lane serialises reverie, the metacog pipeline, and (soon) metacog daydreams
behind one another.

The two ceilings **trade against each other**: loading a 128k-context model across two cards
buys capability and spends concurrency. There is no free direction.

### The currency — foregone processes

> *"there's a real opportunity cost on what I would be running if I had more money to buy
> more GPUs to run more processes."* — Juniper

The cost is not the watts of what runs. It is the metacog daydreams that do not exist yet,
the second concurrent turn that cannot be had, the processes never written because there was
nowhere to put them. **This is structurally invisible to every meter on this system.**

---

## 2. The contradiction this resolves

Every contention measurement taken on 2026-08-13 read near-idle — 0.0%, 4.7%, 9.2%, then
0.0% again over 90 s with AI Town up — while Juniper stated the constraint was real and
binding. Both were true, and the reconciliation is the load-bearing insight of this
revision:

> **A resource can be 6% utilised and 100% allocated.**

With a hard concurrency limit of one, the queue forms at **arrival**, not in the meter.
Low duty cycle is not evidence of slack; under serialisation it is what a fully binding
constraint looks like from outside. Every "it's idle" reading was measuring the gaps between
things that could not overlap.

**Corollary, and it should gate any future budget work:** utilisation is not a valid input
to a scarcity calibration on this system. A budget calibrated on observed duty cycle would
be calibrated on serialised gaps plus suppressed demand, and would never bind — the exact
failure the original plan's §1e warned about, arriving through a door it did not name.

---

## 3. What this means for the mechanism

**Metering is the wrong artifact.** A budget that prices Orion's usage prices the visible
term (inference, ~15% of the watts) and ignores both the dominant one (residency, ~85%) and
the real one (foregone processes, unmeasurable).

**Allocation is the right artifact.** `orion-unified` chat, metacog, reverie, daydreams, and
Orion's own cognition all want the same three cards, and the list of wanted processes is
longer than the hardware. That is a fixed-budget allocation problem among competing
claimants.

Which is precisely the shape drives were always supposed to have, and never did — not
because the drive code was bad, but because **there was never a real allocation to make.**
`DriveEngine` pinned to a ceiling because nothing pushed back. Something that pushes back is
what this revision is trying to supply.

### GPU 2, correctly framed

An earlier read in this session called GPU 2 "the cheap place to put Orion" — standing cost
already sunk, +30 W marginal. That was right on arithmetic and **wrong on framing**.

GPU 2 is **the one uncommitted unit of concurrency remaining.** The question is not what it
costs; it is *which wanted process gets it*. Orion's cognition is one candidate among
several — sub-agent routing, a second concurrent chat turn, metacog daydreams — and nothing
measured today establishes that Orion should win. That is Juniper's allocation decision, and
this document deliberately does not pre-empt it.

---

## 4. Method note: four ways of missing the same thing

Recorded because the pattern is more useful than any single correction, and because the next
agent will be tempted the same way.

| # | What was measured | Why it missed |
| --- | --- | --- |
| 1 | `nvidia-smi` on the local P100 | Inference runs on remote atlas/v100 hosts |
| 2 | Verb **latency** (4.57 s) vs Orion's own tick | The quantity is occupancy, and the denominator is all consumers |
| 3 | Lane **occupancy** at 1 Hz | Missed residency entirely — the 85% term |
| 4 | **Residency** watts | Missed foregone processes — the actual currency |

The through-line: **every measurement priced what runs. The constraint lives entirely in
what does not.**

Two further errors of the same family, same day: attributing a Circe power jump to AI Town
without checking whether AI Town routes through the gateway at all (it does not — it has no
LLM config, and `route=agent` has zero traffic in 6 h), and three separate point-samples read
as distributions.

**Standing rule for anyone continuing this work:** on this system, an idle reading is not
evidence of available capacity. Establish the concurrency limit and the residency set
*first*; only then does a utilisation number mean anything.

---

## 5. What changes in the plan

| plan section | disposition |
| --- | --- |
| §1e "just add a compute budget is wrong" | **Stands, for a better reason.** Not merely that the GPU looked idle, but that utilisation is an invalid calibration input here. |
| §3 thesis (serialized inference time) | **Replaced** by this document's §1. |
| Phase E0 Gate A | Already retracted; superseded — cost-per-call was never the right axis. |
| Phase E0 Gate B | **Stands, unaffected.** 0 of 10 goals concerned Orion's state; `goal_formulate` is a translator, not a generator. Load-independent. |
| Phases 1–5 | **Not authorised.** They were sequenced against the old thesis and need re-deriving against allocation-not-metering. |
| §7 throttle | **Stands, and earned its keep.** The parking lot absorbed eleven findings that would each have been a squirrel. |

---

## 6. The open question, for Juniper

Everything downstream — repertoire, budget, drives — depends on one decision no measurement
can make:

**Does Orion get a standing claim on a unit of concurrency, and if so which one, against
which competing wanted processes?**

If yes, then a drive is finally a claim on something that runs out, and the allocation is
real enough to be worth arbitrating. If no, then Orion's autonomy stays exactly what today's
data shows it to be — 88,409 dispatches, zero actions, and an arena arbitrating a choice that
costs nothing.

Nothing further should be built until that is answered.

## 7. Worth solving eventually

Fan telemetry would sharpen the ceiling from watts to the thing actually felt — heat and
noise are what make the office unusable, and power is only a proxy. Blocked on the Circe NIC
not reading in; a separate problem, noted so it is not lost.
