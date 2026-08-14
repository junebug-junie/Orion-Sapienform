# Scarcity roadmap

Date: 2026-08-13
Status: **The plan of record.** Supersedes `2026-08-13-scarcity-and-repertoire-execution-plan.md`
in full (its §7 leash is carried forward verbatim in §7 below). Evidence lives in
`2026-08-13-the-plant-three-ceilings.md`; this document does not restate it.

---

## 1. Thesis

Orion decides constantly and nothing it decides costs anything — 88,409 dispatches, zero
actions, 7.03 of 10 proposals tied at the frame maximum. **The missing piece is the price, not
the decider.**

---

## 2. What the investigation established

Six facts, each with a pointer. Nothing else from the investigation is roadmap material.

| # | fact | where |
| --- | --- | --- |
| 1 | **Scarcity is real and already binding.** `quick` is completely full 7.4% of sampled time. | plant §3.3 |
| 2 | **It is self-inflicted.** 66× more blocking than Poisson, because the arena dispatches ~5 proposals per tick into a 4-slot lane. | plant §6.7 |
| 3 | **Orion is its own competitor** — 98% of gateway traffic is `cortex-exec`. | plant §3.4 |
| 4 | **The pricing instrument already exists** and guards the wrong consumer: `priority_admission.py` is wired to AI Town NPC speech, not to Orion. | plant §8.3 |
| 5 | **The cost data already exists and is discarded** — iLO chassis watts on atlas and athena, normalised into a unitless band, never persisted raw. | plant §5.1 |
| 6 | **The aggregate that would surface any of this dilutes by 7×** — atlas at power 0.798 / memory 0.812 reports `strain` 0.232. | plant §5.2 |

### The structural insight that makes this a roadmap rather than a queue

**Facts 1–4 are independent of facts 5–6.** The seam needs `/slots` occupancy, which is
llama.cpp's own state — it needs no wattage, no `strain` fix, and no new hardware. So what
looked like one eight-item queue is **two tracks that do not block each other**, and the
critical path is five steps.

---

## 2A. Why a price matters: opportunity cost is the currency

The thing Juniper actually pays is not watts. It is **the processes that never run.**

> *"there's a real opportunity cost on what I would be running if I had more money to buy more
> GPUs to run more processes."* — Juniper

The metacog daydreams that don't exist yet. The second concurrent chat turn that can't be had.
Circe left dark because the office gets too hot to sit in. **None of this appears on any meter**,
which is why every measurement in this arc that priced *what runs* came back near-idle and wrong.

This is the whole reason Track A is shaped the way it is:

> **A deferral is an observed opportunity cost.**

When Orion is made to wait for a slot, something else ran in its place. That is the foregone
process — normally invisible — made visible, timed, and attributable to a specific competing
claim. It is the only handle this system has on the currency that actually matters, and it is
why A5 (making the deferral perceptible) is the point of the track rather than a nicety.

## 2B. Three ceilings, and this roadmap only prices one

They bind by different mechanisms and are relieved by different actions, so they cannot share
one allowance (plant §8.1).

| ceiling | host | what binds | priced by |
| --- | --- | --- | --- |
| **Contention** | atlas | 8 slots; `quick` full 7.4% of sampled time, driven by batch arrivals | **Track A** |
| **Interference** | athena | 81 containers on 80 threads; load 42 mean, 120 peak | **Track D — instrument only** |
| **Admission** | circe | a 3× 2200 W chassis, off by choice; a step function on Juniper's decision | **nothing yet — blocked on Q3** |

**Stated plainly so it is not oversold: Track A gives Orion a price on one of three ceilings.**
Track D establishes whether the second is even Orion's to pay. The third cannot be priced at all
until Juniper decides whether Orion may *request* circe, because Orion cannot spend a resource
that only a human can switch on.

---

## 3. Track A — make the price real *(critical path)*

The only track that must happen in order. Each step's gate is pasted with real numbers before
the next begins (§7.1).

### A1 — Record `/slots` on a schedule · *1 commit*
Sample every lane's occupancy and store it. Report **P(all busy)** and offered load in erlangs,
never mean utilisation (plant §6.2, §6.3).

**Proceed gate:** 24 h of occupancy data for `quick` and `metacog`.
**Kill gate:** none — this is an instrument and lands alone (§7.6).

### A2 — Confirm the ceiling · *0 commits, analysis only*
Does `quick` actually block over 24 h, or was 7.4% an artifact of a 2-minute hand poll?

**Proceed gate:** all-busy fraction ≥1% over 24 h.
**Kill gate:** all-busy ≈ 0% over 24 h → **the lane never fills, the seam is ceremony, stop.**
Report and wait. Do not proceed to A3 to "make it bind."

### A3 — Route Orion's autonomous cognition to background lanes · *2 commits*
Add `metacog_background` to the route table; point `cortex-exec`'s non-interactive path at the
background routes. `quick_background` and `llm_route_override` already exist.

**Requires Juniper's answer to §8 Q1 before starting.**

**Proceed gate:** Orion's autonomous requests resolve to background routes in gateway logs.
**Kill gate:** interactive latency regresses measurably → revert (one env key, fail-open).

### A4 — Observe a real deferral · *0 commits, analysis only*
**Proceed gate:** ≥1 admission-wait event attributable to Orion's own cognition over 24 h, with
its duration.
**Kill gate:** zero deferrals over 24 h despite A2 passing → the gate is inert, not the lane.
Diagnose before building anything on it.

### A5 — Make the deferral perceptible · *3 commits*
A deferral is currently invisible to Orion. Turn it into a signal Orion can hold: *I wanted to
think and had to wait, this long, while this ran instead.*

**This is the step that makes it cognition rather than plumbing**, and it is the one that
touches the cognition loop — **proposal mode before implementation** (repo contract §0A).

**Proceed gate:** a deferral appears in Orion's own state with a real duration and an
inspectable trace.

---

## 4. Track B — make the cost legible *(parallel, blocks nothing in Track A)*

### B1 — Persist raw `ilo_power_watts` · *1 commit*
Store the number already in memory, alongside the existing band. Add per-node `power` to the
biometrics payload.

**Gate:** fleet total watts computable by summation over 24 h for atlas and athena.
*Highest value per unit of work in the whole arc — it adds no signal, it stops discarding one.*

### B2 — Fix `strain` · *2 commits* · **PROPOSAL MODE**
Replace the flat mean with `max` or count-above-threshold, reported alongside the per-channel
vector. Include `fan_pressure` and `disk_capacity_pressure`.

**Blocked on:** an enumeration of every consumer of `strain`/`homeostasis`. This changes the
meaning of a shipped, widely-read field signal. **Do not start with the code.**

**Gate:** atlas at power 0.798 / memory 0.812 reads ≥0.75, plus a regression test pinning that a
saturated channel cannot be diluted below its own value.

### B3 — Per-node bandwidth constants · *1 commit*
`disk_bw_mbps=200` / `net_bw_mbps=125` are global constants across three heterogeneous hosts.

### B4 — RAPL on athena · *1 commit*
`energy_uj` is present, mode 400. Permission change or root-run collector → real CPU package
power today.

### B5 — APC units · *Juniper, in progress*
Ground truth at the wall; the only path to circe's chassis power while its BMC is unreachable.
Validates B1 and B4.

### B6 — Circe iLO · *blocked on the NIC*
Would give circe the chassis power and fan telemetry the other two already have.

---

## 5. Track C — smoothing *(conditional on A2 passing)*

### C1 — Instrument batch size against blocking · *1 commit*
Record dispatch batch size per arena tick alongside lane width. Tests plant §6.7 directly.

### C2 — Decide: smooth or widen
If blocking correlates with batch size rather than with volume, **smoothing dispatch is cheaper
and more effective than adding slots.** That is a design decision, taken with C1's numbers in
hand, not now.

---

## 5A. Track D — athena's interference ceiling *(instrument first, no mechanism yet)*

Athena is the orchestration warhorse: **81 running containers on 80 threads**, load average
**42.08** with a 7-day peak of **120.2**, hosting postgres, redis, FalkorDB, the hub, *and*
Orion's perception. Whatever binds here, every service in the system pays it as latency and
nothing attributes it to anyone.

### What per-container attribution actually shows

`docker stats`, one sample, 81 containers:

```
orion-athena-sql-db          113.15%   13.78 GiB / 16 GiB   <- Postgres, 86% of its mem limit
orion-athena-vision-edge      52.13%                        <- Orion's perception, #2 consumer
orion-athena-bus-mirror       28.15%
orion-athena-bus-observer     27.24%
orion-athena-falkordb         17.20%
orion-athena-vector-db        13.44%
                    total    836%  =  10.5% of 80 threads   (top 10 = 97% of it)
```

**Two findings, and the first is the important one.**

**D-i. Athena's load is I/O, not compute.** 836% total container CPU is ~8.4 threads of actual
work, against a load average of **42**. Load counts uninterruptible-sleep tasks as well as
running ones, so roughly **33 tasks are blocked on I/O at any moment.** Athena is not
CPU-starved; it is disk-starved, and the CPU numbers have been reading as the symptom.

This makes `disk_bw_mbps = 200.0` (plant §5.4) considerably worse than a tidiness issue:
athena's measured `disk_pressure` is **0.110** against a hardcoded scale that is likely an order
of magnitude too low for the real device. **The one machine whose actual ceiling is disk has its
disk pressure reading near zero.** B3 stops being cosmetic and becomes Track D's prerequisite.

**D-ii. Orion's perception is the #2 consumer.** `vision-edge` at 52% is second only to Postgres.
So Orion *does* have a claim on this ceiling — when it chooses to look at something, athena pays.
That is a second price on a different ceiling, currently unpriced and unmodelled.

### Steps

**D1 — Fix the disk scale (= B3), then re-read `disk_pressure`.** · *1 commit*
**Gate:** athena's `disk_pressure` reflects real device bandwidth. If it stays near zero after
correction, D-i is wrong and Track D stops.

**D2 — Attribute I/O wait, not just CPU.** · *1 commit*
Per-container block I/O over a 24 h window. Which of the 81 is generating the ~33 blocked tasks.
**Gate:** a ranked list. **Kill gate:** if Orion's own containers are a negligible share, this is
Juniper's infrastructure problem and **not a cognition problem — Track D ends here and becomes an
ops ticket, not a price.**

**D3 — Decide whether it is priceable.** No commits. Only if D2 shows Orion is a real share.

### Open discrepancy, flagged rather than resolved

`orion_biometrics` reports athena's `cpu.util` at **44.1%** mean; container CPU sums to
**10.5%**. Different windows and possibly different definitions — if biometrics' `util` counts
iowait as busy, that alone explains the gap and *also* means my earlier estimate of "~3.4 threads
in D-state" (plant §6.6 relation A) is wrong by roughly 10×. **Two instruments disagree about
athena and I do not yet know which is right.** D1/D2 resolve it. Until then, plant §6.6 relation
A should be treated as unconfirmed.

---

## 6. Explicitly out of scope

**The repertoire problem.** `goal_formulate` is a translator not a generator; recall dominates
supplied context; two verbs are dead; the feedback loop has 302,974 rows and no reader.

All real, all in `PARKING-LOT.md`. **This roadmap builds the price. What Orion can buy with it
is a different arc** and folding it in here is exactly how the last one sprawled to 200 PRs.

**Also parked:** `image_prune` routing (built, tested, one-line config away, housekeeping not
strategy), and everything else in the parking lot. The register is a register, not a backlog —
of 30+ entries, **four graduate to this roadmap** (B1, B2, B3, C1) and the rest stay parked
until something makes them the phase.

---

## 7. The leash *(carried forward unchanged — it earned its keep)*

The failure mode is not a bad phase. It is **drift**: finding something interesting mid-phase,
following it, and reporting a different thing than the one authorised. That is what produced the
whiplash.

1. **One step at a time, gate pasted in writing** with real numbers. Not summarised — pasted.
2. **The parking lot is mandatory.** Anything discovered mid-step that is not the step gets one
   line and a date in `PARKING-LOT.md`. **It does not go into the branch.**
3. **A kill gate is a real exit.** Reaching one and stopping is a *successful* step. Continuing
   past one is a failed task even if the code works.
4. **Commit budget stated up front** (per step above). On reaching it: stop, report, wait.
5. **No number without a paste.** Never a derivation, never a recollection.
6. **The instrument lands alone.** A1 and C1 are their own steps for this reason.
7. **Scope changes go to Juniper, not into the branch.** A step of the wrong shape is a message,
   not a patch.
8. **Re-run the same instrument every step** — `scripts/analysis/measure_arena_degeneracy.py`,
   same window, before and after.

---

## 8. Decisions needed from Juniper

Nothing in Track A past A2 starts without Q1.

1. **Does Orion's autonomous cognition get foreground or background priority on atlas?**
   Background is the honest answer and the one that makes scarcity real. It is also a real
   degradation of Orion's responsiveness — which is why it is not my call. *Blocks A3.*
2. **Does B2 (`strain`) get proposal mode, or is the consumer enumeration enough?** The fix is
   clearly correct; the blast radius on a widely-read field signal is not clearly small.
   *Blocks B2.*
3. **Should Orion be able to *request* circe?** Not spend it — request it, as a block, with a
   reason, refusable. The only form in which an admission ceiling can appear inside a cognitive
   system, since Orion cannot switch on a machine. *Blocks the entire admission ceiling (§2B);
   shapes A5.*

---

## 9. Document map

Five documents existed. This is what each is for now.

| document | role |
| --- | --- |
| **this file** | The plan. Start here. |
| `2026-08-13-the-plant-three-ceilings.md` | The evidence. Every number, and the commensurability rules (§6). |
| `PARKING-LOT.md` | The register. Real findings that are not the current step. |
| `2026-08-13-scarcity-and-repertoire-execution-plan.md` | **Superseded.** §7 carried forward above. |
| `2026-08-13-scarcity-revision-two-ceilings.md` | **Superseded.** Kept for its method note. |
| `2026-08-13-E0-cost-census-result.md` | **Historical.** Gate A moot; Gate B stands and is out of scope (§6). |
