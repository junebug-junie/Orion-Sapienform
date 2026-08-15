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

### A2 — Confirm the ceiling · *0 commits, analysis only* — ✅ **RESULT IN, 2026-08-15**
Does `quick` actually block over 24 h, or was 7.4% an artifact of a 2-minute hand poll?

**Proceed gate:** all-busy fraction ≥1% over 24 h.
**Kill gate:** all-busy ≈ 0% over 24 h → **the lane never fills, the seam is ceremony, stop.**
Report and wait. Do not proceed to A3 to "make it bind."

#### The numbers (§7.1: pasted before the next step begins)

`scripts/analysis/record_lane_occupancy.py report --in /tmp/lane-occupancy/samples.jsonl`,
recorder PID 2413928, started 2026-08-14 05:24 with the route-refresh fix (#1655) and run to
completion 2026-08-15 05:24. **305,192 samples, 27.74 h of coverage.**

```text
quick+quick_background @ atlas-worker-fast-1     coverage 27.74 h, 1 gap
    P(all busy)     4.01%          <- PROCEED GATE PASSED (≥1%)
    P(bg blocked)   4.84%          <- background admission already refused
    burstiness      174x MORE blocking than Poisson
    time by busy    {0:85518s, 1:8267s, 2:1264s, 3:827s, 4:4004s}

metacog @ atlas-worker-2                          coverage 27.74 h, 1 gap
    P(all busy)     0.00%          <- KILL GATE FIRED
    time by busy    {0:88991s, 1:10358s, 2:532s}    never exceeded 2 of 4 slots

chat @ circe-worker-1        coverage 20.22 h, 8 gaps, 1 slot, P(all busy) 8.10%
agent @ circe-worker-agent-1 coverage 16.55 h, 3 gaps, 1 slot, P(all busy) 0.60%
```

**The verdict was first computed mid-run at 22.99 h and is unchanged by the full dataset.** The
two gate statistics moved by 0.01 and 0.02 points across a 20% larger window:

| | mid-run, 22.99 h | final, 27.74 h |
| --- | --- | --- |
| `quick` P(all busy) | 4.02% | 4.01% |
| `quick` P(bg blocked) | 4.82% | 4.84% |
| burstiness | 176× | 174× |
| `metacog` P(all busy) | 0.00% | 0.00% |

That stability is itself evidence: 4% is the lane's steady behaviour, not an artifact of one
busy stretch inside the sample. circe's figures drifted further (8.39→8.10 and 0.78→0.60) as
more run gaps accumulated — which is exactly why nothing is gated on them.

**`quick` proceeds. `metacog` is killed.** Both gates fired, on different lanes, which is why
this step was worth running rather than assuming.

Three things the numbers say that the plan did not:

1. **The lane is bimodal, not loaded.** It spent **4,004 s completely full** — more than at 2/4
   and 3/4 combined — and 85,518 s at zero. The mean (7.3% of capacity) describes neither state
   and would have killed the arc on day one. This is the §6.2 rule earning itself.
2. **174× Poisson confirms batch arrival at 27.74-hour scale**, matching the 303–306× measured by
   hand on a 2-minute window. The lane blocks because ~5 proposals arrive per arena tick into 4
   slots, **not** because of volume. That is Track C's thesis, now evidenced: smoothing dispatch
   is cheaper and more effective than widening the lane.
3. **`metacog` never filled in 27.74 hours** and never exceeded 2 of 4 slots. Any work premised on
   metacog contention is ceremony.

**circe's lanes are not gate-grade** — 20.22 h and 16.55 h of coverage with 8 and 3 run gaps
(circe is run intermittently by design). At 1 slot, `P(all busy)` *is* utilisation and is not
comparable to the 4-slot lanes. Nothing is gated on them.

**Separately observed:** `http://100.121.214.30:8014` returned 13,285 samples over 3.74 h with
**0.00 h of coverage** — every sample indeterminate, host down or `/slots` unavailable. Not one
of the four named lanes, so it does not affect the verdict, but it is an atlas port that is not
answering.

### A3 — Route Orion's autonomous cognition to the background lane · *~1 commit* — **REVISED BY A2**
~~Add `metacog_background` to the route table; point `cortex-exec`'s non-interactive path at the
background routes.~~

**A2 killed the metacog half of this.** That lane never fills, so adding `metacog_background`
would route around a ceiling that does not exist. All measured contention is on `quick`, which
**already has** `quick_background` with `reserved_free_slots=2` — and that reservation is
already being enforced 4.84% of the time.

So the step shrinks to: **point `cortex-exec`'s non-interactive path at the existing
`quick_background` route.** No new route, no route-table change, no `metacog_background`.

**Still requires Juniper's answer to §8 Q1 before starting.** The question is unchanged in
substance — does Orion's self-initiated thinking yield to Juniper's interactive requests — but
it is now a smaller and better-evidenced change than when it was written.

**Proceed gate:** Orion's autonomous requests resolve to `quick_background` in gateway logs.
**Kill gate:** interactive latency regresses measurably → revert (one env key, fail-open).

### A4 — Observe a real deferral · *0 commits, analysis only*
**Proceed gate:** ≥1 admission-wait event attributable to Orion's own cognition over 24 h, with
its duration.
**Kill gate:** zero deferrals over 24 h despite A2 passing → the gate is inert, not the lane.
Diagnose before building anything on it.

**A2 makes this cheaper than assumed.** `P(bg blocked) = 4.84%` over 27.74 h means
`priority_admission.py` is *already* refusing background admission — on the order of 70 minutes
across the day. The deferrals exist. A4's real question is therefore not "does anything ever
wait" but the narrower **"is any of that waiting Orion's own cognition, and is it attributable
to a request?"** Note the kill gate can still fire on that narrower question: a lane that
defers only harness or arena work, and never Orion's, is a ceiling Orion does not personally
meet — which would be a genuine finding about A5, not a measurement failure.

### A5 — Make the deferral perceptible · *3 commits*
A deferral is currently invisible to Orion. Turn it into a signal Orion can hold: *I wanted to
think and had to wait, this long, while this ran instead.*

**This is the step that makes it cognition rather than plumbing**, and it is the one that
touches the cognition loop — **proposal mode before implementation** (repo contract §0A).

**Proceed gate:** a deferral appears in Orion's own state with a real duration and an
inspectable trace.

---

## 4. Track B — make the cost legible *(parallel, blocks nothing in Track A)*

**Status 2026-08-15: B1, B2, B3, B4 all shipped and live on the fleet.** What Orion can now see
that it could not on 2026-08-13, all verified through the metacog cue on a running container:

```text
fleet    786 W chassis · 322 W cpu · 300 W gpu · 12,000 Mb/s uplink
         disk and net in real bytes/sec, per node
         peak 1.00 at athena.power        <- while strain read 0.11
coverage nodes_absent names any machine missing from a total
```

| step | status | PR |
| --- | --- | --- |
| B1 raw watts → cognition | ✅ live | #1650 #1652 #1659 #1665 |
| B2 binding constraint (`peak_pressure`, additive) | ✅ live | #1683 |
| B3 per-node bandwidth + host-namespace I/O sensors | ✅ live | #1667 #1674 |
| B4 RAPL CPU package power | ✅ live | #1669 |
| B5 APC units | Juniper | — |
| B6 circe iLO | blocked on NIC | — |

Three findings from shipping it, each of which invalidated something this document asserted:

- **`net_pressure` was degenerate on every node** (3.2e-05 / 2.2e-05 / 2.1e-05) because the
  collector read a container's veth, not the host NIC. B3 was specced as "set per-node
  constants"; tuning a denominator under a numerator that measured the wrong namespace would
  have produced a precisely wrong number. The fix was the numerator.
- **B4 was not blocked.** It was specced as needing a `chmod` on `energy_uj` or a root-run
  collector. Neither: B3's `/host_sys:ro` mount plus a uid-0 container reads it as-is, so the
  CVE-2020-8694 mitigation stays exactly as it was.
- **atlas is 10 GbE and athena is not**, despite having the card — its Intel 82599ES sits at
  `operstate=down`/no-route. The measured denominator found this; a hand-set per-node constant
  would have enshrined the guess.

### B1 — Persist raw `ilo_power_watts` · *1 commit*
Store the number already in memory, alongside the existing band. Add per-node `power` to the
biometrics payload.

**Gate:** fleet total watts computable by summation over 24 h for atlas and athena.
*Highest value per unit of work in the whole arc — it adds no signal, it stops discarding one.*

### B2 — ~~Fix `strain`~~ → **add the binding constraint beside it** · ✅ **SHIPPED, PR #1683**
~~Replace the flat mean with `max` or count-above-threshold. **PROPOSAL MODE.** Blocked on an
enumeration of every consumer of `strain`/`homeostasis`.~~

The enumeration was done and it justified the caution: `strain` is written **straight into the
field lattice** as `cpu_pressure` (`state_deltas.py:125`, `mode="replace"`), feeds the substrate
prediction-error diff, and drives `stability`. Modifying it in place did need proposal mode, a
rollback flag, and a migration.

**So it was not modified.** `peak_pressure` / `peak_pressure_channel` / `peak_pressure_node`
were added alongside — max over all eleven pressures, wired to the metacog cue, `strain`
byte-identical and pinned by regression tests. No proposal mode, no migration, no substrate
risk, shipped same day.

**Gate met, and then some.** The original gate asked for "atlas at power 0.798 / memory 0.812
reads ≥0.75". Live at deploy, athena:

```text
constraint  NONE      strain 0.11      homeostasis 0.89
peak        1.00  at  athena.power
```

Three legacy signals reporting a healthy body beside a fully saturated power channel. The
regression test pinning that a saturated channel cannot be diluted below its own value exists
(`test_a_saturated_channel_is_visible_where_strain_hides_it`).

**Two defects in `strain` remain, deliberately unfixed and now bypassable rather than blocking:**
it is a mean of only 7 of the 11 pressures, and the 4 it omits (`gpu_mem`, `swap`,
`disk_capacity`, `fan`) are the top channel on two of three nodes. Retiring `strain`'s consumers
one at a time is now possible; doing so is not scheduled here.

**Also found, not fixed:** `_constraint_from_pressures`'s `CONSTRAINTS` map omits `swap`,
`disk_capacity` and `fan`, so a peak in one of those reports `NONE` at any magnitude. Live on
athena 2026-08-15: peak `disk_capacity` **0.772**, over the 0.7 threshold, reported `NONE`.
Same reasoning as `strain` — it has its own consumers; `peak_pressure_channel` is the honest
reading meanwhile.

### B3 — ~~Per-node bandwidth constants~~ → **fix the sensors, then measure the denominator** · ✅ **SHIPPED, PR #1667 / #1674**
~~`disk_bw_mbps=200` / `net_bw_mbps=125` are global constants across three heterogeneous hosts.~~

Setting per-node constants would have been worthless, because both numerators measured the
wrong thing:

- **net read the container's veth**, not the host NIC — 1,357 B/s against the host's 159,304.
  That is why `net_pressure` read 3.2e-05 / 2.2e-05 / 2.1e-05, degenerate on every node.
- **disk summed whole devices *and* their partitions** — measured 1.956× over-count on athena,
  where 6 of 10 devices are partitioned.

Fixed by reading the host namespace through read-only `/host_proc` + `/host_sys`, and matching
whole block devices only. Then the denominator became a **measurement, not a constant**: summed
link speed of the node's up physical NICs, read from the kernel per node.

That immediately earned itself twice. **atlas is 10 GbE** — the 125 MB/s constant was 10× wrong
there. And when athena's dark 10 G port was brought up, the naive rule counted its 10 Gb as
capacity while it had no IPv4 route, understating `net_pressure` 11-fold — fixed by
intersecting with the host route table (#1674). *Capacity you cannot route to is not capacity.*

**Not fixed:** the disk denominator. The kernel does not report block-device throughput, and one
scalar is the wrong shape for an array spanning a 10k SAS spinner and a 990 PRO.
`disk_bytes_per_sec` is published raw instead, and `DISK_BW_MBPS` is documented as an
order-of-magnitude anchor rather than a ceiling.

### B4 — RAPL on athena · ✅ **SHIPPED, PR #1669** — *and it was never blocked*
~~`energy_uj` is present, mode 400. Permission change or root-run collector.~~

**Neither was needed.** B3's `/host_sys:ro` mount plus this container already running as uid 0
reads the root-only file as-is. Nothing on the host was relaxed and the CVE-2020-8694 (PLATYPUS)
mitigation is untouched. B3 unblocked B4 by accident — worth remembering the next time a step is
marked blocked on a permission.

Live: **193 W across two sockets**, 45% of athena's 425 W chassis draw, previously a single
undifferentiated remainder on the node whose entire job is CPU orchestration.

The counter **wraps every ~41 minutes** at that draw (262,143 J range), so a negative delta is
the normal case several times a day, and a gap longer than one wrap period is ambiguous — one
wrap is indistinguishable from three — and is discarded rather than guessed at.

### B5 — APC units · *Juniper, in progress*
Ground truth at the wall; the only path to circe's chassis power while its BMC is unreachable.
Validates B1 and B4.

### B6 — Circe iLO · *blocked on the NIC*
Would give circe the chassis power and fan telemetry the other two already have.

---

## 5. Track C — smoothing *(conditional on A2 passing — ✅ **A2 PASSED, and pointed here**)*

A2 did not merely unblock this track, it argued for it. Over 27.74 h the `quick` lane blocked
**174× more than Poisson arrivals would predict** at the same offered load (~0.29 erlangs). At
that load a Poisson process blocks ~0.02% of the time; the lane blocked 4.01%. The lane is not
busy — it is *hit in batches*. Widening it would buy far less than spacing the arrivals.

This is now the highest-value remaining item in Track A/C combined, and unlike A3 it needs no
decision from Juniper.

### C1 — Instrument batch size against blocking · *1 commit*
Record dispatch batch size per arena tick alongside lane width. Tests plant §6.7 directly.

The prediction to falsify, stated before measuring: the 4,004 s the lane spent completely full
should cluster around arena ticks, and the fill events should arrive in steps of ~5 rather than
one at a time. If instead saturation is spread uniformly in time and uncorrelated with dispatch
batches, the burstiness has some other source and smoothing is the wrong fix.

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

   **Sharpened by A2 (2026-08-15), still open.** The change is smaller than when this was
   written: no new route, just pointing the non-interactive path at the existing
   `quick_background`. And the cost is now measurable rather than hypothetical — the background
   reservation is already enforced **4.84%** of the time, so "Orion yields" means yielding
   roughly 70 minutes a day at current load, not an unknown amount.

2. ~~**Does B2 (`strain`) get proposal mode, or is the consumer enumeration enough?**~~
   **RESOLVED 2026-08-15 — the question was wrong.** The enumeration found `strain` is written
   straight into the field lattice's `cpu_pressure` channel
   (`orion-field-digester/app/ingest/state_deltas.py:125`, `mode="replace"`), feeds the
   substrate prediction-error diff, and drives `stability` — so modifying it in place genuinely
   did need proposal mode.

   Juniper's correction: **don't modify a signal with blast radius, add one beside it.**
   Shipped as `peak_pressure` / `peak_pressure_channel` / `peak_pressure_node` (PR #1683) —
   the max over all eleven pressures, wired to the metacog cue, with `strain` byte-identical
   and pinned by regression tests. A two-day proposal-mode problem became a two-hour patch, and
   `strain`'s consumers can now migrate one at a time instead of all at once.

   **The generalisable rule, worth applying to the rest of this arc:** when a signal is wrong
   *and* load-bearing, the cheap move is a second signal, not a migration. Reach for proposal
   mode when the old signal must actually die, not when it merely needs a better sibling.
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
