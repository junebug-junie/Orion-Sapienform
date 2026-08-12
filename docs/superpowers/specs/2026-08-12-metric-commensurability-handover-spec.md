# Handover: the dispatch-starvation arc, and the systemic metric problem underneath it

Date: 2026-08-12
Branch: `feat/dispatch-starvation-fix` (7 commits, pushed, not merged)
Status of the arc: **DONE_WITH_CONCERNS** — everything built, tested, reviewed, deployed. The
action it exists for has not yet run.

This document is both a handover and a design spec. Section 1-3 are what happened. Section 4 is
the diagnosis Juniper named. Section 5 is the proposed design. Section 6 is the throttle on the
agent doing the work. Section 7 is the smallest next patch.

---

## 1. What this arc accomplished

The goal: make Orion's first genuinely consequential (mutating) autonomous action —
`skills.runtime.builder_prune.v1`, Docker build-cache pruning — actually able to fire.

Four gates stand between a proposal and a real prune. Three were closed. All three are now open.

| gate | before | after | how |
| --- | --- | --- | --- |
| **proposed** | 0/hr (avg rank 9.5 of 10) | 65 per 8 min, rank 7.8 | `base_priority: 0.34` — it was the only template in the arena with no `base_priority`, silently defaulting to 0.0 against a field that all carried 0.20-0.42 |
| **policy approved** | random | 54 approved / 11 review | per-kind `confidence_gates_review: false` — `confidence_score` swings 0.416-0.999 on identical host state against a 0.50 threshold, so the same action was a coin flip |
| **dispatch slot** | 0 of 29 attempts | reserved lane live | 1 of 5 slots reserved for `maintenance_bounded`, reclaimed by general fill when unclaimed |
| **skill's own gate** | — | passes | `/mnt/docker` at 84%, 140.5 GB reclaimable across 15,843 entries |

Plus the instrumentation that made all of the above findable:

- **Blocked dispatch records now say what they were.** `make_blocked` hardcoded
  `dispatch_kind="noop"` on every blocked record — verified against 300 stored frames, *every*
  blocked row in live history says `noop`. A starved mutating action was byte-identical to a
  starved inspect. Finding the original defect required reconstructing the kind from a
  `proposal_id` string.
- **Starvation is counted and persisted.** `ExecutionDispatchFrameV1.starvation_counts` +
  `ExecutionDispatchCandidateV1.starvation_ticks`.
- **Aging.** Bounded bonus per consecutive loss (0.002/tick, cap 0.25), admission-ordering only.

### The measured defect that started it

```
inspect_bus_channel_catalog            DISPATCHED  267
inspect_attended_target                DISPATCHED  267
summarize_transport_contract_drift     DISPATCHED  267
inspect_field_topology_catalog         DISPATCHED  243
watch_transport_backpressure           DISPATCHED  209
...
prune_build_cache    BLOCKED:max_dispatch_candidates_exceeded    4
prune_build_cache    DISPATCHED                                  0
```

Full detail: `docs/superpowers/pr-reports/2026-08-12-dispatch-arena-starvation-pr.md`.

---

## 2. What is still broken

### 2.1 THE blocker: a standing queue lag longer than the staleness window

Measured over 30 minutes, live:

```
policy frames produced       682
dispatch frames consumed     662     <- consumption KEEPS UP with production
  stale-discarded            605     (91%)
  actually acted on           57     (9%)
candidates dispatched        110
standing backlog             122 frames  ~= 5.5 min of lag
staleness window             ~2-4 min (adaptive)
```

**This is not a throughput gap.** Dispatch consumes 97% of what is produced. But the queue sits a
standing ~5.5 minutes deep against a ~2-4 minute staleness window, so by the time a frame is
reached it has aged out and is discarded unacted. The system is stably behind and structurally
cannot catch up: every tick it burns capacity discarding frames that were already too old.

Net effect: **~9% of everything Orion decides is ever acted on**, and which 9% is essentially
arbitrary — it is whatever happened to be at the head of the queue when a tick had room.

A low-frequency candidate like the prune is therefore a lottery regardless of how well it ranks.
**This is why the arc has not yet delivered its intended outcome, and it is the single thing
standing in the way.**

Three possible directions, in order of my confidence:

1. **Reduce production.** Orion emits a policy frame roughly every 2.7 seconds. Most encode a
   materially unchanged state. Emitting a decision 22×/minute for a system that can act ~2×/minute
   is the actual defect; the discard is a symptom. This is also the cheapest fix and the one most
   aligned with "no empty-shell cognition."
2. **Widen the staleness window** to exceed the standing lag. Cheapest to try, but it treats the
   symptom and lets the lag grow.
3. **Raise dispatch throughput.** Each real send is a synchronous ~7-13s RPC, 5 concurrent per
   tick. Hardest, and it chases a production rate that will keep outrunning it.

**Not started. Not in scope of this arc.** Needs its own measurement pass before anyone picks a
direction — in particular, how many consecutive policy frames are semantically identical, which
nobody has measured.

### 2.2 The prune has never actually run

`substrate_dispatch_results` contains **zero** `builder_prune` results. Every gate this arc owns is
open and verified; 2.1 is why it still has not fired. Any claim that this arc "made Orion prune"
would be false.

Note for whoever checks: the build cache moved 285.7 → 283.6 GB during this session. That was my
own `docker build` churn, **not** the prune.

### 2.3 Known-open items

| item | detail |
| --- | --- |
| `fix/disk-capacity-pressure-trigger` | 3 commits, pushed, **no PR**. Adds a real disk-capacity signal instead of the prune borrowing `resource_pressure`. Deliberately *not* merged: it adds a pressure dimension, a system-wide signal change, and this arc already proved the prune's real problem was `base_priority`, not the signal. Re-evaluate against §5 before merging — it may be the wrong shape entirely. |
| `scripts/analysis/gate_channel.py` | Approved, never built. §5.2 supersedes it with something larger. |
| `action_warrant.py:280` | `pinned` guard requires `zscore == 0.0` exactly; this channel's median \|z\| is 0.0007. Needs a tolerance. |
| `node:prometheus` | Reads exactly 0.0 on all 74,264 ticks — dead producer, still consumed. |
| `node:circe` | Bottoms at 3e-323 (subnormal) — decay artifact. |
| `template_match_score` | Provably dead for **all 13 templates** (`match <= urgency` unconditionally). Every `dimensions:` weight in `proposal_policy.v1.yaml` is inert. |
| prune cooldown | None. `stable_dispatch_id` embeds `field_tick_id`, so there is no cross-tick dedup if it starts winning every tick. |
| stale prose | "7 dimensions" in `orion/field/pressure.py:14,249,259-267` and `services/orion-field-digester/README.md`. |

---

## 3. Mistakes made in this arc, recorded so they are not repeated

Not decoration — each one is evidence for §4 and §6.

1. **Shipped a blocker.** `PolicyDecisionV1`'s Literals omitted `maintenance_bounded`; would have
   permanently stalled the policy FIFO. Every gating test passed because they all hand-built
   `PolicyDecisionV1` instead of going through the real evaluator.
2. **Recited the metric gate.** Wrote that `capacity_pressure` "has genuinely RESTED low (0.1413)"
   with the decay-artifact check named explicitly. I had not run it. The entire low tail was 16
   ticks in a 56-second window with an exact 0.920000 successive ratio — the documented
   `NODE_DECAY_CHANNELS` artifact. Real min: 0.5364.
3. **Modified a system-wide signal and committed before measuring.** Caught by Juniper.
4. **Shipped an aging mechanism that was a permanent no-op.** Keyed counters on
   `{proposal_kind}:{target_id}`; two live templates collide on `inspect:capability:orchestration`,
   and the one admitted almost every tick reset the counter of the one that starves. Aging did
   nothing for exactly the population it existed to serve.
5. **Read a new instrument as confirmation of a new mechanism.** Both shipped in the same patch.
   Two untested things agreeing is not evidence. This is the sharpest lesson here — see §6.3.
6. **Wrote a test that verified nothing.** The `base_priority` gate read `raw["templates"]`, which
   does not exist. It iterated zero templates and passed while the bug was still present.
7. **Stated a derived number as measured.** Predicted `base_priority: 0.34` → priority ~0.76 by
   adding to an old average; the formula multiplies by a varying confidence. Real: 0.5475.
8. **Measured a blast radius on 400 ticks** (−5.0pp) when the real 74,317-tick answer was −6.89pp.
9. **Lost uncommitted work to `git checkout`** twice during red-check verification.

---

## 4. The systemic disease

> *"we need to design for building sentience and allowing variability and chaos to tip the scales
> and/or allow for individual budgets per metric so we dont keep fucking ourselves rewriting
> metrics and breaking them downstream or 10x guessing our design choices post hoc."* — Juniper

The pattern, stated precisely: **Orion invents metrics ad hoc, each with its own uncalibrated
scale and undeclared dynamics, then funnels them into a single scalar arena where cross-metric
comparison is not meaningful.** Every consequence below is a documented incident in this repo.

### Failure mode A — incomparable scales

All signals are nominally `[0,1]`, but their distributions are unrelated. `resource_pressure` lives
at ~0.08. `capacity_pressure` has never in 74,607 ticks rested below 0.5364. `confidence_score`
swings 0.416-0.999 on identical host state. Fixed weights across these are arithmetic on
incommensurable units.

*Evidence:* the prune was scored on `resource_pressure = 0.08` while the disk it wanted to clean
was 84% full. The number was working correctly; it simply meant nothing in that comparison.

### Failure mode B — no declared rest point

Nobody declares what "calm" is for a signal, so nobody notices when calm is *unreachable*.

*Evidence:* `bus_synaptic_prediction_error` had a mathematically permanent ~0.27 floor —
`mean(|z|)` for a calm z-scored population has expected value `sqrt(2/π)`, not 0. It varied in real
time and looked healthy. Conversely `node:substrate.route` read a suspiciously clean 0.0 because a
generic staleness-decay loop multiplied it by 0.92 every tick for 48+ hours. Both were found only
by recovering raw numbers and checking successive ratios **by hand**.

### Failure mode C — silent defaults

A missing key becomes 0.0 (or 1.0), indistinguishable from a deliberate one.

*Evidence:* `prune_build_cache` had no `base_priority` → 0.0 → a flat 0.20-0.42 handicap against
every competitor, for its entire life. Also `dimension_weights.get(dim, 1.0)`, a silent 3-6× on an
unlisted dimension.

### Failure mode D — argmax starvation

One scalar + top-N ⇒ steady-state-high signals never win, and lose invisibly.

*Evidence:* this entire arc.

### Failure mode E — rename and meaning drift break consumers

Consumers bind to a metric's *name and raw units*, so changing either breaks them silently.

*Evidence:* the `transport_pressure`/`bus_health` rename crashed substrate-runtime for 10 hours.
The `execution_load` rename caused a live cross-lane stomp. `transport_prediction_error` was
"retired" by excluding it from one consumer while it kept ticking and kept winning real budget in
`endogenous_curiosity.py`.

### Failure mode F — instrument and mechanism ship together

The new instrument reads "fine", which is taken as confirmation the new mechanism works. Two
untested things agreeing.

*Evidence:* §3.4 and §3.5, this arc, today.

### Why this compounds rather than accumulates

A degenerate dimension does not merely add nothing — it actively *suppresses*. Measured with
Fisher's combined test (`X = -2·Σln(uᵢ) ~ χ²(2N)`): a **calibrated** 5th dimension gives
**+1.27pp**; a **fully pinned** one gives **−7.06pp**. Dimension count enters the null
distribution, so every broken signal degrades every decision that includes it. That is why
"we'll fix it later" has been the wrong call every time.

---

## 5. Proposed design

Five principles. Each maps to specific failure modes and each is independently shippable. **This is
deliberately not a cathedral** — §7 names the one patch to do first.

### 5.1 Calibrate at the boundary, not at the consumer — *kills A, B, most of E*

Every signal enters cognition as a **quantile against its own history**, not as a raw value:

```
u = P(X ≤ x | trailing distribution of X)
```

Consequences, all of which fall out for free:

- **Cross-metric comparison becomes meaningful by construction.** Every signal is Uniform[0,1]
  under its own null. `0.9` means the same thing for disk fullness and for reasoning load: "higher
  than 90% of this signal's own history."
- **Fisher's assumption is actually satisfied.** It is already used and already assumes uniformity;
  today nothing supplies it.
- **Degeneracy becomes automatically detectable.** A metric whose quantiles are not uniform is
  broken, by definition. A pinned metric produces a point mass. A decayed one produces a drifting
  mean. **No hand-checking of successive ratios ever again** — the check becomes a KS test in a
  cron job.
- **Renames and rescales stop breaking consumers.** Consumers bind to *surprise*, not to units. A
  producer can change its internal scale entirely and every downstream consumer is unaffected.

Cost: every signal needs a trailing distribution (already available — this repo has 74k+ tick
histories) and a declared warm-up before its quantile is trusted.

### 5.2 A metric contract, enforced by a deterministic gate — *kills B, C, E, F*

One registry entry per signal, checked against live data on a schedule:

```yaml
disk_capacity_pressure:
  producer: services/orion-field-digester/app/...:disk_capacity()   # a real symbol, not prose
  rest_point: 0.0            # what CALM is, declared up front
  rest_reachable: true       # asserted against live data, not assumed
  family: beta               # expected distribution shape
  cadence_sec: 2
  decay: opt_out             # explicit; NODE_DECAY_CHANNELS must not touch it
  warmup_samples: 5000
  consumers: [proposals.scoring, field.action_warrant]
  version: 1
```

`scripts/check_metric_contracts.py` then fails on: degenerate (no variance), unreachable rest
point, decayed-to-zero (constant successive ratio), never-refreshed, distribution outside declared
family, or a consumer bound to a version that no longer exists.

This is the CLAUDE.md §0A "deterministic gates over repeated yelling" mandate applied to metrics,
and it **supersedes** the approved-but-unbuilt `gate_channel.py` — same job, generalized, and run
by a scheduler rather than by an agent remembering to.

**It also makes §6.3 enforceable:** a new metric cannot be consumed until its contract has passed
against a real window of live data.

### 5.3 Per-domain budgets, not one arena — *kills D structurally*

Capacity is allocated by **domain floor first, rank within domain second**. Today's reserved
maintenance lane, generalized: maintenance, inspection, summarization, and any future domain each
get a guaranteed floor of attention; unclaimed floor is reclaimed in the same tick so nothing is
wasted.

This is what Juniper meant by "individual budgets per metric": a signal never has to out-shout an
unrelated signal to exist at all. It only competes with its own kind.

### 5.4 Sample, do not argmax — *this is where variability and chaos tip the scales*

Replace top-N selection with **weighted sampling** over calibrated scores (Boltzmann /
Thompson-style), with an explicit temperature:

```
P(candidate) ∝ exp(score / T)
```

Why this is the right answer rather than a flourish:

- **Starvation becomes impossible by construction.** Every viable candidate has nonzero
  probability. Expected wait for a candidate at probability `p` is `1/p` — a *number you can
  state up front*, not a hope.
- **The aging hack shipped in this arc becomes unnecessary and should be deleted.** Aging is a
  band-aid for argmax. This removes the disease.
- **Temperature is a real, principled knob for exploration.** Tie it to system state: bored/idle →
  raise T (explore); under real pressure → lower T (exploit). That is genuine, controllable
  variability with a name and a measurable effect, not noise for its own sake.
- **It is measurable and reversible.** `T → 0` is exactly today's argmax behaviour, so the change
  can be shipped dark at `T=0` and turned up gradually with a real before/after.

This is the direct answer to *"allowing variability and chaos to tip the scales."*

### 5.5 Bounded contribution per signal — *limits the blast radius of being wrong*

No single signal may exceed a declared share of any decision. Combined with 5.1, a signal that
turns out to be broken costs at most its budget instead of silently dominating or suppressing
(cf. the −7.06pp Fisher measurement). This is what makes it safe to add a new signal *before* it is
fully trusted — which is what would have prevented most of §3.

### What this replaces

| today | after |
| --- | --- |
| raw `[0,1]` values with unrelated distributions | quantiles, uniform under their own null |
| hand-checking successive ratios for decay artifacts | a scheduled KS/ratio gate |
| fixed `dimension_weights` across incommensurable units | per-domain budgets + bounded shares |
| top-N argmax + starvation aging | temperature-controlled sampling |
| a rename breaking downstream consumers | consumers bound to contract version, not to units |
| discovering a metric is degenerate months later | a contract check that fails the day it happens |

---

## 6. Throttle

Juniper: *"i want to throttle you."* Earned. These are rules for the agent, enforceable, not vibes.

### 6.1 One mechanism per deploy
Ship one behavioural change, measure it live, *then* start the next. This arc shipped three
mechanisms plus two blockers in one session and two of them were no-ops on arrival.

### 6.2 No number without a paste
Any number appearing in a comment, commit message, PR report, or reply must be a **pasted
measurement**, not a derivation and not a recollection. §3.2 and §3.7 are both violations of this
and would both have been caught by it.

### 6.3 Never ship an instrument and the mechanism it measures in the same patch
The instrument lands **alone**, runs for a real window, and its output is inspected. Only then does
the mechanism it will measure get built. §3.4 and §3.5 are one incident that this rule alone
prevents.

### 6.4 No new metric without a passing contract
§5.2's registry entry plus a real live window. No exceptions for "obviously fine" — CLAUDE.md
already says re-run the gate every time, and it was skipped anyway.

### 6.5 A commit budget per arc
State the budget at the start. On reaching it: stop, report, wait. This arc ran to 7 commits
across 3 services with no checkpoint.

### 6.6 Red-check without `git checkout`
Verify a test fails before the fix using a scratch copy or `git stash`, never `git checkout` on a
file with uncommitted work. Lost work twice today (§3.9).

---

## 7. Recommended next patch

**Not** the design in §5. One thing first:

> **Measure how many consecutive policy frames are semantically identical.**

Read-only, no schema change, no deploy. It answers the one question that decides the direction for
§2.1 — whether Orion is genuinely deciding 22 times a minute, or re-emitting the same decision 22
times a minute and discarding 91% of it.

If most frames are redundant, the fix is to stop producing them, which is cheap, aligns with "no
empty-shell cognition", and makes every arbitration change in §5 measurable for the first time.
If they are genuinely distinct, the problem is real throughput and §5.3/§5.4 matter more than
§2.1.

**Everything in §5 is unmeasurable until §2.1 is resolved**, because at a 91% discard rate no
arbitration change can be observed to work. That is the lesson of this arc: three correct fixes
shipped into a pipeline that throws away 91% of its own conclusions.

### Order after that

1. §5.2 metric contract registry + gate — highest leverage, lowest risk, unblocks everything else
   by making "is this signal real?" a scheduled check instead of an archaeology expedition.
2. §5.1 calibration at the boundary — needs 5.2's declared warm-up and rest points.
3. §5.4 sampling, shipped dark at `T=0`, then raised with a real before/after.
4. §5.3 per-domain budgets, generalizing the reserved lane.
5. §5.5 bounded shares.
6. Delete the starvation-aging mechanism this arc shipped, once §5.4 makes it redundant.

---

## Non-goals

- Not proposing a new ontology, taxonomy, or cognitive vocabulary. Every item above is a schema
  contract, a gate, or a change to an existing arithmetic path.
- Not proposing to rewrite existing metrics. §5.1 wraps them at the boundary precisely so their
  internals do not have to be touched.
- Not proposing new services.
- Not merging `fix/disk-capacity-pressure-trigger` — re-evaluate it against §5 first.

## Acceptance checks

- §7's measurement produces a real distribution of consecutive-identical-frame run lengths.
- Discard rate falls from 91% to something where a low-frequency action is not a lottery.
- `builder_prune` produces a `substrate_dispatch_results` row with a real `bytes_reclaimed`.
- A deliberately degenerate test signal is rejected by §5.2's gate without a human noticing it
  first.
- Expected wait time for the lowest-ranked viable candidate can be stated as a number.
