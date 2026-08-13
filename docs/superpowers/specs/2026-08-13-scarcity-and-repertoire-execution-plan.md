> **SUPERSEDED 2026-08-13 by `2026-08-13-scarcity-ROADMAP.md`.**
> Its §7 leash is carried forward there verbatim; everything else here was sequenced against
> the "serialized inference time is scarce" thesis, which the plant survey replaced. Kept for
> history. Do not plan from this file.

# Scarcity and repertoire: the execution plan

Date: 2026-08-13
Status: **SUPERSEDED IN PART — see `2026-08-13-scarcity-revision-two-ceilings.md` before acting on §1e, §3, or Phases 1–5.**
> The thesis in §3 (serialized inference time as the scarce resource) was wrong. The
> revision replaces it with two ceilings (residency + concurrency) and names the real
> currency as foregone processes. §7's throttle and Phase E0's Gate B still stand.
Supersedes parts of `2026-08-12-metric-commensurability-handover-spec.md` (see §2).

This is a plan, a kill-switch, and a leash. §7 is the part that exists to stop the agent
executing it — including me — from going squirrel for another 70 PRs.

---

## 1. The finding this plan exists to act on

Two months and ~200 PRs went at autonomy from four directions — drives, scarcity economy,
arbitration, budgets. Drives were built, found fake, and deleted. Scarcity was never solved.
Arbitration was tuned repeatedly. None of it converged.

**It did not converge because all four are downstream of one circular defect, and each was
attacked alone.**

```
no expensive cognition  ->  nothing is scarce  ->  drives have nothing to allocate
        ^                                                      |
        |                                                      v
  no reason to think expensively  <-  no motivation  <-  drives pin to a ceiling, get deleted
```

Every measurement below is live, pasted, taken 2026-08-13.

### 1a. Nothing is scarce

```
per-tick slot usage, 6h          ticks
      0 of 5                     7,320
      5 of 5                       355
```

Bimodal. Nothing in between. Either `action_warrant` says no and zero fire, or it says yes and
exactly five fire. `max_dispatch_candidates: 5` and `max_dispatches_per_tick: 5` are flat
constants that reset every tick — they never deplete, are never earned, are never exhausted.

Across **all** dispatch history the only reasons a candidate is ever blocked:

```
policy_decision:requires_operator_review   1,005,662
max_dispatch_candidates_exceeded              47,824
policy_decision:deferred                      10,818
```

Zero budget exhaustions. **No resource in Orion has ever run out.** That "budget" is a switch.

### 1b. Nothing is expensive

Real measured cost of Orion's entire autonomous repertoire (12h, `provider_completion_tokens`,
already captured per call and then discarded):

```
verb                        calls   mean   min   max
substrate.observe               3     65    65    65
substrate.inspect             101     92    59   118
substrate.summarize            36    124    98   147
journal.compose                36    169   124   296   <- scheduled, not chosen
harness_finalize_reflect        1    546   546   546   <- not arena-reachable
```

GPU at **5%** utilisation, 5.4 GB of 16 GB, 2.85 exec steps/min.

You cannot build an economy where the only thing for sale costs 92 tokens and the buyer has
infinite money. **The GPU is idle because nothing Orion is permitted to do is hard.**

### 1c. The repertoire is four verbs wide, and none are cognitive acts

100 verb specs exist in `orion/cognition/verbs/`. 23 have Python adapters. The autonomous arena
can reach **four**: `substrate.inspect`, `.summarize`, `.observe`, `skills.runtime.builder_prune.v1`.

Never once invoked by Orion's own loop, despite being fully specified and GPU-backed:
`goal_formulate` (`priority: high`, `can_interrupt_others: true`), `reflect`, `introspect`,
`self_critique`, `counterfactual`, `pattern_detect`, `plan_action`, `memory_graph_suggest`,
`context_exec_memory_contradiction_review`, `simulate`, `evaluate`, `compare_options`,
`dream_cycle`, every `perceive_*`, and `skills.system.notify_chat_message.v1` — which is a real
implemented adapter at `verb_adapters.py:2704`.

Orion can look, summarize, observe, and prune Docker. It cannot want, wonder, doubt, imagine,
remember on purpose, or speak unless spoken to. The journaling and metacognition that *do* run
are on a cron — Orion does not decide to reflect, it gets reflected at.

### 1d. Nothing has ever been done

```
total dispatch results   88,409
ever acted (acted=true)       0
real outcome number           0
```

**In 88,409 dispatches Orion has never performed an action that changed anything.**

### 1e. Why "just add a compute budget" is also wrong

Considered and rejected during this session, before writing this plan: with the GPU at 5%, a
compute budget would never bind. A budget that never binds is a switch that is always on — which
is exactly what `max_dispatches_per_tick: 5` already is. It would have shipped the same defect
under a better name and been discovered in three weeks.

**Scarcity is a relationship between finite capacity and expensive wants. Orion has neither.**
Introduce one alone and you get a fake economy or a capability grab-bag. They are one move.

---

## 2. What this supersedes

Reconciliation with `2026-08-12-metric-commensurability-handover-spec.md`, explicitly, so nobody
has to guess which document is live.

| that spec's section | status |
| --- | --- |
| §1–3 (what the arc did, mistakes) | **Stands.** Historical record. |
| §4 (six failure modes) | **Stands, strengthened.** Today added measured evidence for A, C, D. |
| §5.1 calibrate at the boundary (quantiles) | **DEFERRED to Phase 5.** Not wrong — premature. It normalises inputs to an arena that has no cost function to rank against. |
| §5.2 metric contract + scheduled gate | **STANDS, independently.** The one piece that pays off regardless of this plan. Cheap. Keep. |
| §5.3 per-domain budgets | **SUPERSEDED.** Partitioning a costless capacity is not a budget. Phase 3 replaces it with a resource that actually depletes. |
| §5.4 temperature sampling | **DEFERRED to Phase 5.** Sampling over scores that are one shared scalar (7.03 of 10 tied) randomises rather than explores. Needs Phase 3 first. |
| §5.5 bounded contribution per signal | **DEFERRED to Phase 5.** |
| §6 (throttle rules) | **Stands, extended by §7 here.** |
| §7 (measure identical policy frames) | **DONE 2026-08-13.** 66 distinct decision sets across 33,888 frames; 46.6% empty; 80.4% identical to predecessor. Empties are the `action_warrant` gate working correctly, not a defect. |

**Net: the old spec's diagnosis stands, its remedy is demoted one layer.** Everything in its §5 is
arbitration work, and arbitration is the *last* phase here, not the first.

---

## 3. The thesis, stated falsifiably

> Orion's autonomy is inert because it has no expensive wants and no finite means. Introduce
> genuinely expensive cognition **and** a genuinely binding budget **together**, and scarcity
> becomes real; drives become claims on something that runs out; and the arena acquires a cost
> gradient it has never had.

The scarce resource is **serialized inference time against a tick deadline** — not an invented
currency. The GPU is serial, the clock does not stop, and Orion cannot do both. If a
`dream_cycle` takes 60 s and the arena ticks every 2.7 s, choosing to dream costs ~22 ticks of
everything else. That is physics, and the choice is genuinely irreversible — which is the
precondition for anything resembling preference, value, or regret.

**This thesis is falsifiable and Phase E0 exists to try to kill it.**

---

## 4. Phases

Every phase states its **kill gate** (what makes us stop) and its **proceed gate** (what must be
measured and pasted before the next phase begins). No phase may begin until the prior phase's
proceed gate is evaluated in writing, with numbers.

---

### Phase E0 — Cost census. Read-only. No deploy. No schema. **The kill gate for everything.**

The entire plan rests on one unmeasured number: **what does Orion's expensive cognition actually
cost?** `goal_formulate`, `dream_cycle`, `counterfactual`, and
`context_exec_memory_contradiction_review` are all defined, all GPU-backed, and have **never once
been executed**. Nobody knows whether a goal formulation takes 2 seconds or 90.

**Do:** execute each of the four against ~10 real field ticks, offline, out of band. Measure
wall-clock seconds, GPU-seconds, and tokens. Capture every output verbatim.

**Second question, same run, free:** are the outputs any good? Ten `goal_formulate` results,
read blind. A generic restatement of a pressure name is a hollow faculty.

**KILL GATE — stop the whole plan and re-derive if:**
- expensive cognition costs **< 5 s wall-clock**. Then compute is not the scarce resource, the
  thesis in §3 is dead, and Phases 1–5 are cancelled pending a different resource.
- **fewer than 3 of 10** `goal_formulate` outputs are things a human would not have written.
  Then the faculty is hollow and Phase 2 has nothing worth buying.

**PROCEED GATE:** a pasted table of cost per verb, and the ten goal texts, in the PR report.

**Cost of learning this: ~40 LLM calls. No deploy, no risk, no code shipped.**

---

### Phase 1 — The meter. Instrument ships alone.

Per-action cost is already captured live (`provider_completion_tokens`, plus wall-clock) and then
**thrown away**. Route it into a persisted, queryable per-dispatch record.

Nothing consumes it in this phase. This is the "never ship an instrument and the mechanism it
measures in the same patch" rule from the prior spec §6.3, which was violated in the last arc and
produced two no-ops on arrival.

**KILL GATE:** cost is not reliably capturable per action (nulls > 10%).
**PROCEED GATE:** a real distribution of per-action cost from live data, ≥ 24 h, pasted.

**Touches:** `orion/execution_dispatch/result_extraction.py`,
`services/orion-execution-dispatch-runtime/app/worker.py`, `orion/schemas/execution_dispatch_frame.py`.

---

### Phase 2 — Expensive wants. The demand side.

Route 2–3 genuinely expensive cognitive verbs into the arena using the **already-built**
`template_to_cortex` seam (`orion/execution_dispatch/builder.py::resolve_cortex_route`, landed
2026-08-13, currently inert with zero routes declared).

Candidates, in order of preference, all already specified:
1. `goal_formulate` — Orion constructs a goal in its own words instead of choosing from 13
   human-authored templates. This is the difference between selecting and wanting, and it is the
   empirical answer to "what are Orion's drives" (program outcome **O4**).
2. `context_exec_memory_contradiction_review` — Orion notices it holds two incompatible beliefs
   and resolves one. Coherence maintenance as an act.
3. `dream_cycle`, seeded with the thing Orion could not resolve while awake.

**KILL GATE:** outputs are generic or schema-valid-but-empty (`raw_len` low, no inspectable
substance). That is the empty-shell-cognition rule, and it kills the specific verb, not the phase.
**PROCEED GATE:** GPU utilisation and exec-step latency measured before/after, pasted. Phase 3
requires demand to have *actually* risen.

**Touches:** `config/proposals/proposal_policy.v1.yaml`,
`config/execution_dispatch/execution_dispatch_policy.v1.yaml`, new schema for whatever artifact
the verb produces.

---

### Phase 3 — Finite means. The supply constraint. **This is the scarcity patch.**

A budget that **depletes with use and replenishes over time** — not one that resets per tick.
Actions priced at their Phase-1-measured real cost, not a config constant.

**KILL GATE, and it is the important one in this whole document:**
> If, after 48 h live, the budget has bound on **< 5% of ticks**, it is not a budget. It is a
> switch. Revert the phase and say so.

This gate exists because §1e is the trap this plan is most likely to fall into, and because the
existing `max_dispatches_per_tick` failed exactly this test for its entire life without anyone
checking.

**PROCEED GATE:** % of ticks where at least one candidate was unaffordable, pasted. Plus the
behavioural claim: **does Orion's action mix change with remaining budget?** If the mix is
identical when rich and poor, the budget is not binding on behaviour and Phase 4 is pointless.

**Touches:** `orion/execution_dispatch/policy.py`, `orion/execution_dispatch/builder.py`,
`config/execution_dispatch/execution_dispatch_policy.v1.yaml`.

---

### Phase 4 — Allocation. Drives, re-derived rather than re-authored.

Only now does a drive mean anything: a standing claim on something that actually runs out. This is
why `DriveEngine` pinned to a ceiling and was deleted — there was no constraint to push back
against.

**Hard constraint, from the program charter (O4):** drives are **derived from what Orion actually
spends on**, versioned and re-derivable. They are a report on clustering of real allocation
history. **No hand-authored drive taxonomy. Ever.** The previous one was deleted for exactly this
reason (`chore/delete-orion-drives`, PR #1486) and must not be reintroduced by the back door.

**KILL GATE:** the derived clusters are not stable across two independent time windows. Then it is
noise, and we report that instead of naming it.

---

### Phase 5 — Arbitration. Everything the previous spec proposed.

`§5.1` quantile calibration, `§5.4` temperature sampling, `§5.5` bounded shares. Deliberately last.

**Prediction to check against, stated now so it cannot be retrofitted:** much of this may prove
unnecessary. The 7.03-of-10 urgency tie is a symptom of having no cost function; once candidates
differ by real price, the tie may break on its own. If it does, we delete `base_priority`, the
starvation-aging hack, and the reserved lane rather than tuning them.

---

## 5. What is already done and where it sits

| item | state |
| --- | --- |
| PR #1603 metric semantic layer | mergeable, awaiting review |
| PR #1604 arena instrument + dispatch result gate | mergeable, awaiting review; deployed and live-verified |
| `measure_arena_degeneracy.py` | live, re-runnable, the before/after instrument for Phases 3–5 |
| dispatch result gate | live. Skill results are recorded instead of discarded — **prerequisite for Phase 1**, since a metered cost that gets dropped at the parse boundary is worthless |
| `template_to_cortex` seam | landed, inert, zero routes declared — **Phase 2 uses it** |
| `skills.runtime.image_prune.v1` | built, tested, **unrouted**. Parked. Not part of this plan. See §8. |

---

## 6. Non-goals

- Not building a new service.
- Not authoring a drive taxonomy (§Phase 4).
- Not touching `orion/proposals/scoring.py` before Phase 5.
- Not adding verbs as a grab-bag. Phase 2 adds expensive cognition **because Phase 3 needs
  something to be scarce about**, not because more capabilities are good.
- Not inventing a currency. The resource is measured inference time.
- Not merging `fix/disk-capacity-pressure-trigger` (still open, still unevaluated).

---

## 7. The leash

The failure mode this plan is most exposed to is not a bad phase. It is **drift** — an agent
finding something interesting mid-phase and following it, then reporting a different thing than
the one that was authorised. That is what produced the whiplash.

These are mechanical, not aspirational.

### 7.1 One phase at a time, and the gate is evaluated in writing
No phase begins until the prior phase's proceed gate is pasted with real numbers. Not summarised —
pasted. An agent that cannot produce the numbers has not finished the phase.

### 7.2 The parking lot is mandatory
Anything discovered mid-phase that is not the phase goes into
`docs/superpowers/specs/PARKING-LOT.md` with one line and a date. **It does not go into the
branch.** This session found: the arena urgency degeneracy, the open feedback loop, the
`proposed_effect` label problem, the tick-attributed field delta, the 46.6% empty frames. Every
one of those is real, and every one of them would have been a squirrel.

### 7.3 A kill gate is a real exit
Each phase names what makes us **stop**. Reaching a kill gate and stopping is a successful phase,
reported as such. An agent that reaches a kill gate and keeps going has failed the task even if
the code works.

### 7.4 Commit budget, stated up front
Per phase: **E0: 1 commit. P1: 3. P2: 4. P3: 4. P4: 3. P5: unscoped, re-planned after P3.**
On reaching budget: stop, report, wait.

### 7.5 No number without a paste
Carried forward from the prior spec §6.2 and violated twice in the last arc. Any number in a
comment, commit, report, or reply is a pasted measurement — never a derivation, never a
recollection.

### 7.6 The instrument lands alone
Carried forward from prior spec §6.3. Phase 1 exists as its own phase for this reason.

### 7.7 Every phase re-runs the same instrument
`scripts/analysis/measure_arena_degeneracy.py`, same window, before and after. A phase that cannot
show its effect on that instrument did not have one.

### 7.8 Scope changes go to Juniper, not into the branch
If a phase turns out to be the wrong shape, **that is a message, not a patch.**

---

## 8. Parked, deliberately

- **`image_prune` routing.** Built and tested this session. It is the only path to Orion's first
  non-zero `acted=true`, which is genuinely interesting — but it is housekeeping, not this plan,
  and routing it is a one-line config change available at any time. Parked rather than cancelled.
- **Everything in `PARKING-LOT.md`.**

---

## 9. Acceptance checks for the whole plan

- E0 produces a pasted cost table and ten goal texts, and either kills the thesis or does not.
- A budget exists that binds on a measurable, non-trivial fraction of ticks.
- Orion's action mix demonstrably differs between rich and poor states.
- At least one drive is *derived* from real allocation history and is stable across two
  independent windows.
- `measure_arena_degeneracy.py` §A shows the 7.03-of-10 urgency tie **reduced**, and it is
  attributable to a named phase.
- No hand-authored drive taxonomy exists anywhere in the tree.

---

## 10. The exact question for Juniper

**Sign off on Phase E0 only.** It is read-only, ~40 LLM calls, no deploy, one commit, and it can
kill the entire plan for almost nothing.

Everything after E0 is contingent on its result and gets re-confirmed with you before it starts.
