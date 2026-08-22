# PR: the allocator — spend the allowance on what is worth learning

Branch: `feat/motor-allocator`
Step 3 of 3. Depends on #1813 (per-action cost) and #1818 (the allowance).

## Summary

- Scores candidates on **expected information per motor-second**, refuses on an
  **absolute bar**, and gates on measured harm.
- Advisory. Logs what it would admit and drop, every tick.
- **Its verdict on Orion's current vocabulary is: admit nothing.**
- Review found 5 HIGH / 6 MEDIUM / 6 LOW. The maths was verified correct; the
  wiring on top of it was inverted. All HIGHs fixed in-branch.

## Outcome moved

```
motor_allocator_preview pending=5 would_admit=0 would_drop=5 nats=0.0000
  refusals={'unmeasurable': 5}
  refusals={'unmeasurable': 4, 'below_information_floor': 1}
```

Every live candidate is either **unmeasurable** (declares no signal, so no
posterior can ever exist for it — 61% of candidate slots) or **below the
information floor** (already measured; nothing left to learn). This is the
first time the system can say *"none of these were worth doing"*, and the
first thing it says it about is everything.

Both refusals are actionable: `unmeasurable` is fixed by declaring signals on
templates; `below_information_floor` is fixed by having actions worth doing.

## The design problem

Every action measures ~zero against a control arm (prune -0.0073 ± 0.0342,
containers -0.0405 ± 0.0581, inspect +0.0386 ± 0.0508 — all inside their own
error bars). Rank by value-per-cost and you rank noise; put an absolute bar on
value and it refuses everything.

Resolution: score on the **epistemic** term of expected free energy. For the
Normal-Normal posteriors already in place,

    E[KL(posterior ‖ prior)] = ½·ln(1 + σ²/τ²)   nats

Derived, then Monte-Carlo'd at four magnitudes of σ². The review re-derived it
independently and confirmed it, and mutation-tested the test: it catches every
structural error (missing ½, wrong τ scaling, a 3% multiplicative drift), and
`seed=11`/`rel=0.02` are not slack.

Pragmatic value is a **gate**, not a term — no exchange rate between
signal-units and nats is invented, because hand-typing that is how
`risk_score` happened.

## Review findings fixed

**H1 — the allocator was doing the exact inverse of its purpose, live.**
An action declaring no signal can never acquire a posterior
(`outcome_resolution` skips it with `no_declared_signal`), so its variance sat
pinned at the cold-start *maximum* forever and it won every slot — while every
action that *did* declare a signal got measured, its variance shrank, fell
below the floor, and was refused permanently. Self-reinforcing: omitting
`expected_effect` bought a 100–350x score multiplier.

Verified from the running container, not inferred: across 57 consecutive
previews every admitted candidate's `nats` was an exact integer multiple of
0.9905007. Not one admission was ever scored on real data.

Fix: `posterior_variance: float | None`; `None` means **unmeasurable**, scores
0, and refuses with its own reason. "We have learned what this does" and "this
can never be scored" need different fixes.

**H2 — wrong variance.** Used `pooled_treated_mean().variance`, which is
`Σ(n_b/N)²·var_b` — the sampling variance of a mean *across* bins, divided by
roughly the bin count. The next observation lands in **one** bin. So an action
read as better-known the more conditions it had run under. Live: scored
`maintain/host:docker_containers` at 0.0091 nats/s (refused) against 0.192
per-bin — top-ranked in the whole set. A 21x error that changed the answer.
`contrast.py`'s own docstring warned about it: *"the contrast is what a BUDGET
reads; the pooled mean is what a PREDICTION claims."*

**H3 — a claim in commit `4b28d1369` that was false.** It said the harm gate
fires on the live prune — *"first time anything here can refuse an action for
being BAD"*. The gate has **no producer anywhere in the repo**: `contrast()` is
called only from an eval, nothing persists a `ContrastEstimate`, and the worker
passes `None`. It has never fired and cannot. The numbers cited were the raw
pre-contrast delta hand-typed into a unit test, presented as live behaviour.
Retracted in the module docstring, marked `UNVERIFIED`, with two hazards
recorded for whoever wires it (a standard error shrinking as 1/√n against
admitted residual confounding; ~1.7 standing false positives at 2σ across ~75
simultaneous tests, permanent because nothing re-tests a contrast).

**H4** — the working-tree changes and an already-applied migration were
uncommitted; the reviewed HEAD shipped the known-bad cost query. Committed.

**H5 — fail-open, with a comment asserting the opposite.** On a spend-read
failure `_derive_motor_budget` returned `None`, and the caller's
`if motor is not None` skipped the enforce branch too — so a transient database
error removed the ceiling entirely. Now fails **closed** when enforcing.

**M5** — `agree=` was a tautology (admitted ⊆ pending by construction, so it
equalled `would_admit` on all 33 logged lines) and `frame.dispatched_candidates`
is always empty at that call site. It was logged as the metric the enforce flip
gets decided on. Removed rather than faked.

**M6** — retired the flat-p50 `would_refuse` projection now the allocator
computes the same thing from each action's own measured cost.

**Plus a NameError I shipped** (`_blended_variance` without `self.`) that threw
on every tick. The advisory try/except caught it and dispatch kept running —
that guard is the only reason a broken preview was not a broken motor path.

## The structural fix behind three of those

All three — the inversion, the pooled variance, and the NameError — lived in
one method on a class needing a database and a settings object to instantiate.
**Nothing could test it**, and every allocator test built its inputs directly,
bypassing exactly the code that was wrong.

Extracted to `allocator.candidate_from_dispatch()`, a pure function, with a
whole-frame test reproducing the live population (four undeclared inspects
winning every slot against one measured prune being refused) asserting the
admission now goes the other way.

## Not fixed — recorded

- **Epistemic-only scoring may be a dodge, and I think the review is right.**
  Epistemic value is instrumental in active inference; scoring on it alone
  swaps the objective from *reach preferred states* to *be surprised*. Two
  concrete symptoms, both verified: nothing reads `posterior.mean` (the
  allocator reads only `.variance`), so it is a closed loop; and variance only
  ever shrinks, so retirement is **absorbing** — an action falls below the
  floor at ~n ≥ 4.4 observations per bin and never returns, in a
  non-stationary world.
- **The right next patch is `randomized_holdback`**, which exists in a type
  union and nowhere else. It is what makes the pragmatic term real rather than
  routing around its absence.
- M1–M4, L1–L6 as listed in the review, including: the 36-motor-hour allowance
  is extrapolated from ~1 hour of data and currently does not bind
  (`pace≈0.96x`, `would_refuse=0`); `sum_motor_seconds_for_day` re-scans the
  day on >96% of ticks that have no candidates; and
  `substrate_dispatch_results_action_cost_idx` has `idx_scan=0` and is unused.

## Tests run

```text
pytest tests/test_motor_allocator.py tests/test_motor_budget.py
       tests/test_execution_dispatch_runtime_{worker,store}.py -q
136 passed
```

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
live: motor_allocator_preview emitting every tick, 0 preview_failed
```

## Restart required

```text
Already applied.
```

## Risks / concerns

- Severity: **HIGH** — if `ORION_DISPATCH_MOTOR_BUDGET_ENFORCE` were flipped
  today, the flag does **not** enable the allocator; it enables one line
  (`motor.exhausted` → send nothing for the rest of the UTC day). All-or-
  nothing, no reserve, no urgent path, first-come-wins. And the allowance
  does not currently bind, so flipping it would produce no observable change —
  which would read as "safe" and license removing the risk cap on no evidence.
- Severity: **HIGH** — were the allocator itself enforced today, Orion would
  do nothing at all. That is the honest verdict on the current vocabulary, not
  a bug, but it must be a decision rather than a surprise.
- Severity: MEDIUM — `no_cost_estimate` is a bootstrap deadlock for a genuinely
  novel action: no cost row → refused → never runs → never gets a cost row.
  Needs a probationary allowance before the allocator is ever enforced.

## PR link

<filled in on push>
