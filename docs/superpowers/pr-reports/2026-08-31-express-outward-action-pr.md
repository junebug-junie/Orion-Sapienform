# PR #2004 — Orion chooses: `express`, the pre-filter, and the cold-start absorbing state

Merged as `71d0849ce` from `feat/orion-express-action-prefilter`. Builds directly
on [#2002](2026-08-30-motor-budget-enforcement-pr.md).

## The result first

```text
motor_allocator_preview  pending=11  would_admit=1  would_drop=10  nats=0.9905
allocator_cold_start_exemption  dispatch_id=…  rate=0.018519  floor=0.020000
motor_allocator_enforced  skipped=10  sending=1
reverie_visual_chain  adae700f-0072-4ad3-98e7-148708a04864  max_steps
```

**Out of eleven things it could have done, Orion spent 53 motor-seconds making an
image and refused the other ten.** No cron fired. No operator asked. The visual
chain's 600 s timer is **off** (`ORION_VISUAL_CHAIN_ENABLED=false`) — the run
happened because the action won a budget slot on value-per-second.

Artifact:
`/mnt/storage-lukewarm/orion/reverie-visual/3ecb12a5ae0c45b310450f2aa35ee5487964a1e6d425953ffc2d308b506ee409.png`
— it added paper lanterns to a scene it was carrying forward from its own prior
description, so the previous run's output became this run's prompt.

Measured against the roadmap's own standard (*"did this make some choice real that
was previously scheduled?"*): **yes, literally.** A cron interval became a
competitive bid.

## Summary

- Added `express` — the first dispatch kind whose effect is **outward** rather than
  introspective. Every one of Orion's other ~17 action families reads, summarizes,
  measures, or tidies; none of them produce anything that leaves the machine.
- Wired it end to end: proposal template → policy decision → dispatch route →
  cortex-exec verb → `POST /visual-chain/run-once` on `orion-thought`.
- **Removed the priority pre-filter's grip on the value scorer** (`max_dispatch_candidates`
  5 → 50) so a hand-typed float stops deciding what the allocator is allowed to see.
- **Exempted never-measured actions from the information floor**, which was an
  absorbing state that made expensive actions permanently unschedulable.

## The two ordering flaws this found

### 1. The pre-filter ran before the value scorer

`max_dispatch_candidates: 5` truncated the candidate list by `base_priority` — a
hand-authored constant — *before* `allocate()` ever saw it. So the
value-of-information machinery was being fed a list already chosen by the thing it
exists to replace. `express` was cut at that stage and never scored at all.

Raised to 50, above the 11/frame actually observed live. The pre-filter was not
deleted; the admission machinery under it is good. **The real send cap is
untouched** — `max_dispatches_per_tick: 5` is a separate limit, as are the motor
budget and the allocator. This widens what is *considered*, not what runs.

The `base_priority: 0.55` bump added earlier in the session to survive that cut was
reverted to `0.30`. Keeping it would have been crowning the action instead of
letting it earn a slot.

### 2. The information floor is an absorbing state for expensive actions

This is the one that actually blocked it, and my first diagnosis was wrong. Aggregate
refusal histograms said `{'unmeasurable': 2}`, from which I inferred `express` had no
posterior to score. Per-candidate DEBUG logging said otherwise:

```text
render_scene  variance=0.25  cost_sec=53.49  nats_per_sec=0.018517  measurable=True
```

Not unmeasurable. **Measured, and 71× better per second than the best alternative**
(0.000259) — and refused alongside it.

A cold prior yields a fixed 0.99 nats. Divided by cost, that means **any action
costing more than ~49 s sits below a 0.02 nats/sec floor forever**, regardless of how
good it is, because it can never accumulate the observations that would raise its
variance estimate. The floor was reporting "nothing here is worth doing" while the
best available option was two orders of magnitude ahead of the runner-up.

Fix: `Candidate.cold_start` (set where `blended is None`), and an exemption in
`allocate()` — an action with **zero** observations cannot be refused on the
information floor, because the rate being tested is entirely *prior*. The floor's
job is retiring things we have **learned** are uninformative; it cannot do that job
on something never measured, and refusing it forever is precisely what prevents the
evidence from existing.

**I did not lower the floor.** That would have admitted the actions we correctly
learned are worthless. The exemption ends the moment one observation exists.

## Files changed

| Path | Why |
|---|---|
| `config/execution_dispatch/execution_dispatch_policy.v1.yaml` | `max_dispatch_candidates` 5→50; `allow_express_dispatch`; `approved_express` decision; `render_scene` route (`express_bounded`, 170 s RPC timeout) |
| `config/proposals/proposal_policy.v1.yaml` | `render_scene` template — `kind: express`, `target_id: host:circe_gpu`, `base_priority: 0.30` |
| `config/policy/substrate_policy.v1.yaml` | `express` rule, `default_decision: approved_express` |
| `orion/autonomy/allocator.py` | `Candidate.cold_start`; the exemption + its log line |
| `orion/schemas/{proposal,execution_dispatch,policy_decision}_frame.py` | `"express"`, `"approved_express"`, `"express_bounded"` added to the closed `Literal` vocabularies |
| `orion/policy/{evaluator,builder}.py` | `approved_express` collected into `approved_decisions` |
| `orion/cognition/verbs/skills.imagination.render_scene.v1.yaml` | **New** verb; `requires_gpu: true`, `timeout_ms: 180000` |
| `services/orion-cortex-exec/app/verb_adapters.py` | `RenderSceneVerb` → `POST {thought_service_url}/visual-chain/run-once` |
| `services/orion-thought/app/main.py` | The endpoint. Returns the readout on a refusal too, so a thermal refusal is not mistaken for a crash |
| `tests/test_express_outward_action.py` | **New**, 21 tests incl. `TestColdStartExemption` |

## Schema / bus changes

Added to closed `Literal` vocabularies: `express` (proposal + dispatch kind),
`approved_express` (autonomy tier), `express_bounded` (allowed scope).

**Compatibility hazard, hit live during this session.** I shipped the config before
the schema. A stale `policy-runtime` rejected the *whole frame* on the unknown
literal — not just the new candidate — and live dispatch went to `candidates=0`.
Fixed by redeploying policy-runtime. `policy_decision_frame.py` already carried a
comment describing this exact failure from 2026-08-12. **Deploy consumers before
producers when widening a closed vocabulary.**

## Tests run

```text
pytest tests/ services/orion-execution-dispatch-runtime/tests -q    # 188 passed
```

Tests assert the exemption does **not** bypass the harm gate, the cost requirement,
the ordering, or the daily allowance — only the information floor, and only at
zero observations.

## Review findings fixed

- **Finding:** the exemption test survived gutting the guard to `if False:` — it was a
  substring check on a log line, not a test of behavior.
  - **Fix:** rewritten to assert on the guard's condition via AST.
  - **Evidence:** mutation-tested against the real file, per
    [Mutation-test static gates against the real file].
- **Finding:** hand-computed fixture asserted `0.0446` nats/sec for a cold action;
  the real value is `0.198`.
  - **Fix:** the test now carries the number `settings.py` independently records.
- **Finding:** `ProposalCandidateV1.model_fields["kind"]` — the field is `proposal_kind`.
- **Finding (live):** `await persist_reverie_visual_chain(chain)` on a **sync**
  function, inside `suppress(Exception)` — every refusal would have failed to persist,
  silently.
  - **Fix:** `asyncio.to_thread`, matching the existing call site.

## The lesson worth keeping

I spent hours inferring from `refusals={'unmeasurable': 2}` — an aggregate that tells
you *how many* but never *which*. One DEBUG line per candidate answered it in seconds.
Same shape as the mutation-pipeline starvation found the same day
([#1999](2026-08-30-mutation-signal-starvation-pr.md)): **an aggregate that cannot
distinguish causes will hide one indefinitely.** That per-candidate logging is in this
PR, at DEBUG.

## Restart required

```bash
# already applied live; listed for reproducibility
scripts/safe_docker_build.sh orion-policy-runtime up -d --build      # FIRST — schema consumer
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- **Severity: high.** The `visual_chain` single-flight lock can wedge. Observed live:
  `already_in_flight` returned while circe was completely idle; cleared only by a
  container restart. A severed HTTP request appears to leave the lock held, which would
  silently make the action permanently unschedulable with no error anywhere. **Unfixed.**
- **Severity: medium.** `express`'s 53 s cost was measured from *my* manual runs, not
  Orion's. Honest, but operator-seeded.
- **Severity: low.** `starvation_aging` remains a no-op under the allocator, which is
  priority-blind. Wire it in or retire it.

## Status

DONE_WITH_CONCERNS — merged and live-verified; the single-flight lock is a real defect.
