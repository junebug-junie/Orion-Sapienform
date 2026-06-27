# PR: substrate self-modeling loop — prediction-error feedback → governed self-revision

> This file is the PR description report, committed to the branch because no
> GitHub auth was available to write the PR body directly. Paste the body section
> into the PR at:
> https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/substrate-self-modeling-loop

---

## Summary

Turns the substrate from a feed-forward pipeline into a closed, self-correcting
loop. Prediction error (how wrong Orion's model just was) was previously measured
and exported but never acted on. This PR wires the feedback edges so surprise
shapes attention, the self-model is bound into Orion's beliefs, and *sustained*
self-model drift produces governed, draft-only change proposals.

Honest framing: this is not a claim about phenomenal experience. It implements the
*functional* properties (predictive feedback, integration/higher-order self-model,
metacognitive self-revision) that theories of mind treat as load-bearing — each as
a thin, inspectable, tested seam riding existing machinery.

## What this PR closes (loop, end to end)

```
prediction error → pressure (rung 1)
    → self-model bound into unified beliefs, carrying prediction_error (rung 2)
    → sustained drift → governed DRAFT proposal, never auto-applied (rung 6)
```

## Changes

| Rung | Change | Files |
|------|--------|-------|
| 1 | `prediction_error_pressure()` as a third seed source in the dynamics engine; BFS-propagated, age-decayed; gated by `PressureConfig.prediction_error_weight` | `orion/substrate/pressure.py`, `orion/substrate/dynamics.py` |
| 2 | `self_state` producer + adapter binding Orion's self-model into the unification layer; each node carries standing `prediction_error` (durable-node completion of rung 1) | `orion/substrate/relational/adapters/self_state_ctx.py`, `orion/cognition/projection_builder.py` |
| 6 | `prediction_error_mutation_signals()`: sustained self-dimension drift → `MutationSignalV1` → existing `PressureAccumulator` → `ProposalFactory` **draft** (`draft_only_not_applied`) | `orion/substrate/mutation_self_revision.py` |
| docs | Full ladder + per-rung specs (rungs 3/4/5 + rung-2 lanes) for follow-up | `docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`, `docs/plans/substrate/CONTINUATION.md` |

## Safety / governance

- **No auto-application.** Rung 6 emits *signals* only; thresholding, cooldown,
  draft generation, trials and rollback remain owned by the unchanged mutation
  pipeline. A single surprising tick stays below threshold; only persistent error
  drafts a proposal, and every proposal is `draft_only_not_applied`.
- **No invented surfaces.** Self-dimensions map only to already-supported cognitive
  mutation surfaces; unmapped dimensions are ignored.
- **No silent mutation** of accepted claims/specs (Knowledge Forge rule preserved).
- Rung 1 is gated by `prediction_error_weight` (set 0 to disable).

## Tests

`12/12` passing:
- `tests/test_cognitive_substrate_phase4_dynamics.py` — seed + propagation + age-decay + no regression (7).
- `orion/substrate/relational/tests/test_self_state_adapter.py` — self-model nodes carry prediction_error; dict/JSON input; absent → None (2).
- `orion/substrate/tests/test_mutation_self_revision.py` — supported-surface routing; calm → nothing; sustained drift → draft proposal, never applied (3).

```
cd <worktree> && .venv/bin/python -m pytest \
  tests/test_cognitive_substrate_phase4_dynamics.py \
  orion/substrate/relational/tests/test_self_state_adapter.py \
  orion/substrate/tests/test_mutation_self_revision.py -q
```

## Follow-up (not in this PR — see CONTINUATION.md)

- Rung 1 runtime bridge: worker writes `prediction_error` onto durable substrate
  nodes (currently lands in a field-digester receipt only).
- Rung 2 remainder: biometrics/execution/transport lane adapters.
- Rung 4 episodic continuity; rung 3 continuous broadcast; rung 5 endogenous
  agency (build flag-gated, **do not enable without sign-off**).

## Reviewer notes

- The repo is concurrently mutated by an automation user; this branch was built in
  an isolated worktree to avoid corruption.
- Commits: `85d29c62` (rung 1), `ca176e4e` (specs), `d9dbe624` (rung 2), `de21cb3a`
  (rung 6).
</content>
