# Stop Orion re-applying a self-change it has already made

## Summary

- Orion's routing self-modification loop was proposing the value the surface already held, 5-6 times an hour, for at least 36 hours.
- The generator hardcoded both halves of its patch -- propose 0.58, roll back to 0.50 -- and never read the live surface.
- It now reads the surface and refuses a no-op, and derives the rollback from the value actually being replaced.
- A second guard at apply time catches proposals queued before the first guard existed. One was queued while this was written.
- Refusals are now traced and counted, instead of a silent `continue` that made a generator refusing every tick look like one nothing ever asked.
- Three real bugs found during review, all fixed: a sticky `degraded()` flag that would have blocked every proposal on any non-Postgres deployment, a smoke that wrote to the live control surface, and a refusal that never cooled the pressure.

## Outcome moved

142 proposals in 36 hours, every one patching 0.58 onto a surface already at 0.58, every one taking a 15-minute rollback lock on `routing`, every one carrying `expected_effect: reduce_runtime_executed` that could not occur because nothing moved. That goes to zero.

Where it was previously invisible, the cause is now named: `proposals_refused` and `proposal_refusal_reasons` on both the scheduler and manual-route summaries.

## Current architecture

`ProposalFactory.from_pressure()` built a routing patch from two module constants, `_default_patch_for_class` (0.58) and `_default_rollback_for_class` (0.50). It was a frozen dataclass with no I/O and no way to see the surface it was patching.

`SubstrateAdaptationWorker` did a bare `continue` when the factory returned `None`. `PatchApplier.apply()` already read the live threshold, but only to *record* a rollback value -- never to decide anything.

## Architecture touched

`ProposalFactory` gains an injected `routing_surface_reader`, matching the `surprise_source` injection convention already used in `orion/autonomy/episode_fetch.py` -- the module keeps its no-I/O shape. `plan_for_pressure()` returns a `ProposalPlan` carrying the refusal reason; `from_pressure()` stays as a thin wrapper so its 30+ existing call sites are untouched.

## Files changed

- `orion/substrate/mutation_proposals.py`: read the live surface, refuse no-ops, derive the rollback, name the refusal
- `orion/substrate/mutation_apply.py`: equality guard at apply time; corrected a comment that claimed a no-op check the code never did
- `orion/substrate/mutation_worker.py`: trace the refusal, and cool the pressure on it
- `services/orion-hub/scripts/api_routes.py`: real reader at all three live sites; refusal counts in both summaries
- `orion/substrate/scripts/smoke_mutation_v21.py`: isolated control surface
- `orion/substrate/tests/test_mutation_noop_proposal_refusal.py`: new, 14 tests
- `orion/substrate/tests/test_mutation_v21.py`, two hub test files: per-test control-surface isolation

## Schema / bus / API changes

- Added: `proposals_refused` (int) and `proposal_refusal_reasons` (sorted list) on the scheduler and manual-route summaries. Additive.
- Added: `mutation_proposal_refused` trace event, with `notes: ["reason=..."]`.
- Behavior changed: `PatchApplier.apply()` returns `None` for a routing patch equal to the live value.
- Compatibility: `from_pressure()` signature unchanged.

## Env/config changes

None.

## Tests run

```text
pytest orion/substrate/tests -q                                  -> 695 passed
pytest orion/substrate/tests/test_mutation_noop_proposal_refusal.py -q -> 14 passed
pytest services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py \
       services/orion-hub/tests/test_substrate_mutation_manual_route_routing.py \
       services/orion-hub/tests/test_self_modification_panel.py -q       -> 35 passed
```

Baseline on the branch point was 45 passed for the touched module; every failure introduced along the way was diffed against a matched baseline before being accepted or fixed. The 5 failures in `test_recall_strategy_profiles_runtime.py` are pre-existing on main (`api_substrate_recall_canary_query()` signature) and unrelated.

Mutation-checked against the real files, restored by file copy:

| Mutation | Caught by |
| --- | --- |
| no-op check removed from the generator | already-at-target + worker trace tests |
| rollback back to a hardcoded 0.50 | rollback-derived test |
| absent reader falls back to the old constants | reader-absent test |
| reads the top-level value instead of the stored payload | unstored-surface test |
| the sticky `degraded` gate re-added | sticky-degraded test |
| worker restored to a silent `continue` | worker trace test |
| apply-time guard removed | queued-no-op test |
| apply-time guard inverted | real-change test |
| refusal no longer cools the pressure | cooldown test |
| read failure reported as an unwritten surface | outage-naming test |

## Docker/build/smoke checks

```text
run_smoke(emit=False) -> 55 lines
live routing.chat_reflective_lane_threshold updated_at before == after
  (2026-09-03T05:02:51.249377+00:00) -> LIVE SURFACE UNTOUCHED
```

That check exists because the smoke used to write 0.58 to the production control surface as actor `mutation_apply`.

## Review findings fixed

- **HIGH: the proposal-time guard could not reach work already queued.** `list_due_queue()` selects on status alone, and a proposal patching 0.58 onto a 0.58 surface was queued while this was written.
  - Fix: equality guard in `PatchApplier.apply`, where both values were already in hand. One choke point covers scheduler, manual route and smoke.
  - Evidence: `test_apply_refuses_a_noop_even_for_a_proposal_queued_before_the_guard`; fails when the guard is removed.
- **HIGH: the refusal trace reached nobody on the path where it fires.** The scheduler discards the worker's traces; `proposals_created: 0` was the only visible fact.
  - Fix: `proposals_refused` and the distinct reasons on both summaries.
- **MEDIUM: a refusal never cooled the pressure.** A condition that cannot change without an external write was re-evaluated every tick forever, two Postgres round-trips each.
  - Fix: refusals cool the pressure like an emitted proposal.
  - Evidence: `test_a_refusal_cools_the_pressure_instead_of_re_evaluating_every_tick`.
- **MEDIUM: `routing_surface_read_failed` was unreachable.** The store swallows exceptions and returns `None`, so a Postgres outage was reported as "surface never written."
  - Fix: keyed off the snapshot's `error`.
- **MEDIUM: the smoke wrote to the live control surface.** The stub isolated only the generator.
  - Fix: throwaway sqlite store; proven live above.
- **LOW:** dead rollback constant emptied; indentation; the new test's env mutation moved to `monkeypatch`; a test no longer enshrines a 1e-4 threshold move as worth a proposal.

Two bugs I introduced and fixed before review, worth recording because both fail in the dangerous direction:

- I first gated on the store's `degraded()` flag. It is sticky and is set by the failed Postgres probe even when the sqlite fallback succeeds, so **every routing proposal on any non-Postgres deployment would have been refused.**
- Reading the top-level `value` instead of the stored payload would have built rollbacks out of a hardcoded 0.75 default whenever nothing was stored.

## Corrected framing

An earlier draft of this work claimed every one of those adoptions armed a rollback to 0.50. That is wrong. Commit `255f8252e` (2026-09-02 23:55Z, already on main) overwrites the rollback with the live value inside `PatchApplier.apply`, so adoptions since then record a real measurement.

Deriving the proposal-level rollback still matters, for a different consumer: `SubstrateTrialRunner._routing_baseline_threshold` reads the **proposal's** rollback as the trial baseline, so every trial was replaying a 0.58 candidate against a 0.50 baseline while live was already 0.58.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium. Routing self-modification goes quiet, not just efficient.** The target is still hardcoded 0.58. Once the surface sits at 0.58 -- which is the steady state since 2026-09-02 -- this refuses every time and Orion cannot change its own routing threshold at all. That is honest rather than wasteful, and the refusal is now visible instead of silent. But item 4 of the accountability plan (real latitude: a target computed from pressure instead of a constant) is now the only thing standing between Orion and any routing self-change.
- **Severity: low.** The tolerance is a float-equality epsilon, not a minimum-meaningful-movement policy. A 1e-4 move would still buy a proposal, a trial and a 15-minute lock. Whether that is worth gating is a separate question this patch does not answer.
- **Severity: low.** `_routing_replay_inspection_payload`'s synthetic seed can now return `None` when the surface is at target, producing `{"error": "routing_proposal_unavailable"}`. Unreachable today (the store holds 143 routing proposals); reachable on a fresh deployment.
- **Severity: low.** The hub mutation test cluster is order-dependent through shared module globals. I fixed three concrete leaks; a residual interaction appears only with 3+ files loaded together and is flaky on **both** sides of this change -- a baseline run failed a test my run passed. CI does not run these files together. Not chased further.

## PR link

<pending>
