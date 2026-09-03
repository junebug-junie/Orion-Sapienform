# Stop generating the no-op that #2058 now blocks

## Relationship to PR #2058

#2058 (merged 2026-09-03) stops a patch that changes nothing being **adopted**. This is the other half: stopping it being **generated**.

They compose. With only #2058, the generator still emits 5-6 no-op proposals an hour and each one spends a proposal, a trial and a decision before being blocked at the end. With both, the cycle refuses at the start and says why.

An earlier revision of this branch duplicated #2058's apply-time guard. That work is dropped — theirs is merged, better factored, and records `apply_blocked` into the store, which mine did not. `mutation_apply.py` is untouched here.

## Summary

- The generator hardcoded both halves of its routing patch: propose 0.58, roll back to 0.50. It never read the live surface.
- Confirmed live: 142 proposals in 36 hours, every one patching 0.58 onto a surface that had been at 0.58 since 2026-09-02T04:11:17.
- It now reads the surface and refuses a no-op, deriving the rollback from the value actually being replaced.
- Refusals are traced and counted instead of a bare `continue`.
- A refusal now cools the pressure, so a condition that cannot change without an external write is not re-evaluated every tick.

## Outcome moved

The 142 proposals go to zero at the source rather than being blocked at the end.

`SubstrateTrialRunner._routing_baseline_threshold` reads the **proposal's** rollback as its trial baseline. Every trial was replaying a 0.58 candidate against a 0.50 baseline while live was already 0.58. #2058 does not touch this; deriving the rollback here fixes it.

## Files changed

- `orion/substrate/mutation_proposals.py`: injected `routing_surface_reader`, no-op refusal, derived rollback, named refusal reasons
- `orion/substrate/mutation_worker.py`: `mutation_proposal_refused` trace, pressure cooldown on refusal, real reader at the default construction
- `services/orion-hub/scripts/api_routes.py`: real reader at all three live sites; `proposals_refused` / `proposal_refusal_reasons` on both summaries
- `orion/substrate/scripts/smoke_mutation_v21.py`: real reader (its control surface is already isolated by #2058); resets the isolated surface before the rollback-required step, which step 3 had moved to 0.58
- `orion/substrate/tests/test_mutation_noop_proposal_refusal.py`: new, 12 tests
- four existing test modules: pass a stub reader

## Schema / bus / API changes

- Added: `proposals_refused` (int), `proposal_refusal_reasons` (sorted list) on the scheduler and manual-route summaries. Additive.
- Added: `mutation_proposal_refused` trace event with `notes: ["reason=..."]`. Distinct from #2058's `mutation_apply_blocked`.
- `from_pressure()` signature unchanged; `plan_for_pressure()` added alongside it, so 30+ existing call sites are untouched.

## Env/config changes

None.

## Tests run

```text
pytest orion/substrate/tests -q                                          -> 701 passed
pytest services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py \
       services/orion-hub/tests/test_substrate_mutation_manual_route_routing.py \
       services/orion-hub/tests/test_self_modification_panel.py -q       -> 35 passed
```

Mutation-checked against the real files, restored by file copy:

| Mutation | Caught by |
| --- | --- |
| no-op check removed | already-at-target + worker trace tests |
| rollback back to a hardcoded 0.50 | rollback-derived test |
| absent reader falls back to the old constants | reader-absent test |
| reads the top-level value instead of the stored payload | unstored-surface test |
| sticky `degraded` gate re-added | sticky-degraded test |
| worker restored to a silent `continue` | worker trace test |
| refusal no longer cools the pressure | cooldown test |
| read failure reported as an unwritten surface | outage-naming test |

## Docker/build/smoke checks

```text
run_smoke(emit=False) -> passes via test_smoke_script_trace_and_invariants
live routing.chat_reflective_lane_threshold updated_at unchanged across the
  full suite (2026-09-03T05:02:51.249377+00:00)
```

## Review findings fixed

Reviewed at `high` against the earlier revision; the apply-side findings were resolved by dropping that half in favour of #2058. The findings that still applied:

- **The refusal trace reached nobody.** The scheduler discards the worker's traces, so `proposals_created: 0` was the only visible fact. Both summaries now carry the count and the distinct reasons.
- **A refusal never cooled the pressure.** Re-evaluated every tick forever at two Postgres round-trips each. Refusals now cool it like an emitted proposal.
  - Evidence: `test_a_refusal_cools_the_pressure_instead_of_re_evaluating_every_tick`.
- **`routing_surface_read_failed` was unreachable.** The store swallows exceptions and returns `None`, so a Postgres outage read as "surface never written". Keyed off the snapshot's `error`.
- **Dead 0.50 rollback constant** emptied; a test no longer enshrines a 1e-4 threshold move as worth a proposal; env mutation moved to `monkeypatch`.

Two bugs I introduced and fixed before review, both failing in the dangerous direction:

- Gating on the store's `degraded()` flag. It is sticky and set by the failed Postgres probe even when the sqlite fallback succeeds, so **every routing proposal on any non-Postgres deployment would have been refused.**
- Reading the top-level `value` rather than the stored payload would have built rollbacks from a hardcoded 0.75 default whenever nothing was stored.

## Corrected framing

An earlier draft claimed every adoption armed a rollback to 0.50. Wrong: `255f8252e` (2026-09-02 23:55Z, already on main) overwrites it with the live value at apply time. The surviving benefit is the trial baseline, named above.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium. Routing self-modification goes quiet.** The target is still hardcoded 0.58, and the surface has been at 0.58 since 2026-09-02, so this refuses every time. Orion cannot change its own routing threshold at all until the target is computed from pressure rather than typed in. That is honest rather than wasteful, and now visible rather than silent -- but it is the remaining blocker to any routing self-change.
- **Severity: low.** The tolerance is a float-equality epsilon, not a minimum-meaningful-movement policy.
- **Severity: low.** `_routing_replay_inspection_payload`'s synthetic seed can return `None` when the surface is at target, yielding `{"error": "routing_proposal_unavailable"}`. Unreachable while the store holds routing proposals.

## PR link

<pending>
