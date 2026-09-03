# Don't adopt a patch that changes nothing

Follows PR #2050 (release the surface lock) and #2057 (show it). Fixes what the first working cycle revealed.

## Summary

- Minutes after the surface lock was released for the first time, Orion adopted `0.58` over a live `0.58` and wrote a history row reading `0.58 -> 0.58`. Left alone that repeats every rollback window forever.
- `PatchApplier.apply` now declines a patch that would leave the surface unchanged. The worker records the refusal through `record_apply_blocked` — the same channel every other blocked apply uses.
- Fixes three separate paths by which test and smoke code could write Orion's **live** routing threshold.
- Fixes a panel field name from #2057 (`current.source` → `current.source_kind`).

## Outcome moved

| | before | after |
| --- | --- | --- |
| adoptions per rollback window, steady state | 1 (no-op) | 0 |
| history rows recording no change | 1 per window | 0 |
| surface lock held by a non-change | 900s per window | never |
| smoke can write the live threshold | **yes**, wherever `DATABASE_URL` is set | no |

Live confirmation of the problem, `conjourney`, 2026-09-03: `substrate_mutation_adoption` gained a second row at 04:19:55 with `applied_patch = rollback_payload = {"chat_reflective_lane_threshold": 0.58}`, and the first-ever history row reads `0.58 -> 0.58` by `mutation_apply`.

## Current architecture

`_default_patch_for_class` (`mutation_proposals.py:104`) returns a hardcoded `0.58` for every `routing_threshold_patch` — it is the only routing patch the pipeline can propose. Once the surface reaches that value, every later proposal re-applies the number already live, takes the one-live-mutation-per-surface lock for the length of its window, and blocks real proposals behind a change that is not a change. That is the empty-shell shape AGENTS.md §0A rules out: a loop reporting self-modification while modifying nothing.

## Files changed

- `orion/substrate/mutation_apply.py`: `_is_noop`; `apply` declines; `noop_reason` explains.
- `orion/substrate/mutation_worker.py`: records the refusal when `apply` declines.
- `orion/substrate/scripts/smoke_mutation_v21.py`: real control-surface isolation.
- `orion/substrate/tests/test_mutation_v21.py`: per-test control-surface isolation.
- `orion/substrate/tests/test_noop_mutation_guard.py`: new, 8 tests.
- `services/orion-hub/static/js/app.js`: `source_kind` field name.

## Schema / bus / API changes

- Added: `PatchApplier._is_noop`, `PatchApplier.noop_reason`. No schema, bus or route changes.
- Behaviour changed: an auto-promoted routing proposal whose patch equals the live value no longer produces an adoption. It produces a `substrate_mutation_apply_block` row with `reason=patch_is_noop:...`.
- Compatibility: additive. A skipped proposal's queue status stays `approved`, identical to what the existing `active_surface` block already does.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest orion/substrate/tests -q          -> 689 passed  (x2 runs, stable)
pytest <hub mutation quad> -q -p no:randomly          -> 0 failures  (x3 runs, stable)
pytest services/orion-cortex-orch/tests/test_control_surface_isolation_guard.py -q -> 3 passed
all 10 gates in .github/workflows/orion-static-gates.yml -> exit 0
```

Live isolation check — the smoke run with `DATABASE_URL` deliberately in scope:

```text
live threshold BEFORE : 0.58 | mutation_apply
live threshold AFTER  : 0.58 | mutation_apply
history rows by the smoke: 0
apply path still exercised: True
```

Mutation-checked against the real files, restored by file copy (never `git stash`, shared across worktrees here):

| Mutation | Caught |
| --- | --- |
| `apply` no longer declines a no-op | no-op tests |
| multi-key patch judged on one key | multi-key test |
| out-of-range patch not clamped before comparing | clamp test |
| `noop_reason` compares the patch to itself | 2 tests |
| worker records the block but does not skip | worker test |
| class gate deleted | uncomparable-surface test |
| smoke does not restore the global store | pinning test |
| smoke never swaps the global store | pinning test |

One mutation deliberately **not** claimed as covered: removing the env-clearing from `_isolated_control_surface` leaves the suite green, and that is correct — with an explicit `sql_db_path`, `__post_init__` never consults the environment. The env clearing is defence-in-depth against a future change to that precedence; the runtime `assert isolated.postgres_url is None and source_kind() == "sqlite"` is what actually enforces isolation.

## Evals run

```text
none — orion/substrate/evals/ has no mutation-runtime harness
```

## Docker/build/smoke checks

```text
orion/substrate/scripts/smoke_mutation_v21.py  -> apply path exercised, live surface untouched
node --check services/orion-hub/static/js/app.js -> OK
```

## Review findings fixed

- **Finding (CRITICAL): the smoke's isolation did not isolate.** `RuntimeControlSurfaceStore(sql_db_path=None, postgres_url=None)` is not an isolation request — `__post_init__` fills either slot from ambient env. With `DATABASE_URL` set, which is exactly where orion-hub runs, the "isolated" surface resolved to live Postgres and the smoke moved the real threshold. The reviewer reproduced it.
  - Fix: clear the env keys, pass an explicit throwaway path, assert the resolution before writing.
  - Evidence: live run above — production row untouched, zero smoke history rows.
- **Finding (CRITICAL): the pinning test had the same hole** and reproduced the leak it existed to catch.
  - Fix: runs with an ambient control-surface key deliberately in scope, asserting the ambient store keeps only the operator's row.
- **Finding (HIGH): the commit regressed a hub test three modules away.** Root-caused: not the guard's logic — a bare extra `get_chat_reflective_lane_threshold()` call at the same point reproduces it, and the guard never fired in the failing run. Reading the surface binds the module-global store, and this suite's fixtures assign that global by raw assignment without restoring it, so behaviour depended on when the surface was first touched.
  - Fix: decide inside `apply`, where the value is already read, so the common path costs no extra read. `noop_reason` is asked only after `apply` declines.
  - Evidence: 3 consecutive clean runs of the exact failing batch; `git`-verified 4/4 failing before, 4/4 clean after.
- **Finding (MEDIUM): `test_mutation_v21` shared one control surface across all its tests**, so one apply moved the starting value for the next. Four tests failed for that reason rather than for what they test.
  - Fix: autouse per-test isolation seeded to 0.5.
- **Finding (MEDIUM): a multi-key patch was judged on one key**, silently dropping a real change to `autonomy_route_threshold`.
- **Finding (MEDIUM): one guard test was vacuous** — it never seeded the surface, so the values differed regardless and it passed with the class gate deleted.
- **Finding (LOW): raw patch compared instead of the clamped value**, so `1.5` over a saturated `1.0` would mint an adoption for a change that did not happen.
- **Finding (LOW): exact float equality is correct here** — verified the JSON/JSONB round trip is exact on the live database. A tolerance would suppress genuine small adjustments. No change made.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

After deploy the steady state should be: no new adoptions, no new history rows, and `substrate_mutation_apply_block` rows carrying `patch_is_noop:...`.

## Risks / concerns

- **Severity: medium.** A skipped proposal's queue status stays `approved` forever — never applied, never rejected. Byte-identical to what the existing `active_surface` block already does, so not a regression, but `approved` is now a lie about a proposal that will never apply. Follow-up: a terminal status.
- **Severity: low.** With the guard live, Orion's pipeline runs end to end and produces no change at all, because the only patch it can propose is a constant it has already applied. The loop is correct and inert. That is not hidden by this PR — it is what the blocked-apply rows now say plainly — but it means item 4 is the critical path, not a refinement.
- **Severity: low.** Three surface-leak paths are fixed here; the hub suite still assigns the control-surface global by raw assignment in three fixtures without restoring it. Not load-bearing after this change, but the next extra read anywhere will expose it again. Follow-up: an autouse restore fixture in `services/orion-hub/tests/conftest.py`, matching cortex-orch.

## Follow-ups

1. **Item 4, real latitude** — the patch value is a constant and the confidence it is compared against is a keyword lookup table (`decision_router.py:237-255`); `AUTO_ROUTER_LLM_ENABLED=false`.
2. **Feed the monitor** — nothing supplies a post-adoption delta, so settlement is still "time passed", not "it helped".
3. Terminal status for a permanently-skipped proposal.
4. Autouse control-surface restore in the hub conftest.

## PR link

<pending>
