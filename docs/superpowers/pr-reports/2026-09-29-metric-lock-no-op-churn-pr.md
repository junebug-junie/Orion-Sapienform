## Summary

- Branches that change no metric definitions no longer have to re-run `scripts/check_definition_drift.py --update` after main moves, and no longer edit the lock file at all.
- The lock's `_last_change` block no longer stores a merge-base commit hash. The hash is printed on the console instead.
- `--update` on a no-change branch keeps the merge base's block as-is. The file stays byte-identical, so there is nothing to commit.
- `--gate` also accepts the block committed at the merge base, but only when the branch's locked definitions equal the base's. A real change still has to be re-locked and stated.
- The gate now compares the whole block (counts too), not only the sentences.

## Outcome moved

On 2026-09-29, three PRs (#2400, #2401, #2402) each re-locked with `change_count: 0` before and after. Earlier, #2325, #2327 and #2329 did the same. Every one of those re-locks was churn: each main merge forced every open branch to re-lock, and two no-change PRs conflicted on the same lines. Both of those are gone now. This PR removes the legacy hash line once. Any branch doing that produces the identical edit, so the edits merge cleanly.

## Current architecture

`_last_change` was a derived block holding `base: "merge base <sha> (origin/main)"`, the change count and sentences. `--gate` recomputed "what this branch changes relative to the merge base" and accepted only that. When main's committed block described another PR's change, a no-change branch recomputed "no definition changes", the two disagreed, and the gate went red until someone re-locked.

## Architecture touched

- `scripts/check_definition_drift.py` only. There are no other consumers of `_last_change` (checked with grep).
- CI step "Static repo gates": the command is unchanged, and so is the step comment.

## Files changed

- `scripts/check_definition_drift.py`: removes the hash from the block, adds the inherit path to `--update`, adds a second acceptance route to `--gate` (gated on the definitions being equal), compares the whole block, loads the base lock once per run, and updates the docstring.
- `tests/test_metric_definition_drift.py`: adds a real temp-git-repo replay of the incident and a route-2 pin. `_stub_base` now also stubs `_base_last_change`.
- `config/metrics/metric_definitions.lock.json`: drops the legacy `base` hash line. No definition changes.
- `Makefile`, `.github/workflows/orion-static-gates.yml`: rule text now says to re-lock only when you change a definition.

## Schema / bus / API changes

- Added: none
- Removed: the `_last_change.base` field in the lock file (informational only; nothing reads it)
- Renamed: none
- Behavior changed: the gate accepts an untouched inherited block on a branch that changes no definitions
- Compatibility notes: branches locked before this fix still carry a `base` line. The gate ignores it. Their first merge of main after this lands may conflict textually on that one line. To resolve: take main's side (`git checkout --theirs`), commit the merge, then run `--update`. The file ends up identical to main's.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed)
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest tests/test_metric_definition_drift.py -q   -> 56 passed
.venv/bin/python -m pytest tests/test_metric_definition_drift.py tests/test_metric_lineage.py -q -> passed (92 before review additions)
python scripts/check_definition_drift.py --gate                        -> definition drift gate: PASS
python scripts/check_definition_drift.py --update (twice)              -> second run leaves the file byte-identical
```

Incident replay against the pre-fix script (from origin/main): 4 new tests fail there, which is the incident reproduced:
- no-change branch after main records another PR's high-severity change: fails with "committed _last_change does not match the merge-base diff"
- two no-change branches: the lock carries a hash, so it gets a diff
- legacy hash stripped identically on every branch
- untouched lock inheriting the base block

Mutation checks (each mutant was caught):
- M1, inherit path without the definitions-equal condition: caught by `test_relocked_definitions_with_main_block_copied_over_still_fail`
- M2, dropping the definition-diff failure: caught by `test_real_definition_change_with_stale_lock_fails` (shape 2)
- M3, `--update` never inherits: caught by the shape 1 and shape 3 tests
- M4, gate accepts any block: caught by the hand-edit tests

## Evals run

```text
No eval harness applies: this is a static CI gate, and the tests above replay the incident against a real git repo.
```

## Docker/build/smoke checks

```text
Not applicable. This is a static repo gate with no runtime service.
```

## Review findings fixed

- Finding: the PR conflicted with main on the lock's `base` line (#2402 rewrote it).
  - Fix: merged origin/main, took main's lock, then ran `--update`. That strips only the hash line.
  - Evidence: `--gate` PASS against merge base 4ed5ee36e.
- Finding: only `changes` was compared, so the counts could be hand-zeroed.
  - Fix: the gate compares the whole block minus the legacy `base` line.
  - Evidence: `test_untouched_lock_inheriting_the_base_block_is_accepted` (second half).
- Finding: two new tests also passed on the pre-fix script and did not pin route 2.
  - Fix: added a route-2 pin test. It fails on the pre-fix script.
  - Evidence: the pre-fix replay above.
- Finding: the git test fixture inherited `GIT_DIR`/`GIT_INDEX_FILE`, which is a risk if pytest ever runs inside a hook.
  - Fix: the fixture clears those and sets `GIT_CONFIG_GLOBAL=/dev/null` and `GIT_CONFIG_NOSYSTEM=1`.
  - Evidence: tests pass.
- Finding: the failure message named only the recomputed block.
  - Fix: it now also prints the merge base's block when that one is acceptable.
  - Evidence: code in the gate block.
- Finding: the per-run cache is not keyed on HEAD.
  - Fix: documented that it is only valid within one `main()` run. It is reset at the top of `main()`.
  - Evidence: docstring.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: open branches that re-locked before this fix may hit a one-time textual conflict on the `base` line.
  - Mitigation: take main's side, commit the merge, then run `--update`. The result is identical everywhere.
- Severity: low
  - Concern: if main itself is left inconsistent (registries do not match the lock), no-change branches still see drift. That behavior predates this PR.
  - Mitigation: the gate on the definition-changing PR prevents that state.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2403

🤖 Generated with [Claude Code](https://claude.com/claude-code)
