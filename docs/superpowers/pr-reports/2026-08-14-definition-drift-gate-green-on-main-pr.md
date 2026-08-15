# The definition-drift gate could never be green on main

## Summary

- `main` is currently red on Static repo gates, and has been since the gate landed.
  Both post-gate commits fail it: `4cca1eb5f` (PR #1666, which introduced the gate)
  and `b4a697a9d` (current HEAD). Every open PR inherits the failure, since CI runs
  the workflow against the merge with main.
- Two distinct causes, both fixed here.
- **Cause 1, real drift:** the committed lock still records
  `declared_consumers: ['orion-state-service']` for `orion:spark:signal`, which PR
  #1665 removed from `orion/bus/channels.yaml`. Re-locked, so the change is stated.
- **Cause 2, structural:** the gate recomputes `_last_change` from the merge base and
  fails if the committed block disagrees. On `main` the merge base *is* HEAD, so the
  recomputed block collapses to "no definition changes" while the committed block
  states what the branch that landed actually changed. They can never agree — so the
  gate goes red on main after **every** definition change, by construction.
- Fixing cause 1 alone would have been a treadmill: green for one PR, then red again
  the moment it merged, and red for every branch cut from it afterwards.

## Outcome moved

Static repo gates can be green on `main` again, which unblocks every open PR. The
gate's actual purpose — an agent quietly changing a metric's meaning gets caught — is
preserved intact.

## Current architecture

`scripts/check_definition_drift.py` fingerprints 595 metric definitions into
`config/metrics/metric_definitions.lock.json`. `--update` re-locks and derives a
`_last_change` block; `--gate` recomputes that block and fails on disagreement. That
recomputation is what makes the alert a constraint rather than a convention — hand-
editing the block to erase an alert fails the gate.

## Architecture touched

Only the gate's base-branch handling and the lock's content. No metric definition,
registry, schema, bus channel, or service behaviour changes.

## Files changed

- `scripts/check_definition_drift.py`: `_base_is_head()`, and skip the derived-block
  recomputation when HEAD is the merge base — with the skip printed, never silent.
- `config/metrics/metric_definitions.lock.json`: re-locked. Drops the stale
  `declared_consumers` for `orion:spark:signal` and states the change.
- `tests/test_metric_definition_drift.py`: two regression tests.

## Why the skip does not weaken the gate

The recomputation answers "what does this **branch** change relative to the merge
base?". That question is meaningful on a PR branch and meaningless on the base branch
itself, where the diff is empty by construction.

On a real PR branch HEAD is never the merge base, so tampering is still caught exactly
where it matters — in the PR that does the tampering. `test_off_the_base_branch_a_hand_edited_alert_still_fails`
pins that, and it was verified by hand as well: hand-editing `_last_change` to
`["nothing to see here"]` on a PR branch still fails.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behaviour changed: `--gate` no longer recomputes the committed `_last_change` when
  HEAD is the merge base. Prints an explicit NOTE instead.
- Compatibility notes: the lock format is unchanged.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not applicable, no env surface touched.
- local `.env` synced: not applicable.
- skipped keys requiring operator action: none.

## Tests run

```text
$ pytest tests/test_metric_definition_drift.py -q
47 passed in 5.16s        # 45 pre-existing + 2 new

# every gate the CI job runs, locally:
PASS  check_metric_lineage.py --gate
PASS  check_definition_drift.py --gate
PASS  check_inner_state_registry.py
PASS  check_scripts_dir_no_stdlib_shadow.py
PASS  check_service_hostname_refs.py
PASS  check_journal_dispatch_registry.py
PASS  check_daily_schedule_collisions.py
```

## Evals run

No eval harness applies. This is a deterministic repo gate; the two regression tests
plus the seven live gate runs above are the coverage.

## Docker/build/smoke checks

Not applicable — no runtime, container, or service behaviour is touched. Evidence was
gathered by running the gate itself against three real repository states:

```text
(a) pristine origin/main b4a697a9d          -> FAIL   (this is the bug)
(b) this branch, base b4a697a9d             -> PASS
(c) origin/main ref moved to HEAD,
    simulating main after this merges       -> PASS + explicit NOTE
(d) _last_change hand-edited on a PR branch -> FAIL   (anti-tamper intact)
```

## Review findings fixed

Self-reviewed. One caught during writing: the first version of
`test_on_the_base_branch_the_derived_block_is_not_recomputed` hand-wrote a
`definitions` dict that does not match `build_lock`'s fingerprint shape, so the gate
failed on ordinary drift and the test would have passed for the wrong reason once
inverted. Rebuilt from the stubbed graph via `build_lock(build_graph())`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: on the base branch the committed `_last_change` is no longer
  verified at all, so a block that was already wrong when it merged stays unnoticed
  there. Mitigation: it is verified on the PR branch that writes it, which is the only
  place it can be recomputed — and the skip is printed rather than silent.
- Severity: low. Concern: the `orion:spark:signal` consumer removal is acknowledged by
  this PR rather than by #1665, which made it. It is a real, intentional change
  (biometrics hub-mode flip; the channel has no subscribers), but the acknowledgement
  is landing one PR later than it should have.
