# PR: the catalog decides `expected_online`, not a nine-day-old cache

## Summary

- `sweep_absent_nodes()` and `invoke_biometrics_pressure()` now resolve `expected_online`
  from `config/biometrics/node_catalog.yaml` for any node the catalog knows, instead of the
  copy cached on the biometrics projection.
- Caught by verifying #1940 against live data before trusting it, not by a test.

## Outcome moved

**Without this, the first thing Orion's new absence detector reports is a permanent false
alarm on a machine that no longer physically exists.**

`atlas` was decommissioned 2026-08-21 -- its GPUs moved into circe, its disks into athena --
and the catalog was set to `expected_online: false` with the note "It will never report
again." But `NodeBiometricsStateV1.expected_online` is a cache that
`orion/substrate/biometrics_loop/node_reducer.py:97` refreshes **only when the node sends an
event**, and atlas's last biometrics row is `2026-08-20T22:43:26Z` -- the day *before* the
catalog change. Live check 2026-08-29:

```text
node_catalog.yaml : expected_online: false
live projection   : true          (last_seen_at 2026-08-20, ~9 days stale)
```

`sweep_absent_nodes()` read the cache, so #1940 as merged would have swept atlas on every
tick, forever. The flag whose entire purpose is to describe a node that has stopped reporting
could never be updated for a node that had stopped reporting.

## Files changed

- `orion/substrate/biometrics_loop/pressure_organ.py`: `_expected_online()` resolver; used by
  the sweep and by the organ's Rule A/B branch, which had the identical stale-cache
  preference at line 183 (a decommissioned node could never reach Rule A
  `node_pressure_suppressed` and would raise availability concerns indefinitely)
- `services/orion-substrate-runtime/app/worker.py`: pass the catalog through
- `tests/test_capability_absence_signal.py`: +3 tests, existing sweep tests updated
- `tests/test_absence_sweep_wiring.py`: signature update

## Schema / bus / API changes

None. `sweep_absent_nodes()` gains a required `catalog` argument; both call sites updated.

## Env/config changes

None.

## Tests run

```text
pytest tests/test_capability_absence_signal.py tests/test_absence_sweep_wiring.py -q
                                                          -> 22 passed
pytest services/orion-substrate-runtime/tests -q --continue-on-collection-errors
                                                          -> 305 passed, 17 failed, 1 error
```

The 17 + 1 are pre-existing: verified by reverting both source files to `origin/main` in
place and re-running -- byte-identical.

`test_decommissioned_node_is_never_swept_despite_a_stale_cached_flag` reproduces the exact
live shape (atlas `expected_online: true` cached, `last_seen_at` 9 days old, alongside a
healthy circe). Mutation-tested: reverting the resolver to `state.expected_online` fails that
test and `test_a_catalog_change_takes_effect_without_the_node_reporting`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Verify: `docker logs --since 10m orion-athena-substrate-runtime | grep biometrics_absence_sweep`
should be **silent** while every real node is reporting -- and must never name `atlas`.

## Risks / concerns

- Severity: low. A node the catalog does not know still falls back to the stored flag, so an
  uncatalogued node is not silently ignored. Covered by a test.
- Severity: low, NOT fixed. The stale `expected_online: true` **row for atlas remains in the
  projection**; this patch stops it being read, it does not heal it. Same class as the known
  `reconcile_field_state_with_lattice()` gap (fills missing keys, never heals persisted stale
  ones). Harmless now that the catalog wins, but it is still a wrong value at rest.
- Severity: low, unchanged. This is still a signal, not a notification. Nothing yet turns a
  capability transition into something that reaches Juniper, and `notify_attempts` has 0 rows
  ever.
