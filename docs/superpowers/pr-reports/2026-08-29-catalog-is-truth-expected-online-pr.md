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

## Second half: clearing the residue the false alarm already wrote

Confirmed live after #1940 deployed without this fix: the sweep fired **142 times in 20
minutes**, every one naming `atlas`, and wrote real state to the projection:

```text
atlas -> pressures=["strain","availability"]
         impacts=["capability:batch_inference","capability:embedding",
                  "capability:local_llm_heavy","capability:local_llm_quick"]
```

Making the catalog authoritative stops the sweep -- but it does **not** clear what was
already written, and nothing else can: `node_availability_recovered` only fires when a node
starts reporting again, and a decommissioned box never will. The row would freeze forever and
keep feeding the concept graph via `biometrics_ctx.py`.

- `sweep_suppressible_nodes()`: retired (`expected_online: false`) + stale + **still carrying
  state**. Self-terminating by construction -- a node only qualifies while it still has
  something to clear, so it goes quiet on its own and cannot become the permanent alarm it
  exists to remove.
- `node_pressure_suppressed` (Rule A) now clears `capability_impacts` and sets
  `availability_status="suppressed"`. It is the only reachable clearing path for a node that
  will never report again.

**Silver lining worth stating plainly:** this false alarm is also the first live proof the
detector works end to end. `capability_impacts` had been `[]` in every row ever written; the
full chain (sweep -> synthetic trigger -> organ -> reducer -> projection) fired correctly. It
found a genuinely absent node. The node was just the wrong one.

### Three stale tests this surfaced

`test_biometrics_pressure_organ.py` had three tests asserting a world where **circe is
offline and atlas is online** -- the pre-2026-07-18 catalog. They passed only because the
organ read the test's own projection value instead of the catalog. Once the catalog became
authoritative their premises were simply false. Retargeted each to a node whose catalog entry
actually matches the rule under test; the rules themselves are unchanged.

## Risks / concerns

- Severity: low. A node the catalog does not know still falls back to the stored flag, so an
  uncatalogued node is not silently ignored. Covered by a test.
- Severity: low. The stale `expected_online: true` value on atlas's projection row is still
  wrong at rest -- this patch stops it being *read* and clears the pressure state it caused,
  but does not rewrite the cached flag itself. Same class as the known
  `reconcile_field_state_with_lattice()` gap (fills missing keys, never heals persisted stale
  ones).
- Severity: low, unchanged. This is still a signal, not a notification. Nothing yet turns a
  capability transition into something that reaches Juniper, and `notify_attempts` has 0 rows
  ever.
