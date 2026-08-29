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

## Third: the last hop -- a dark node now reaches Juniper

The arc's actual ask. The substrate has detected absence since #1940, but that produced a
signal Orion could read and nothing that reached anyone: during the ~45 minute outage of the
entire local GPU fleet, `notify_requests` recorded **nothing at all**.

`_node_availability_checks()` adds one `HealthCheck` per catalogued node whose projection
carries an `availability` pressure, naming the node, the staleness threshold, and **which
capabilities were lost** -- an operator needs to know what went away, not just that a box is
quiet.

**Reuses `HealthMonitor` instead of adding a second notifier.** Every property this needs is
already there and is hard to get right: edge-triggered (fires on transition, never per tick --
a per-tick alert on a 45-minute outage would be ~90 pages), a recheck debounce, restart
handling via `_has_open_alert`, retry-on-failure, and a recovery note when the node returns.

**A state transition, not a threshold -- there is no number to tune.** That is the point of
the whole design. The obvious alternative, alerting on transport-error rate, was measured and
**rejected**: over 315 hours the real outage hour scored *below* the p95 of ordinary hours,
because the trigger rate is cooldown-capped at 120/hr. No cut separates the classes.

Suppressed (decommissioned) nodes are skipped, so atlas can never page again.

**The sink is verified, not assumed.** A real notification was sent end to end on 2026-08-29
(`severity=error`, `channels=[email, in_app]`, `recipient_group=juniper_primary`) and Juniper
confirmed receipt by email. That check was the stated prerequisite for building this at all.

### Notify observability gap found while verifying (NOT fixed here)

- `notify_attempts` has a table definition in `services/orion-notify-digest/app/db_models.py`
  and **no writer anywhere in the repo**. Zero rows in its lifetime because nothing was ever
  built to fill it -- so "0 attempts" was never evidence of non-delivery.
- `notify_requests.status` is written once as `"pending"` and never updated; all 10,671 rows
  since 2026-07-20 read `pending` regardless of outcome.
- The notify service's own `[NOTIFY]` logger is not wired to stdout -- not one of its
  decision lines (`email_send_eligible`, `email_send_attempted`) appears in
  `docker logs`, only uvicorn access lines. That is why this went unnoticed for months.

Delivery works. Its accounting does not exist. Worth its own patch.

## Risks / concerns

- Severity: low. A node the catalog does not know still falls back to the stored flag, so an
  uncatalogued node is not silently ignored. Covered by a test.
- Severity: low. The stale `expected_online: true` value on atlas's projection row is still
  wrong at rest -- this patch stops it being *read* and clears the pressure state it caused,
  but does not rewrite the cached flag itself. Same class as the known
  `reconcile_field_state_with_lattice()` gap (fills missing keys, never heals persisted stale
  ones).
- Severity: medium. The alert rides `health_check_interval_sec`, default **900s**. Worst-case
  time-to-page for a node going dark is therefore ~15 minutes plus the recheck delay, not
  ~3 minutes. Detection is fast; paging is on the health tick's clock. Lowering that interval
  is an operator call, not something this patch should decide unilaterally.
- Severity: low. Alert text names the lost capabilities from `capability_impacts`, which is
  only correct because Rule F expands from the catalog. If a node's catalog entry is wrong,
  the page is wrong in the same way.
