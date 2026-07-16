# PR report — attention broadcast typing pinned to stale raw prediction_error

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/dynamic-pressure-reason-typing)
Branch: `fix/dynamic-pressure-reason-typing`
Status: **DONE**

## Summary

- Fixed a review finding from PR #1061: `orion/substrate/attention_broadcast.py::_node_salience()`'s
  `kind` (used for `target_type_hint`: `"anomaly"` vs `"concept"`) stayed
  pinned to `"prediction_error"`/anomaly forever once any nonzero raw
  `metadata["prediction_error"]` was ever observed on a node — even after
  the node's *current* `dynamic_pressure` was being driven entirely by an
  unrelated source (drive seed, contradiction propagation). Salience
  magnitude/ranking was already correct (fixed in #1061); this only
  affected display/typing.
- `orion/substrate/dynamics.py::SubstrateDynamicsEngine.tick()` already
  computed a `reasons: dict[str, str]` per node inside `_compute_pressures()`
  (values like `"prediction_error_seed"`, `"drive_seed"`,
  `"contradiction_unresolved"`, `"prediction_error_propagation:{predicate}"`,
  `"drive_propagation:{predicate}"`, `"contradiction_involved"`) but
  discarded it after use. Now persisted onto the node as
  `metadata["dynamic_pressure_reason"]` at the same `node.model_copy()`
  call site that already writes `metadata["dynamic_pressure"]`.
- `_node_salience()` now derives `kind` from that reason string
  (`startswith("prediction_error")` → anomaly typing, else → concept
  typing) instead of the raw presence check, with a fallback to the old
  presence-check behavior for nodes that predate any dynamics tick (no
  `dynamic_pressure_reason` key at all).
- Code review (high effort) found a second, related bug while reviewing
  this same seam: the drive_seed and contradiction passes in
  `_compute_pressures()` wrote their reason string unconditionally
  whenever their own seed/amp was positive — even when a different pass's
  already-larger pressure value was the actual dominant driver for that
  node. Only the prediction_error pass had the correct
  `if seed > pressure[node_id]:` guard. Fixed both to match that pattern.
  This bug predates this PR (it lived in `_compute_pressures()` before the
  `reasons` dict was ever consumer-visible) but this PR is what made it
  typing-relevant for the first time.

## Outcome moved

A node's anomaly-vs-concept typing in the substrate attention broadcast
now reflects what is *currently* driving its pressure, not a raw seed
field that never decays and never clears once set. Concretely: a node
that once had a prediction-error surprise but is now purely
drive-pressured (or contradiction-pressured) will correctly type as
`"concept"`/`"pressure"` instead of permanently as `"anomaly"`.

## Current architecture

`SubstrateDynamicsEngine.tick()` (`orion/substrate/dynamics.py`) is
instantiated and ticked in the live worker loop
(`services/orion-substrate-runtime/app/worker.py:798`), not just in
tests — a real runtime seam. It writes `metadata["dynamic_pressure"]`
onto substrate nodes each tick. `attention_broadcast.py::_node_salience()`
reads that metadata (via `substrate_pressure_signals()` →
`build_substrate_attention_frame()`) to build workspace competition
signals; `kind` feeds both `target_type_hint` and `signal_kind` on the
resulting `AttentionSignalV1`.

## Architecture touched

`orion/substrate/dynamics.py` (writer), `orion/substrate/attention_broadcast.py`
(reader/typing), plus their test suites. No schema/bus/env changes — this
is a new key inside the existing flexible `metadata: dict` field already
carried by substrate nodes.

## Files changed

- `orion/substrate/dynamics.py`: `tick()` now writes
  `metadata["dynamic_pressure_reason"] = pressure_reasons.get(node_id, "none")`
  alongside `metadata["dynamic_pressure"]`. `_compute_pressures()`'s
  drive_seed and contradiction passes now guard their reason write with
  `if seed/amp > pressure[node_id]:`, matching the prediction_error pass's
  existing pattern, so the reason attributed to a node is always the
  actually-dominant source, not whichever pass ran last.
- `orion/substrate/attention_broadcast.py`: `_node_salience()` derives
  `kind` from `metadata["dynamic_pressure_reason"]` when present
  (`startswith("prediction_error")` → anomaly), falling back to the old
  raw-presence check when the key is absent (pre-tick node). Docstring
  updated to explain the staleness problem and the fallback.
- `orion/substrate/tests/test_attention_broadcast.py`: updated
  `test_prediction_error_beats_equal_plain_pressure` and
  `test_salience_uses_decayed_pressure_not_raw_prediction_error` to set
  `dynamic_pressure_reason` explicitly (realistic post-tick node state,
  not relying on the fallback). Added 4 new regression tests: drive_seed
  reason types as concept despite stale raw prediction_error;
  prediction_error_propagation reason types as anomaly; no-reason-metadata
  node falls back to old presence-check behavior.
- `tests/test_cognitive_substrate_phase4_dynamics.py`: added
  `test_dynamic_pressure_reason_is_persisted_on_node_metadata` (covers
  drive_seed/prediction_error_seed/prediction_error_propagation/no-driver
  cases via a real `tick()`) and
  `test_dynamic_pressure_reason_attributes_to_the_dominant_source_not_the_last_pass`
  (regression for the review-found second bug — a single node with a
  dominant prediction_error seed and a smaller, unrelated open
  contradiction must keep `dynamic_pressure_reason == "prediction_error_seed"`,
  not have it clobbered by the later-running contradiction pass).

## Schema / bus / API changes

- Added: `metadata["dynamic_pressure_reason"]` (string) on substrate
  nodes, written by `SubstrateDynamicsEngine.tick()`. No schema/model
  change needed — `metadata` is an already-flexible dict field on
  `BaseSubstrateNodeV1`.
- Removed: none.
- Renamed: none.
- Behavior changed: `_node_salience()`'s `kind` (and therefore
  `AttentionSignalV1.target_type_hint`/`signal_kind`) now reflects the
  current dominant pressure driver instead of a raw, never-clearing
  metadata field, for any node that has been through at least one
  dynamics tick.
- Compatibility notes: fully backward compatible. Nodes without the new
  key (never ticked) fall back to the exact pre-patch behavior.

## Env/config changes

None. No `.env_example` changes; `python scripts/sync_local_env_from_example.py`
not applicable.

## Tests run

```text
.venv/bin/python -m pytest orion/substrate/tests/ tests/test_cognitive_substrate_phase4_dynamics.py tests/test_cognitive_substrate_phase5_graph_cognition.py -q
=> 223 passed

.venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py -q
=> included in the 227-pass run below

.venv/bin/python -m pytest orion/substrate/tests/ tests/test_cognitive_substrate_phase4_dynamics.py tests/test_cognitive_substrate_phase5_graph_cognition.py services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py -q
=> 227 passed
```

Broader sweep (`services/orion-substrate-runtime/tests/`, excluding a
Postgres-dependent integration test file) surfaced 14 failing tests
unrelated to this diff (cursor/reducer-health/quarantine tests requiring
a live Postgres on `localhost:5432`, refused). Confirmed pre-existing on
unmodified `main` by running the same specific failing tests from the
original `/mnt/scripts/Orion-Sapienform` checkout — same failures, same
reasons, before this branch existed.

Sanity-checked the new
`test_dynamic_pressure_reason_attributes_to_the_dominant_source_not_the_last_pass`
regression test actually catches the review-found bug: temporarily
reverted just the contradiction-pass guard, reran the test, confirmed it
failed with `AssertionError: assert 'contradiction_unresolved' == 'prediction_error_seed'`,
then restored the fix and reran the full suite green.

## Evals run

None applicable — no eval harness exists for `orion/substrate/dynamics.py`
or `attention_broadcast.py`; this is deterministic unit-testable logic,
fully covered by the gate tests above.

## Docker/build/smoke checks

Not run — pure Python logic change inside `orion/substrate/` (shared
package code consumed by `orion-substrate-runtime`'s worker loop). No
config, dependency, port, or compose changes to validate.

## Review findings fixed

Code-review skill (subagent, high effort) against the diff:

- **Finding** (MATERIAL): `_compute_pressures()`'s drive_seed pass
  (`dynamics.py`, was line ~195) and contradiction pass (was line ~256)
  wrote `reasons[node_id]` unconditionally whenever their own seed/amp
  was positive, even when `pressure[node_id]` was already higher from a
  different pass — only the prediction_error pass correctly gated with
  `if seed > pressure[node.node_id]:`. Concrete impact: a node whose
  pressure is genuinely dominated by `prediction_error_seed` but also has
  a smaller, unrelated open contradiction would get its reason silently
  overwritten to `"contradiction_unresolved"` by the third pass, flipping
  its typing from anomaly to concept — directly undermining this patch's
  stated purpose.
  - **Fix**: mirrored the existing `if seed > pressure[node_id]:` pattern
    in both the drive_seed and contradiction passes.
  - **Evidence**: new test
    `test_dynamic_pressure_reason_attributes_to_the_dominant_source_not_the_last_pass`
    reproduces the exact scenario (single node with both a dominant
    prediction-error seed and a smaller open contradiction), fails without
    the fix (`AssertionError: assert 'contradiction_unresolved' ==
    'prediction_error_seed'`, verified by temporary revert), passes with
    it.
- **Finding** (MINOR, accepted as-is, not fixed): a node whose pressure
  coincidentally settles within `1e-6` of its previous value while the
  underlying driver silently changes in the same tick keeps a stale
  `dynamic_pressure_reason` (the pre-existing early-continue in `tick()`
  skips the whole metadata rewrite, not just `dynamic_pressure`). Same
  class of approximation the pre-existing `dynamic_pressure` skip already
  makes; requires a near-exact magnitude coincidence across a driver
  change in the same tick — far narrower than the material finding above.
  Left as-is to keep the patch thin; documented in this report rather than
  silently ignored.
- Confirmed not issues (reviewer verified, no action needed): the
  `"none"`/absent-key distinction in `_node_salience()`'s fallback logic
  is correct (metadata key is only ever written as a string once ticked,
  never `None`); `str(reason).startswith("prediction_error")` is
  exhaustive against all six possible reason values with no
  false-positives; the modified/new tests exercise real code paths, not
  vacuous fixtures; the runtime wiring is real (`worker.py:798`), not an
  empty-shell change.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not run this session — pure logic change inside `orion/substrate/`
(shared package), takes effect on next `orion-substrate-runtime` restart/
redeploy. No other services import `dynamics.py`/`attention_broadcast.py`
directly.

## Risks / concerns

- Severity: low
- Concern: the `1e-6` early-continue staleness case described above
  (minor review finding, not fixed) means `dynamic_pressure_reason` can
  theoretically lag one tick behind a driver change in a narrow coincidence
  case.
- Mitigation: not fixed in this patch to keep it thin; same tolerance the
  existing `dynamic_pressure` value itself already accepts. Worth a
  one-line follow-up docstring note in `tick()` if it ever becomes a real
  observed issue.

## PR link

`gh` is authenticated in this environment — see below.
