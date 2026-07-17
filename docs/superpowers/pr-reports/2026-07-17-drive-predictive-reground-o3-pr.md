# PR report: give predictive drive a primary tension source (O3)

Branch: `feat/drive-predictive-reground-o3`
PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1114
Series: follows `docs/superpowers/pr-reports/2026-07-16-drive-economy-desaturation-o1-o4-pr.md`
(O1 + O4, merged 2026-07-16). This patch is O3 -- one of the two named follow-ups
in that report's diagnosis (O3: predictive re-grounding; O2: event-rate
normalization, still open, separate patch).

## Summary

- `predictive` (one of `DriveEngine`'s 6 drives, `orion/spark/concept_induction/drives.py`)
  previously never had a primary tension source -- it only received secondary
  `drive_impacts` weights (0.4-0.7) riding other dimensions' tensions
  (`tension.contradiction.v1`, `tension.identity_drift.v1`) and sat dead at a
  0.016 median pressure, permanently inactive.
- `extract_tensions_from_self_state()` (`orion/spark/concept_induction/tensions.py`)
  now mints a new `tension.prediction_surprise.v1` tension directly off
  `self_state.overall_surprise` -- an already-clamped [0,1] top-level
  prediction-error aggregate on `SelfStateV1` that this function previously
  never read at all, despite being published on every `substrate.self_state.v1`
  tick this worker already consumes.
- Firing is delta-gated (`surprise_delta > 0.05` once a previous self-state
  exists; absolute-threshold `overall_surprise > 0.30` fallback only when
  there is none), matching the pattern every sibling block in this function
  already uses. An earlier version of this patch fired on the absolute level
  alone, unconditionally, every tick -- code review (see below) found and
  fixed this before merge.
- `drive_impacts={"predictive": 1.0}` -- single-drive, unambiguous primary
  source, no dilution across other drives.
- No wiring changes needed: `ConceptWorker._tensions_from_self_state()` already
  calls this function on every self-state tick and publishes whatever it
  returns through the existing tension -> `DriveEngine.update()` ->
  drive_state/drive_audit path unchanged.

## Outcome moved

This is O3 from the 2026-07-16 drive-economy desaturation diagnosis
(`orion/autonomy/drives_and_autonomy_retrospective.md` §5a). O1/O4 (merged
same day) fixed the dominance-attribution poisoning mechanism; this is half of
the second, still-open mechanism -- pressure pinning from a starved
`predictive` drive. O2 (event-rate normalization) is the other half and is
intentionally not in this patch -- it is a riskier change to the worker's
`DriveEngine.update()` call cadence (touches the live publish-per-event
contract) and is being designed separately before implementation.

## Current architecture

`extract_tensions_from_self_state()` built tensions from
`SelfStateV1.dimensions[key].score` deltas only (`coherence`,
`agency_readiness`/`social_pressure`, `uncertainty`,
`resource_pressure`/`execution_pressure`). `SelfStateV1.overall_surprise` and
`SelfStateV1.prediction_error_scores` (both populated upstream by the
substrate self-state builder) were never read by this function.

## Architecture touched

`orion/spark/concept_induction/tensions.py` only. No bus contract, schema
registry, channel, or env changes -- `tension.prediction_surprise.v1` is a
payload-level `kind` string inside the already-registered
`memory.tension.event.v1` envelope, the same pattern five sibling kinds in
this file already use with zero registry entries (`tension.contradiction.v1`,
`tension.distress.v1`, `tension.identity_drift.v1`, `tension.cognitive_load.v1`,
`tension.satisfaction.v1`).

## Files changed

- `orion/spark/concept_induction/tensions.py`: new block in
  `extract_tensions_from_self_state()`, purely additive, delta-gated (see
  Review findings below for the revision history within this patch)
- `orion/spark/concept_induction/tests/test_prediction_surprise_tension.py`:
  new file, 8 tests -- the original 5 (fires above threshold with correct
  magnitude, silent below threshold, trajectory multiplier scaling, clamps
  at 1.0, silent at default 0.0) plus 3 added during review to cover the
  delta-gated path (sustained-unchanging elevated surprise does not re-fire,
  fires on a genuine rise, small delta below the gate does not fire)

## Schema / bus / API changes

- Added: `tension.prediction_surprise.v1` as a new `TensionEventV1.kind` value
  (payload-internal tag, not a new bus envelope kind or channel)
- Removed: none
- Renamed: none
- Behavior changed: `predictive` now gets a primary growth signal it never
  had; no change to any existing tension kind's behavior
- Compatibility notes: none -- purely additive, no existing consumer branches
  on an exhaustive kind enum (verified: `TensionEventV1.kind` is a free string
  field, not a pydantic `Literal`)

## Env/config changes

None.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-drive-predictive-reground-o3
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/spark/concept_induction/tests -q
150 passed, 16 warnings in 18.76s
```

(147 before the review fix below, 150 after 3 regression tests were added for
the delta-gated path.)

## Evals run

```text
No eval harness exists for this surface (tracked as issue #1066, same gap
noted in the O1/O4 PR). Post-deploy live re-measurement
(measure_autonomy_gate.py against fresh drive_audits) is the behavioral
check, same as O1/O4.
```

## Docker/build/smoke checks

```text
Not run -- pure Python logic change in an already-running consumer path
(ConceptWorker), no config/dependency/port/compose changes. Restart of
orion-spark-concept-induction picks this up.
```

## Review findings fixed

This patch went through two review passes: an initial manual medium-effort
pass (line-by-line + cross-file trace) at first commit, then a full 8-angle
`/code-review medium` pass (3 correctness angles, 3 cleanup angles, altitude,
conventions; 1-vote verify on the surviving candidate) requested separately.
The second pass caught a real bug the first pass missed.

- **Finding (HIGH, confirmed): the original absolute-threshold-every-tick
  firing would saturate `predictive` near 1.0, not fix its starvation.**
  `orion/spark/concept_induction/tensions.py`, the new block. The first
  version fired unconditionally whenever `overall_surprise > 0.30`, with no
  gating on `previous_self_state` (unlike the other 4 blocks in this same
  function, which all switch to delta-based firing once a previous state
  exists). Verification traced `self_state.overall_surprise`'s producer
  (`orion/self_state/prediction.py`) and found it has no smoothing/decay of
  its own -- it's a fresh, potentially-noisy one-step-prediction error every
  tick -- and computed the leaky-integrator math (`DriveEngine.update()`,
  `decay_tau_sec=1800.0` vs ~4.6s bus-tick cadence): a repeated full-weight
  impulse pins pressure above 0.95 within ~5 ticks (~23s). This is the exact
  cpu/gpu_pressure saturation mechanism fixed in PRs #1108-1111, reintroduced
  from the opposite direction -- `predictive` would flip from "permanently
  dead" to "permanently pinned," undoing the point of this patch.
  - Fix: switched to the same delta-gated pattern every sibling block in this
    function already uses (`surprise_delta > 0.05` once `previous_self_state`
    exists; absolute `> 0.30` fallback only on the very first tick with no
    prior state). 3 new regression tests cover it: sustained-unchanging
    elevated surprise does not re-fire, a genuine rise fires on the delta
    (not the absolute level), a small delta below the gate does not fire.
  - Evidence: `150 passed` (up from 147), including the 3 new tests exercising
    the previously-untested repeated-tick path.

- Checked and cleared: `orion/spark/concept_induction/drive_attribution.py`'s
  `_PRIMARY_DRIVE_BY_KIND` map (used only as a tie-break shortcut in
  `dominant_drive_from_attribution`) does not have an entry for the new kind.
  Traced the full cascade: `compute_tick_attribution` sums
  `magnitude x drive_impacts` weight directly and is unaffected by the map.
  When `tension.prediction_surprise.v1` is the tick's lead tension and
  `predictive` ties at max attribution, the map lookup misses and falls
  through to the cascade's step 2, which ranks tied drives by the lead
  tension's own `drive_impacts` weights (`{"predictive": 1.0}`) -- the same
  correct result the map would have given directly. No behavioral gap in any
  constructible scenario; not fixed because there is nothing to fix.
- Checked and cleared: `TensionEventV1.kind` is a free string field, not a
  `Literal`/enum, and no registry/catalog file tracks individual tension kind
  strings (verified against `orion/bus/channels.yaml`, which registers the
  envelope kind `memory.tension.event.v1`, not payload-internal kind tags) --
  consistent with 5 sibling kinds already following this same pattern.
- Accepted, not changed (LOW, established file-wide convention): the new
  block's `if fire_pred and mag_pred > 0.0:` guard is technically redundant
  in every branch (a positive delta or absolute threshold times a positive
  `traj_mul` is always `> 0.0`) -- but every one of the 4 sibling blocks in
  this same function has the identical redundant guard shape. Diverging from
  it here would reduce consistency for no real benefit.
- Accepted, not changed (trivial): `test_surprise_at_max_clamps_at_one`
  doesn't exercise a true out-of-range clamp (Pydantic already constrains
  `overall_surprise <= 1.0`), so it can't prove the `clamp01()` call does
  anything beyond the schema's own guarantee. Left in place -- it still
  pins down exact boundary behavior (`magnitude == 1.0` with default
  `traj_mul`), and removing it buys nothing.
- Checked and cleared (Reuse angle): the new test file's `_self_state()`/
  `_envelope()` helpers duplicate near-identical local helpers already
  present in 3+ other files under `orion/spark/concept_induction/tests/`
  (no shared `conftest.py` exists in that directory). Pre-existing pattern
  across the whole test directory, not something this PR introduced --
  flagged as a real, low-priority, separate cleanup opportunity (a shared
  fixture module), not blocking this patch.
- Checked and cleared (Altitude angle): the delta-gated pattern (vs. this
  repo's more principled `DeviationGate` EWMA/z-threshold mechanism used
  elsewhere) and the choice to ground on the scalar `overall_surprise` (vs.
  the more granular `prediction_error_scores` dict) both match every sibling
  block's existing altitude in this same function -- not a new shortcut
  introduced by this patch.
- Checked and cleared (Conventions angle): no CLAUDE.md violation found
  against the repo-root `CLAUDE.md` (no keyword-cathedral, env/config,
  bus/schema-contract, or PR-report-template rule broken; no directory-scoped
  CLAUDE.md exists under `orion/spark/concept_induction/`).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: LOW. Concern: both the `0.30` absolute-fallback threshold and the
  `0.05` delta gate are first-pass calibrations with no live `overall_surprise`
  distribution to validate against (same honesty standard as O4's SATURATED
  thresholds, which were explicitly flagged the same way). Mitigation: plain
  module-level literals, trivially tunable; post-deploy re-measurement against
  `drive_audits` (same command used to verify O1/O4) will show whether
  `predictive` gets real, non-pinned variance now.
- Severity: LOW (down from the pre-fix HIGH). Concern: even delta-gated,
  `predictive` could in principle still accumulate meaningful pressure if
  `overall_surprise` genuinely oscillates every tick during a volatile
  episode (delta-gated blocks fire intermittently in that case, not on every
  tick, per the verifier's trace -- but not literally zero). Mitigation: this
  is the same accepted risk profile every other block in this function
  already carries; a general per-drive rate-limit/cadence-fold across all
  tension sources is O2's explicit scope, not this patch's.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1114
