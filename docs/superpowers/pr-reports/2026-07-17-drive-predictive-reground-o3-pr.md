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
- Absolute-threshold firing (`overall_surprise > 0.30`) is intentional, not
  delta-based: unlike the other dimensions in this function (quality scores
  where a *drop* signals tension), `overall_surprise` is already a magnitude
  of surprise/error -- the absolute level itself is the tension.
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
  `extract_tensions_from_self_state()` (26 lines, purely additive)
- `orion/spark/concept_induction/tests/test_prediction_surprise_tension.py`:
  new file, 5 tests (fires above threshold with correct magnitude, silent
  below threshold, trajectory multiplier scaling, clamps at 1.0, silent at
  default 0.0)

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
147 passed, 16 warnings in 14.09s
```

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

Reviewed at medium effort (manual line-by-line + cross-file trace, given the
diff is a 26-line additive block). No findings survived verification.

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

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: LOW. Concern: the `0.30` absolute threshold is a first-pass
  calibration with no live `overall_surprise` distribution to validate
  against (same honesty standard as O4's SATURATED thresholds, which were
  explicitly flagged the same way). Mitigation: plain module-level literal,
  trivially tunable; post-deploy re-measurement against `drive_audits` (same
  command used to verify O1/O4) will show whether `predictive` actually gets
  real variance now.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1114
