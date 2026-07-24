# Phase 3 shadow-measure finding: biometrics prediction-error vs. drives bucket-vote

`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
Phase 3 asks whether field-native prediction-error routing gives a recognizably different
(and better) picture than the old drives bucket-vote output, for the same real ticks. All
five domains' producer instrumentation shipped 2026-07-21 (charter §6 item 3); this is the
comparison itself, run for the first time, 2026-07-24, real Postgres data, 18h window
post-rebuild.

## What was compared

- **New side**: `biometrics_prediction_error()` (`orion/substrate/prediction_error.py`),
  written to `substrate_field_state`'s `node:substrate.biometrics` node.
- **Old side**: `drive_audits.drive_pressures` (`capability`/`continuity`), the only two
  drives `biometrics_state` deviations route into
  (`config/autonomy/signal_drive_map.yaml`, confirmed live 2026-07-24). No dedicated
  biometrics tension kind exists (`TensionEventV1.kind` for this path is the generic
  `"tension.signal.v1"`, `orion/autonomy/signal_tension.py`) -- `dominant_drive`/
  `drive_pressures` is the only axis the old system exposes for this domain.
- Both sides confirmed to have zero SelfStateV1 dependency before building this (see the
  charter's 2026-07-24 status note under §6 item 2).

Script: `scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py`. Read-only,
no writes/events/flags/consumer changes. 19 unit tests, all passing, no DB required.

## Real numbers (18h window, 2026-07-24, post-rebuild)

- 41,919 `drive_audits` rows, 31,793 `substrate_field_state` rows, 100% join coverage
  (within a 30s staleness bound).
- Both signals show real, non-degenerate variance (biometrics_prediction_error:
  0.0-0.4856; capability/continuity pressure: 0.0-1.0).
- **Correlation** (biometrics_prediction_error vs. max(capability, continuity) pressure):
  Pearson r = 0.0198, n = 41,919. Essentially no linear relationship.
- **Split comparison**: mean biometrics_prediction_error when `dominant_drive` in
  {capability, continuity} = 0.0259 (n=1,610); when some other drive dominates = 0.0332
  (n=31,322). Difference is small and in the *wrong* direction -- the new signal reads
  slightly lower, not higher, when the old system attributes pressure to the drives
  biometrics feeds.

This is not a small-sample artifact -- n is large on both tests, coverage is complete, and
neither series is degenerate.

## Interpretation -- genuinely ambiguous, not a clean verdict

Two real, competing explanations, and this measurement cannot distinguish between them on
its own:

1. **The signals are genuinely unrelated.** `biometrics_prediction_error` may not be
   capturing whatever real phenomenon the old system's capability/continuity pressure
   responds to.
2. **The old baseline is too polluted to be a fair comparison partner.**
   `signal_drive_map.yaml` also routes `mesh_health` (weight 0.5) and `failure_event`
   (weight 0.5) into `capability` -- the same bucket biometrics feeds. A drive-audit event
   dominated by capability could be driven entirely by a mesh-health or failure event with
   zero biometrics contribution that tick. If so, the near-zero correlation reflects the
   old system's many-to-one bucket-vote design diluting domain-specific signal, not a
   defect in the new one -- which would itself be a concrete, real-data illustration of the
   exact problem Phase 3-5 exist to fix (a single dominant_drive label can't attribute back
   to which of 3 real domains actually caused it).

## What this does NOT decide

- Does not authorize migrating biometrics to a live field-native path (Phase 4).
- Does not by itself justify retiring any part of the bucket-vote layer (Phase 5).
- Does not conclude `biometrics_prediction_error` is broken or unvalidated -- its own
  shadow-design doc (2026-07-21) already established it as a real, non-degenerate signal
  on its own terms; this is a *relationship* question, not a re-litigation of that.

## Recommended next step

Disentangle explanation 1 from 2 before drawing further conclusions: repeat this comparison
against a drive-audit subset filtered to events where `mesh_health`/`failure_event` tensions
were NOT also present in `tension_kinds` for that fold window (isolating capability/
continuity movements attributable to biometrics alone). If correlation improves
substantially on that filtered subset, explanation 2 (bucket dilution) is supported --
independent confirmation the bucket-vote layer loses domain attribution, strengthening the
case for Phase 4 migration. If it stays near zero even isolated, explanation 1 is
supported -- worth a second look at `biometrics_prediction_error`'s own formula before
trusting it as this domain's field-native replacement. Not built in this patch --
next real step, not assumed.

## Source material

- `scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py` -- the
  measurement itself.
- `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
  -- Phase 3's scope and acceptance framing.
- `docs/superpowers/specs/2026-07-21-biometrics-prediction-error-shadow-design.md` -- the
  new signal's own metric-quality-gate validation.
- `config/autonomy/signal_drive_map.yaml` -- the real biometrics-to-drive routing weights.
