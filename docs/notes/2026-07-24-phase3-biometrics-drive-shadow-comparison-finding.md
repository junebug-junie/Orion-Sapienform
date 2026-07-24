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
trusting it as this domain's field-native replacement.

## Resolved same-day (2026-07-24): the disentangling step was built and run

`scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py` gained
`BIOMETRICS_ISOLATED_TENSION_KINDS`/`is_biometrics_isolated_event()`/`filter_isolated()`.
Traced every real tension kind that can carry a `capability`/`continuity` drive_impact --
not just `mesh_health`/`failure_event` as originally scoped, but also
`tension.chat_evidence.v1` (chat_social_hazard), `tension.cognitive_load.v1`/`tension.
distress.v1`/`tension.identity_drift.v1` (turn-effect deltas), and `tension.satisfaction.v1`
(action outcomes) -- `orion/spark/concept_induction/tensions.py`, confirmed live 2026-07-24.
`tension.drive_competition.v1` (empty `drive_impacts`, a pure competition marker) and
`tension.contradiction.v1` (coherence/predictive only) don't pollute and don't disqualify a
tick.

**Correction (same-day review, 2026-07-24): the isolation is weaker than first claimed.**
The filter (`tension_kinds == {"tension.signal.v1"}`) does NOT isolate biometrics-only
events. `mesh_health` deviations also emit this same generic `"tension.signal.v1"` kind in
practice (`orion/signals/adapters/equilibrium.py` -> `"orion:signals:equilibrium"` channel
-> `bus_worker.py`'s generic "signal" rail -> `signal_to_tension()`,
`orion/autonomy/signal_tension.py`) -- the originally-assumed `"tension.health.v1"` kind for
mesh_health is dead code (`SignalTensionSource.from_equilibrium` has zero live callers).
`drive_audits` has no `signal_kind`/`evidence_text`/`related_nodes` column to disambiguate
after the fact (only a generic `summary` like "pressure concentrates on capability"), and
the one channel that could independently confirm mesh_health's real firing rate
(`"orion:equilibrium:snapshot"`) is not durably logged to Postgres (consumer is
`orion-cortex-orch` only) -- checked and ruled out, no way to bound the contamination rate
with real data. The isolated subset (n=1,607-1,612 depending on run) is therefore
"capability/continuity movement attributable to biometrics_state OR mesh_health, nothing
else" -- narrower evidence than a clean biometrics-only isolation, not a false result.

**Result: correlation did NOT meaningfully improve when isolated.** Full dataset: r≈0.016-
0.020 (n≈41,900). Isolated subset: r≈0.035-0.046 (n≈1,607-1,612). Both remain close to
zero; the change (~+0.02-0.03) is small, well under this script's own
`MIN_ISOLATED_N_FOR_INTERPRETATION`-gated "substantial improvement" bar.

**This weakly favors explanation 1 over explanation 2 -- weakly, given the mesh_health
caveat above.** Removing chat-hazard/turn-effect/action-outcome pollution (though not
mesh_health specifically) and still finding near-zero correlation is real, if imperfect,
evidence the old bucket's dilution isn't the whole story -- `biometrics_prediction_error`'s
own formula not capturing whatever the old system's capability/continuity pressure responds
to remains the better-supported explanation, just not an airtight one. This does NOT mean
the old drives system is validated or should stay -- it means the specific claim "the new
signal is secretly the same thing, just measured more cleanly" isn't well supported by this
data, with the caveat that a truly clean biometrics-only test was not achievable with data
this script has access to. Worth a second look at `biometrics_prediction_error`'s formula
(`orion/substrate/prediction_error.py`) before treating it as this domain's field-native
replacement -- not built in this patch, a real next step. A cleaner disambiguation (e.g. a
durable log of `"orion:equilibrium:snapshot"`, if that's ever added) would strengthen this
further but isn't required to act on the current, weaker-but-real signal.

## Source material

- `scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py` -- the
  measurement itself.
- `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
  -- Phase 3's scope and acceptance framing.
- `docs/superpowers/specs/2026-07-21-biometrics-prediction-error-shadow-design.md` -- the
  new signal's own metric-quality-gate validation.
- `config/autonomy/signal_drive_map.yaml` -- the real biometrics-to-drive routing weights.
