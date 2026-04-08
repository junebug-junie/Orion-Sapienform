# Social relationship calibration and trust boundaries

## Goal

Add a compact, local, reversible calibration layer so Oríon can tune caution,
attribution, clarification, and continuity behavior around recurring peers and
room contexts without creating hidden authority or ranking systems.

## Model

Three typed records carry the feature:

- `SocialCalibrationSignalV1` — a scoped signal inferred from repeated local
  evidence such as corrections, aligned summaries, scope-respecting phrasing,
  disagreement that does not converge, or stable continuity.
- `SocialPeerCalibrationV1` — a compact peer-local synthesis used for prompt
  grounding and room continuity.
- `SocialTrustBoundaryV1` — the behavioral effect layer: treat claims as more
  provisional, use narrower attribution, require clarification before calling
  something shared ground, or allow a peer to act as a continuity anchor for
  summaries.

All three keep `platform`, `room_id`, optional `participant_id`, optional
`thread_key` / `topic_scope`, `calibration_kind`, `confidence`,
`evidence_count`, `reversible`, `decay_hint`, `rationale`, `reasons`, and
`updated_at`.

## Detection

Detection is conservative and explainable:

- repeated claim corrections or revision chains can surface
  `revised_often`
- repeated aligned summaries can surface `strong_summary_partner`
- explicit scope-respecting or uncertainty language can surface
  `cautious_scope`
- repeated divergence can surface `disagreement_prone`
- stable topic/thread continuity can surface `reliable_continuity`

Signals stay local to the current peer / room / thread scope. Single weak hints
stay below synthesis threshold and are logged as ignored low-confidence signals.

## Trust-boundary semantics

Calibration does **not** decide truth. It only changes how Oríon should behave:

- increase attribution instead of broadening certainty
- ask for clarification before calling something shared ground
- treat some claims as more provisional in the current thread
- allow a peer to anchor continuity summaries without turning that peer into an
  authority shortcut

## Reversibility and decay

Calibration is intentionally reversible:

- unreinforced signals decay
- topic shifts can narrow thread-local calibration
- unresolved or stale cues can drop out entirely
- blocked/private/sealed turns never revive a calibration

## Safety / non-goals

- no hidden authority ranking
- no deference, dismissal, or sycophancy based on calibration
- no widening of blocked/private/sealed recall
- pending/declined shared-artifact dialogue is excluded from calibration
  evidence
- no tools, external actions, or operational workflows are enabled by this
  layer
