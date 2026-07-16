# orion/self_state

Builds `SelfStateV1` (Layer 6: the mood) from `FieldStateV1` (the body,
`orion-field-digester`) and `FieldAttentionFrameV1` (per-node/per-capability
attention scoring, `orion-attention-runtime`). Consumed by the
`orion-self-state-runtime` service.

- `builder.py` — `build_self_state()`, the per-tick composer.
- `scoring.py` — per-dimension scoring formulas, channel merge/max rules.
  For what each of the 29 raw `field_channel_corpus.v1` channels means, how
  it's calculated, and which dimension (if any) it feeds, see
  `services/orion-field-digester/README.md`'s "Field channel glossary".
- `policy.py` — loads `config/self_state/self_state_policy.v1.yaml`
  (`channel_dimension_map`, `evidence_channel_map`, `dimension_worse_direction`).
- `deviation.py` — Phase 2 deviation-probe instrumentation (measurement only,
  reuses `DeviationGate`'s EWMA-baseline mechanism).
- `prediction.py` — inter-tick prediction/surprise loop.
- `transport.py` — `transport_integrity` dimension helpers.

## `inner_state_registry.py`

`SelfStateV1` is the one schema every cognition-facing prompt-builder
(metacog templates, chat prompts) is expected to read from — directly, or
via phi's `InnerStateFeaturesV1`. This module is the registry of every
"what does Orion currently feel/perceive" signal in the repo, not just the
ones `SelfStateV1` itself composes: `FieldStateV1`, `FieldAttentionFrameV1`,
`DriveStateV1`, `AutonomyStateV2`, phi (both the trained encoder and the
surviving heuristic slice), `BiometricsClusterV1`, the L7–L11 ladder, and
`mood_arc_corpus.v1` — a training-data sink, not a bus signal, for a
not-yet-built downstream model.

Each `InnerStateSignal` entry names its producer service, cadence, and one
of four composition statuses:

- `COMPOSED` — has a named field on `SelfStateV1` (or, for phi, is wired
  into a cognition-facing narrative). `field_attention_frame.v1` moved from
  `SHADOW` to `COMPOSED` in Phase 1 (2026-07-12): `builder.py` previously
  read the full per-node/per-capability `FieldAttentionTargetV1` list and
  kept only bare `target_id` strings on `dominant_attention_targets`,
  discarding `pressure_score`/`dominant_channels`/`reasons`. Structured
  per-target data now survives, additively, on
  `SelfStateV1.dominant_attention_target_details`
  (`AttentionTargetSummaryV1`: `target_id`, `target_kind`, `pressure_score`,
  top `dominant_channel`, top `reason`) — same target_ids, same order, as
  the existing bare-string list. Phase 2 (2026-07-12) builds on this:
  `PhiIntrinsicRewardV1.dominant_node`/`dominant_node_reason`
  (`orion-spark-introspector`) name the most salient real hardware node,
  filtered to `target_kind == "node"` and excluding two synthetic
  pseudo-nodes — confirmed live that a `target_kind == "system"` entry
  frequently wins the #1 salience slot, so `target_kind` filtering matters
  as much as the pseudo-node exclusion. Phase 3 (2026-07-12) closes the loop:
  `dominant_node`/`dominant_node_reason` are threaded through
  `SparkStateSnapshotV1` (the relay schema `orion-cortex-exec` actually
  reads) into `spark_embodiment_narrative`, rendered into both metacog
  prompt templates alongside `spark_phi_narrative` — confirmed live in
  production before this phase shipped (`node:atlas`/`node:circe`
  alternating as `capability:llm_inference`'s real GPU contention winner,
  81.7%/18.3% split over a 295-tick window — see
  `docs/notes/2026-07-12-phase4-attention-provenance-crosscheck.md`).
- `SHADOW` — real, live, deliberately **not** composed, with a required,
  stated reason (`shadow_reason`). No current entry holds this status: the
  model case was phi's `valence` heuristic, justified on the claim that no
  trained latent correlates with anything hedonic-adjacent — checked
  directly against the active encoder's real `probes.json` on 2026-07-13
  and found **false** (`agency_readiness` is a real encoder input feature,
  correlating with 6 of 8 latents up to `|r|=0.686`). Flipped to
  `COMPOSED` the same day (`fix/valence-probe-readout`, PR #985) via a
  probe-weighted readout, `_agency_valence_proxy()`. Left here as the
  standing example of why `shadow_reason` claims need re-checking against
  real artifacts, not just cited from memory.
- `DUPLICATE` — an unresolved overlap with another entry (`duplicate_of`),
  e.g. `drive_state.v1`/`autonomy_state_v2` — same 6-drive taxonomy, two
  independent reducers, not yet reconciled (traffic-gated decision, on
  record separately — this registry makes the fact visible, it doesn't
  force the answer).
- `REHEARSAL` — computed, verified to reach no cognition consumer at all.
  Five current entries: the L7–L11 ladder; `mood_arc_corpus.v1`
  (2026-07-13) — an append-only training-data sink for a not-yet-built
  windowed felt-state autoencoder
  (`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`),
  deliberately dark until real hours of data accumulate. Superseded
  (2026-07-13, same day, later session) by `field_channel_corpus.v1` as
  roadmap item 2's actual intended input — a full pass at item 2 found
  that `mood_arc_corpus.v1`'s apparent "trajectory structure" was almost
  entirely explained by `orion-field-digester`'s own `apply_decay(0.92)`
  mechanism, because it captured `_phi_from_self_state()`'s already-
  smoothed, already hand-weighted 4-scalar *output* rather than raw
  substrate. `field_channel_corpus.v1` (producer: `orion-field-digester`,
  not `orion-spark-introspector`) instead captures
  `collect_field_channel_pressures()`'s flat, variable-width channel dict —
  the layer before that hand-weighting, still carrying the same
  unavoidable `apply_decay` smoothing but not the additional composite.
  `mood_arc_corpus.v1` itself is untouched and keeps collecting; this is
  additive, not a disable; `mood_arc_encoder.v1`
  (2026-07-13, same day, roadmap item 2 now built) —
  `scripts/fit_mood_arc_encoder.py` trains a windowed autoencoder over
  `mood_arc_corpus.v1` rows and writes a dark, disk-only
  `MoodArcEncoderManifestV1` artifact triad (`manifest.json`/`weights.npz`/
  `probes.json`); no bus publish, no cognition consumer, same REHEARSAL
  reasoning as its corpus sibling. That same session found the spec's
  original single shuffle-baseline gate too weak on its own — this corpus's
  real autocorrelation is largely explained by a known, deliberate
  leaky-integrator decay mechanism (`BIOMETRICS_FIELD_DECAY_RATE=0.92`,
  `services/orion-field-digester/app/digestion/decay.py`), so an encoder
  could pass the original gate purely by learning that already-known filter.
  Addressed with a two-tier gate (a shuffle floor, hard-gated, unchanged
  threshold from the spec; an AR(1)-surrogate ceiling, diagnostic-only, not
  yet calibrated) plus a purged/embargoed temporal train/held-out split
  (naive random window sampling leaks given measured autocorrelation out to
  lag ~10-15 ticks across 50%-overlapping windows) and a block-bootstrap
  confidence interval on the floor ratio — none of this is in the original
  written spec doc, it is stricter than what item 2 originally asked for;
  see `MoodArcEncoderManifestV1`'s docstring
  (`orion/schemas/telemetry/mood_arc.py`) for the field-level rationale so a
  future reader doesn't have to re-derive it; and `chat_stance_disposition`
  (2026-07-13) — the Thought proceed/defer/refuse decision per unified-turn
  chat turn, real and correctly computed but dead-ends at the raw
  `active_chat_session` ledger row today. A composition route into
  `SelfStateV1.social_pressure` was considered and rejected (that dimension
  is already excluded from φ's live trainable feature set — see
  `docs/superpowers/specs/2026-07-13-stance-disposition-inner-state-path.md`
  for the full trace and the candidate paths forward, none chosen yet).

This was built because the same failure mode — a real signal silently
duplicating another, or never reaching cognition — was independently
rediscovered five times by manual grep-archaeology in one session
(`docs/superpowers/specs/2026-07-12-inner-state-unification-design.md`).
It is deliberately **not** a merge into `orion/schemas/registry.py` (that
file is a general-purpose name→class lookup with hundreds of unrelated
entries; wrong blast radius for this) and deliberately **not** a service —
merging `orion-spark-introspector` (offline-trained), `orion/autonomy`
(chat-turn-gated), and `orion-attention-runtime` (per-tick) into one
codebase would trade five duplicated computations for one badly-scoped
service.

`scripts/check_inner_state_registry.py` (`make check-inner-state-registry`)
runs two checks against this module: a rot check (every registered schema
still imports and is a real `BaseModel`) and a best-effort new-duplicate
heuristic (a new bus channel or schema whose name matches an inner-state
keyword, with no registry entry, fails the gate by name). The heuristic's
real limitation is documented in the script itself — it is a maintained
keyword list, not a formal proof.
