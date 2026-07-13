# orion/self_state

Builds `SelfStateV1` (Layer 6: the mood) from `FieldStateV1` (the body,
`orion-field-digester`) and `FieldAttentionFrameV1` (per-node/per-capability
attention scoring, `orion-attention-runtime`). Consumed by the
`orion-self-state-runtime` service.

- `builder.py` — `build_self_state()`, the per-tick composer.
- `scoring.py` — per-dimension scoring formulas, channel merge/max rules.
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
  Two current entries: the L7–L11 ladder, and `mood_arc_corpus.v1`
  (2026-07-13) — an append-only training-data sink for a not-yet-built
  windowed felt-state autoencoder
  (`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md`),
  deliberately dark until real hours of data accumulate.

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
