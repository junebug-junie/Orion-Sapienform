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
surviving heuristic slice), `BiometricsClusterV1`, and the L7–L11 ladder.

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
  the existing bare-string list.
- `SHADOW` — real, live, deliberately **not** composed, with a required,
  stated reason (`shadow_reason`). The model case: phi's surviving
  `valence` heuristic, explicitly justified because no trained latent
  correlates with anything hedonic-adjacent.
- `DUPLICATE` — an unresolved overlap with another entry (`duplicate_of`),
  e.g. `drive_state.v1`/`autonomy_state_v2` — same 6-drive taxonomy, two
  independent reducers, not yet reconciled (traffic-gated decision, on
  record separately — this registry makes the fact visible, it doesn't
  force the answer).
- `REHEARSAL` — computed, verified to reach no cognition consumer at all
  (the L7–L11 ladder).

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
