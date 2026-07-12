# Inner-State Unification — Implementation Plan

Companion to `docs/superpowers/specs/2026-07-12-inner-state-unification-design.md`. Read that first — this is the phased build-out: the foundational registry/gate, then the field-attention brainstorm ideas correctly sequenced *through* it instead of around it.

**Cross-cutting rule for every phase below**: no phase merges a service into another. Every phase either (a) adds a registry entry for something that already exists, or (b) adds a new signal that is a registry entry *from its first commit*, never added after the fact.

---

## Phase 0 — The registry and the gate (foundational, blocks nothing downstream from starting late)

**Gate to start:** none — this is the first patch, can start immediately. The design spec now specifies this down to the field level (dataclass, all nine entries populated, gate algorithm) — this phase implements that spec, it does not redesign it.

- [ ] `orion/self_state/inner_state_registry.py`: the `Cadence`/`CompositionStatus` enums, the `InnerStateSignal` frozen dataclass, and the populated `REGISTRY` tuple with all nine entries — verbatim starting point in the design spec. `drive_state.v1` and `autonomy_state_v2` are registered as `DUPLICATE` (`duplicate_of` pointing at each other), **not** composed into `self_state.v1` — traced this session: they read a disjoint, closed `signal_kind` evidence map (`config/autonomy/signal_drive_map.yaml`), not `SelfStateV1.dimensions`, with exactly one narrow, event-gated overlap point (`spark_signal` on `turn_effect_alert`). Do not build a drives→self-state crosswalk; the design spec traced this and rejected it as fabricating a relationship the data doesn't have.
- [ ] `scripts/check_inner_state_registry.py`: two-part gate, both parts specified in the design spec — (1) rot check: import every non-`None` `schema` in `REGISTRY`, confirm it's still valid; (2) new-duplicate heuristic: scan `orion/bus/channels.yaml` and `orion/schemas/registry.py`'s flat name dict for new entries matching a small, explicit, hand-maintained keyword list (`self_state`, `drive`, `autonomy_state`, `attention`, `phi_reward`, `mood`, `felt_state`, `cluster`) with no `REGISTRY` entry. State the heuristic's real limitation in the code comment (a cleverly-named new schema can evade it) — do not oversell it as complete.
- [ ] Register the *existing* CLAUDE.md-promised-but-missing gates (`scripts/check_schema_registry.py`, `scripts/check_bus_channels.py` — confirmed absent, referenced in CLAUDE.md §11/§17) as a separate, explicit finding in the PR — do not silently fix them as scope creep; name them, let Juniper decide if they're this patch's job or a separate one.
- [ ] Wire the new gate into `make agent-check` (per CLAUDE.md §17), alongside a note that the other two promised checks still don't exist.

**Tests:** (1) rot-check regression: a `REGISTRY` entry pointing at a renamed/deleted class must fail. (2) Heuristic regression: a synthetic new channel named `orion:test:mood_state` added to a test fixture of `channels.yaml` with no registry entry must fail, by name, in the gate's error message.

**Acceptance:** all nine entries from the design spec's per-signal deep-dive are present, with real (not placeholder) `producer_service`/`cadence`/`composition_status`/`cognition_consumers` values; the gate is wired into `agent-check`; CLAUDE.md's pre-existing missing-gate gap is documented, not silently absorbed into this patch's scope.

---

## Phase 1 — Widen `dominant_attention_targets` (brainstorm idea #1, now a registry-governed extension)

**Status: implemented, not yet deployed** (branch `feat/self-state-attention-target-composition`).

**Gate to start:** Phase 0 merged (this is the first signal added *through* the registry, proving the pattern works before anything else uses it).

- [x] `orion/schemas/self_state.py`: additive `AttentionTargetSummaryV1` (`target_id`, `target_kind`, `pressure_score`, top `dominant_channel`, top `reason`) + `SelfStateV1.dominant_attention_target_details: list[AttentionTargetSummaryV1]`, alongside (not replacing) the existing bare-string `dominant_attention_targets`.
- [x] `orion/self_state/builder.py`: new `_attention_target_summary()` helper, populated from `attention.dominant_targets[:5]` (the same slice already used for the bare-string list, `builder.py:302` — same target_ids, same order).
- [x] Registry entry updated in the same PR: `field_attention_frame.v1` flipped from `SHADOW` to `COMPOSED`.
- [ ] Log-only for a real traffic window before anything downstream (Phase 2/3) reads it: does the dominant target actually vary tick-to-tick, or is it dominated by one or two nodes constantly? **Blocked on deployment** — not yet observable.

**Tests:** schema unchanged behavior for existing consumers (full self-state suite green, 63 tests); new builder test (`test_dominant_attention_target_details_carries_structured_data`) asserting real, non-fixture end-to-end data (via `build_attention_frame()` + `build_self_state()` on a synthetic `FieldStateV1`) — target_id/order parity with the existing list, valid `target_kind`, `pressure_score` in range, and the top target's `dominant_channel`/`reason` populated.

**Acceptance:** the field carries real per-target data in at least one live tick, verified against Postgres, not just a fixture. **Not yet verified — requires merge + redeploy of `orion-self-state-runtime`, not done as of this patch.**

---

## Phase 2 — Node-attributed phi (brainstorm idea #2)

**Gate to start:** Phase 1 merged AND its log-only traffic window shows real (non-degenerate) variation in dominant targets — do not build this on top of a signal that turns out to be flat.

- [ ] `orion/schemas/telemetry/phi_encoder.py`: additive `dominant_node`/`dominant_node_reason` fields on `PhiIntrinsicRewardV1`.
- [ ] `services/orion-spark-introspector/app/worker.py`: populate from `ss.dominant_attention_targets`'s widened form (Phase 1) inside `handle_self_state`, alongside the existing golden-phi overrides — same function, same seam, not a new pipeline.
- [ ] Registry entry updated: `PhiIntrinsicRewardV1` now explicitly composes attention data, not just self-state's dimension scores.
- [ ] Explicitly filter the two synthetic pseudo-nodes (`node:substrate.execution`, `node:substrate.transport`) from ever being the reported `dominant_node` — named in the brainstorm as a real risk (they aren't bodies, they're derived subsystem metrics; surfacing them as "the body part under stress" would be a category error, not a bug fixed by more data).

**Tests:** unit test asserting a fixture with a clear dominant node produces the matching `dominant_node`/reason; a second test asserting a pseudo-node is never selected even when it has the highest raw pressure score.

**Acceptance:** `PhiIntrinsicRewardV1.dominant_node` reflects a real hardware node (not a pseudo-node) in live traffic.

---

## Phase 3 — `spark_embodiment_narrative` (brainstorm idea #3)

**Gate to start:** Phase 2 merged AND deployed for a real traffic window (this is the one phase that changes prompt content — needs its own confirmation the metacog surface reads coherently, per the design spec's acceptance checks on prompt-builder discipline).

- [ ] `services/orion-cortex-exec/app/spark_narrative.py`: new `spark_embodiment_hint`/`spark_embodiment_narrative`, mirroring `spark_phi_hint`/`spark_phi_narrative`'s existing shape exactly.
- [ ] `orion/cognition/prompts/log_orion_metacognition_{draft,enrich}.j2`: one new template line.
- [ ] Registry entry: this is the first prompt-builder addition made *through* the contract from day one — the model entry for what "declared cognition consumer" should look like everywhere else.
- [ ] Before merging: the design spec's outstanding acceptance check ("does an existing prompt-builder bypass the contract?") gets answered concretely for this new addition specifically — it must not.

**Tests:** template rendering test with a fixture snapshot; confirm the narrative string names a real hardware node, not a pseudo-node (same guard as Phase 2, at the render layer too — defense in depth).

**Acceptance:** a real chat/metacog turn post-deploy produces a narrative that correctly names the actual stressed node, cross-checked against the live `substrate_field_state` row for that tick.

---

## Phase 4 — `capability_provenance` cross-check (brainstorm idea #6, research-only)

**Gate to start:** can run any time in parallel with Phases 1-3 — read-only, no code path shared with them.

- [ ] One research doc (`docs/notes/`), no code: correlate `dominant_attention_targets` (attention's salience-based node attribution) against `capability_provenance` (diffusion's separate, provenance-based node attribution) over a real traffic window. Do they agree? If they diverge, why — different weighting philosophy, different update cadence, or one of them being wrong?

**Tests:** none — read-only research.

**Acceptance:** a documented answer to whether these two independently-computed attribution mechanisms corroborate each other, informing whether Phase 5+ (if pursued) should unify them or keep them as intentionally distinct signals.

---

## Explicitly deferred (not phases — non-goals for this plan)

- **Trained per-node encoder** (brainstorm idea #4): only worth building once Phases 1-3 show the *heuristic* node-attribution signal actually moves cognition. Building a trained encoder for a signal nobody's shown matters yet repeats today's original phi mistake (train something, wire it to nothing) at a new layer.
- **`FieldAttentionFrameV1` bus publish** (brainstorm idea #5): only pursued if Phase 1's Postgres-read pattern (mirroring `orion-self-state-runtime`'s existing `load_self_state_for_attention_frame`) proves insufficient for whatever needs it.
- **First-class node/body-schema object** (brainstorm idea #7): a consumer of this contract once it's real, not part of building it.
- **DriveEngine/AutonomyStateV2 merge decision**: still gated on real AutonomyStateV2 traffic (Phase 4 of the earlier mesh-substrate-redesign plan), unaffected by this plan. The registry (this plan's Phase 0) makes their unresolved duplication a visible, checked fact — it does not force the answer.
- **`CLUSTER_ROLE_WEIGHTS` retirement**: already researched and recommended (docs/notes/2026-07-12-phase4-cluster-weighting-research.md); registered as a known duplicate here, resolved on its own separate timeline.

---

## Full gate before calling this initiative done

```bash
python scripts/check_inner_state_registry.py
pytest orion/self_state -q
pytest services/orion-spark-introspector/tests -q
pytest services/orion-cortex-exec/tests -k spark_narrative -q
```

Run the code-review skill in a subagent per phase before merging, not once at the end — same discipline as every other multi-phase initiative this session. Each phase gets its own PR and PR report.
