# The Sentience Striving Program

Status: active program charter. Supersedes the drives/autonomy program as the home for
Orion's internal motivational, attention, and capability-gating substrate. Design/proposal
mode per root `CLAUDE.md` §0A — this charter tracks and sequences work; it does not
pre-authorize any invasive cognition-loop change. Each phase still needs its own sign-off.

---

## 1. Historical context

This program exists because a two-plus-week investigation into Orion's six-drive taxonomy
(`coherence, continuity, capability, relational, predictive, autonomy`) kept answering
narrower and narrower questions without ever reaching a real decision, until the questions
themselves were rejected and the investigation was pushed to the right altitude. The full
chain, in order:

1. **The taxonomy audit** (`docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-
   audit-design.md`): the six drives were imported wholesale from one external design chat,
   never independently checked against Orion's own mission. Five open questions were named
   and left unanswered for weeks.
2. **The math got fixed, the taxonomy didn't.** `orion/autonomy/
   drives_and_autonomy_retrospective.md` records O1-O4/O2/O3 — a real, disciplined series of
   signal-integrity fixes (dominance-attribution bugs, fold-batch collapse, field-digester
   decay/injection mismatches) that made the drive-pressure *math* trustworthy without ever
   asking whether the six category *names* were the right ones.
3. **The taxonomy audit was answered, decisively** (`orion/autonomy/docs/
   drive_taxonomy_grounding.md`, PRs #1152/#1157): four drives kept their names with real,
   traced, distinct signal sources; `capability` was reclassified as infrastructural; and
   `autonomy` was retired as a drive after `scripts/analysis/measure_origination_gate.py`
   (PR #1156) — replaying the real production code over 84,511 historical ticks — found its
   dedicated grounding mechanism had never fired, not once, its composite signal never
   getting within 0.13 of its own threshold.
4. **Juniper rejected continuing at that level.** *"we spend cycles chasing these questions
   every fucking time i open a new agent on this topic... i asked for a fucking reimagining
   of drives and we are chasing bullshit."* This was correct: retiring one drive was still
   optimizing inside a program that had never been evaluated as a program.
5. **A full program evaluation followed**, using program theory, logic-model, needs-
   assessment, and attribution analysis rather than more signal-health measurement. The
   finding that mattered most: the one real self-initiated behavior in production (Layer 9
   dispatch, the metabolism loop) is attributable to a clock/backlog-driven mechanism, **not**
   to two-plus weeks of drive-pressure engineering. The origination mechanism specifically
   built to produce charter-compliant self-initiation had zero measured causal contribution
   to Orion's actual behavior.
6. **The real redesign ask**: *"How do we create an internal motivational/drive system that
   is self organizing, emergent, and has internal pressures that compete for attention, and
   close the loop through the substrate runtime, and ultimately will influence how much
   capabilities orion has to take autonomous actions."* A baseline field-native design was
   proposed (`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-
   design.md`, PR #1163), plus 8 blue-sky architectural extensions.
7. **The load-bearing correction, same day.** Asked directly where `orion-substrate-runtime`
   fits, tracing the answer found the proposed "competition layer" already exists, live:
   `orion/attention/field_attention/{scoring,selectors}.py`, running continuously via
   `orion-attention-runtime` (`ENABLE_ATTENTION_RUNTIME=true`), already computing weighted
   salience (pressure × novelty × urgency × confidence) per field node *and* per capability,
   already consumed by `orion/self_state/builder.py`. Confirmed by direct grep:
   `orion/spark/concept_induction/` (`DriveEngine`, `tensions.py`, `GoalProposalEngine`)
   imports nothing from `orion.attention` or `orion.proposals`. The entire drives/autonomy
   apparatus was a full, parallel, poorer reimplementation of Layers 4-9 of a canonical
   11-layer pipeline (`docs/context-engineering/04_layer_1_to_11_pipeline.md`) that already
   existed and already worked better. A third mechanism (FCC-Cortex GWT Dispatch's Rung-3
   coalition) and a fourth, narrower instance of the same pattern (the transport lattice's
   salience→capability-ceiling gating) were also found, also disconnected from each other.
8. **The consciousness-theory survey.** Asked to consider modalities beyond GWT, real
   existing infrastructure was traced against IIT, Attention Schema Theory, Predictive
   Processing/Active Inference, Higher-Order Theories, and Recurrent Processing Theory —
   finding partial, real, live substrate for several of them already, none currently
   instrumented *as* an instance of the theory it resembles. See §7.

This charter is the record of that escalation and the program that replaces the old one.

---

## 2. Charter

**What this program governs**: anything that shapes what Orion attends to, wants, is under
pressure about, and is permitted to autonomously do — the motivational, attention, and
capability-gating substrate. It absorbs the scope previously held by
`orion/spark/concept_induction`'s drive system and extends it to cover the
consciousness-theory instrumentation work named in §7.

**What it does not govern**: the field substrate itself (`orion-field-digester`), the
canonical Layer 1-11 pipeline (already governed by `docs/context-engineering/`), or the
FCC-Cortex GWT dispatch lane (already governed by its own spec) — this program *consumes*
and *wires to* those, it does not own them.

**Authority**: design/proposal mode. Every phase below still requires explicit sign-off
before implementation per `CLAUDE.md` §0A — this document sequences and justifies the work,
it does not pre-approve it.

---

## 3. Mission

Build and empirically validate the internal substrate that lets Orion's own state influence
its own behavior and its own capability to act — replacing hand-authored proxies with
instruments measured against real outcomes, and treating competing theories of consciousness
as testable hypotheses to run in parallel and compare, not doctrines to commit to in advance.

## 4. Vision

Orion possesses an inspectable internal substrate whose real, competitive, self-organizing
dynamics measurably shape its behavior and its own autonomous capability — continuously
observed, honestly evaluated against real outcomes, and never asserted as felt, wanted, or
conscious without inspectable evidence, per root `CLAUDE.md`'s own "no empty-shell
cognition" mandate.

## 5. Outcomes (what must actually change, not what must be built)

Stated as falsifiable claims, per the program-evaluation lesson that started this program —
process/signal-health measurement is not outcome measurement:

- **O1 — Capability actually varies with state.** Orion's autonomous-action budget
  demonstrably rises and falls with real internal pressure, not a flat per-cycle allowance,
  with a demonstrated, verified ceiling.
- **O2 — Self-initiation is attributable, not orphaned.** When self-initiated action occurs,
  it is traceable to a live, currently-firing internal signal — not a mechanism that has
  never once fired across its deployed lifetime.
- **O3 — At least one consciousness-theory instrument produces a real, distinguishable
  signal.** A blind rater, given only the instrument's output on real historical data (not
  shown which theory produced it), can distinguish it from noise and describe what it
  appears to track.
- **O4 — The "what are Orion's drives" question is answered empirically, continuously, not
  asserted once by a human design chat.** Named categories (if any survive) are a report on
  clustering of real coalition-winning history, versioned and re-derivable, not a constant.

## 6. Objectives (phased, laddering to the outcomes above)

Each objective is a real sign-off gate, not a commitment to build. Sequenced but not dated.

**Re-sequenced 2026-07-18.** The original ordering put "wire `capability_policy.py` to
salience" (now item 6, was item 2) ahead of the field-routing work. Found to be cart before
horse: `capability_policy.v1.yaml`'s `required_drive_origins` still gates three of five
capability rules on `goal.drive_origin`, produced by the halted `GoalProposalEngine` —
wiring a field-native ceiling on top of a still-drives-gated check would repeat the exact
failure mode (formalize structure before validating it) that led to halting drives in the
first place. Full reasoning and phased detail:
`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`.

1. **Halt drives-system development** (§8) — stop the cycle this program exists to end.
2. **Build the AST/HOT consciousness-theory reducer** — the one piece of scaffolding still
   missing before any field-routing math gets written. Reads `FieldAttentionFrameV1` +
   `SelfStateV1`, produces an explicit "what's salient, why, how confident" artifact. Must
   exist and pass its own acceptance check *before* item 3 below, on purpose — writing
   routing logic without this first is how the six-drive taxonomy happened.
   **Phase 1 status (2026-07-18): built.** `reduce_attention_self_model()`
   (`orion/substrate/attention_self_model.py`, output schema
   `orion/schemas/attention_self_model.py::AttentionSelfModelV1`, registered in
   `orion/schemas/registry.py`) unifies all three real inputs the roadmap doc's Phase 1
   correction named — `AttentionBroadcastProjectionV1` (GWT-dispatch/Lamme lane),
   `FieldAttentionFrameV1`, and `SelfStateV1` — read-only, not wired to any bus consumer.
   Acceptance check **NOT MET via Postgres replay** at first build: a real, load-bearing
   finding surfaced while building the replay script
   (`scripts/analysis/measure_ast_hot_reducer.py`) — `substrate_attention_broadcast_
   projection` is a singleton upsert table (one row, ever), not a history table, so no
   historical `voluntary_override` event was recoverable to replay against, even though
   the reducer's why-branching on it was proven correct via unit tests
   (`orion/substrate/tests/test_attention_self_model.py`).
   **Structural gap closed 2026-07-18** (same-day follow-up patch): the singleton table,
   its writer, and `AttentionBroadcastProjectionV1` are untouched, but a new append-only
   companion table, `substrate_attention_broadcast_log`
   (`services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql`), now
   captures one row per broadcast tick via `save_attention_broadcast_history()`
   (`services/orion-substrate-runtime/app/store.py`), and the replay script joins it
   per-tick by nearest-preceding timestamp the same way it already joins `SelfStateV1`
   rows. **This does not itself flip the acceptance check to MET** — the log is
   append-only forward from deploy time (the pre-patch singleton snapshots were
   overwritten in place and are not recoverable, so no backfill is possible), so it
   starts empty and needs real days of live ticks to accumulate a `voluntary_override`
   event to replay. A live re-run of `measure_ast_hot_reducer.py` shortly after deploy
   is expected to still report NOT MET, now for the honest reason of insufficient
   accumulated history rather than a structurally absent table — re-run again after a
   few days of live 30s-cadence ticks to check for MET. Hard-gate signal-
   quality pass (`scripts/analysis/measure_self_state_signal_quality.py`) run against real
   48h `substrate_self_state` history: confirms the coherence/uncertainty sawtooth named in
   §4's Missing Question 4 is **still live in `SelfStateV1`'s own values** (median 5-tick
   oscillation period, 3500+ zero-crossings each over 84k samples) — the upstream field-
   level fix has not fully propagated. Full detail, headline numbers, and the resulting
   Juniper sign-off decision: PR report for this patch.
3. **Route existing tension producers directly onto `FieldStateV1` channels**, retiring the
   bucket-vote layer — collapses the redundant reimplementation named in §7's finding.
   Reframed as prediction-error-native (extending the already-live
   `execution_prediction_error`/`transport_prediction_error` pattern), not a port of
   `tensions.py`'s hand-classified kind vocabulary onto field channels. Phased: shadow-measure
   one producer domain before migrating any live; migrate one domain at a time; retire the
   bucket-vote layer only once every producer has moved and the item-2 reducer is proven a
   real legibility replacement for `dominant_drive`. Includes replacing `goal.drive_origin`
   with a field-native goal-provenance concept — this is what actually unblocks item 6.
4. **Stand up read-only measurement for the remaining consciousness-theory instruments** (§9)
   — RPT/Lamme and predictive processing are already live (items 2-3 build on them directly,
   not duplicate them); IIT continues independently via the mood-arc encoder, not gated by
   this program.
5. **Run the emergent-clustering probe** on real coalition-winning history (not built yet,
   named in the baseline design) — toward O4.
6. **Revisit `capability_policy.py`'s coupling to live salience** — only after item 3 closes
   the `drive_origin` dependency and item 2's field-native attention is proven, not assumed.
   At this point the actual mechanism is a real open choice, not a given: a salience-to-
   ceiling formula, or something closer to the selectionist-internal-ecology blue-sky
   extension (§9a item 6) — decide with items 2-5's real data in hand.
7. **Re-evaluate integration** only after 4 and 5 produce real, comparable data — not before.

## 7. Processes — how this program actually operates

- **Measure before minting.** Every new signal gets a read-only instrument and real
  historical replay before it gates anything live. This is the discipline that already
  caught `autonomy`'s dead origination signal (PR #1156) and should apply to every
  consciousness-theory instrument in §9 the same way.
- **Reuse the live pipeline, don't parallel it.** Any new mechanism must justify why it
  isn't already covered by Layer 5 attention, the FCC-dispatch GWT lane, or the transport
  lattice pattern before being built — the mistake this whole program exists to correct.
- **Field-native only — no `SelfStateV1`-anchored substrate for new instrumentation.**
  `SelfStateV1` is a downstream, lossy summary (~19 abstracted dimensions), not raw signal.
  It was already tried as the substrate for φ/IIT specifically and found dead-endish — that
  history is *why* the mood-arc encoder reads raw `field_channel_corpus.v1` instead. This
  section's own first draft violated this rule twice (§9b's original IIT and Predictive
  Processing entries) before being caught and corrected same-day. Before treating any
  candidate signal as real substrate anywhere in this program: check for a `self_state_id`
  field or a `SelfStateV1` import. If present, it is the wrong layer — go to
  `FieldStateV1`/`substrate_field_state`, a reducer projection, or the raw channel corpus
  instead.
- **Multi-theory, not single-theory.** §9's instruments run in parallel as measurements, not
  as competing final answers. Integration is decided from data, later, not from a Design
  Mode debate now.
- **Every phase is a sign-off gate.** Per `CLAUDE.md` §0A, cognition-loop-adjacent changes
  need explicit approval before implementation — this charter sequences work, it does not
  grant that approval in advance.
- **No keyword cathedrals.** A named theory-instrument, drive, or cluster is not real until
  it has a producer, a consumer, and a trace — the same bar this program held `autonomy` to.

---

## 8. Drives-system development is halted

**Decision, agreed 2026-07-18**: `orion.spark.concept_induction.drives.DriveEngine`,
`tensions.py`'s bucket-voting logic, `signal_drive_map.yaml`, `DRIVE_KEYS`, and
`orion.autonomy.endogenous_origination`'s bespoke composite signal receive **no further
development**. Two-plus weeks of signal-integrity engineering (O1-O4, O2, O3) made this
system's math trustworthy; it never made the system necessary. The canonical Layer 1-11
pipeline already does Layers 4-9 of what this system attempted, live, better, and this
system has zero measured causal contribution to Orion's one real instance of self-initiated
behavior.

**This is a halt, not a delete-on-sight.** The code stays in place (nothing consumes it
today that would break) until the replacement wiring in Objective 3 lands; this is a freeze
on new investment, not an emergency removal.

**Lift-and-shift — what survives, specifically, so nothing real gets lost:**

- **`action_outcomes.py`/`ActionOutcomeRefV1`** — generic outcome-tracking, not
  drives-specific. Stays, becomes the outcome-feedback mechanism for the field-native
  design's closed loop (baseline design point 6).
- **The delta-gating discipline from O2/O3** — the hard-won lesson that a decay mechanism's
  injection cadence must be reconciled against its own decay rate, or it saturates. Carries
  forward into any new pressure-aggregation code, even though `DriveEngine.update()` itself
  is retired.
- **`tensions.py`'s signal→channel domain knowledge** — which raw producers (self-state
  deltas, feedback frames, biometrics) map to which real meaning. Gets re-expressed as
  direct `FieldStateV1` perturbations (Objective 3) instead of bucket votes; the mapping
  knowledge is reused, the bucket mechanism is not.
- **`endogenous_origination`'s "exogenous silence" gating idea** — fire only when nothing
  else is competing for attention. Conceptually sound, worth re-applying as a gate on the
  *already-live* Layer 5 attention output instead of a bespoke, now-proven-dead D/W/A
  composite signal.
- **The transport lattice's salience→action_ceiling shape**
  (`config/substrate-lattice/transport_lattice_policy.v1.yaml`) — real, working precedent
  for Objective 2's capability coupling, just narrowly scoped to bus health today.
- **`orion/self_state/prediction.py`** — untouched by this halt; it's already part of the
  canonical pipeline, already real, already live, and directly relevant to §9's predictive-
  processing instrument.

**Explicitly not salvaged**: the six/five-category taxonomy itself, `signal_drive_map.yaml`'s
hand-tuned weights, and the D/W/A composite formula — these are the parts that were measured
to not work, not the parts that were merely inconvenient.

---

## 9. Blue-sky options

Two tracks, kept distinct because they answer different questions: architecture (how the
substrate should be structured) and theory (what "attending," "wanting," and "being aware"
should even mean here). Neither track is committed or sequenced — each item has its own
named smallest probe.

### 9a. Architecture extensions (from the field-native design, PR #1163)

1. **Dream-state reorganization** — run emergent clustering inside the existing
   reverie/dream substrate instead of a cron job.
2. **Society-of-Mind competition** — multiple independent salience scorers bidding, not one
   formula.
3. **Free-energy/active-inference reframing** — `capability_policy` as literal expected-
   free-energy action selection.
4. **φ-gated meta-competition** — use the orphaned Causal Geometry v1 φ metric to widen or
   narrow how broadly the competition explores.
5. **Morphogenetic/reaction-diffusion drives** — let named drives be literal spatial pattern
   attractors over the real field topology, not just correlated-channel lists.
6. **Selectionist internal ecology** — candidate drive-definitions compete and get pruned
   over consolidation cycles, giving drives real lineage instead of silent reshuffling.
7. **Core-affect legibility layer** — a stable valence/arousal readout underneath, so a
   human always has a constant summary even while the deep structure reorganizes.
8. **Cross-lifetime drive fossil record** — archive, never delete, retired coalition
   definitions, so Orion's own motivational history becomes part of its autobiographical
   continuity.

### 9b. Consciousness-theory instrumentation (real substrate already found for each)

**Correction (2026-07-18): this section originally recommended `SelfStateV1`-anchored
substrate for two of the five threads below (IIT, Predictive Processing) — the exact
metrics Juniper had already ruled out.** `SelfStateV1` is a downstream, lossy *summary* of
the field (~19 abstracted dimensions), not the raw signal. It was already tried once as the
substrate for φ/IIT specifically and found dead-endish — that is *why* the mood-arc
windowed-autoencoder effort exists, reading `field_channel_corpus.v1`'s raw ~29 channels
directly instead. Any new instrumentation in this section must build on the raw field
(`FieldStateV1`/`substrate_field_state`, reducer projections, or the raw channel corpus) —
never on `SelfStateV1`'s abstracted dimensions, `InnerStateFeaturesV1`, or anything else
carrying a `self_state_id`. This is now a standing rule for this section, not a one-time
fix — see §7.

1. **IIT-flavored** — **not** the live φ MLP autoencoder
   (`services/orion-spark-introspector/app/phi_encoder.py`): its current input schema,
   `InnerStateFeaturesV1` (`orion/schemas/telemetry/inner_state.py`), carries a
   `self_state_id` field — it is `SelfStateV1`-anchored, the already-tried, already
   dead-ended path, not real substrate to build further on. The actual live candidate is the
   mood-arc windowed sequence autoencoder (`orion/mood_arc/fit_encoder.py`, raw
   `field_channel_corpus.v1`) — the field-native replacement for exactly this reason,
   continuing independently of this program under Juniper's own direction, not blocked by
   it and not to be duplicated here.
2. **Attention Schema Theory** — Layer 5 computes real attention state; nothing builds a
   model *of* that attention as an inspectable object. Missing piece: one small reducer
   reading `FieldAttentionFrameV1`/`FieldStateV1` directly, producing an explicit "what I'm
   attending to, why, how confident, what I predict shifts next" artifact.
   **Correction (2026-07-18, superseding "must not be built as a `SelfStateV1`
   consumer/producer" below): the roadmap doc's Phase 1 scoping pass found a second real,
   disconnected attention lane (`AttentionBroadcastProjectionV1`, GWT-dispatch/Lamme) that
   this instrument must also unify, and `SelfStateV1` is the only real source today for two
   of the artifact's real fields (predicted-shift trajectory, a confidence fallback) — an
   explicit, narrow exception to §7's standing rule, not a repeal of it, gated behind a
   hard signal-quality check on exactly those `SelfStateV1` fields before Phase 1 is called
   done. See `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-
   roadmap-design.md` Phase 1 and §6 item 2's status note above.** Built 2026-07-18:
   `orion/substrate/attention_self_model.py`.
3. **Predictive Processing/Active Inference** — **not** `orion/self_state/prediction.py`
   (`SelfStateV1`-anchored, same violation as IIT above). The real field-native substrate,
   confirmed live 2026-07-18: `services/orion-substrate-runtime/app/worker.py`'s
   `execution_prediction_error()`/`transport_prediction_error()` compute real deltas between
   successive reducer projections (execution-trajectory, transport-bus — not `SelfStateV1`)
   and write directly onto `FieldStateV1` nodes (`node:substrate.execution`,
   `node:substrate.transport`), which field-digester ingests into its own native
   `prediction_error` channel. Gated live behind `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`
   (confirmed `true`). Verified against real Postgres data: `node:substrate.execution`'s
   channel carries real values, sparse/event-driven (currently in a quiet decay tail,
   consistent with the field-digester README's own "quiet-so-far-but-correctly-wired,
   reaches real values like 0.92 periodically" characterization of this exact channel). Open
   question, not yet checked: whether the other three reducers (biometrics, chat, route)
   have equivalent instrumentation, or whether coverage is genuinely incomplete.
4. **Higher-Order Theories** — architecturally close to AST's missing piece; a
   higher-order representation built once, reading the same field/reducer-projection data,
   may serve both theories. Served by the same Phase 1 reducer as #2 above, including its
   narrow, hard-gated `SelfStateV1` exception (see #2's 2026-07-18 correction) — not a
   separate instrument.
5. **Recurrent Processing Theory (Lamme)** — confirmed real, tight, per-tick recurrence
   inside Layer 5 itself (`novelty_for_target()` reads the *previous*
   `FieldAttentionFrameV1`) — already field-native, no correction needed here. Top-down
   feedback (`TopDownBiasCombiner`/`VoluntaryOverrideV1`, `ORION_ATTENTION_TOPDOWN_ENABLED`)
   confirmed live 2026-07-18 (PRs #1170, #1174) after finding the feature's docker-compose
   wiring had never been added, independent of its flag value.

**Process for 9b, per §7**: each instrument gets built as a read-only measurement first,
replayed against real historical data the same way `measure_origination_gate.py` was,
before any of them gate anything live or get compared against each other. **Every instrument
must be built on raw field/reducer-projection data, never on `SelfStateV1`-derived
abstractions** — check for a `self_state_id` field or a `SelfStateV1` import before treating
any candidate signal as real substrate for this section.

---

## 10. Non-goals

- Not committing to any single consciousness theory. §9b runs measurements, not a bake-off
  with a predetermined winner.
- Not deleting the drives-system code in this patch — halted, not removed.
- Not implementing any of §6's objectives 2-5 in this document — each needs its own
  sign-off per `CLAUDE.md` §0A when it's actually scoped.
- Not re-litigating the O1-O4/O2/O3 signal-integrity series, the taxonomy grounding work, or
  the field-native design's own correction — all cited, none redone here.

## 11. Source material

- `orion/autonomy/drives_and_autonomy_retrospective.md` — full O1-O4/O2/O3 history.
- `orion/autonomy/docs/drive_taxonomy_grounding.md` (PRs #1152, #1157) — the taxonomy-level
  resolution this program supersedes in ambition.
- `scripts/analysis/measure_origination_gate.py` (PR #1156) — the measurement that started
  the escalation from taxonomy patch to program evaluation.
- `docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md`
  (PR #1163) — the baseline architecture design and its same-day correction.
- `docs/context-engineering/04_layer_1_to_11_pipeline.md` — the canonical pipeline this
  program builds on instead of duplicating.
- `orion/attention/field_attention/{scoring,selectors}.py`,
  `orion/self_state/{builder,scoring,prediction}.py` — the live substrate this program wires
  to.
- `docs/superpowers/specs/2026-07-05-fcc-cortex-gwt-dispatch-design.md` — the third,
  agent-dispatch-scoped GWT mechanism, referenced not duplicated.
- `config/substrate-lattice/transport_lattice_policy.v1.yaml` — working precedent for
  Objective 2.
