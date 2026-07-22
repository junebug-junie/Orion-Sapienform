# Objective 3 roadmap — consciousness-theory scaffolding before math/logic lock-in

Status: design mode. No code changes proposed here. Reorders the charter's own Objective
2/3 sequencing based on a real, named risk: building Objective 3's field-routing mechanics
before any consciousness-theory scaffolding exists would repeat the exact failure mode that
killed the drives system — formalizing structure (which channels matter, how they combine,
what "tension" means) before validating it against anything, and then being stuck defending
that structure for weeks once code and habit accumulate on top of it.

## Arsonist summary

Objective 2 (wire `capability_policy.py` to field-attention salience) was found, on inspection,
to still depend on the halted drives system underneath — `capability_policy.v1.yaml`'s
`required_drive_origins` checks `goal.drive_origin`, which is produced by `GoalProposalEngine`,
which lives inside the halted `orion/spark/concept_induction/` directory. Wiring a genuinely
field-native ceiling on top of a still-drives-gated check is cart before horse — the ceiling
would be dressing, not substance.

More importantly: Objective 3 (route tension producers directly onto `FieldStateV1`, retire
the bucket-vote layer) is *itself* at risk of the same failure mode if it starts with "port
`tensions.py`'s kind-labels onto field channels" as the design. That would just rebuild the
six-drive taxonomy's category-first, validate-never problem one layer down — new hand-picked
channel mappings, new hand-picked aggregation formulas, locked in before anything checks
whether they're the right structure. The explicit ask: use the consciousness-theory threads
already scaffolded this program (RPT/Lamme live, predictive-processing substrate live) plus
one more (AST/HOT) *before* writing Objective 3's actual routing math, so the math has
something real to be informed by instead of being invented from scratch and then defended.

## Current architecture (what's already real, to build on top of, not duplicate)

- **RPT/Lamme — live.** `TopDownBiasCombiner` (`orion/substrate/attention/top_down.py`),
  real biased-competition attention, `ORION_ATTENTION_TOPDOWN_ENABLED=true` confirmed live
  (PRs #1170, #1174). A goal's priority × relevance already competes against bottom-up
  salience and can flip the winner — a real competition mechanism, not hypothetical.
- **Predictive processing — live, field-native.** `execution_prediction_error()`/
  `transport_prediction_error()` (`orion-substrate-runtime/app/worker.py`) compute real
  deltas between successive reducer projections and write directly onto `FieldStateV1`
  nodes, flowing into field-digester's own `prediction_error` channel. Confirmed live
  against real Postgres data this session.
  **Correction (2026-07-22): `execution_prediction_error()`'s "confirmed live" status above
  was wrong for that one function.** It diffed by exact `trace_id` match, but real
  cortex-exec runs are single-shot creates (a fresh trace_id every time) — the match
  structurally never occurred, so this instrument returned `0.0` in perpetuity regardless
  of real execution volume, not "live and validated." Fixed in
  `orion/substrate/prediction_error.py` (fallback to the most-recently-updated prev run
  when no exact trace_id match exists) — see `services/orion-substrate-runtime/README.md`'s
  matching section for the full trace. `transport_prediction_error()`'s live/validated
  status is unaffected — its key (`bus_id`) genuinely does recur across polls, this was a
  defect specific to execution's (and, same design, route's) trace_id-keyed instruments.
- **AST/HOT — not built.** The identified gap: Layer 5 computes real attention state
  (`FieldAttentionFrameV1`); nothing builds a model *of* that attention as an inspectable
  object. This is the one piece of scaffolding actually missing before Objective 3 should
  start writing routing logic.
- **`capability_policy.py`'s drives dependency — not yet resolved.** `required_drive_origins`
  gates three of five capability rules on `goal.drive_origin`, sourced from the halted
  `GoalProposalEngine`. Objective 3, done with this in mind, is the natural place to also
  replace `drive_origin` with a field-native goal-provenance concept.

## Missing questions

1. Does the AST/HOT reducer need to be fully built before Phase 2 starts, or can Phase 2
   (prediction-error-native tension reframing) proceed in parallel, with AST/HOT as the
   read/legibility layer arriving before Phase 4 (real migration) rather than before Phase 2?
2. What replaces `chat_stance.py`/Hub UI's `dominant_drive` display once the bucket-vote
   layer is gone? The AST/HOT reducer is the natural candidate, but this needs an explicit
   design pass of its own before Phase 5 (retirement) is safe to execute.
3. Is a single AST/HOT reducer sufficient, or does genuine legibility require the core-affect
   layer (blue-sky idea 7) too, as a stable, non-churning summary underneath a
   possibly-reorganizing deeper structure? Worth deciding before Phase 4, not assumed now.
4. **Added 2026-07-18 (Juniper).** Is `SelfStateV1` itself trustworthy enough to keep building
   on, or does it need a v2? Real signal-quality problems have already been found in it this
   session, not hypothetical: the `confidence`/`available_capacity` merge-polarity masking bug,
   several dead/folded-away `channel_dimension_map` entries, and the pre-fix coherence/
   uncertainty sawtooth (fixed at the field-digester level — not yet independently confirmed
   that fix fully propagated through to `SelfStateV1`'s own dimension values, only that the
   field-level mechanism causing it is closed). This is a **hard gate**, not a someday item —
   see the callout in Phase 1 below.

## Proposed roadmap

### Phase 0 — already done, name it as the starting scaffold, don't rebuild it

RPT/Lamme (live) and predictive processing (live, field-native) are the two consciousness
threads already real enough to build on. Nothing to do here except treat them as load-bearing
inputs to Phase 1, not independent side threads.

### Phase 1 — build the AST/HOT reducer, coordinated with Lamme, not parallel to it

**Correction (2026-07-18), found while scoping the coordination Juniper asked for.**
Lamme's `TopDownBiasCombiner` does not operate on `FieldAttentionFrameV1` (Layer 5, general
field attention) — it operates on a *different* attention frame entirely:
`AttentionFrameV1`/`OpenLoopV1` (the FCC-Cortex GWT-dispatch lane,
`orion/schemas/attention_frame.py`), whose `voluntary_override: VoluntaryOverrideV1 | None`
field is set exactly when top-down bias flips the winner. These are two of the
three-plus disconnected attention mechanisms already found in this program's field-native
correction (PR #1163) — building AST's reducer against only one of them, blind to the other,
would recreate that same disconnection at the self-modeling layer instead of resolving it.

The good news, found in the same pass: the coordination surface already exists, mostly built.
`AttentionBroadcastProjectionV1` (same schema file) wraps `AttentionFrameV1` (hence Lamme's
`voluntary_override`) together with `attended_node_ids`, `dwell_ticks`,
`coalition_stability_score`, and a bounded `coalition_history` — explicitly documented as
"rung 3: the selected coalition of the latest workspace competition, re-broadcast as a
single queryable projection." Confirmed live: real producer
(`orion/substrate/attention_broadcast.py`), persisted to
`substrate_attention_broadcast_projection`, and confirmed to have carried a genuine non-null
`voluntary_override` at least once in the last 24h — Lamme's mechanism has already fired
live and reached this projection, not just the eval suite. Cadence caveat: this projection
updates on agent-dispatch-lane activity, not a continuous ~2s tick the way
`FieldAttentionFrameV1`/`orion-attention-runtime` does — the two inputs Phase 1 unifies are
real but not equally frequent, and the reducer needs to handle "no new GWT-lane activity
since last frame" as an honest, distinct state from "nothing salient," not silently treat
them the same.

Revised scope: one reducer, two real inputs, explicitly unified rather than picking one —
`AttentionBroadcastProjectionV1` (GWT-dispatch-lane self-model, already carries the Lamme
override/coalition-stability story) *and* `FieldAttentionFrameV1` + `SelfStateV1` (general
field self-model, Layer 5/6). Output: one explicit, inspectable artifact answering what's
currently salient, why (including *whether* a goal's top-down bias was the reason, when
`voluntary_override` is present), how confident, what's predicted to shift next — genuinely
coordinated with what Lamme already produces, not a second, disconnected attention-schema
built next to it. Read-only, measured against real historical data before anything
downstream consumes it, matching this program's own §7 process rule.

**Acceptance check**: on a real historical window that includes at least one real
`voluntary_override` event (already confirmed to exist in live data), the reducer's output
correctly narrates *that* event as the reason for the attention shift, not just a generic
salience reading — proving the two inputs are actually unified, not just both present.

**Hard gate (added 2026-07-18, Juniper): `SelfStateV1` signal-quality assessment, before
this phase is considered done — not before it starts.** This reducer takes `SelfStateV1` as
one of its two real inputs. Real signal-quality problems in `SelfStateV1` are already known
from this session, not hypothetical: the `confidence`/`available_capacity` merge-polarity
masking bug, several dead/folded-away `channel_dimension_map` entries, and a pre-fix
coherence/uncertainty sawtooth whose upstream field-level cause is closed but whose full
propagation through to `SelfStateV1`'s own values was never independently re-checked. Before
Phase 1's acceptance check above is signed off as met, run a real, live-data jitter/signal-
quality pass over `SelfStateV1`'s actual dimensions — noise floor, drift, oscillation period,
the same kind of measurement discipline used everywhere else in this program (e.g. the
"quiet decay tail" vs. "genuinely dead" distinction already used for `prediction_error`,
extended systematically across every dimension `SelfStateV1` exposes). If that assessment
finds `SelfStateV1` itself — not just the field beneath it — needs replacing, a `SelfStateV1`
v2 becomes a blocking prerequisite for Phase 1's reducer output to be trusted for anything
downstream, not an optional follow-up. Do not let Phase 2 onward build on a self-state layer
whose own signal quality was assumed rather than measured.

### Phase 2 — reframe tension-minting as prediction error, not hand-classified kinds

Do not port `tensions.py`'s kind vocabulary (`contradiction`, `distress`, `cognitive_load`,
etc.) onto field channels — that would recreate a hand-authored taxonomy one layer down,
the exact pattern this whole investigation exists to stop repeating. Instead, extend the
already-live, already-validated predictive-processing pattern
(`execution_prediction_error`/`transport_prediction_error`) to the producers currently
feeding `tensions.py`: self-state deltas, feedback frames, biometrics. A producer mints a
field perturbation when its own domain's prediction error crosses a real, measured
threshold — not when a human-authored classifier decides it matches a named "tension kind."

**Acceptance check**: for at least one producer domain, real prediction-error-based
perturbations are shown, on real historical replay, to correlate with the same real events
`tensions.py`'s old kind-based classifier fired on (contradiction/distress moments) — proving
the reframing captures the same real signal without needing the taxonomy, before trusting it
for the rest.

### Phase 3 — shadow-measure, don't wire live yet

Route one real producer domain (start with self-state deltas, the richest and best-understood)
onto real field channels in observe-only mode, alongside the still-live bucket-vote system,
and compare — using Phase 1's AST/HOT reducer as the read model — whether field-native
routing produces a recognizably different (and better) picture of "what Orion is currently
under pressure about" than the old drive-bucket output did for the same real ticks. This is
the actual test of whether Objective 3's whole premise holds, run before any producer is
actually migrated off the bucket system.

### Phase 4 — migrate producers one at a time, live

Only after Phase 3's shadow comparison is genuinely convincing: migrate real producers off
`tensions.py`'s bucket-vote path onto direct field perturbations, one domain at a time
(self-state deltas first, then feedback frames, then biometrics), each with its own
acceptance check against real live data, matching this program's own §7 discipline
(measure before minting, applied to each migration step, not just the whole effort once).
Include the goal-provenance replacement for `drive_origin` named in Missing Question 2/Current
Architecture as part of this phase, not deferred further — this is what actually frees
`capability_policy.py` from the halted system, closing the gap that made Objective 2
premature in the first place.

### Phase 5 — retire the bucket-vote layer for real

Once every producer is migrated and the AST/HOT (plus, if Missing Question 3 resolves yes,
core-affect) legibility layer is proven to give equal-or-better visibility than
`dominant_drive` did, delete `DriveEngine`/`tensions.py`'s bucket logic outright — not
halt, actually remove, per the charter's own §8 framing ("halt, not delete... until the
replacement wiring lands" — this is that landing).

### Phase 6 — revisit Objective 2 with a real foundation

Only now does Objective 2 stop being cart-before-horse. `capability_policy.py` no longer
depends on `goal.drive_origin` (Phase 4 closed that). Field-native salience has been
validated, not assumed (Phase 3). Whether the actual coupling mechanism should still be a
hand-tuned salience-to-ceiling formula, or something closer to the selectionist-internal-
ecology blue-sky idea (candidate access-rules competing and getting reinforced by real
outcomes, not a formula written down in advance), is a real, open design choice at this
point — worth deciding with Phase 1-5's real data in hand, not before.

## Non-goals

- Not building Phase 1's AST/HOT reducer in this document — scoped, not implemented.
- Not deciding Missing Question 3 (core-affect layer necessity) here — named for Phase 4 to
  resolve, not guessed now.
- Not touching `capability_policy.py`, `DRIVE_KEYS`, `signal_drive_map.yaml`, or any
  halted-directory code in this document — design only, per `CLAUDE.md` §0A.
- Not re-litigating why drives was halted or why Objective 2 was found premature — both
  already established this session, cited not redone.

## Acceptance checks (program-level, across all phases)

- Each phase has its own named acceptance check above — no phase proceeds without its
  predecessor's check passing against real data, not asserted.
- The whole roadmap succeeds if, by Phase 5, `chat_stance.py`/Hub UI have a real,
  field-native replacement for `dominant_drive` that a human finds at least as legible as
  the original, and `capability_policy.py` has zero remaining `drive_origin` references.
- The whole roadmap fails safely if Phase 3's shadow comparison doesn't show field-native
  routing is genuinely better — in which case the honest outcome is documenting that finding
  and reconsidering, not forcing Phase 4 anyway.

## Recommended next patch

Phase 1: the AST/HOT reducer, read-only, scoped as its own small patch — the one piece of
consciousness-theory scaffolding still missing before any of Objective 3's actual routing
logic gets written. Everything after this phase is explicitly gated on its own acceptance
check passing first.

## Source material

- `orion/sentience_striving_program/README.md` — the program charter this roadmap
  re-sequences (Objectives 2 and 3, §6).
- `orion/substrate/attention/top_down.py`, PRs #1170/#1174 — the live RPT/Lamme substrate.
- `services/orion-substrate-runtime/app/worker.py`'s `execution_prediction_error`/
  `transport_prediction_error` — the live, field-native predictive-processing substrate.
- `orion/autonomy/capability_policy.py`, `config/autonomy/capability_policy.v1.yaml` — the
  `required_drive_origins` coupling that makes Objective 2 premature until Phase 4 lands.
- `docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md` — the
  baseline field-native design this roadmap operationalizes into phased, gated steps.
