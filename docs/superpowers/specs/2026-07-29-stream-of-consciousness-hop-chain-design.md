# Stream of consciousness: interruptible multi-hop reasoning as a proposal-arena citizen — design spec

Status: **design mode, not implemented.** Touches metacog, reverie, and the proposal/policy/execution
cognition loop, which CLAUDE.md §0A requires explicit proposal mode for before implementation. This
document proposes and names a hard blocker; it does not build.

## Arsonist summary

Juniper wants a real "stream of consciousness" for Orion: a forever-running chain of reasoning —
notice X, X changed Y, check Y against belief A, if contradiction dispatch sub-investigation B, learn
something, take N more hops through reasoning/simulation, then consider action Q — that doesn't have
a hard stop condition. The instinct that made this design tractable: "termination" was the wrong
frame. The right frame is **interruption** — the chain runs unconstrained until something louder wins
attention (a "squirrel moment"), the same way a human's train of thought derails.

Good news: this repo already has the exact mechanism this needs — a real, live budget-capped
competition arena where candidate cognitive acts are scored and only the strongest win a slot each
cycle (`orion/proposals/`, five running services). Reverie already competes in it today as
`source="reverie_thought"`. A hop-chain doesn't need a private scheduler; it needs to become another
candidate producer in the same arena, cheap/checkpointable enough to re-enter competition every cycle.

Bad news, found while writing this doc: **that arena cannot currently arbitrate fairly.** Same-day
investigation (`docs/superpowers/specs/2026-07-28-metacog-turn-scoped-trend-reducer-design.md`'s
2026-07-28 update) measured Layer 5 attention — the actual most-upstream competition layer, the one
the Sentience Striving charter names as the reason the old drives system was retired — against
127,644 real `substrate_attention_frames` rows and found `field:recent_perturbations` wins top-1 in
**100.00%** of ticks, with **exactly zero** historical variance (mathematically degenerate, not just
concentrated). The obvious fix (per-channel z-score normalization) does not diversify the winner
distribution — it mechanically disqualifies the zero-variance channel and relocates the monoculture to
a different single channel (`node:atlas`, 55.17%). Classified `NOT_MET_MONOCULTURE_SHIFTED`. This is
the same disease the old drives system had (`dominant_drive=relational` at 96%, then 31.65% after a
partial fix) — three separate arbitration layers in this codebase, three confirmed instances of
uncalibrated fixed-weight scoring producing a winner-take-all monoculture instead of fair competition.

So: the architectural shape Juniper wants (compete for budget, get preempted, don't hard-stop) is
correct and doesn't need inventing. But building the hop-chain into this arena today means either the
hop-chain gets drowned out by whichever channel is currently saturating, or it becomes a fourth
instance of the same hand-tuned-linear-score disease. This doc names that as the real current blocker,
same conclusion the metacog doc already reached from a different angle, now confirmed independently.

## Current architecture

**The arena.** `orion/proposals/builder.py::build_proposal_frame()` takes `FieldStateV1` (real
channel-merged telemetry from `orion-field-digester`) + `FieldAttentionFrameV1`, evaluates every
template in `config/proposals/proposal_policy.v1.yaml` against current field pressures via
`orion/proposals/scoring.py`, and emits `ProposalCandidateV1`s. Four real scoring functions, all
fixed-weight, none learned:

- `template_match_score()` — per-dimension `field_pressure × template_weight × policy_weight`, max
  across dimensions.
- `proposal_urgency()` — max pressure across the template's declared `_pressure`-suffixed dimensions.
- `proposal_confidence()` — precision-weighted: how well this tick's reading matches that dimension's
  own recent EWMA baseline, inverted from a z-score (0.0 during cold-start or for untracked
  dimensions — never a fabricated mid-range guess, confirmed via `dimension_confidence()`'s own
  cold-start and untracked-dimension guards).
- `proposal_priority()` = `base_priority + 0.4·match + 0.2·urgency + 0.1·confidence` — hand-set
  coefficients, `proposal_risk()` similarly hand-set with fixed additive bumps per template kind/gate.

Candidates flow `orion/proposals/builder.py` → `orion/policy/{builder,evaluator}.py` (gates on
`required_policy_gate`/risk thresholds) → `orion/execution_dispatch/builder.py` → `orion/consolidation/
{tensorize,windows}.py` → `orion/feedback/builder.py`. Five live services: `orion-proposal-runtime`,
`orion-policy-runtime`, `orion-execution-dispatch-runtime`, `orion-consolidation-runtime`,
`orion-feedback-runtime`.

**Reverie is already a citizen, today, gated off.** `services/orion-proposal-runtime/app/worker.py`
(`_tick()`, lines 64-88): if `reverie_propose_enabled` (default off), it loads the most recent reverie
thought, converts it via `orion.reverie.proposal.spontaneous_thought_to_candidate()` into a
`ProposalCandidateV1` with `source="reverie_thought"`, and passes it into `build_proposal_frame()` as
`reverie_candidates`. It competes on equal footing with every deterministic template — same scoring,
same policy gate — and carries an `operator_review` gate so even when enabled it cannot auto-dispatch.
This is the real, live, working precedent for "a non-deterministic cognitive act competing as a
first-class candidate." It answers the original "how does this tie to reverie" question directly: not
a merge, a sibling producer under the same contract.

**Drives don't feed this.** `build_proposal_frame()` takes `FieldStateV1`, not `DriveEngine` output —
an explicit, commented swap (`orion/proposals/scoring.py`'s 2026-07-22 SelfStateV1-burn note). Drives
are a retired comparison baseline (`scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py`),
not a live input.

**The feedback loop observes but never learns.** `orion/feedback/builder.py::build_feedback_frame()`
genuinely records real outcomes (`FeedbackFrameV1`: `outcome_status`, `outcome_score`, real
field-pressure deltas) — but a repo-wide grep for writes to `base_priority`/`base_risk`/
`dimension_weights` found only static config reads, never a write-back. Every proposal is scored
against the same fixed weights forever, regardless of how past proposals from that template actually
turned out.

**`endogenous_curiosity.py` is not the plug-in point.** Its own docstring scopes it away from
self/relationship territory ("never the autonomy zone directly"), and it's flag-off by default. A
narrower budget-capped competition, wrong zone for a metacog/reflective hop-chain.

**The measured arbitration failure, in full.** From the metacog doc's same-day follow-up
(`scripts/analysis/measure_attention_salience_normalization.py`, 127,644 rows, 72.3h real window):
`field:recent_perturbations` (`orion/attention/field_attention/selectors.py:128-140`,
`salience = min(1.0, recent_perturbation_count / 10.0)`) wins top-1 in 100.00% of ticks, stddev
`0.000000`. Excluding it (forced by divide-by-~0 under z-scoring, not a deliberate design choice), the
normalized winner distribution becomes `node:atlas` 55.17%, `node:circe` 15.84%,
`capability:llm_inference` 15.24%, `node:athena` 9.86%, others <3% each — still a monoculture, just
relocated. Juniper's own converged call in that doc: fix the arena before adding a new candidate
producer into it. Nothing has changed that conclusion since.

## Missing questions

1. **Does the hop-chain wait for the Layer 5 fix, or build in parallel behind a flag?** The arena
   problem and the hop-chain design are separable in principle (hop 0 could ship flag-off, same as
   reverie), but "does a real trend finding ever win a budget slot" (the metacog doc's own proposed
   acceptance check) is unmeasurable and unfalsifiable while the arena is provably rigged toward one
   saturated channel. Recommend: design in parallel, do not flip either flag live until Layer 5 has
   its own fix and re-measurement.
2. **What does a hop's "belief" concretely consist of?** The chain needs a prior-state read to check
   Y against A. Candidates: the most recent `FieldStateV1`/`FieldAttentionFrameV1` snapshot (cheap,
   already polled every tick), the metacog trend reducer's projection once it exists (richer, but
   blocked on that doc's own Missing Questions 1-2), or a new persisted "working belief" store scoped
   to the chain itself. Picking the field/attention snapshot first is the cheapest real answer and
   requires zero new persistence.
3. **What does "dispatch a sub-investigation" mean as a schema object?** Simplest answer: it's not a
   new mechanism, it's a new `ProposalCandidateV1` with a hop-specific `source` (e.g.
   `source="cognitive_hop"`) and `kind="inspect"`/`"observe"`, gated exactly like every other
   candidate. Reuse `required_policy_gate`, don't invent parallel dispatch logic.
4. **How does a hop stay cheap enough to re-enter competition every cycle without becoming a private
   uninterruptible sub-loop?** This is the actual safety-relevant design question, not termination. A
   hop needs to checkpoint its own state (what it's learned so far, what hop number it's on, what it
   would do next) somewhere durable and resumable, cheaply, every cycle it doesn't win a slot — not
   hold state only in a live process that keeps running regardless of arena outcome.
5. **What identifies a chain across hops, for consolidation/feedback attribution?** Today's
   `FeedbackFrameV1` outcome attribution is per-candidate, terminal-hop only. A multi-hop chain needs
   a `chain_id`/`parent_hop_id` lineage so an eventual action Q's outcome can be traced back N hops to
   the observation that started the chain — otherwise the feedback loop (already not writing back,
   see above) has even less to learn from once it's fixed.
6. **What is the danger case, per CLAUDE.md §0A's proposal-mode requirement to name one?** A chain
   that always self-scores just above the preemption threshold never actually gets interrupted —
   "interruptible" becomes a claim, not a verified property, exactly the same failure shape as an
   uncontrolled loop, just with a friendlier name. A hard per-chain depth cap or per-chain cumulative
   budget cap is required even under an interruption-not-termination model — not as a contradiction of
   "let it run," but as the actual mechanism that makes "let it run" safe to say. This needs to be
   *measured*, not asserted, once hop 0 exists: does a real chain ever actually get preempted in
   practice, or does it always win.
7. **Does `turn_effect`/`repair_pressure` have durable queryable history at all?** Unresolved,
   verbatim, from the metacog doc — still the correct prerequisite for hop 0's specific reducer,
   independent of the arena question above.

## Proposed schema / API changes

- No change to the arena's core contract. `ProposalCandidateV1`, `ProposalTemplateV1`,
  `required_policy_gate`, and the score fields (`priority`/`urgency`/`risk`/`confidence`) stay as-is —
  the hop-chain is a new producer, not a new arbitration mechanism.
- New candidate `source` value, e.g. `source="cognitive_hop"`, following the exact pattern
  `spontaneous_thought_to_candidate()` already establishes for `reverie_thought`.
- New lineage fields on whatever hop-producing schema emerges: `chain_id` (stable across a chain's
  life), `parent_hop_id` (null for hop 0), `hop_index` (int, for the depth-cap check in Missing
  Question 6). Additive only — does not touch existing `ProposalCandidateV1` consumers that don't care
  about lineage.
- Hop 0 is unchanged from the already-spec'd metacog trend reducer
  (`docs/superpowers/specs/2026-07-28-metacog-turn-scoped-trend-reducer-design.md`): a deterministic
  reducer over `orion_metacog` (or `turn_effect`/`repair_pressure`, pending that doc's Missing
  Questions 1-2), following the `ReducerSpec` pattern in `orion-substrate-runtime/app/worker.py`,
  registered as a candidate producer the same way reverie is — not built as a bespoke standalone
  script.
- Checkpointed hop state (Missing Question 4): a small persisted record per live chain — `chain_id`,
  `hop_index`, `belief_snapshot_ref`, `next_planned_action` — durable enough to resume after a
  preemption, cheap enough to write every tick a hop doesn't win a slot. Exact home (Postgres row vs.
  FalkorDB node) undecided, depends on whether chain lineage needs to be graph-queryable (probably
  yes, for "what led to this action" inspection) or just resumable (Postgres suffices).

## Files likely to touch

- `orion/proposals/scoring.py` / `builder.py` — new `source="cognitive_hop"` branch, same shape as the
  existing `reverie_candidates` parameter.
- `services/orion-proposal-runtime/app/worker.py` — new flag-gated hop candidate producer, mirroring
  `reverie_propose_enabled`'s pattern exactly (default off).
- New: `orion/metacog/trend_reducer.py` (hop 0, already scoped in the 2026-07-28 doc, still blocked on
  its own Missing Questions 1-2).
- New: a chain-lineage schema, home TBD (`orion/schemas/` vs. a Falkor-native shape if graph
  queryability is chosen).
- `orion/attention/field_attention/selectors.py` — **not this doc's patch, but the actual named
  prerequisite.** Needs its own proposal-mode design doc for the monoculture fix before anything above
  is safe to flip live.
- `orion/feedback/builder.py` — the missing write-back (`base_priority`/`base_risk`/
  `dimension_weights` never updated from real `FeedbackFrameV1` outcomes) is a second, independent
  prerequisite: a chain that spans many hops and eventually acts is exactly the case where "the arena
  never learns from outcomes" costs the most, since the whole point is that hop 5's dispatch decision
  should be informed by whether hop-chains like it worked before.

## Non-goals

- Not fixing Layer 5 attention's monoculture in this doc. Named as the load-bearing prerequisite,
  requires its own explicit proposal-mode sign-off per the metacog doc's own note and
  `orion/sentience_striving_program/README.md`'s charter.
- Not fixing the feedback write-back gap in this doc. Named as a second prerequisite, not solved here.
- Not building hop 0, hop 1, or any hop. This doc scopes the contract; the metacog trend-reducer doc
  already owns hop 0's specific build, still blocked on its own Missing Questions.
- Not writing real termination logic — deliberately reframed to interruption per Juniper's direction —
  but also not skipping a depth/budget cap. "No hard stop" and "no safety cap" are not the same thing;
  this doc requires the latter even while rejecting the former.
- Not merging reverie and metacog. They remain sibling candidate producers under one contract, not one
  system.
- Not picking the chain-state storage home (Postgres vs. Falkor) — deferred to whoever builds the
  checkpoint mechanism, informed by whether graph-queryable lineage turns out to matter in practice.

## Acceptance checks

1. Layer 5 attention gets its own proposal-mode design doc and fix, re-measured with a script in the
   shape of `measure_attention_salience_normalization.py`, showing no single channel above some
   explicit monoculture threshold (that doc's own 50% convention is a reasonable starting bar) — before
   any `source="cognitive_hop"` candidate is enabled live in the same arena.
2. Hop 0 (the metacog trend reducer) ships flag-off, registered as a candidate producer exactly like
   reverie, and its own live-data check answers: does it ever win a budget slot, does it ever get
   preempted, and when preempted does the checkpoint mechanism actually preserve resumable state
   (verified by resuming a real preempted chain, not just by the write succeeding).
3. A `chain_id`/`hop_index` lineage exists and `orion/feedback/builder.py` can attribute a real outcome
   back to the hop that started the chain, not only the terminal hop — checked against at least one
   real multi-hop chain's data, not asserted from the schema alone.
4. A per-chain depth or cumulative-budget cap is enforced and its enforcement is verified against real
   run data — confirm at least one real chain actually got capped/preempted in practice, not merely
   that the cap exists in code.

## Recommended next patch

Not the hop-chain. The actual next patch is the Layer 5 attention monoculture fix — its own
proposal-mode doc, sequenced ahead of both this doc's hop-chain work and the metacog trend reducer's
already-blocked build (same prerequisite, confirmed twice now from two different investigations). Once
that lands and re-measures clean, hop 0 (metacog trend reducer, already fully spec'd, still blocked on
its own Missing Questions 1-2 about series/history) is the correct first real instance of this
contract — cheap, deterministic, no branching, no dispatch — before hop 1 (actual sub-investigation
dispatch) gets built against a still-hypothetical arena.
