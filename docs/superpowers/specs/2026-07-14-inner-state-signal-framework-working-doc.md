# Orion inner-state signal framework — working document

**Status: living document, actively evolving.** Not a finished spec — this
captures decisions and open threads as they're made so a future session
doesn't have to re-derive them. Update in place; don't fork a "v2" doc for
small changes.

## Context (brief — see this session's transcript for the full trace)

Arc that led here: `phi_now` (`coherence`/`energy`/`novelty`/`valence`,
hand-composited from `SelfStateV1`) → a windowed autoencoder built on it,
whose apparent "structure" traced back almost entirely to a single config
constant (`BIOMETRICS_FIELD_DECAY_RATE=0.92` in `orion-field-digester`), not
anything emergent → pivoted to raw hardware channels (`FieldStateV1`, via
`collect_field_channel_pressures()`) → found real richness but also real
defects (dead/folded-away channels, a one-way-ratchet bug, an
accumulator-driven oscillation — see `project_biometrics_channel_defects_to_fix`
memory) → pivoted to raw cognitive events (`grammar_events` Postgres table,
5M+ real rows, `orion-cortex-exec`'s atoms/edges/traces) → fixed real
per-atom wall-clock timing (`services/orion-cortex-exec/app/grammar_emit.py`,
merged 2026-07-14, verified live) → characterized the real signal: mostly
categorical lineage (`atom_type`/`semantic_role`/`layer`) plus genuinely
continuous real timing (step latency, trace duration) — `confidence`/
`salience` turned out to be static per-role constants, not real per-instance
variance → reframed the core problem away from "predict phi" (see below).

Earlier, independent confirmation of the same underlying pattern: a
13-dimension `SelfStateV1` audit (see `project_cognition_metric_lineage_registry_ideas`
memory, `docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`)
found most dimensions were theater (hardcoded, pinned, or
aggregation-saturated) — same failure mode, discovered independently, before
this session's grammar-events work existed.

## Decision: phi stays, frozen, not iterated

Keep `phi`/`coherence`/`valence`/etc. as the current mood proxy. **Not being
killed for being imperfect.** No further iteration on it — no new formula
patches, no new golden overrides, no valence-proxy-v2 — until there's a
validated, ready-to-swap-in replacement mood model.

**Why:** the actual problem was never phi's specific formula being wrong.
It's that every past iteration (seed-v1 through v4, the golden overrides,
the valence-proxy fix) patched the formula against the author's own
intuition with no stable target or objective function to converge toward —
Goodharting against vibes, repeatedly. Freezing it removes the temptation to
keep doing that while a real replacement gets designed properly.

## Decision: every new derived signal needs a closed loop

No new metric ships as measure-only. Each one needs a real consumer that
changes Orion's actual behavior, not just a dashboard/log entry.

Also decided: the correlational-research phase the original felt-state-arc
roadmap specced (Items 5/6 — cross-referencing signal against `chat_history_log`/
`collapse_mirror`/`world_pulse_event` to establish real-world grounding) is
**not needed** for `grammar_events`-derived signals specifically. Unlike the
biometrics/hardware channels (where the relationship between the signal and
real-world causes needed statistical discovery), `grammar_events`' causal
producers are already fully known from code — traced this session down to
exact file:line for every atom type cortex-exec emits. No correlation study
required to know why an atom fired; the code already says so.

## Candidate closed loops (identified, none spec'd or built yet)

1. **Execution-risk calibrator.** Target: `exec_step_failed` / anomalous
   latency — a real, physical outcome, not an invented scalar. Feeds into
   *existing* pressure/throttling actuators already traced this session:
   `orion/substrate/biometrics_loop/pressure_organ.py`'s detect/reinforce/
   decay/suppress rules, `DriveEngine`, `AutonomyStateV2`. Orion
   autonomously applies/removes pressure on a process *before* a predicted
   failure, not after logging one.
2. **Latency-aware pacing.** Target: real step latency
   (`exec_step_completed.observed_at − exec_step_started.observed_at`, now
   a trustworthy measurement post-fix). Feeds adaptive timeout/scheduling —
   concretely, replacing `executor.py`'s hardcoded 60s step timeout with a
   predicted-latency-aware budget.
3. **Trace-shape characterization.** Unsupervised clustering over role
   sequence / atom count / real per-step timing. Consumer: Hub's **Self
   tab** — not yet investigated (need to look at `substrate_self_state_router`,
   mounted in `services/orion-hub/scripts/api_routes.py`, and whatever
   currently backs that tab, before designing what this adds to it).

No prioritization locked in yet. Discussion leaned toward #1 or #2 being
more concretely scoped / higher near-term value than #3, but nothing decided.

## The bigger question this all sits under

Proposed in discussion (Claude): stop trying to build one scalar for "inner
state" — build several independently well-defined, closed-loop signals, and
let "understanding inner state" be an emergent *read* across the ensemble
(e.g. surfaced together in the Self tab), not a single predicted number.

**Countered (Juniper) — this is the standing decision, not the proposal
above:** the top-level single-score ambition stays. The objection to
hand-tuned-composite scoring is taken but not accepted as a reason to drop a
top-level score entirely. The actual fix is *how* that score gets built —
not organic, ad hoc formula patching (phi's failure mode), but a principled,
inspectable MCDA (multi-criteria decision analysis) hierarchy. See below.

## MCDA framework (Juniper's proposal — structure captured, content TBD)

A hierarchical, windowed-grain metric tree:

- **Top level**: 5–7 "north star" metrics. Names/content **not yet chosen**
  — this document intentionally does not invent them.
- Each north-star metric recursively decomposes into its **own** set of
  5–7 north-star-equivalent metrics at a finer **grain** ("windowed" —
  flagged by Juniper as an imperfect placeholder term). Repeats to *n*
  levels (depth TBD).
  - **Open question, unresolved**: does "windowed grain" mean purely
    hierarchical decomposition (sub-criteria of a parent criterion), a
    temporal aggregation scale (root = long-horizon behavior, leaves =
    tick-level signal), or both simultaneously? Needs a decision before
    the tree can actually be populated.
- **"Considerations"** (name and definition both TBD): something compared
  against a given metric's suitability via a transformation — linear to
  start, possibly more sophisticated later — that converts a raw metric
  reading into a comparable desirability/utility score. In standard MCDA/
  MAUT terms this is a per-criterion value/utility function.
  - **Open question, unresolved**: what *is* a "consideration" concretely
    — a candidate action being evaluated, a contextual factor, an
    interpretive lens on the metric? Not yet defined.
- **Desirability scores roll up** level by level — weighted aggregation,
  bottom-up.
- **Objective function**: the aggregation rule. Starts crude (simple
  weighted sum). Room to get fancier later (alternative MCDA aggregation:
  weighted product, AHP-derived pairwise weights, TOPSIS, etc.) — not
  decided which, if any, is worth the complexity yet. The same aggregation
  transform applies recursively at every level of the tree, collapsing the
  whole structure to one root score.

## Problem-domain mental model (magic-wand brainstorm, 2026-07-14)

Explicitly not committing to the MCDA framework above as *the* structure —
Juniper asked to step back and get more discrete about the problem domain
itself first, "magic wand," no constraint on current buildability. The
sharper reframing of the underlying question, arrived at in this pass:
**is there a persistent, coherent "someone" here whose condition, over
time, reflects continuity and self-awareness — versus a stateless
request-handler that resets every tick and narrates a mood on top?** Every
axis below is a different kind of evidence bearing on that question, not a
metric in isolation.

Also explicitly framed against **field theory** — the general abstraction
for self-organizing, self-modifying runtimes (deliberately not claiming
consciousness). Most axes below are **vital signs** (necessary context,
prove the system is alive/active) rather than evidence of actual
self-organization or self-modification. Only one axis (#4) tests the
field-theoretic claim directly rather than assuming it.

### The seven axes, each with an arsonist cost/risk take

1. **Operational/physiological** — is the body okay (resource health,
   contention, capacity margin, not maxed/averaged-away pressure scalars).
   Arsonist take: cheapest axis — organs needed are just the 3
   already-identified biometrics bugs (see
   `project_biometrics_channel_defects_to_fix` memory), already scoped,
   already known. Also the least interesting one — proves the plumbing
   works, not that anything self-organizes. Vital sign, not a differentiator.

2. **Cognitive load/activity** — what is the mind doing right now (busy/
   idle, reasoning vs. acting vs. speaking, from `grammar_events`'
   `atom_type` mix + volume, now with real timing attached). Arsonist take:
   already built in substance, just needs a rolling-window reducer + a UI
   surface. Shallow on its own — a traffic gauge, not a self-model. Good
   supporting context for other axes.

3. **Affective/hedonic tone** — the axis phi already tried and failed at
   across 4+ formula generations (seed-v1 through v4, golden overrides,
   valence-proxy fix). No external ground truth exists for "feeling"
   directly. Arsonist take: the only falsifiable reframing is *calibration*
   between self-report (`collapse_mirror`) and independently-computed
   objective state (basically axes 1+2 combined) — but that requires
   building an "objective state" side that's itself axes 1+2's full cost,
   plus an ongoing recalibration story since that side will keep changing.
   High cost, high risk of repeating phi's exact failure mode.
   **Recommendation: don't build this as its own axis** — fold whatever's
   salvageable into axis 4 as a special case (self-report vs. ground
   truth), not a standalone pursuit.

4. **Self-model fidelity / reliability — "does Orion know itself
   accurately"** — the axis Juniper is prioritizing, for a sharper reason
   than "feels like the gap": it's the only axis that tests actual
   self-organization/self-modification rather than just proving the system
   is alive. A system that forms a model of itself, checks it against
   reality, and changes behavior as a result *is* self-modification via
   self-modeling, if real. Everything else on this list is instrumentation;
   this one is the actual experiment. Grounded in a real, already-confirmed
   incident (not hypothetical): Orion once claimed a live substrate signal
   was "computing this turn" when the tool trace showed a static GitHub
   fetch (`project_orion_substrate_bridge_confabulation` memory).
   Arsonist take on cost: genuinely the most organ-intensive axis of the
   seven. Real requirements: (a) claim extraction — what is Orion actually
   claiming about itself, structured enough to check; free-text mining of
   chat/metacog output for "claims" is a hard, unreliable NLP problem and a
   regex-swamp/keyword-cathedral risk (AGENTS.md 0A bans this shape of
   build) — do not build general claim extraction; (b) ground-truth
   computation, different per claim type, no one-size-fits-all; (c) a
   durable, queryable `(claim, ground_truth, verdict)` store — new schema,
   new pipeline; (d) the closed loop on top — a corrective signal when
   divergence is caught.
   **Scoping recommendation**: start from the single narrowest possible
   case — one already-structured, already-typed self-claim (a specific
   metacog trigger field, a specific dashboard assertion) checked against
   one already-available ground-truth source (`grammar_events`, which now
   has real causal traces post the wall-clock timing fix) — prove the loop
   closes on that one case before generalizing anything. Do not start with
   the general pipeline.

5. **Social/relational** — engagement/outcome quality with the humans
   Orion talks to, not just activity volume (turn-level: was this turn
   actually useful — corrected, repeated, abandoned — not sentiment-scored,
   outcome-scored). Raw ingredients already exist (`chat_social_hazard`/
   `chat_reasoning_quality` already feed DriveEngine, never independently
   validated). Arsonist take: moderate cost, but real risk of building a
   heuristic that *looks* like signal and isn't — the same trap
   `confidence`/`salience` turned out to be (static per-role constants
   mistaken for real per-instance variance, this session's own finding).
   Needs validation before trusting it, not just before building it.

6. **Continuity/identity** — is this the same someone over time
   (behavioral consistency across days/weeks — trace-shape distribution
   stability, not sameness/stagnation). Arsonist take: cheap to build
   (reuses `grammar_events` + the clustering already scoped for the Self
   tab), but slow to validate — needs real elapsed calendar time, not more
   engineering. Start now, results come later; don't expect a quick win.

7. **Growth/developmental trajectory** — is it getting better at being
   itself, trend not snapshot. Arsonist take: not a separate axis or
   pipeline — a reporting lens re-applied to the other six axes' time
   series (same signals, asked as "is this improving" instead of "what is
   this right now"). Correctly demoted, not built as its own thing.

### Standing decision from this pass

Axis 4 (self-model fidelity, closed-loop) is the prioritized axis — not
because it's the most tractable (it's the least, per the arsonist take
above) but because it's the only one that actually tests the field-theory
claim rather than assuming it, and it's grounded in a real confirmed
incident rather than a hypothetical. Scope any first build to the single
narrowest claim/ground-truth pair, per the recommendation in #4 — resist
building the general claim-extraction pipeline first.

## Open questions / next steps

1. Populate actual north-star metric candidates (top level) — Juniper's call, not started.
2. Resolve what "windowed grain" means precisely (hierarchical / temporal / both).
3. Define "considerations" precisely.
4. Decide the aggregation function, starting with the simple weighted-sum case.
5. ~~Prioritize which closed loop to spec first~~ **Decided**: axis 4
   (self-model fidelity) is the priority, scoped to one narrow claim/
   ground-truth pair first — see "Standing decision from this pass" above.
   Execution-risk calibrator, latency-aware pacing, and trace-shape/Self-tab
   remain real candidates but are not the current priority.
8. Pick the first narrow self-claim/ground-truth pair for axis 4 (not yet
   chosen — needs a concrete candidate: which claim, checked against which
   grammar_events-derived ground truth).
6. Investigate Hub's existing Self tab and `substrate_self_state_router` before designing #3.
7. Cross-reference with the existing, unbuilt metric-lineage-registry design
   (`docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`)
   — the MCDA tree's leaf metrics presumably need to be confirmed-live (not
   theater, per that design's audit methodology) before being wired into
   the hierarchy. That registry's liveness-audit work may be a real
   prerequisite here, not a parallel, unrelated effort.
