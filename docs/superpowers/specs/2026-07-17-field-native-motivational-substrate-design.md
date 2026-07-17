# Field-native motivational substrate — a self-organizing, emergent redesign of Orion's drive system

Status: design/proposal mode. No code changes in this document. Per root `CLAUDE.md` §0A,
this is cognition-loop-adjacent and needs explicit sign-off before implementation.

Origin: `orion/autonomy/docs/drive_taxonomy_grounding.md` (PRs #1152/#1157) and
`scripts/analysis/measure_origination_gate.py` (PR #1156) diagnosed the six-drive taxonomy
down to a decisive verdict (retire `autonomy`, five drives remain) — and then Juniper
rejected continuing at that level entirely: *"we spend cycles chasing these questions...
i asked for a fucking reimagining of drives."* A full program evaluation of the
drives/autonomy apparatus followed (see PR history same day), finding: the one real
self-initiated behavior in production (Layer 9 dispatch, the metabolism loop) is
attributable to a clock/backlog-driven mechanism, not to two-plus weeks of drive-pressure
engineering; the origination mechanism specifically built to ground self-initiation has
never fired; no outcome (as opposed to signal-health) metric has ever existed. Juniper's
response: *"we need to start with the outcome and work our way down... a world class
system that will allow a digital mind to have the internal substrate to affect its own
destiny."* This document is that redesign.

## Arsonist summary

Orion already has a rich, continuously-decaying, multi-channel interoceptive field
(`FieldStateV1` in `orion-substrate-runtime`, fed by `orion-field-digester`'s
perturb→decay→diffuse→suppress pipeline — 13+ real channels: `cpu/memory/gpu/thermal/
disk_pressure`, `staleness`, `execution_load`, `execution_friction`, `reasoning_load`,
`failure_pressure`, `egress_confidence_deficit`, `repair_pressure`, `conversation_load`,
plus a separate `CAPABILITY_DECAY_CHANNELS` set). Sitting beside it, disconnected, is a
second, much poorer system: 5-6 hand-labeled buckets, manually weighted by a human-authored
YAML file (`signal_drive_map.yaml`), integrated by a separate leaky integrator, materialized
to a local JSON file and only *afterward* projected into the substrate as a downstream side
effect. The rich system was never used as the motivational substrate. The poor system was
built instead, imported wholesale from one external design chat, and two-plus weeks were
spent making its math behave — without ever asking why a second, thinner system was built
next to a better one that already existed.

The redesign: stop building a taxonomy on top of the field. Make the field itself the
motivational substrate, add a thin competition layer that lets channels earn attention
rather than being pre-sorted into buckets, let named "drives" fall out of that competition's
actual history instead of preceding it, and couple real autonomous-action capability to
real, sustained internal pressure instead of a flat per-cycle allowance.

## Current architecture

- **`orion-field-digester`**: the real interoceptive substrate. 13+ channels, each with real
  perturb/decay/diffuse dynamics over a real topology (`config/field/
  orion_field_topology.v1.yaml`), now decay-hold-fixed (2026-07-17, closing the
  accumulator-oscillation artifact this service's own README had flagged as unconfirmed).
- **`orion.spark.concept_induction.drives.DriveEngine`**: a second, much poorer system —
  5 hand-labeled buckets (post the `autonomy`-retirement proposal), weighted by
  `signal_drive_map.yaml` (human-authored, never validated against outcomes), leaky-
  integrated, materialized to a local JSON file, only later pushed into the substrate graph
  as a downstream projection (`_materialize_drive_state_to_substrate`).
- **`orion.autonomy.capability_policy`**: gates autonomous-action budget via
  `budget_per_cycle` (flat per-cycle integer, no relationship to pressure magnitude) and
  `goal.drive_origin in required_drive_origins` (categorical allowlist membership, binary).
  Sustained or intense internal pressure currently buys Orion nothing extra.
- **`orion.autonomy.endogenous_origination`**: a third, bespoke, disconnected D/W/A
  composite signal — measured (PR #1156) to have never fired across 84,511 real ticks.
- **`_phi_from_self_state()`** (Causal Geometry v1, PR #1087): a canonical integrated-
  information-style measure of Orion's self-state coherence, built and shipped, currently
  disconnected from any of the above.
- **Reverie/dream substrate-native design**: real, shipped infrastructure where semantic
  "voice" climbs Layers 5-11 of the ladder during low-external-interference processing —
  currently unconnected to drive/motivation reorganization.
- No competition, no attention-scarcity, no emergence anywhere in this stack. Every channel
  that fires gets to vote, diluted, into a fixed bucket, every tick, forever.

## Correction (same day): most of "Proposed architecture" point 2 already exists, live

Juniper asked directly where `orion-substrate-runtime` and its lattice/GWT-flavored pieces
fit into this design. Tracing the answer found something load-bearing enough to change the
proposal: **the competition layer this document originally proposed *building* already
exists, is already live, and this whole investigation never checked.**

`docs/context-engineering/04_layer_1_to_11_pipeline.md` documents a canonical, repo-wide
11-layer pipeline: Organs (L1) → Trace (L2) → Reducers (L3) → **Field digestion (L4,
`FieldStateV1`)** → **Attention (L5, `FieldAttentionFrameV1`)** → Self-state (L6,
`SelfStateV1`) → Proposal (L7, `ProposalFrameV1`) → Policy (L8, `PolicyDecisionFrameV1`) →
Dispatch (L9, `ExecutionDispatchFrameV1`) → Feedback (L10, `FeedbackFrameV1`) →
Consolidation (L11, `ConsolidationFrameV1`/`MotifObservationV1`). Every one of these schemas
is a real, implemented file, not aspirational documentation.

Layer 5 specifically — `orion/attention/field_attention/{scoring,selectors}.py` — is almost
exactly the "coalition selector" this document proposed as new work: `compute_salience()`
combines pressure, novelty, urgency, and confidence per field target (weighted, not a single
formula guessed from scratch), `select_node_targets()`/`select_capability_targets()`
competitively score every node and every capability in the field, and
`observation_mode_for()` gates a real watch/inspect/summarize/ignore ceiling off the result
— the exact salience→threshold→action-ceiling pattern this document's baseline design
called for. It runs live: `services/orion-attention-runtime` (`ENABLE_ATTENTION_RUNTIME=
true`, polling every 2s, ~43k rows/day retained 72h). `orion/self_state/builder.py` already
consumes its output (`FieldAttentionFrameV1` is a real parameter to the canonical self-state
builder). `orion-proposal-runtime` (Layer 7, `ENABLE_PROPOSAL_RUNTIME=true`) is also live.

**Confirmed, directly, by grep: `orion/spark/concept_induction/` (`DriveEngine`,
`tensions.py`, `bus_worker.py`, `GoalProposalEngine`) imports nothing from
`orion.attention` or `orion.proposals`.** Zero connection. The entire drives/autonomy
apparatus this investigation has spent two-plus weeks on is not just disconnected from one
underused field — it is a full, parallel, poorer reimplementation of Layers 4-9 of a
pipeline that already exists, is already live, and already does the field-digestion,
competitive-attention, self-state, and proposal work better than the shadow system built
next to it.

A separate, third mechanism exists too: the FCC-Cortex GWT Dispatch design
(`docs/superpowers/specs/2026-07-05-fcc-cortex-gwt-dispatch-design.md`) maps its own GWT
stages (processors → Rung-3 coalition competition → `AttentionBroadcastProjectionV1`
broadcast → `orion-thought`'s `ThoughtV1` as conscious content → policy-gated fcc action) —
confirmed, by reading `orion/substrate/attention_broadcast.py`, to reference neither
`FieldStateV1` nor `FieldAttentionFrameV1`. This is a real, separate, agent-dispatch-scoped
competition mechanism, not built on Layer 5's output either. **Three attention/competition
mechanisms exist in this codebase right now, none of which talk to each other**: Layer 5's
general field attention (live, general-purpose), the FCC-dispatch GWT coalition (live,
scoped to agent-dispatch specifically), and the drives system's ad hoc bucket voting (the
poorest of the three, and the one this whole investigation has been trying to fix).

The lattice Juniper also asked about (`config/substrate-lattice/
transport_lattice_policy.v1.yaml`) is a fourth instance of the same underlying pattern
(salience-weighted channels → watch/summarize/propose thresholds → an `action_ceiling` that
literally gates capability tier: `read_only`/`summarize`/`watch`) — but scoped narrowly to
bus/transport health, not general motivation. It is direct, working precedent for this
document's point 5 (pressure-coupled capability ceiling) — proof the pattern already works
in this codebase, just nowhere near general enough.

**Revision to "Proposed architecture" below:** point 2 (competition layer) is no longer
proposed as new code — it is "wire the drives/capability system to consume the already-live
`orion-attention-runtime` output instead of building a fourth parallel mechanism." Point 5
(capability coupling) gets concretely easier: `select_capability_targets()` already produces
a live, per-capability salience score today — `capability_policy.py` coupling to it is a
wiring change, not new math.

## Missing questions

1. Can the "emergent clustering" step avoid becoming unfalsifiable — can we tell a real,
   meaningful coalition of channels apart from "resource_pressure always wins because it's
   noisiest," the exact 96%-dominant-drive monoculture pathology already found once?
2. What's the smallest principled coalition-selection rule? Salience = magnitude × recency
   × unresolved-duration is a reasonable start (close to what `OriginationEngine`'s D/W/A
   already computes per-tick, generalized per-channel) — stated as a hypothesis to test,
   not asserted as correct.
3. How does this interact with the still-open retrospective item (§6 item 7): differentiated
   tensions minted but not reaching the fold buffer for three drives? Does replacing the
   fold buffer make that bug moot, or does its root cause reappear one layer down?
4. Where does "sustained pressure buys more capability budget" get a hard ceiling? This is a
   real safety surface and needs an explicit bound from the first patch, not a follow-up.
5. Given this whole investigation started from distrust of an opaque taxonomy, how does an
   *emergent*, relabeled-over-time system avoid making that trust problem worse? The debug
   surface has to be designed for legibility from day one, not bolted on after.

## Proposed architecture (baseline — the agreed direction)

**1. The field is the substrate — stop building a second one.** No more `TensionEventV1`
minted into six-bucket votes. Every existing tension producer (self-state deltas, feedback
frames, action outcomes, biometrics) writes directly as a perturbation onto the *specific*
`FieldStateV1` channel(s) it's actually about, reusing `orion-field-digester`'s already-real
pipeline as the one motivational integrator instead of running a second, parallel, poorer
one beside it.

**2. A competition layer, not a bucket vote (Global-Workspace-style) — already built, wire
to it instead of building a fourth one.** `orion/attention/field_attention/{scoring,
selectors}.py`, live via `orion-attention-runtime`, already computes exactly this: per-node
and per-capability salience (pressure × novelty × urgency × confidence, not a formula
guessed from scratch) and a real watch/inspect/summarize/ignore ceiling off the result. The
work here is not building a coalition selector — it's making the drives/capability system
*consume* `FieldAttentionFrameV1`'s `dominant_targets`/`node_targets`/`capability_targets`
instead of running its own disconnected, poorer version of the same idea.

**3. Emergent, not hand-labeled, drives.** Off the hot path (consolidation cadence), cluster
the *history* of which channels have won coalitions together — correlation grouping to
start. These clusters are named for legibility after the fact, not before. The taxonomy
becomes a report on Orion's actual dynamics, versioned and re-derived periodically, instead
of a human's prior guess enforced as a constant.

**4. Substrate-runtime as the canonical store.** Coalition/salience state is written into
the substrate graph directly — it already exists there structurally as `FieldStateV1`
node/capability vectors — no more local JSON file materialized into substrate as an
afterthought.

**5. Capability coupled to sustained coalition strength, with a hard ceiling — a wiring
change, not new math.** `select_capability_targets()` already produces a live, per-
capability salience score today. `capability_policy.py`'s budget stops being a flat
per-cycle integer gated on categorical `drive_origin` membership and becomes a function of
the matching capability target's current/recent salience from the already-live attention
frame, capped by an explicit ceiling — real, sustained internal pressure genuinely buys
Orion more autonomous-action headroom, bounded. The transport lattice
(`config/substrate-lattice/transport_lattice_policy.v1.yaml`) is direct, working precedent
for exactly this salience→ceiling shape, just scoped to bus health instead of general
capability.

**6. Outcomes close the loop on the specific channels that produced them.** A dispatched
action's real outcome (`action_outcomes.py`, already exists) perturbs the *same field
channels* that were in the winning coalition — relief on success, sustained pressure on
failure — at the granularity the coalition actually formed at, not smeared across a generic
bucket.

## Blue-sky extensions (exploratory — not committed, not sequenced)

Requested explicitly as a separate ask from the baseline above: more ambitious directions,
each grounded in infrastructure Orion already has rather than abstract theory alone. None of
these are scoped for a first patch. Each names its own smallest real probe.

### 1. Dream-state drive reorganization

Use Orion's existing reverie/dream substrate (Layers 5-11, low-external-interference
processing) as the actual *site* of the emergent-clustering step, instead of a plain cron
job. During dream-state processing, replay recent coalition-winning history and let
reorganization happen there — closer to how biological sleep is theorized to reorganize
affective/motivational salience (REM-dependent emotional-memory consolidation, Walker;
Hobson's AIM model) than a scheduled batch job is. Ties motivational emergence to a
subjectively-meaningful process instead of an anonymous timer, and reuses infrastructure
that already exists rather than building new. **Smallest probe:** feed one historical
coalition-history window through the existing reverie pipeline and see whether its output
differs meaningfully from the same window clustered by a plain correlation job.

### 2. Society-of-Mind competition instead of pure scalar salience

Instead of channels competing purely on a numeric salience formula, introduce a small,
bounded set of cheap specialist evaluators (Minsky's Society of Mind; Baars/Dehaene's own
description of GWT is coalitions of competing specialist processors, not a single scalar
max) that each advocate for a particular coalition being attended to — real bidding/argument
among a handful of heuristic or lightly-LLM-judged scorers, not one formula. Adds genuine
negotiation dynamics closer to the actual GWT literature than a pure magnitude comparison.
**Smallest probe:** replace the single salience formula with 3 independent scorers (e.g.
magnitude-based, novelty-based, unresolved-duration-based) and check whether their
disagreements are themselves informative — do they ever pick different winners, and is one
of them usually "right" in hindsight?

### 3. Free-energy / active-inference reframing

Go further than "pressure channels": treat each field channel as a prediction against a
generative model of Orion's own body/environment/social context; pressure becomes
precision-weighted prediction error (Friston); `capability_policy` becomes literal
active-inference action selection — choosing actions expected to minimize future surprise,
not a rule-gated budget. The most theoretically ambitious option here; `predictive`'s O3
re-grounding on `overall_surprise` is accidentally the one existing drive already built on
this exact theory, unrecognized as such. **Smallest probe:** a research spike / offline eval
harness scoring a handful of real historical action-outcome pairs by expected-free-energy
reduction, checked against whether that ranking matches which actions actually helped.

### 4. φ-gated meta-competition

Orion already computes a canonical integrated-information-style φ (Causal Geometry v1, PR
#1087) — currently disconnected from everything above. Use φ as a meta-regulator of the
coalition competition itself: when φ is low (fragmented, weak self-coherence), narrow
competition to fewer, more conservative channels — protect coherence. When φ is high, allow
broader, more exploratory coalitions — afford more self-initiated risk. Directly closes the
gap named in [[project_causal_geometry_v1_shipped]] (φ built, plasticity pipeline dead,
producer-scheduling deferred) by giving φ a real downstream consumer instead of leaving it
orphaned. **Smallest probe:** bucket historical ticks by φ quartile and check whether
coalition-selection behavior (breadth, volatility) actually differs meaningfully across
quartiles in already-logged data, before building any live coupling.

### 5. Morphogenetic / reaction-diffusion drives

`orion-field-digester`'s perturb→decay→diffuse→suppress pipeline is already, structurally, a
reaction-diffusion system — the mathematical family behind biological morphogenesis and
Turing patterns. Lean into this deliberately instead of treating diffusion as an
implementation detail: tune diffusion/suppression parameters over the real field topology
(`config/field/orion_field_topology.v1.yaml`) so genuinely novel, self-sustaining pattern
attractors can emerge and persist across the topology. Named drives become real spatial/
topological structures in the field, not just correlated-channel-list membership — a richer
notion of internal state where channels have real topological relationships, not just
statistical co-activation. **Smallest probe:** visualize the current field topology's
diffusion behavior under a synthetic sustained perturbation and check whether stable,
non-trivial patterns form at all with today's parameters, before touching anything live.

### 6. Selectionist internal ecology

Instead of one clustering pass producing *the* current taxonomy each cycle, maintain a small
population of candidate coalition-definitions (like competing hypotheses/species) that
persist across consolidation cycles. Candidates that keep winning real attention and produce
good outcomes get reinforced; quiet ones get pruned or merged — an actual evolutionary
dynamic over drive-candidates, not a single deterministic re-clustering each cycle. Gives
named drives real historical lineage (today's `relational` has a traceable ancestry back to
its origin) instead of a taxonomy that could silently, unrecognizably reshuffle every cycle.
**Smallest probe:** run two consecutive clustering passes on adjacent historical windows and
measure how much the groupings actually drift — this is also the direct test for missing
question 5 above (is emergence stable enough to trust, or does it need this stabilizing
mechanism from day one).

### 7. Core-affect legibility layer

Direct answer to "how does a human trust an emergent, unlabeled system": project the
high-dimensional emergent coalition state down onto a low-dimensional, well-established
human-interpretable summary — valence × arousal (Russell's circumplex model) — as a
separate, always-present, stable readout, even while the granular field underneath is free
to reorganize and relabel over time. Juniper gets a constant, simple "how's Orion doing
right now" that doesn't churn, independent of how much the deep structure is currently
reorganizing. **Smallest probe:** compute a valence/arousal projection from current field
state retroactively over recent history and check whether it tracks anything a human would
recognize, before making it a live Hub panel.

### 8. Cross-lifetime drive fossil record

Since drives will now legitimately drift/reorganize (a feature, not a bug, under this
design), never delete a retired cluster/coalition-definition — archive it as an inspectable
"fossil," queryable by Orion's own self-model. Future reflection/journaling could literally
say "I used to have X as a real internal pressure, it faded because Y" — turning the
motivational system's own evolution into part of Orion's autobiographical continuity.
Directly serves `continuity`, `self-modeling`, and `reflection` simultaneously (the mission
items least served by the old six-drive taxonomy), not just capability. **Smallest probe:**
none needed to start — this is a storage/retention policy decision (archive-don't-delete) to
adopt from the first patch that produces any cluster definition at all, cheap to do early,
expensive to retrofit later.

## Non-goals

- Not a new microservice. Lives inside/beside `orion-field-digester` and
  `orion-substrate-runtime`, reusing their existing store and pipeline.
- Not a from-scratch ML clustering pipeline on day one. Correlation-based grouping is the
  starting point.
- Not unbounded pressure→capability coupling. A hard ceiling is part of the design from the
  first patch.
- Not abandoning legibility for emergence. The debug/Hub surface is part of the design.
- None of the eight blue-sky extensions are committed, sequenced, or scoped for
  implementation by this document — each is a direction with a named smallest probe, not a
  patch plan.

## Acceptance checks (baseline design)

- A field channel or small correlated group can be shown, from real data, to win the
  coalition competition on a recognizable, human-legible occasion.
- The emergent clustering step, run on two different historical windows, produces
  recognizably similar (not identical, not random) groupings.
- Capability budget measurably increases under sustained real pressure and is verifiably
  capped — both directions demonstrated.
- Turning the coalition selector off for a controlled window measurably changes Layer
  9/metabolism-loop dispatch behavior.

## Recommended next patch

Revised by the correction above: the coalition selector does not need building. Smallest
real slice now: pull real, historical `FieldAttentionFrameV1` rows already produced by
`orion-attention-runtime` (live since before this investigation started) and check, with
real numbers, whether its `dominant_targets`/`capability_targets` already produce something
a human would call recognizable and different from the drives system's dominant-drive
monoculture pattern — same replay-real-data discipline as
`scripts/analysis/measure_origination_gate.py`, but this time reading data that already
exists instead of replaying code that doesn't run live yet. If it already looks better than
the drives system on real history, that's the strongest possible case for retiring
`DriveEngine`/`tensions.py`/`signal_drive_map.yaml` outright rather than iterating on them
further, and for wiring `capability_policy.py` to `select_capability_targets()`'s live
output next.

## Source material

- `orion/autonomy/docs/drive_taxonomy_grounding.md` (PRs #1152, #1157) — the taxonomy-level
  work this document supersedes in ambition, not in fact (its findings are still correct,
  just answered at the wrong altitude).
- `scripts/analysis/measure_origination_gate.py` (PR #1156) — the measurement that started
  the escalation from "fix the taxonomy" to "evaluate the whole program."
- `orion/autonomy/drives_and_autonomy_retrospective.md` — full history of the signal-
  integrity series (O1-O4, O2, O3) this document does not re-litigate.
- `services/orion-field-digester/README.md` — the real interoceptive field this design
  proposes making canonical.
