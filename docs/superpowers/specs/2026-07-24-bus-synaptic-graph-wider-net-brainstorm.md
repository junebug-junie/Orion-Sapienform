# Bus synaptic graph — wider-net brainstorm (G/F/B sequenced, A/C/D/E additive)

Status: **brainstorm output, sequencing decided, implementation follows in this same
arc.** Follow-up to `docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`'s
"Wider net — additional ideas surfaced pushing past the first pass" section. That doc named five
ideas (anomaly propagation, node centrality, chain-shape clustering, cross-node imbalance,
time/session-conditioned baselines) as Phase 3+, none built. This doc grounds each one against the
*current* live graph state (not the state at the time of the original brainstorm), finds one new
problem the grounding pass surfaced that the original doc never named, and sequences a first-build
path: **G → F → B**, with **A, C, D, E carried forward as additive, independently buildable
candidates** — nothing here is exclusive-or with anything else; every idea in this doc can ship on
its own schedule without blocking or being blocked by the others, except where a specific
prerequisite is named below.

## Arsonist summary

This arc has now shipped, in order: Phase 1 (`PUBLISHES`/`CAUSALLY_FOLLOWED_BY` edges + census, PR
#1323), Phase 2 (in-flight chain tracking + `EXECUTES_VERB` per-verb latency, PR #1328, signal-
quality fix PR #1333), channel-name normalization (PR #1337), and a live reasoning-consumer path for
`orion-recall` (PR #1342/#1345/#1347/#1348, all merged and live-verified against real production
data, including two live-caught deploy-time bugs). `graph_writer.py`'s own module docstring already
names the five wider-net ideas as explicitly out of scope for what's built — this doc is not
rediscovering them, it's the planned next pass. Grounding against the *live* graph today (not the
brainstorm-time snapshot) surfaced a real, unplanned problem: 9,434 `Channel` nodes exist against a
264-entry real catalog, because PR #1337's `normalize_channel_name()` fix only applies at write time
going forward — pre-fix literal per-UUID reply-channel nodes were never cleaned up and still pollute
the graph. This blocks the most natural next wider-net idea (node centrality) on the `PUBLISHES`
side, so it gets sequenced explicitly rather than discovered mid-build.

## Current architecture

- `services/orion-bus-mirror/app/graph_writer.py` — `BusSynapticGraphWriter`, `ChainTracker`,
  `extract_verb_step_facts`. Tracks `hop_count`/`real_hop_count`/duration per `correlation_id` but
  **not** the sequence of organs touched (chain-shape clustering has no data to build on yet). No
  edge carries a `node`/host property today (cross-node imbalance has zero existing plumbing).
- `services/orion-hub/scripts/bus_synaptic_graph_routes.py` — read-only debug API:
  `/summary`, `/hot-organs` (raw `PUBLISHES` out-degree — a partial, already-shipped instance of
  "node centrality," built on the polluted `Channel` side), `/hot-edges`, `/anomalies`.
- **Live graph state as of this doc** (queried via `/api/bus-synaptic-graph/summary`): 51 organs,
  **9,434 Channel nodes**, 9 verbs, 9,487 `PUBLISHES` edges, **134 `CAUSALLY_FOLLOWED_BY` edges**, 11
  `EXECUTES_VERB` edges. The `CAUSALLY_FOLLOWED_BY` subgraph (organ-to-organ only) is small and
  unaffected by the Channel-node bloat — this is why Idea B is sequenced to start there.
- Chain termination is still genuinely undefined (`ChainTracker`'s own docstring: "there is no
  terminal marker... a chain simply stops being tracked once it goes quiet"). `causality_chain`
  field prevalence in real traffic is still unmeasured — same open question the parent brainstorm
  doc already named and left unresolved.

## Missing questions

Carried from the parent doc, still unresolved, plus one new one from this pass:

- Does the mesh actually run multi-node (athena/atlas) in practice right now? Determines whether
  Idea D (cross-node imbalance) is real or moot — unchecked.
- What does real betweenness-centrality cost against this graph's size (pre- and post-cleanup)? No
  native FalkorDB algorithm for it; would need a full-graph load into an external library
  (networkx or similar) — cost unmeasured, why Idea B's first slice is degree-only.
- Real `causality_chain` field prevalence in current live traffic — resolves the confidence question
  for Idea A (anomaly propagation) directly. **This doc's Idea G answers it.**
- **New, found during this pass's grounding check, not in the parent doc:** is the ~9,170-node excess
  Channel-node count still growing today (some write path bypassing the normalize fix), or frozen
  since PR #1337 shipped? Changes Idea F's dry-run logic — a live leak needs a different fix than a
  one-time historical cleanup.

## Proposed schema / API changes

**Sequenced first (this session, in order):**

**Idea G — `causality_chain` prevalence audit.** Read-only measurement of how often real envelopes
populate `causality_chain` vs. relying purely on `correlation_id` co-occurrence. Resolves a named,
unresolved Missing Question cheaply; informs whether `CAUSALLY_FOLLOWED_BY` edges need a confidence
property before Idea A ever trusts them for multi-hop propagation. One-off script, no service
changes, no schema change.

**Idea F — stale pre-normalization Channel-node cleanup.** A one-off migration to identify and
delete literal per-UUID `Channel` nodes created before PR #1337's fix, that no longer match the real
264-entry catalog and show low/one-shot `PUBLISHES` counts. Dry-run first (print candidates, no
`DELETE`), confirm live growth-rate question above before finalizing delete criteria, get explicit
go-ahead before any actual deletion against the live graph (this is a destructive operation against
production data per this repo's own safety rules — dry-run output is not itself authorization to
delete).

**Idea B — node centrality, starting from the clean subgraph.** Real in+out degree over the
`CAUSALLY_FOLLOWED_BY` organ-to-organ subgraph (134 edges, cheap, does not depend on Idea F). New
Hub route alongside the existing `/hot-organs`/`/hot-edges`/`/anomalies`. Betweenness deferred to a
later pass pending the graph-library cost question above.

**Carried forward as additive, independently buildable — not exclusive with anything above, no
required build order relative to each other or to G/F/B beyond what's named:**

- **Idea A — anomaly propagation trace.** Bounded BFS (`max_hops=2`) outward from a currently-
  anomalous edge via `CAUSALLY_FOLLOWED_BY`, checking each hop's own z-score, exposed as a new
  read-only Hub route. Benefits from Idea G's answer (edge confidence) but does not require it to
  ship a first version — can use unweighted co-occurrence edges as-is, just with a documented caveat.
- **Idea C — chain-shape clustering (in-progress shapes).** Extend `_ChainEntry` with a bounded
  organ-sequence list; bucket currently-open chains by sequence-so-far, report frequency. Explicitly
  labeled "in-progress-shape distribution," not "completed-routine distribution" — chain termination
  is still undefined, this sidesteps rather than solves that tension.
- **Idea D — cross-node imbalance.** Add `source_node`/`target_node` properties to
  `CAUSALLY_FOLLOWED_BY` on write if `envelope.source.node` is populated; group by node pair in a new
  route. Real value conditional on the still-unchecked "does the mesh run multi-host today" question
  above — verify before treating as a priority.
- **Idea E — time/session-conditioned baselines (measure first, don't build the mechanism yet).**
  Log-only correlation check between open-chain count and edge z-score spikes, to test whether
  load-conditioning would actually change any current anomaly classification before investing in
  bucketed-EWMA machinery. The full mechanism risks breaking the "state size bounded by topology, not
  message count" sustainability invariant this arc has held since Phase 1 if not capped carefully.

## Files likely to touch

- Idea G: new one-off script, likely `services/orion-bus-mirror/scripts/`, no service changes.
- Idea F: new one-off script, `services/orion-bus-mirror/scripts/`, dry-run mode default.
- Idea B: `services/orion-hub/scripts/bus_synaptic_graph_routes.py` (new route).
- A/C/D/E (when picked up): `services/orion-bus-mirror/app/graph_writer.py` (C, D), 
  `services/orion-hub/scripts/bus_synaptic_graph_routes.py` (A, D), tests for whichever ships.

## Non-goals

- Not building full betweenness centrality in this pass — degree-only first slice for Idea B; the
  graph-library cost question is unresolved, not assumed cheap.
- Not solving chain termination in this pass — Idea C's in-progress-shape sidestep is a deliberate,
  named compromise, not a fix.
- Not committing to a build order among A/C/D/E — each ships independently when picked up, per the
  user's explicit direction that this wider-net set is additive, not sequenced against each other.
- Not deleting any live graph data as part of writing this doc — Idea F's actual `DELETE` step
  requires its own explicit go-ahead after a dry-run pass, not blanket authorization from this spec.

## Acceptance checks

Same discipline as the rest of this arc:
- Live-data sanity check against the real running mesh before calling any of G/F/B done — G and F are
  inherently data-grounded (that's their entire point); B gets checked against real
  `CAUSALLY_FOLLOWED_BY` data, not a fixture.
- Idea F: dry-run output reviewed and explicitly approved before any live `DELETE` executes.
- Code review via subagent for B (new production Cypher/route code) before merge; G and F are one-off
  scripts, reviewed inline given their scope.

## Recommended next patch

Build G, then F, then B, in that order, this session — G is free and de-risks F/A; F is necessary
hygiene, low-risk with dry-run-first; B is the smallest wider-net idea that most directly answers the
parent design doc's own stated goal (a real alternative to `ORGAN_REGISTRY`'s hand-authored edges).
A/C/D/E remain named, gated, additive candidates for a later pass — not scheduled here.

## Idea E measurement result (same session, after G/F/B/A/C/D shipped)

Per Idea E's own scoping ("measure first, don't build the mechanism yet"), attempted the smallest
buildable version: a live correlation check between concurrent open-chain count
(`orion-bus-mirror`'s existing periodic in-flight-chain-summary log line, PR #1328) and
currently-elevated edges (`/api/bus-synaptic-graph/anomalies`).

**Available data**: ~30 minutes of real `open_count` history from container logs, ranging 121→135
(a ~12% swing, gradual drift, no sharp bursts). Paired against a live `/anomalies` snapshot at the
same rough window: 7 elevated edges (1 publish-gap, 6 causal-latency).

**Finding: inconclusive, not negative.** The available observation window never contained a real
burst in open-chain-count (the kind of sharp swing the hypothesis — "the mesh is busy because
Juniper is actively chatting" vs. "busy for no reason" — actually needs to be tested against). A
single snapshot pair, or a slow 12% drift, cannot distinguish "load-conditioning would matter" from
"it wouldn't" — there simply wasn't enough dynamic range in this window to test either way. This is
a real, honest result per Idea E's own escape valve ("if it turns out the load-conditioning question
can't be cheaply answered right now, that's still a fine deliverable"), not a disguised negative.

**Recommendation**: do not build the bucketed-EWMA mechanism yet. If revisited, the right next step
is a longer-window, real-burst-inclusive sample (e.g. an active development/chat session, which this
arc's own traffic patterns suggest produces sharper open-chain-count swings than a quiet period) —
not more measurement during another quiet window, which would likely reproduce the same
inconclusive result. No code shipped for Idea E in this session, per its own non-goal.
