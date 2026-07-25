# Bus synaptic graph — verb-latency slicing gaps + causal-DAG/content-drift (item 2)

Status: **scoping + first build.** Follow-up to this session's post-arc audit (after G/F/B/A/C/D
shipped and live-verified, item 1 — `bus_mirror.sqlite` retention — fixed separately). Covers four
named-but-unbuilt gaps from `services/orion-bus-mirror/README.md`'s "Phase 2" section and the parent
`2026-07-24-bus-vitality-field-signal-brainstorm.md`'s "Signal families" list:

1. **Per-node verb-latency partitioning** — `EXECUTES_VERB` edges capture `last_node` but do not
   partition the EWMA baseline by it; an organ running the same verb on two hosts blends both
   hosts' latencies into one baseline. Named explicitly as a "Real limitation, not solved here" in
   the README.
2. **Preceding-verb / `model_used` / concurrent-load slicing** — same README paragraph, same
   "not built" status. A flat per-verb average can't separate "this verb is slow" from "it's slow
   *because* of what ran before it, or which model served it, or how loaded the mesh was."
3. **Causal-DAG empirical verification** — `orion/signals/registry.py`'s `ORGAN_REGISTRY`
   `causal_parent_organs` edges are hand-authored, "first-pass structural approximations" per that
   file's own comment. The live `CAUSALLY_FOLLOWED_BY` edges are the mechanism to verify them
   against real traffic; parked as "eventual, not prioritized" in the parent brainstorm doc.
4. **Content-drift signals** — three concrete, non-phi ideas named in the parent doc, none built:
   verb-sequence drift (does the *set* of distinct chain shapes change over a window — Idea C's
   `summarize_chain_shapes` now gives this a real data source it didn't have before), payload schema
   drift (do the *keys present* in a channel's payload change over time), error-kind recurrence
   (which `error`/`status=failed` messages repeat verbatim vs. are novel).

**Sequencing, same "additive, not exclusive" framing as the wider-net doc:** build (1) now — data is
already captured (`last_node` exists on every `EXECUTES_VERB` edge today), the fix is purely how the
graph key partitions it, no new extraction logic needed. (2), (3), (4) are scoped below but not built
in this pass — each is either a bigger mechanism (2), explicitly "not prioritized" already (3), or a
fresh extraction pipeline with no existing data source (4). None block each other.

## Ground truth

`BusSynapticGraphWriter.record_verb_step()` (`app/graph_writer.py`) currently:
```cypher
MATCH (:Organ {organ_id: $organ_id})-[e:EXECUTES_VERB]->(:Verb {verb_name: $verb_name})
```
— one edge per `(organ_id, verb_name)` pair, full stop. `node` is written as a mutable `SET`
property (`e.last_node = $node`), overwritten on every observation regardless of which host actually
produced it — so the *baseline itself* (`latency_ewma_sec`/`latency_var`/`latency_zscore`) is a blend
across every host that organ has ever run on, and `last_node` only ever reflects whichever host
happened to report most recently.

## Idea 2.1 — per-node verb-latency partitioning (built this pass)

**What:** include `node` in the `EXECUTES_VERB` relationship's identity, not just as a mutable
property — `MERGE (o)-[e:EXECUTES_VERB {node: $node}]->(v)` instead of
`MERGE (o)-[e:EXECUTES_VERB]->(v)` + `SET e.last_node`. Creates one distinct edge per
`(organ_id, verb_name, node)` triple (a real multigraph between the same two nodes, which Cypher/
FalkorDB supports natively), each with its own independent EWMA baseline.

**Why:** the README already names this as the specific thing preventing "this verb is slow" from
being separated from "this specific machine is loaded" — the original motivation for capturing
`node` at all. Directly closes that gap with no new data collection.

**Smallest buildable version:** exactly the MERGE/MATCH pattern change above, in both the read
(`MATCH ... {node: $node}`) and write (`MERGE ... {node: $node}`) queries. `node=None` (a real,
common case — verb-step payloads don't always carry it) becomes its own valid partition (an
"unknown-node" bucket), not an error or a dropped observation.

**Files:** `services/orion-bus-mirror/app/graph_writer.py` (`record_verb_step`), its tests, README.

## Idea 2.2 — preceding-verb / model_used / concurrent-load slicing (scoped, not built)

**What:** three separate slicing axes on `EXECUTES_VERB`, each a real, distinct signal:
- Preceding verb: partition by `(organ, verb, preceding_verb)` — needs the chain-shape machinery
  Idea C just built (`organ_sequence`) extended to track *verb* sequence, not just organ sequence,
  since "preceding verb" isn't the same as "preceding organ."
- `model_used`: partition by whatever field in the step payload names the serving model (needs a
  live check of whether `model_used` is actually populated in real `steps[]` payloads — unverified,
  same class of question Idea G resolved for `causality_chain`).
- Concurrent-load: z-score a verb's latency against how loaded the mesh was *when it ran* (ties
  directly to the in-flight-chain-count signal Phase 2 already built) — this is structurally the
  same mechanism the wider-net doc's Idea E scoped and found inconclusive for edge-level anomalies;
  the same measure-first caution applies here.

**Why not built now:** each of these needs its own live-data check before committing to a graph
shape (matching this arc's own repeated lesson: don't build partitioning on an axis until you know
it's actually populated/meaningful in real traffic). Real added complexity too — 2.1 alone already
splits one edge into N; stacking more partition axes multiplies edge count fast.

**Smallest next step if picked up:** a read-only prevalence check (same shape as Idea G) — sample
real `steps[]` payloads, check how often `model_used` is populated and how many distinct
preceding-verb pairs actually recur, before deciding whether either axis is worth the edge-count
cost.

## Idea 2.3 — causal-DAG empirical verification (built, this session -- diagnosed, not fixed)

**What:** compare `ORGAN_REGISTRY`'s hand-authored `causal_parent_organs` edges
(`orion/signals/registry.py`) against the live `CAUSALLY_FOLLOWED_BY` edges — which hand-authored
edges are confirmed by real traffic, which are missing from real traffic, which real edges exist
that the hand-authored graph never named.

**Built**: `orion/signals/scripts/causal_dag_empirical_verification.py`, read-only, no code/schema
change. Live organ_id is derived from each registry entry's `service` field (`"orion-cortex-exec"`
-> `"cortex-exec"`), not the registry key itself — several registry entries (`graph_cognition`,
`autonomy`, `cortex_exec`) model internal reasoning stages of the *same* physical service, so a
passive wiretap can never observe an edge between them (one `organ_id` per real bus service). Those
edges are tracked in their own `same_service_internal_edges` bucket, not counted as false negatives.

**Live result** (152 real `CAUSALLY_FOLLOWED_BY` edges, 30 registry entries, run 2026-07-25):

- **1 unmapped** registry entry (`journaler` — `service` field is a free-text library reference, not
  a real bus service name).
- **1 same-service internal edge** (`cortex-exec -> cortex-exec`, from `graph_cognition`/`autonomy`).
- **3 confirmed**: `cortex-exec -> llm-gateway`, `cortex-exec -> recall`,
  `vision-council -> spark-introspector`.
- **22 missing from live**: registry claims these edges exist, no live evidence found (e.g.
  `biometrics -> collapse-mirror`, `recall -> dream`, `social-memory -> recall`). Some of this is
  likely an artifact of this arc's own known limitation (only *explicitly propagated*
  `correlation_id`s ever produce a real hop — most bus envelopes carry a fresh, never-repeated one),
  not necessarily proof the causal relationship is false; not disambiguated further in this pass.
- **149 observed, not in registry**: real edges the hand-authored DAG never named at all, including
  entire organs central to current real traffic that aren't represented as registry
  causal-parent/child relationships the way they're actually wired today — `landing-pad`, `actions`,
  `orion-harness-governor`, `sql-writer`, `orion-vector-host`.

**Verdict**: `ORGAN_REGISTRY`'s own trailing comment ("first-pass structural approximations... must
verify") is confirmed accurate and, if anything, understated — only 3 of its mappable edges have any
live confirmation, and the real mesh has ~50x more causal structure than the registry documents.
**Diagnosed, not fixed** — correcting/expanding `ORGAN_REGISTRY` to match real topology is a
substantially bigger task than this verification pass (which organs to add, which edges to remove
vs. mark unconfirmed-but-plausible, whether `journaler`'s missing bus-service mapping means it's
dead code or just legitimately non-bus) and deserves its own scoping decision, not silent scope
creep from "verify" into "rewrite."

**Files:** `orion/signals/scripts/causal_dag_empirical_verification.py` (new), `orion/signals/tests/test_causal_dag_empirical_verification.py` (new, 7 tests).

## Idea 2.4 — content-drift signals (scoped, not built)

**What:** three concrete alternatives, all content-grounded rather than volume-grounded:
- Verb-sequence drift: does the *set* of distinct chain shapes change over a window? Idea C's
  `summarize_chain_shapes` (shipped) is the data source this now has that it didn't before — a
  never-before-seen shape appearing, or a previously-common one disappearing, is itself a signal.
  This is the closest of the four to already having a foundation.
- Payload schema drift: do the *keys present* in a channel's payload change over time, independent
  of what the values say — cheap key-set diffing, catches producer changes without reading every
  changelog.
- Error-kind recurrence: which `error`/`status=failed` messages repeat verbatim vs. are novel over
  time — a real regression-detector shape.

**Why not built now:** all three need a persistence layer this arc hasn't built (a rolling window of
*sets*, not scalars — different shape than every EWMA-based signal shipped so far). Real, worthwhile
direction, genuinely bigger than a single-patch slice.

**Smallest next step if picked up:** verb-sequence drift specifically, since Idea C already produces
the raw material (`summarize_chain_shapes`' output) — would need a small persistence addition
(remember the shape set from N ticks ago, diff against current) rather than a whole new extraction
pipeline like the other two.

## Non-goals

- Not building 2.2/2.3/2.4 in this pass — each named, scoped, sequenced as a future candidate, not
  silently dropped.
- Not attempting to unify all four into one mechanism — they are genuinely different shapes (edge
  partitioning, DAG diffing, set-drift detection), forcing a shared abstraction would be premature.

## Acceptance checks

Same discipline as the rest of this arc: live-data sanity check against the real running graph
before calling 2.1 done (confirm distinct per-node baselines actually diverge on a real multi-host
verb, not just that the query executes).

## Recommended next patch

2.1 now (this session). 2.2/2.3/2.4 remain named, scoped, additive candidates — no build order
imposed among them, same as A/C/D/E in the wider-net doc.
