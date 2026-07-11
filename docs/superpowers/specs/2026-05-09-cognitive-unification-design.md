# Cognitive Unification — Complete Substrate + Relational Read Model

**Date:** 2026-05-09
**Status:** Approved for implementation
**Phase:** Unified Cognitive Substrate — Relational Completion

---

## Problem

Orion has a multi-phase Unified Cognitive Substrate program (Phases 1–21): canonical
`SubstrateGraphRecordV1` contracts, adapters for concept induction, autonomy, and Spark,
materialization/reconciliation, GraphDB/SQL policy, and mutation control plane. The adapters
and materializer exist and are correct.

What is missing is operational completion: `chat_stance.py` still composes 8+ independent
producer lanes (YAML fallbacks, local JSON concept profiles, autonomy SPARQL across three
named graphs, `orionmem` SPARQL, social bridge dicts, recall bundle, reasoning repo) without
shared provenance, trust ordering, or a single inspectable read path.

Specifically:
- No canonical read model for "what Orion believes, with provenance, per anchor"
- No declared trust tiers or merge policy across producers
- No promotion rules from noisy signals into durable relational facts
- Five producer lanes (identity YAML, self-study RDF, orionmem, recall, social) have no
  substrate adapter at all — their contributions are invisible to the substrate
- Operators and downstream agents cannot rely on one graph or API

---

## Goal

Deliver bounded unification that:

1. Defines one canonical read model (`UnifiedRelationalBeliefSetV1`) for stance, recall, and
   workflows, backed by the substrate-first architecture
2. Wires all producer lanes (autonomy, Spark ConceptProfile, concept induction, identity YAML,
   self-study RDF, orionmem, tri-layer recall, social memory) into the substrate with explicit
   trust tiers and merge policy
3. Documents scope and non-scope boundaries (social-memory consent, self-study folding,
   ephemeral vs. durable tier routing)
4. Provides verification: per-adapter unit tests, merge policy tests, and a golden path
   integration scenario (multi-producer update → materialize → read in stance)

---

## Non-Goals

- Replacing all SQL/vector recall infrastructure
- Collapsing the social-memory privacy model or consent boundary
- Full automatic extraction of sensitive personal facts without human/operator promotion
- New Hub UI surfaces (observability routes through existing Hub substrate inspection and Mind tab)
- Breaking changes to prompt contracts or downstream stance field schema

---

## Architecture

### New Module

```
orion/substrate/relational/
  __init__.py
  registry.py          # ProducerRegistryV1, ProducerEntryV1, TrustTierV1
  layer.py             # CognitiveUnificationLayer — warm path + cold fallback coordinator
  beliefs.py           # UnifiedRelationalBeliefSetV1, AnchorBeliefSliceV1
  adapters/
    __init__.py
    identity_yaml.py   # YAML kernel → operator_static tier
    self_study.py      # self-study RDF → graphdb_durable, folds into orion anchor
    orionmem.py        # AffectiveDisposition SPARQL → snapshot_ephemeral
    recall.py          # tri-layer recall bundle from ctx → snapshot_ephemeral
    social.py          # social inspection/bridge from ctx → snapshot_ephemeral
  tests/
    test_adapters.py
    test_layer.py
    test_golden_path.py
```

Existing `orion/substrate/adapters/` (autonomy, spark, concept_induction) are unchanged.
They are registered in `ProducerRegistryV1` without modification.

### Data Flow

```
Producers                      CognitiveUnificationLayer        Substrate Store
──────────────────────         ─────────────────────────        ─────────────────────
identity YAML (startup)   →    materialize (operator_static) →  GraphDB + in-mem cache
autonomy adapter          →    materialize (graphdb_durable) →  GraphDB + in-mem cache
concept induction adapter →    materialize (concept_induced) →  GraphDB + in-mem cache
Spark adapters            →    materialize (concept_induced) →  GraphDB + in-mem cache
self-study RDF adapter    →    materialize (graphdb_durable) →  GraphDB + in-mem cache
orionmem SPARQL adapter   →    materialize (snapshot_ephemeral)→ in-mem only
recall bundle adapter     →    materialize (snapshot_ephemeral)→ in-mem only
social adapter            →    materialize (snapshot_ephemeral)→ in-mem only
                               ↓ warm path (GraphDB + cache first)
                               ↓ cold/stale fallback (on-demand per anchor)
                               → UnifiedRelationalBeliefSetV1
                                           ↓
                               chat_stance.py unified_beliefs_for_stance(ctx)
```

### Placement

`orion/substrate/relational/` is a subpackage of the existing substrate module — library
code, not a service. `services/orion-cortex-exec` imports it for stance. Recall, Hub
inspection, and future agents can import it for their own queries without a network hop.

---

## Trust Tiers

Four tiers, ordered by durability and operator authority:

| Tier | Name | Written to GraphDB | Description |
|---|---|---|---|
| 1 | `operator_static` | Yes, write-through | Operator-authored facts: identity YAML, manually promoted relational facts. Highest authority; never auto-overwritten by lower tiers. |
| 2 | `graphdb_durable` | Yes, write-through | Machine-produced durable facts: autonomy artifacts, self-study RDF. Overwritable only by `operator_static`. |
| 3 | `concept_induced` | Yes, write-through | Induction outputs: Spark ConceptProfile, concept induction adapter. Lower confidence; subject to decay. |
| 4 | `snapshot_ephemeral` | No — in-memory only | Transient signals: orionmem dispositions, recall fragments, social memory. Inform stance but do not persist to the durable graph. |

### Merge Policy

When the materializer encounters a conflict between an existing node and an incoming node
for the same identity key, tier rank governs: a lower-numbered tier (higher authority) always
wins on `signals.confidence` and `temporal.valid_from`. A `concept_induced` write cannot
raise the confidence of an `operator_static` node.

This is encoded as an optional `tier_rank: int` field added to `ProvenanceV1` (zero-cost
for existing nodes that omit it). The merge function in `reconcile.py` is extended with a
single tier-rank guard before applying the existing max-confidence merge.

---

## `ProducerRegistryV1`

```python
@dataclass(frozen=True)
class TrustTierV1:
    name: str        # "operator_static" | "graphdb_durable" | "concept_induced" | "snapshot_ephemeral"
    rank: int        # 1–4; lower = higher authority
    write_through: bool  # whether to persist to GraphDB

@dataclass(frozen=True)
class ProducerEntryV1:
    producer_id: str               # e.g. "identity_yaml", "autonomy", "spark_concept"
    trust_tier: TrustTierV1
    anchor_scopes: tuple[str, ...] # anchors this producer covers
    freshness_ttl_sec: int         # staleness threshold
    pull_on_cold: bool             # whether to fan out when anchor is cold
    adapter_fn: Callable[[dict[str, Any]], SubstrateGraphRecordV1 | None]
    # adapter_fn receives the ctx dict from beliefs_for_stance; network-based adapters
    # (autonomy, self_study, orionmem) ignore ctx and make their own calls; ctx-based
    # adapters (recall, social, identity_yaml) read from it directly.
```

The registry is constructed once at process startup and injected into `CognitiveUnificationLayer`.

---

## `CognitiveUnificationLayer`

### Public Interface

```python
class CognitiveUnificationLayer:
    def __init__(
        self,
        registry: ProducerRegistryV1,
        store: SubstrateGraphStore,
        ephemeral_store: InMemorySubstrateGraphStore | None = None,
    ) -> None: ...

    def beliefs_for_stance(
        self,
        *,
        anchors: Sequence[str] = ("orion", "relationship", "juniper"),
        ctx: dict[str, Any] | None = None,
        timeout_sec: float = 5.0,
    ) -> UnifiedRelationalBeliefSetV1: ...
```

### Warm Path

For each requested anchor, check the most recent `temporal.observed_at` among nodes for
that anchor in the store. If within the declared TTL for all producers covering that anchor,
query the store directly — no producer fan-out occurs.

### Cold/Stale Fallback (per anchor)

If any anchor is cold or stale: identify producers covering that anchor with
`pull_on_cold=True`, fan them out concurrently via a bounded `ThreadPoolExecutor` (same
pattern as existing autonomy subject fan-out in `chat_stance.py`), materialize results via
`SubstrateGraphMaterializer`. Durable tiers write through to GraphDB; `snapshot_ephemeral`
nodes go to the ephemeral in-memory store only.

### Failure Posture

Any producer fan-out failure is caught, logged, and recorded as `degraded=True` on the
affected anchor's slice and in `UnifiedRelationalBeliefSetV1.degraded_producers`. The call
always returns. Consistent with the existing graceful-degradation pattern throughout
`chat_stance.py`.

---

## `UnifiedRelationalBeliefSetV1`

```python
@dataclass(frozen=True)
class AnchorBeliefSliceV1:
    anchor: str
    concepts: list[BaseSubstrateNodeV1]   # ConceptNodeV1 + HypothesisNodeV1
    tensions: list[BaseSubstrateNodeV1]   # TensionNodeV1
    goals: list[BaseSubstrateNodeV1]      # GoalNodeV1
    drives: list[BaseSubstrateNodeV1]     # DriveNodeV1
    snapshots: list[BaseSubstrateNodeV1]  # StateSnapshotNodeV1
    degraded: bool
    tier_outcomes: list[str]  # e.g. ["operator_static_protected:2", "concept_induced_accepted:5"]

@dataclass(frozen=True)
class UnifiedRelationalBeliefSetV1:
    anchors: dict[str, AnchorBeliefSliceV1]
    generated_at: str
    cold_anchors: list[str]        # anchors that required on-demand fallback
    degraded_producers: list[str]  # producer_ids that failed
    lineage: list[str]             # human-readable provenance summary for logging
```

---

## New Adapters

### `identity_yaml.py` — `operator_static`

- Input: identity YAML dict sourced from three ctx keys that Hub is responsible for
  populating before the first `beliefs_for_stance` call:
  `ctx["orion_identity_summary"]`, `ctx["juniper_relationship_summary"]`,
  `ctx["response_policy_summary"]`. There is no disk-reading fallback — if these keys
  are absent the adapter returns `None` and no operator_static identity snapshot is
  seeded. Callers that pre-populate the store externally (e.g. tests) bypass this
  requirement. Disk-based startup seeding is explicitly deferred.
- Output: `ConceptNodeV1` nodes for temperament, cognitive pillars, communication style;
  one `StateSnapshotNodeV1` for the overall identity snapshot
- Anchor: `orion`
- Nodes protected from overwrite by all lower tiers

### `self_study.py` — `graphdb_durable`

- Input: SPARQL query against existing self-study named graph in GraphDB
- Output: self-study induced concepts → `ConceptNodeV1` with `anchor_scope="orion"`,
  `source_kind="self_study"` in provenance
- Folds into the `orion` anchor; no separate anchor for self-study
- `pull_on_cold=True`; TTL: 5 minutes (same as autonomy)

### `orionmem.py` — `snapshot_ephemeral`

- Input: SPARQL against `orionmem:AffectiveDisposition` named graphs (replaces inline
  SPARQL block `fetch_chat_stance_memory_graph_hints` in `chat_stance.py`)
- Output: one `StateSnapshotNodeV1` per disposition with `trustPolarity` in signals metadata
- Anchor: `orion` or `relationship` depending on named graph
- In-memory only; TTL: 2 minutes
- `pull_on_cold=True`

### `recall.py` — `snapshot_ephemeral`

- Input: `ctx["recall_bundle"]` — already present at stance time, no network call
- Output: recall fragments mapped by source tag:
  `journal`/`metacog` → `ConceptNodeV1`, `tension` → `TensionNodeV1`, `dream` → `EventNodeV1`
- Anchor determined by fragment subject metadata
- In-memory only; `pull_on_cold=False` (always fresh from ctx when present)

### `social.py` — `snapshot_ephemeral`

- Input: `ctx["social_inspection_snapshot"]`, `ctx["social_stance_snapshot"]`,
  `ctx["social_turn_policy"]`, `ctx["social_peer_style_hint"]`,
  `ctx["social_context_window"]`, `ctx["social_thread_routing"]`,
  `ctx["social_repair_decision"]`
  (The last three are required to fully replace `_social_bridge_summary` — they carry
  peer reply mode, room reply mode, de-escalation flag, and context-exclusion signals.)
- Output: disposition/posture fields → `StateSnapshotNodeV1` with `anchor_scope="relationship"`
- Supersedes `_social_summary` and `_social_bridge_summary` in `chat_stance.py` with
  fallback to the original ctx paths when the adapter returns empty
- In-memory only; `pull_on_cold=False`

### Existing Adapters (registered, unchanged)

| Adapter | Tier | Anchors | `pull_on_cold` | TTL |
|---|---|---|---|---|
| `autonomy.py` | `graphdb_durable` | orion, relationship, juniper | Yes | 5 min |
| `concept_induction.py` | `concept_induced` | orion, relationship, juniper | Yes | 5 min |
| `spark.py` | `concept_induced` | orion | Yes | 5 min |

---

## `chat_stance.py` Integration

### Change Summary

The eight independent producer read lanes are replaced by a single call:

```python
beliefs = _unified_beliefs_for_stance(ctx)
```

Each existing private function is replaced by a projection helper that reads from `beliefs`
instead of calling the producer directly. Every downstream field in `build_chat_stance_context`
(concept, social, autonomy, reflective, etc.) keeps its existing return shape. Zero prompt
contract changes. Zero downstream field schema changes.

The identity YAML / `identity_kernel_with_fallbacks` path reads from
`beliefs.anchors["orion"].snapshots` with the existing hardcoded strings as last-resort
fallback if the substrate is entirely cold.

**Net effect:** ~8 fan-out call sites removed, 1 `_unified_beliefs_for_stance` call added,
~8 projection helpers added (5–15 lines each). Total line count of `chat_stance.py` decreases.

---

## Observability

### Tier Conflict Tracking

`NodeMergeDecision` in `reconcile.py` gains two optional fields:
- `tier_conflict: bool` — True when incoming tier rank > existing tier rank (i.e., a lower-authority producer is attempting to modify a higher-authority node; the existing node wins)
- `tier_outcome: str` — e.g. `"operator_static_protected"`, `"concept_induced_accepted"`

These flow through the existing `MaterializationResultV1` that the Hub substrate inspection
route already reads.

### Mind Bus Telemetry

Tier conflict counts are emitted on the existing Mind bus channel alongside other
substrate metrics at the end of each `beliefs_for_stance` call (cold path only — skip
when `cold_anchors` is empty). No new bus channel required.

Emit site: `CognitiveUnificationLayer.beliefs_for_stance`, after slice assembly.

Event payload schema (dict passed to the existing Mind bus emit function):

```python
{
    "event": "substrate.tier_outcomes",
    "generated_at": "<ISO-8601 UTC>",          # from UnifiedRelationalBeliefSetV1.generated_at
    "cold_anchors": ["orion", ...],            # anchors that triggered fan-out
    "tier_outcomes": {                         # per-anchor; keys match AnchorBeliefSliceV1.tier_outcomes
        "orion": ["operator_static_protected:2", "concept_induced_accepted:5"],
        "relationship": [],
    },
    "degraded_producers": ["autonomy"],        # empty list if none
}
```

Only emitted when `cold_anchors` is non-empty (i.e. fan-out occurred). Warm-path calls
produce no bus event.

### Manual Promotion / Override

Operator promotion of a `concept_induced` or lower fact to `operator_static` goes through
the existing mutation control plane (`mutation_control_surface.py`, Phase 21). The relational
layer does not own or duplicate that surface.

---

## Scope and Non-Scope Boundary

### In scope

- `ProducerRegistryV1` declaring all producer lanes with tiers, TTLs, anchor scopes
- `CognitiveUnificationLayer` with hybrid warm/cold routing
- `UnifiedRelationalBeliefSetV1` output schema
- Five new adapters: identity_yaml, self_study, orionmem, recall, social
- Registration of three existing adapters (autonomy, concept_induction, spark)
- Tier-aware merge policy extension in `reconcile.py`
- `chat_stance.py` integration replacing all independent fan-out lanes
- Tests: per-adapter, merge policy, golden path scenario
- Observability extensions to `NodeMergeDecision` and Mind telemetry

### Out of scope

- Social-memory consent model (social adapter reads from ctx only, no SQL write path)
- Replacing SQL/vector recall infrastructure
- Full automatic sensitive fact extraction without operator promotion
- New Hub UI surfaces
- Prompt contract changes
- **Reasoning repo** (`InMemoryReasoningRepository` / `ReasoningSummaryCompiler`) — reasoning
  artifacts are control-plane operational state, not relational identity knowledge. The
  `_reflective_summary` and reasoning projection paths in `chat_stance.py` are preserved
  as-is and do not route through the unified layer. This is an explicit decision, not an
  oversight.
- **Identity YAML hot-reload** — the YAML adapter seeds at startup only; file-watch
  re-seeding is deferred.

---

## Tests and Golden Path

### Unit Tests — Per Adapter (`test_adapters.py`)

One test per adapter: correct node kinds, correct `anchor_scope`, correct tier in provenance
metadata, empty input returns empty record without raising. Smoke test confirming three
existing adapters emit valid records through the registry.

### Unit Tests — Layer (`test_layer.py`)

- Warm path returns beliefs without calling any producer
- Cold path triggers fan-out, nodes appear in output
- Tier protection: `operator_static` node not overwritten by `concept_induced` node on same identity key
- Ephemeral isolation: `snapshot_ephemeral` nodes absent from durable store snapshot post-materialization
- Degraded producer: one adapter raises, output `degraded=True` for that anchor only
- TTL staleness: node older than TTL triggers re-materialization

### Golden Path Scenario (`test_golden_path.py`)

1. Construct `ProducerRegistryV1` with three producers: identity_yaml (operator_static),
   autonomy (graphdb_durable), concept_induction (concept_induced)
2. Instantiate `CognitiveUnificationLayer` with `InMemorySubstrateGraphStore` (no GraphDB needed)
3. Call `beliefs_for_stance(anchors=["orion"])` on cold store → all three fan-outs triggered
4. Assert: concepts non-empty from concept induction; drives present from autonomy; identity
   snapshot present with `tier_rank=operator_static`
5. Assert: identity snapshot `signals.confidence` ≥ any `concept_induced` node (tier protection)
6. Assert: `cold_anchors == ["orion"]`
7. Call `beliefs_for_stance` second time → assert no producer fan-out (warm path)
8. Assert: `lineage` names all three sources with their tiers
