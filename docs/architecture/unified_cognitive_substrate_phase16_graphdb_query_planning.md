# Unified Cognitive Substrate — Phase 16: GraphDB Query Planning, Reuse, and Source-Honest Performance

## Why this phase exists

Phase 15 made GraphDB-backed semantic reads the primary basis for cognition-facing consumers.
That closed the source-of-truth gap, but left a practical gap: repeated query construction, duplicated reads across consumers, and inconsistent visibility into reuse/degraded paths.

Phase 16 hardens that semantic spine with bounded planning and explicit reuse metadata.

## Problem statement

Without a shared query-planning discipline:

- graph cognition, consolidation, and curiosity each rebuild similar bounded query sets,
- repeated query shapes can trigger redundant GraphDB reads in the same cognitive pass,
- operators cannot easily inspect whether a semantic read path was graphdb-primary, reused from bounded cache, mixed, or degraded.

## Phase 16 design

### 1) Bounded query planning

A shared planner normalizes common cognitive intents into deterministic bounded plans:

- `graph_view_basis`
- `consolidation_region`
- `curiosity_seed`

Each plan has explicit bounded limits and a deterministic signature for equivalent requests.

### 2) Query-shape discipline + composition

A semantic read coordinator executes plan steps using canonical query kinds:

- hotspot
- concept
- contradiction
- provenance neighborhood
- focal slice

The coordinator centralizes dispatch, metadata collation, and source honesty for composed reads.

### 3) Explicit bounded reuse cache

The coordinator includes in-process, bounded query-result reuse keyed by plan-step shape.

- GraphDB remains the semantic authority.
- Reuse is only an optimization of already-executed bounded reads.
- Reuse is explicit in execution metadata (`reused_cache`, step-level reuse notes).

No hidden stale shadow is introduced as an alternative truth source.

### 4) Performance and degraded visibility

Semantic read execution metadata now includes:

- `plan_kind`
- `source_kind` (`graphdb`, `cache_reuse`, `mixed`)
- `degraded`
- `truncated`
- `reused_cache`
- coarse `duration_ms`
- per-step source/reuse notes

Consumers surface this metadata into their own notes/audit outputs.

## Consumer integration

Phase 16 routes core readers through shared planning/reuse paths:

- graph cognition view basis composition,
- consolidation region selection,
- frontier curiosity seed and focal region selection,
- runtime review audit propagation (via consolidation semantic metadata).

This is plumbing/performance work only; cognitive policy semantics are intentionally unchanged.

## Boundedness and source honesty guarantees

- All semantic query plans retain explicit `max_nodes` / `max_edges`-style bounds.
- Cache reuse does not silently replace GraphDB authority.
- Degraded paths remain explicit and inspectable in result metadata and notes.
- Control-plane ownership (queue/budget/suppression/telemetry) remains SQL/runtime-side, not moved into GraphDB.

## Follow-on work

- plan-level TTL and bounded size controls for long-lived coordinator scopes,
- deeper GraphDB query specialization per dominant consumer intents,
- operator surfaces for semantic read latency/reuse/degraded trend summaries.
