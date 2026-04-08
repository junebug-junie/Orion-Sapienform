# Unified Cognitive Substrate — Phase 14: GraphDB Read/Query Layer and Hub Wiring

## Why Phase 14

Phase 13 made GraphDB a durable write target for substrate materialization. However, semantic reads were still cache-dependent or placeholder-based in Hub semantic endpoints.

Phase 14 completes the semantic path by making GraphDB the primary bounded read source for substrate inspection.

## Persistence vs read completion

- **Phase 13:** durable GraphDB persistence behind materialization seam.
- **Phase 14:** real GraphDB-backed bounded semantic reads (cold-start safe) and Hub semantic wiring.

## Bounded query semantics

GraphDB-backed store now exposes bounded query operations:

- `query_focal_slice`
- `query_hotspot_region`
- `query_contradiction_region`
- `query_concept_region`
- `query_provenance_neighborhood`

Each query is bounded, deterministic, and reports metadata via `SubstrateQueryResultV1`:

- query kind
- source kind (`graphdb`, `fallback`, `cache`)
- degraded/error state
- truncation
- limits
- generation timestamp

## Cold-start posture and cache role

GraphDB is the **primary semantic source** for query operations.

Cache remains:

- performance optimization,
- fallback path for degraded behavior,
- deterministic dev/test seam.

Fallbacks are explicit in metadata and never silent.

## Hub `/substrate` wiring posture

Semantic endpoints now call substrate query APIs:

- `/api/substrate/overview`
- `/api/substrate/hotspots`

These responses now include explicit semantic source/degraded/truncation metadata.

## GraphDB semantic vs SQL operational split (preserved)

- **GraphDB semantic reads:** overview, hotspots, semantic region slices.
- **SQL operational reads:** review queue, runtime review executions, telemetry summary, calibration.

No control-plane migration to GraphDB is introduced in this phase.

## Non-goals

- no broader runtime autonomy expansion
- no large UI rewrite
- no ontology fork
- no SQL operational/control-plane migration
