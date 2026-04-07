# Reasoning Materialization Phase 2

Phase 2 operationalizes the Phase 1 canonical reasoning schemas with a bounded write/materialization seam.

## What landed

- `orion/core/schemas/reasoning_io.py`
  - `ReasoningWriteContextV1`
  - `ReasoningWriteRequestV1`
  - `ReasoningWriteResultV1`
  - `ReasoningArtifactV1` discriminated union over canonical reasoning artifact models
- `orion/reasoning/repository.py`
  - `ReasoningRepository` protocol
  - `InMemoryReasoningRepository` deterministic local seam (`write_artifacts`, `list_latest`, `list_by_scope`, `list_by_type`)
- `orion/reasoning/materializer.py`
  - `ReasoningMaterializer.materialize(...)` as the narrow typed write path
- adapters:
  - `orion/reasoning/adapters/concept_induction.py`
  - `orion/reasoning/adapters/autonomy.py`
  - `orion/reasoning/adapters/spark_state.py`

## Mapping policy used in this phase

- Conservative status policy: adapter output defaults to `proposed`/`provisional`; no silent canonical promotion.
- Provenance preserved with source family/kind/channel/producer and source references where available.
- Anchor scope preserved from producer shape (`subject` mapping for concept/autonomy, fixed `orion` for spark snapshots).

## Drift found and handling decision

1. **Concept induction drift**
   - Source family has rich `ConceptItem`/`ConceptCluster` contracts but Phase 1 reasoning family does not yet include a dedicated `ConceptV1` payload model.
   - **Decision now:** translate concept items to conservative `ClaimV1` (`claim_kind="concept_item"`) and cluster membership to `RelationV1`.
   - **Follow-up:** add explicit concept artifact payload model in a future phase to reduce translation loss.

2. **Spark source fragmentation**
   - Spark appears as both direct `SparkStateSnapshotV1` and embedded under `SparkTelemetryPayload.metadata["spark_state_snapshot"]`.
   - **Decision now:** map both through `spark_state` adapter seam; no fake producer introduced.

## Why this is bounded

- No broad runtime routing cutover was performed.
- No promotion engine or mentor gateway runtime writes were introduced.
- This layer is ready for Phase 3 policy/promotion logic to sit above write/materialization and repository seams.
