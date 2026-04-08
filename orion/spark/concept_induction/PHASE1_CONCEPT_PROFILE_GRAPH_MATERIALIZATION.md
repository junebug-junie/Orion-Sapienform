# Phase 1: Spark ConceptProfile Graph Materialization

## What was added

Spark `ConceptProfile` writes now include an additive graph materialization step.

- Runtime/read behavior is unchanged in this phase.
- `LocalProfileStore` remains the source of truth for current workflow/runtime reads.
- Graph write is emitted as `rdf.write.request` to `orion:rdf:enqueue` using kind `spark.concept_profile.graph.v1`.

## Graph mapping shape

Each profile materialization emits explicit RDF nodes for:

- Profile root (`orion:SparkConceptProfile`)
- Subject anchor (`orion:SparkConceptProfileSubject`)
- Concepts (`orion:SparkConcept`)
- Clusters (`orion:SparkConceptCluster`)
- Optional state estimate (`orion:SparkStateEstimate`)
- Materialization provenance (`orion:MaterializationProvenance`)

Structured fields that need graph querying are first-class triples (subject/revision/time window/concept links/cluster links).
Nested/free-form metadata remains JSON literals (`*MetadataJson`, `dimensionsJson`, `trendJson`) to avoid premature ontology bloat.

## Failure policy

Graph materialization is additive and isolated:

- local profile persistence still occurs through the existing path,
- profile publish and downstream local behavior are preserved,
- graph materialization failures are logged as `concept_profile_graph_materialization ... graph_write_succeeded=False` with `error_kind`.

## Intentionally not done in this phase

- No graph-backed read repository/query layer
- No workflow read cutover (`concept_induction_pass`, `chat_stance` remain local-backed)
- No RPC retrieval boundary for concept profiles
