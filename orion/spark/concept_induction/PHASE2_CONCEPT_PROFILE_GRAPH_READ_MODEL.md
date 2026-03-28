# Phase 2: Spark ConceptProfile Graph Read Model

## What was added

A graph-backed Spark concept-profile read backend now exists behind the existing repository seam.

- `LocalConceptProfileRepository` remains the default backend.
- `GraphConceptProfileRepository` can be selected explicitly.
- `ShadowConceptProfileRepository` is available for parity diagnostics while still returning local results.

## Why this shape

Workflow/runtime consumers continue to call the same repository contract.
Raw SPARQL and GraphDB result parsing are isolated inside bounded query + mapper modules:

- `graph_query.py` for SPARQL query construction/execution.
- `graph_mapper.py` for deterministic mapping from SPARQL rows into `ConceptProfile`.

This keeps graph query assumptions explicit and out of workflow logic.

## Latest semantics

“Latest profile” is deterministic per subject:

1. highest `revision`
2. if tied, latest `createdAt`
3. if tied, lexicographically highest `profileId`

This ordering is encoded in the graph latest-profile SPARQL filter.

## Availability semantics

Graph backend distinguishes:

- `available`: profile row resolved and mapped
- `empty`: backend query succeeded but no profile for subject
- `unavailable`: graph unconfigured or query failure

## Observability

Repository emits compact logs:

- `concept_profile_repository_status`
- `concept_profile_repository_parity` (shadow mode)

## Intentionally not done in Phase 2

- No workflow/runtime cutover to graph by default
- No RPC/service boundary
- No LocalProfileStore removal
- No generic graph framework beyond Spark concept-profile retrieval
