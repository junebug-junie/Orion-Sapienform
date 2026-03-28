# Phase 4: ConceptProfile Runtime Cutover (concept_induction_pass)

## Scope
- `concept_induction_pass` is the first consumer with runtime cutover controls.
- `chat_stance` is intentionally excluded in this phase and continues to follow global repository backend behavior.

## Backend resolution order
For `concept_induction_pass`, backend is resolved in this order:
1. `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS` (if non-empty)
2. `CONCEPT_PROFILE_REPOSITORY_BACKEND`
3. implicit local default when neither is explicitly set

For all other consumers (including `chat_stance`), the global backend setting remains authoritative.

## Fallback policy semantics (graph requests only)
`CONCEPT_PROFILE_GRAPH_CUTOVER_FALLBACK_POLICY` controls behavior when requested backend resolves to `graph` and graph retrieval is unavailable (`query_error`, `graph_not_configured`, etc.).

- `fail_open_local`
  - retries through the local repository seam
  - returns local-backed workflow output
  - marks fallback explicitly in metadata/logs

- `fail_closed`
  - workflow fails explicitly
  - no silent local fallback

Note: graph `empty` responses are not treated as graph-unavailable errors and therefore do not trigger fallback policy.

## How to verify backend truth
Inspect:
- log event: `concept_profile_repository_resolution`
- workflow metadata path:
  - `metadata.workflow.concept_profile_resolution.requested_backend`
  - `metadata.workflow.concept_profile_resolution.resolved_backend`
  - `metadata.workflow.concept_profile_resolution.fallback_used`
  - `metadata.workflow.concept_profile_resolution.unavailable_reason`

## Rollback
Rollback is config-only:
- set `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS=` (empty), or
- set `CONCEPT_PROFILE_BACKEND_CONCEPT_INDUCTION_PASS=local`

No code rollback is required to return `concept_induction_pass` to local behavior.
