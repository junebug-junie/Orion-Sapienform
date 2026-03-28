# Phase 3A: Shadow rollout for Spark ConceptProfile repository

## Purpose

Phase 3A enables shadow-mode reads in real runtime consumers so local-vs-graph parity can be validated on live traffic **without changing user-visible behavior**.

## What changed

- `concept_induction_pass` now calls the repository seam with observer metadata (`consumer`, `correlation_id`, `session_id`).
- `chat_stance` now calls the repository seam with observer metadata from request context.
- Shadow repository parity logs now include bounded mismatch semantics and unavailable reasons.

## Operational model

- `CONCEPT_PROFILE_REPOSITORY_BACKEND=local` (default): local-only result path.
- `CONCEPT_PROFILE_REPOSITORY_BACKEND=shadow`: local result is returned; graph result is compared and logged.
- `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph`: graph-only read path (manual verification mode).

## Parity diagnostics

`concept_profile_repository_parity` emits:

- consumer
- backend
- subjects_requested
- local_profiles_returned
- graph_profiles_returned
- mismatch_count
- mismatch_fields
- unavailable_reason
- correlation_id / session_id

## Evidence needed before cutover

Before any graph-default cutover:

1. Sustained shadow traffic with low mismatch counts.
2. Clear handling of graph unavailability without user-visible regressions.
3. Confidence that mismatch fields are understood and actioned.

## Explicit non-goals in Phase 3A

- No default cutover to graph
- No LocalProfileStore removal
- No RPC/service boundary
