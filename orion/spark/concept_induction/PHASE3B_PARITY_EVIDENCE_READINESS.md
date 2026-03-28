# Phase 3B: Parity evidence and cutover-readiness model

## Goal

Phase 3B converts shadow-mode parity logs into bounded operational evidence so cutover readiness can be evaluated explicitly per consumer.

## Evidence collected (bounded)

Per consumer (`concept_induction_pass`, `chat_stance`):

- total comparisons
- exact matches
- mismatches
- graph unavailable count
- empty-on-local-only count
- empty-on-graph-only count
- mismatch class frequencies
- recent subject samples
- last updated timestamp

This store is process-local and reset-on-restart by design in this phase.

## Stable mismatch classes

- `revision_mismatch`
- `profile_missing_on_graph`
- `profile_missing_on_local`
- `concept_count_mismatch`
- `cluster_count_mismatch`
- `concept_identity_mismatch`
- `state_estimate_mismatch`
- `freshness_window_mismatch`
- `graph_unavailable`
- `query_error`

## Readiness model

Readiness is computed per consumer and is **advisory only**:

- minimum sample count met
- mismatch rate <= configured threshold
- graph unavailable rate <= configured threshold
- no critical mismatch classes present

Default thresholds are configurable through environment.

## Inspectable surfaces

1. `concept_profile_parity_evidence` summary logs (periodic interval)
2. `orion.cortex.orch.info.request.v1` response now includes a parity evidence snapshot

## Why this is intentionally bounded

- no dashboard product
- no heavy persistence database
- no automatic cutover logic

This phase provides enough operational signal for deliberate cutover decisions while preserving local-return behavior.

## Non-goals retained

- no default graph cutover
- no LocalProfileStore removal
- no RPC boundary
