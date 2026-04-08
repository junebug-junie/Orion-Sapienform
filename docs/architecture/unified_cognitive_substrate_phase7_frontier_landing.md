# Unified Cognitive Substrate — Phase 7: Frontier Delta Landing and Governance

## Why Phase 7 exists

Phase 6 introduced structured frontier graph-delta generation. Phase 7 adds the missing governed landing path so frontier deltas can be evaluated, gated, and (when allowed) materialized into substrate state.

## Generation vs landing

- **Generation (Phase 6):** frontier proposes typed delta bundles.
- **Landing (Phase 7):** deterministic policy decides per-item outcomes:
  - reject / proposed_only / provisional / materialize_now / hitl_required / blocked

This separation prevents frontier from becoming a sovereign writer.

## Typed landing contracts

Phase 7 introduces:

- `FrontierLandingRequestV1`
- `FrontierDeltaLandingDecisionV1`
- `FrontierLandingResultV1`

These contracts make decisions, blocked reasons, HITL posture, and materialization summaries explicit.

## Zone-aware policy behavior

Landing now uses real zone rules:

- `world_ontology`: most permissive materialization thresholds
- `concept_graph`: moderate thresholds
- `autonomy_graph`: stricter, review/HITL-biased
- `self_relationship_graph`: strictest; hypothesis/proposal posture and no direct protected canonical writes

## Materialization path

When decisions allow `materialize_now`, candidates are materialized via existing substrate materialization infrastructure (`SubstrateGraphMaterializer`) to preserve identity reconciliation and store semantics.

Each landed artifact carries frontier provenance metadata for inspectability.

## Contradictions and evidence gaps

Contradiction/evidence-gap items are never treated as direct facts. Where allowed, they land as bounded hypothesis markers rather than canonical truth claims.

## Non-goals

- no broad runtime auto-invocation rollout
- no learned landing policy
- no bypass of strict-zone governance/HITL expectations

## Forward path

Later phases can add:

- HITL workflow wiring for queued decisions
- promotion-integration automation from landing outputs
- runtime-triggered frontier landing orchestration with bounded policy hooks
