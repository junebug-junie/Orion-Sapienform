# Unified Cognitive Substrate — Phase 15: GraphDB Cognitive Re-anchoring

## Why this phase exists

Phase 14 made GraphDB semantic reads real for operator inspection, but core cognitive consumers still had residual dependence on process-local substrate snapshots.

Phase 15 re-anchors cognition-facing consumers to GraphDB-backed semantic queries as the primary read basis, with explicit bounded fallback.

## Re-anchored consumers

- **Graph cognition view construction:** can now build from substrate query-backed semantic slices (`build_graph_views_from_store`) instead of only local snapshots.
- **Consolidation evaluator:** selects review regions from semantic query APIs first, with explicit `local_fallback` when query regions are empty/degraded.
- **Frontier curiosity evaluator:** derives candidate regions/signals from semantic query APIs first, preserving bounded region limits and strict-zone guardrails.
- **Runtime review execution:** audit summary now surfaces semantic source/degraded basis from consolidation execution.

## Source-of-truth posture

Primary semantic source: **GraphDB-backed substrate query layer**.

Fallback source: bounded local snapshot/cache only when required, with explicit metadata:

- `semantic_source`
- `semantic_degraded`

No silent source swaps.

## GraphDB semantic vs SQL operational split (preserved)

- **GraphDB semantic:** cognition/consolidation/curiosity region reads.
- **SQL operational/control-plane:** queue, budgets, suppression, telemetry, calibration.

Phase 15 does not move control-plane ownership into GraphDB.

## Boundedness and safety

- Query and region selection remain bounded (`max_nodes`, `max_edges`, limited focal refs).
- Runtime review still executes a single bounded cycle.
- Existing strict-zone/operator-only behavior remains intact.

## Follow-on work

- performance tuning for high-volume GraphDB query mixes,
- richer semantic region scoring over query results,
- broader consumer migration once confidence and observability are sufficient.
