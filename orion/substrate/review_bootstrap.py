from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from orion.core.schemas.substrate_consolidation import GraphConsolidationDecisionV1, GraphConsolidationResultV1
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.store import SubstrateGraphStore, SubstrateQueryResultV1


@dataclass(frozen=True)
class GraphReviewBootstrapExecutionV1:
    items_before: int
    items_after: int
    items_enqueued: int
    due_after: int
    scheduled_decision_count: int
    semantic_source: str
    semantic_degraded: bool
    notes: list[str]


class GraphReviewBootstrapper:
    """Explicit operator-invoked bootstrap for initial review frontier seeding."""

    def __init__(self, *, scheduler: GraphReviewScheduler, semantic_store: SubstrateGraphStore) -> None:
        self._scheduler = scheduler
        self._store = semantic_store

    def bootstrap(self, *, now: datetime | None = None, query_limit: int = 12) -> GraphReviewBootstrapExecutionV1:
        t = now or datetime.now(timezone.utc)
        scheduling_now = t - timedelta(days=1)
        limit_nodes = max(1, min(32, int(query_limit)))
        limit_edges = max(4, min(64, int(limit_nodes * 2)))

        queue_before = self._scheduler.queue.snapshot(limit=200).queue_items
        before_ids = {item.queue_item_id for item in queue_before}

        contradiction = self._store.query_contradiction_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
        hotspot = self._store.query_hotspot_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
        concept = self._store.query_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)

        seed_specs = [
            ("contradiction_region", contradiction, "concept_graph"),
            ("hotspot_region", hotspot, "autonomy_graph"),
            ("concept_region", concept, "world_ontology"),
        ]

        scheduled_decision_count = 0
        notes: list[str] = []

        for query_kind, query_result, target_zone in seed_specs:
            decision = self._build_decision(query_kind=query_kind, query_result=query_result, target_zone=target_zone)
            if decision is None:
                notes.append(f"seed_skipped:{query_kind}")
                continue

            anchor_scope, subject_ref = self._anchor_subject(query_result)
            if anchor_scope is None:
                notes.append(f"seed_skipped:{query_kind}:missing_anchor_scope")
                continue

            request_id = f"graph-review-bootstrap-{query_kind}-{uuid4()}"
            consolidation_result = GraphConsolidationResultV1(
                request_id=request_id,
                decisions=[decision],
                outcome_counts={decision.outcome: 1},
                regions_reviewed=[query_kind, f"nodes:{len(query_result.slice.nodes)}", f"edges:{len(query_result.slice.edges)}"],
                unresolved_regions=[query_kind] if decision.outcome in {"requeue_review", "maintain_priority"} else [],
                confidence=decision.confidence,
                degraded=bool(query_result.degraded),
                notes=["operator_bootstrap_seed", f"query_kind:{query_kind}", f"source:{query_result.source_kind}"],
            )
            scheduled = self._scheduler.apply_consolidation_result(
                consolidation_result=consolidation_result,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                now=scheduling_now,
                invocation_surface="operator_review",
            )
            scheduled_decision_count += len(scheduled.schedule_decisions)

        queue_after = self._scheduler.queue.snapshot(limit=200).queue_items
        after_ids = {item.queue_item_id for item in queue_after}
        due_after = len(self._scheduler.queue.list_eligible(now=t, limit=200))

        semantic_source = "mixed"
        semantic_sources = {contradiction.source_kind, hotspot.source_kind, concept.source_kind}
        if len(semantic_sources) == 1:
            semantic_source = next(iter(semantic_sources))
        semantic_degraded = bool(contradiction.degraded or hotspot.degraded or concept.degraded)

        if not notes:
            notes.append("bootstrap_seeded")

        return GraphReviewBootstrapExecutionV1(
            items_before=len(queue_before),
            items_after=len(queue_after),
            items_enqueued=len(after_ids - before_ids),
            due_after=due_after,
            scheduled_decision_count=scheduled_decision_count,
            semantic_source=semantic_source,
            semantic_degraded=semantic_degraded,
            notes=notes[:16],
        )

    def _build_decision(
        self,
        *,
        query_kind: str,
        query_result: SubstrateQueryResultV1,
        target_zone: str,
    ) -> GraphConsolidationDecisionV1 | None:
        nodes = query_result.slice.nodes
        if not nodes:
            return None

        node_ids = sorted({node.node_id for node in nodes if node.node_id})[:8]
        if not node_ids:
            return None

        if query_kind == "contradiction_region":
            max_pressure = max(float(node.metadata.get("dynamic_pressure") or 0.0) for node in nodes)
            unresolved = any(not bool(node.metadata.get("resolved", False)) for node in nodes)
            high_pressure = max_pressure >= 0.6
            outcome = "maintain_priority" if unresolved and high_pressure else "requeue_review"
            reason = "bootstrap contradiction frontier"
            priority = 90 if outcome == "maintain_priority" else 76
            evidence = f"contradictions:{len(nodes)} pressure:{max_pressure:.3f}"
            notes = ["bootstrap_seed", "contradiction_frontier"]
        elif query_kind == "hotspot_region":
            outcome = "requeue_review"
            reason = "bootstrap hotspot frontier"
            priority = 72
            evidence = f"hotspots:{len(nodes)}"
            notes = ["bootstrap_seed", "hotspot_frontier"]
        else:
            outcome = "requeue_review"
            reason = "bootstrap concept frontier"
            priority = 68
            evidence = f"concepts:{len(nodes)}"
            notes = ["bootstrap_seed", "concept_frontier"]

        return GraphConsolidationDecisionV1(
            target_refs=node_ids,
            outcome=outcome,
            reason=reason,
            confidence=0.72,
            zone=target_zone,
            priority=priority,
            notes=notes,
            evidence_summary=evidence,
        )

    @staticmethod
    def _anchor_subject(query_result: SubstrateQueryResultV1) -> tuple[str | None, str | None]:
        for node in query_result.slice.nodes:
            if node.anchor_scope:
                return node.anchor_scope, node.subject_ref
        return None, None
