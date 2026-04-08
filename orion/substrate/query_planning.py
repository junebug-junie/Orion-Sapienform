from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .store import SubstrateGraphStore, SubstrateQueryResultV1


@dataclass(frozen=True)
class SubstrateQueryPlanStepV1:
    query_kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> str:
        pairs = [f"{key}={self.params[key]!r}" for key in sorted(self.params)]
        return f"{self.query_kind}|" + "|".join(pairs)


@dataclass(frozen=True)
class SubstrateQueryPlanV1:
    plan_kind: str
    steps: tuple[SubstrateQueryPlanStepV1, ...]

    @property
    def signature(self) -> str:
        return f"{self.plan_kind}::" + ";".join(step.cache_key() for step in self.steps)


@dataclass(frozen=True)
class SubstrateQueryStepExecutionMetaV1:
    query_kind: str
    source_kind: str
    degraded: bool
    truncated: bool
    reused_cache: bool
    duration_ms: float
    limits: dict[str, int]
    details: dict[str, Any]


@dataclass(frozen=True)
class SubstrateQueryExecutionMetaV1:
    plan_kind: str
    source_kind: str
    degraded: bool
    truncated: bool
    reused_cache: bool
    duration_ms: float
    step_meta: tuple[SubstrateQueryStepExecutionMetaV1, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubstrateQueryExecutionV1:
    plan: SubstrateQueryPlanV1
    results: tuple[SubstrateQueryResultV1, ...]
    meta: SubstrateQueryExecutionMetaV1


class SubstrateQueryPlanner:
    @staticmethod
    def graph_view_basis(*, subject_ref: str | None, max_nodes: int, max_edges: int) -> SubstrateQueryPlanV1:
        bounded_nodes = max(1, int(max_nodes))
        bounded_edges = max(1, int(max_edges))
        return SubstrateQueryPlanV1(
            plan_kind="graph_view_basis",
            steps=(
                SubstrateQueryPlanStepV1("hotspot_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
                SubstrateQueryPlanStepV1("concept_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
                SubstrateQueryPlanStepV1("contradiction_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
                SubstrateQueryPlanStepV1(
                    "provenance_neighborhood",
                    {"evidence_ref": str(subject_ref or ""), "limit_nodes": bounded_nodes, "limit_edges": bounded_edges},
                ),
            ),
        )

    @staticmethod
    def consolidation_region(*, target_zone: str, focal_node_refs: list[str], max_nodes: int, max_edges: int) -> SubstrateQueryPlanV1:
        bounded_nodes = max(1, int(max_nodes))
        bounded_edges = max(1, int(max_edges))
        if focal_node_refs:
            return SubstrateQueryPlanV1(
                plan_kind="consolidation_region",
                steps=(
                    SubstrateQueryPlanStepV1(
                        "focal_slice",
                        {"node_ids": list(focal_node_refs), "max_edges": bounded_edges},
                    ),
                ),
            )
        if target_zone in {"concept_graph", "world_ontology"}:
            step = SubstrateQueryPlanStepV1("concept_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges})
        else:
            step = SubstrateQueryPlanStepV1("hotspot_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges})
        return SubstrateQueryPlanV1(plan_kind="consolidation_region", steps=(step,))

    @staticmethod
    def curiosity_seed(*, max_nodes: int, max_edges: int) -> SubstrateQueryPlanV1:
        bounded_nodes = max(1, int(max_nodes))
        bounded_edges = max(1, int(max_edges))
        return SubstrateQueryPlanV1(
            plan_kind="curiosity_seed",
            steps=(
                SubstrateQueryPlanStepV1("hotspot_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
                SubstrateQueryPlanStepV1("concept_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
                SubstrateQueryPlanStepV1("contradiction_region", {"limit_nodes": bounded_nodes, "limit_edges": bounded_edges}),
            ),
        )


class SubstrateSemanticReadCoordinator:
    """Bounded query-plan executor with explicit in-process query reuse metadata."""

    def __init__(self, *, store: SubstrateGraphStore, cache_enabled: bool = True) -> None:
        self._store = store
        self._cache_enabled = bool(cache_enabled)
        self._cache: dict[str, SubstrateQueryResultV1] = {}

    def execute(self, plan: SubstrateQueryPlanV1) -> SubstrateQueryExecutionV1:
        started = perf_counter()
        results: list[SubstrateQueryResultV1] = []
        step_meta: list[SubstrateQueryStepExecutionMetaV1] = []
        reused_any = False

        for step in plan.steps:
            key = step.cache_key()
            step_started = perf_counter()
            if self._cache_enabled and key in self._cache:
                result = self._cache[key]
                reused = True
                reused_any = True
            else:
                result = self._dispatch(step)
                if self._cache_enabled:
                    self._cache[key] = result
                reused = False
            duration_ms = (perf_counter() - step_started) * 1000.0
            results.append(result)
            step_meta.append(
                SubstrateQueryStepExecutionMetaV1(
                    query_kind=step.query_kind,
                    source_kind=result.source_kind,
                    degraded=result.degraded,
                    truncated=result.truncated,
                    reused_cache=reused,
                    duration_ms=duration_ms,
                    limits=dict(result.limits),
                    details=dict(result.details),
                )
            )

        duration_ms = (perf_counter() - started) * 1000.0
        degraded = any(item.degraded for item in results)
        truncated = any(item.truncated for item in results)

        if all(item.source_kind == "graphdb" for item in results):
            source_kind = "graphdb"
        elif reused_any and all(item.source_kind in {"graphdb", "cache"} for item in results):
            source_kind = "cache_reuse"
        elif not self._cache_enabled:
            source_kind = "graphdb_no_cache"
        else:
            source_kind = "mixed"

        meta = SubstrateQueryExecutionMetaV1(
            plan_kind=plan.plan_kind,
            source_kind=source_kind,
            degraded=degraded,
            truncated=truncated,
            reused_cache=reused_any,
            duration_ms=duration_ms,
            step_meta=tuple(step_meta),
            notes=tuple(
                [f"cache_enabled:{self._cache_enabled}"]
                + [f"step:{item.query_kind}:{item.source_kind}:reuse={item.reused_cache}" for item in step_meta]
            ),
        )
        return SubstrateQueryExecutionV1(plan=plan, results=tuple(results), meta=meta)

    def _dispatch(self, step: SubstrateQueryPlanStepV1) -> SubstrateQueryResultV1:
        if step.query_kind == "hotspot_region":
            return self._store.query_hotspot_region(
                min_salience=float(step.params.get("min_salience", 0.6)),
                limit_nodes=int(step.params.get("limit_nodes", 32)),
                limit_edges=int(step.params.get("limit_edges", 64)),
            )
        if step.query_kind == "concept_region":
            return self._store.query_concept_region(
                limit_nodes=int(step.params.get("limit_nodes", 32)),
                limit_edges=int(step.params.get("limit_edges", 64)),
            )
        if step.query_kind == "contradiction_region":
            return self._store.query_contradiction_region(
                limit_nodes=int(step.params.get("limit_nodes", 32)),
                limit_edges=int(step.params.get("limit_edges", 64)),
            )
        if step.query_kind == "provenance_neighborhood":
            return self._store.query_provenance_neighborhood(
                evidence_ref=str(step.params.get("evidence_ref", "")),
                limit_nodes=int(step.params.get("limit_nodes", 32)),
                limit_edges=int(step.params.get("limit_edges", 64)),
            )
        if step.query_kind == "focal_slice":
            return self._store.query_focal_slice(
                node_ids=list(step.params.get("node_ids") or []),
                max_edges=int(step.params.get("max_edges", 64)),
            )
        raise ValueError(f"unsupported_query_kind:{step.query_kind}")
