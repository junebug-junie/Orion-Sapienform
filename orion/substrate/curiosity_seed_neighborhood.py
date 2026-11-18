"""Attach each stored curiosity seed's accepted-claim neighborhood (design PR #2593, patch 1).

Runs AFTER the frontier decision, on the list about to be stored, so it cannot change
what curiosity picks or in what order. For every endogenous seed it runs one bounded
``neighborhood`` plan step (query_planning.py) over the seed's focal nodes and copies
back, onto a copy of the signal:

- ``focal_edge_refs``: links with both ends among the focal nodes;
- ``boundary_edge_refs`` / ``neighbor_node_refs``: links with one end outside, and that end;
- ``projection_endpoint_node_refs``: nodes present only because an accepted claim's
  projection touches them.

Only accepted-claim links are kept (``edge_role == "semantic_projection"``; the read
already walks those only while their Assertion is provisional/canonical at the projected
revision). Legacy unreviewed edges the default read may also return are dropped and
counted in the receipt (Juniper's decision 3, 2026-10-10).

Failure is local: a read error leaves that signal exactly as it was and is counted in
``degraded_reasons``; nothing here raises.
"""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any, Sequence

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.substrate.query_planning import (
    SubstrateQueryPlanStepV1,
    SubstrateQueryPlanV1,
    SubstrateSemanticReadCoordinator,
)

# The reading design's experimental caps (2026-10-06 §1), also NeighborhoodRequestV1's
# defaults. Not tuned, not learned. They match the schema's max_length on the new fields.
INTERNAL_EDGE_LIMIT = 12
BOUNDARY_EDGE_LIMIT = 16
NEIGHBOR_NODE_LIMIT = 16
MAX_FOCAL_PER_READ = 16  # NeighborhoodRequestV1.focal_node_ids max_length
# Whole-tick wall budget for these reads. Live p50 3 ms, max 125 ms per read (design
# PR #2593); a tick stores at most ~10 seeds. Past this, remaining seeds are stored as-is.
TICK_TIME_BUDGET_MS = 2000.0

ACCEPTED_ROLE = "semantic_projection"
SEED_NOTE = "endogenous_seed"


def _is_seed(sig: Any) -> bool:
    return isinstance(sig, FrontierInvocationSignalV1) and SEED_NOTE in (sig.notes or [])


def _plan(focal: Sequence[str]) -> SubstrateQueryPlanV1:
    return SubstrateQueryPlanV1(
        plan_kind="curiosity_seed_neighborhood",
        steps=(
            SubstrateQueryPlanStepV1(
                "neighborhood",
                {
                    "focal_node_ids": tuple(focal),
                    "internal_edge_limit": INTERNAL_EDGE_LIMIT,
                    "boundary_edge_limit": BOUNDARY_EDGE_LIMIT,
                    "neighbor_node_limit": NEIGHBOR_NODE_LIMIT,
                },
            ),
        ),
    )


def _empty_receipt() -> dict[str, Any]:
    return {
        "reads": 0,
        "nonempty": 0,
        "internal_edges": 0,
        "boundary_edges": 0,
        "projection_endpoints": 0,
        "legacy_edges_excluded": 0,
        "truncated": 0,
        "degraded_reasons": {},
        "duration_ms": 0.0,
        "max_read_ms": 0.0,
    }


def attach_seed_neighborhoods(
    signals: Sequence[Any],
    *,
    store: Any,
    time_budget_ms: float = TICK_TIME_BUDGET_MS,
) -> tuple[list[Any], dict[str, Any]]:
    """Return (same signals in the same order, seeds enriched) and a per-tick receipt."""
    out = list(signals)
    receipt = _empty_receipt()
    reasons: Counter[str] = Counter()
    if store is None:
        reasons["no_store"] += 1
        receipt["degraded_reasons"] = dict(reasons)
        return out, receipt
    coordinator = SubstrateSemanticReadCoordinator(store=store, cache_enabled=False)
    started = perf_counter()
    for index, sig in enumerate(out):
        if not _is_seed(sig):
            continue
        focal = list(dict.fromkeys(str(r) for r in sig.focal_node_refs if r))[:MAX_FOCAL_PER_READ]
        if not focal:
            reasons["no_focal_nodes"] += 1
            continue
        if (perf_counter() - started) * 1000.0 >= time_budget_ms:
            reasons["skipped:tick_time_budget"] += 1
            continue
        read_started = perf_counter()
        try:
            execution = coordinator.execute(_plan(focal))
            result = execution.results[0]
            details = dict(execution.meta.step_meta[0].details)
        except Exception as exc:  # noqa: BLE001 - counted, signal left unchanged
            reasons[f"unavailable:{type(exc).__name__}"] += 1
            continue
        finally:
            read_ms = (perf_counter() - read_started) * 1000.0
            receipt["reads"] += 1
            receipt["max_read_ms"] = round(max(receipt["max_read_ms"], read_ms), 3)
        if result.degraded and result.error:
            reasons[str(result.error)] += 1
        if result.truncated:
            receipt["truncated"] += 1
        roles = {e.edge_id: (e.edge_role or "legacy_unreviewed") for e in result.slice.edges}
        ends = {e.edge_id: (e.source.node_id, e.target.node_id) for e in result.slice.edges}
        internal = [e for e in details.get("focal_edge_refs") or [] if roles.get(e) == ACCEPTED_ROLE]
        boundary = [e for e in details.get("boundary_edge_refs") or [] if roles.get(e) == ACCEPTED_ROLE]
        receipt["legacy_edges_excluded"] += sum(1 for role in roles.values() if role != ACCEPTED_ROLE)
        focal_set = set(details.get("focal_node_refs") or [])
        neighbors: list[str] = []
        for edge_id in boundary:
            src, dst = ends[edge_id]
            outside = dst if src in focal_set else src
            if outside not in neighbors:
                neighbors.append(outside)
        touched = {n for edge_id in internal + boundary for n in ends[edge_id]}
        endpoints = [n for n in details.get("projection_endpoint_node_refs") or [] if n in touched]
        if not (internal or boundary):
            continue
        out[index] = sig.model_copy(
            update={
                # Kept on top of anything the seed already carried (none do today).
                "focal_edge_refs": list(dict.fromkeys([*sig.focal_edge_refs, *internal[:INTERNAL_EDGE_LIMIT]]))[:64],
                "boundary_edge_refs": boundary[:BOUNDARY_EDGE_LIMIT],
                "neighbor_node_refs": neighbors[:NEIGHBOR_NODE_LIMIT],
                "projection_endpoint_node_refs": endpoints[:NEIGHBOR_NODE_LIMIT],
            }
        )
        receipt["nonempty"] += 1
        receipt["internal_edges"] += len(internal)
        receipt["boundary_edges"] += len(boundary)
        receipt["projection_endpoints"] += len(endpoints)
    receipt["duration_ms"] = round((perf_counter() - started) * 1000.0, 3)
    receipt["degraded_reasons"] = dict(reasons)
    return out, receipt
