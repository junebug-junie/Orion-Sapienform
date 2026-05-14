"""CognitiveUnificationLayer — warm path + cold fallback coordinator.

This is the single inspectable read path for what Orion believes, with provenance,
per anchor.  It replaces the 8+ independent producer fan-outs in chat_stance.py
with a unified, TTL-aware, tier-respecting materialization pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.substrate.materializer import MaterializationResultV1, SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore, SubstrateGraphStore

from .beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1
from .registry import SNAPSHOT_EPHEMERAL, ProducerRegistryV1

logger = logging.getLogger("orion.substrate.relational.layer")

_DEFAULT_ANCHORS: tuple[str, ...] = ("orion", "relationship", "juniper")


def _skip_unified_beliefs_ctx(ctx: dict[str, Any]) -> bool:
    """Heavy substrate / GraphDB work must not run for spark introspection or explicit skips."""
    if bool(ctx.get("skip_unified_beliefs")):
        return True
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    if bool(opts.get("skip_unified_beliefs")):
        return True
    verb = str(ctx.get("verb") or ctx.get("requested_verb") or "").strip().lower()
    if verb == "introspect_spark":
        return True
    lane = str(
        ctx.get("execution_lane")
        or ctx.get("llm_lane")
        or opts.get("execution_lane")
        or opts.get("llm_lane")
        or ""
    ).strip().lower()
    return lane == "spark"


def _lightweight_belief_set(anchors: Sequence[str]) -> UnifiedRelationalBeliefSetV1:
    slices = {a: AnchorBeliefSliceV1(anchor=a) for a in anchors}
    return UnifiedRelationalBeliefSetV1(
        anchors=slices,
        cold_anchors=[],
        degraded_producers=[],
        lineage=["skipped:introspect_spark_or_unified_beliefs_disabled"],
    )


def _aggregate_tier_outcomes(
    results: list[MaterializationResultV1],
    node_id_to_anchor: dict[str, str],
) -> dict[str, list[str]]:
    """Aggregate NodeMergeDecision.tier_outcome counts into per-anchor tier_outcomes lists.

    Returns ``{anchor: ["operator_static_protected:2", "concept_induced_accepted:5", ...]}``
    matching the spec format.
    """
    anchor_counters: dict[str, Counter[str]] = {}
    for result in results:
        for decision in result.node_decisions:
            if not decision.tier_outcome:
                continue
            anchor = node_id_to_anchor.get(decision.canonical_node_id)
            if anchor is None:
                continue
            if anchor not in anchor_counters:
                anchor_counters[anchor] = Counter()
            anchor_counters[anchor][decision.tier_outcome] += 1

    return {
        anchor: [f"{outcome}:{count}" for outcome, count in sorted(counter.items())]
        for anchor, counter in anchor_counters.items()
    }


class CognitiveUnificationLayer:
    """Warm/cold hybrid coordinator for all substrate producer lanes.

    Usage::

        layer = CognitiveUnificationLayer(registry, store)
        beliefs = layer.beliefs_for_stance(ctx=ctx)

    Thread-safety: ``beliefs_for_stance`` is safe to call from multiple threads;
    the ephemeral store is rebuilt on each call.
    """

    def __init__(
        self,
        registry: ProducerRegistryV1,
        store: SubstrateGraphStore,
        ephemeral_store: InMemorySubstrateGraphStore | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._base_ephemeral = ephemeral_store  # prototype; rebuilt per call
        self._durable_materializer = SubstrateGraphMaterializer(store=store)
        # Per-producer last-materialized timestamps for accurate per-producer TTL checks.
        # Updated after each successful cold fan-out materialization.  Falls back to
        # anchor-node age estimate for producers not yet tracked (e.g. first call after
        # restart, or when the store was pre-populated externally).
        self._last_materialized_at: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def beliefs_for_stance(
        self,
        *,
        anchors: Sequence[str] = _DEFAULT_ANCHORS,
        ctx: dict[str, Any] | None = None,
        timeout_sec: float = 5.0,
    ) -> UnifiedRelationalBeliefSetV1:
        """Return the unified belief set for the requested anchors.

        Always returns a valid result.  Individual producer failures are caught,
        logged, and recorded in ``degraded_producers``; the affected anchor slice
        is marked ``degraded=True``.
        """
        ctx = ctx if isinstance(ctx, dict) else {}

        if _skip_unified_beliefs_ctx(ctx):
            anchors_resolved = tuple(anchors) if anchors else _DEFAULT_ANCHORS
            logger.info(
                "cognitive_unification_layer_skip reason=spark_or_unified_beliefs_disabled verb=%s correlation_id=%s",
                ctx.get("verb"),
                ctx.get("correlation_id") or ctx.get("trace_id"),
            )
            return _lightweight_belief_set(anchors_resolved)

        # Fresh ephemeral store per call (snapshot_ephemeral nodes are never cached)
        ephemeral_store = InMemorySubstrateGraphStore()
        ephemeral_materializer = SubstrateGraphMaterializer(store=ephemeral_store)

        cold_anchors: list[str] = []
        degraded_producers: list[str] = []
        lineage: list[str] = []
        materialization_results: list[MaterializationResultV1] = []

        # ---- Warm path check (per-producer TTL) ----
        now = datetime.now(timezone.utc)
        durable_snapshot = self._store.snapshot()
        for anchor in anchors:
            cold_producers = self._registry.cold_producers_for_anchor(anchor)
            if not cold_producers:
                continue

            anchor_nodes = [n for n in durable_snapshot.nodes.values() if n.anchor_scope == anchor]
            if not anchor_nodes:
                cold_anchors.append(anchor)
                continue

            is_stale = False
            for p in cold_producers:
                last_mat = self._last_materialized_at.get(p.producer_id)
                if last_mat is None:
                    # Not yet tracked by this layer instance — fall back to the
                    # most-recent anchor-node timestamp as a conservative proxy.
                    latest_observed = max(n.temporal.observed_at for n in anchor_nodes)
                    age_sec = (now - latest_observed).total_seconds()
                    if age_sec > p.freshness_ttl_sec:
                        is_stale = True
                        break
                else:
                    age_sec = (now - last_mat).total_seconds()
                    if age_sec > p.freshness_ttl_sec:
                        is_stale = True
                        break

            if is_stale and anchor not in cold_anchors:
                cold_anchors.append(anchor)

        # ---- Cold/stale fan-out (pull_on_cold=True producers) ----
        if cold_anchors:
            cold_producer_map: dict[str, Any] = {}
            for anchor in cold_anchors:
                for p in self._registry.cold_producers_for_anchor(anchor):
                    cold_producer_map[p.producer_id] = p

            if cold_producer_map:
                max_workers = min(len(cold_producer_map), 6)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_producer = {
                        executor.submit(p.adapter_fn, ctx): p
                        for p in cold_producer_map.values()
                    }
                    try:
                        done_iter = as_completed(future_to_producer, timeout=timeout_sec)
                    except TypeError:
                        done_iter = as_completed(future_to_producer)

                    try:
                        for future in done_iter:
                            producer = future_to_producer[future]
                            try:
                                record = future.result(timeout=0)
                            except FuturesTimeoutError:
                                logger.warning("producer_fan_out_timeout producer_id=%s", producer.producer_id)
                                degraded_producers.append(producer.producer_id)
                                continue
                            except Exception as exc:
                                logger.warning("producer_fan_out_failed producer_id=%s error=%s", producer.producer_id, exc)
                                degraded_producers.append(producer.producer_id)
                                continue

                            if record is not None:
                                try:
                                    if producer.trust_tier.write_through:
                                        mat_result = self._durable_materializer.apply_record(record)
                                    else:
                                        mat_result = ephemeral_materializer.apply_record(record)
                                    materialization_results.append(mat_result)
                                    self._last_materialized_at[producer.producer_id] = datetime.now(timezone.utc)
                                except Exception as exc:
                                    logger.warning("producer_materialize_failed producer_id=%s error=%s", producer.producer_id, exc)
                                    degraded_producers.append(producer.producer_id)
                                    continue
                                lineage.append(f"{producer.producer_id}:{producer.trust_tier.name}")
                    except FuturesTimeoutError:
                        # One or more producers did not complete before the deadline.
                        # Mark all unfinished futures as degraded and proceed with
                        # whatever results were already collected.
                        for future, producer in future_to_producer.items():
                            if not future.done() and producer.producer_id not in degraded_producers:
                                logger.warning("producer_fan_out_deadline producer_id=%s", producer.producer_id)
                                degraded_producers.append(producer.producer_id)

        # ---- Always-refresh ctx-based ephemeral producers (pull_on_cold=False) ----
        seen_ephemeral: set[str] = set()
        for anchor in anchors:
            for producer in self._registry.ephemeral_ctx_producers_for_anchor(anchor):
                if producer.producer_id in seen_ephemeral:
                    continue
                seen_ephemeral.add(producer.producer_id)
                try:
                    record = producer.adapter_fn(ctx)
                    if record is not None:
                        mat_result = ephemeral_materializer.apply_record(record)
                        materialization_results.append(mat_result)
                        pid = producer.producer_id
                        if not any(e.startswith(pid + ":") for e in lineage):
                            lineage.append(f"{pid}:{producer.trust_tier.name}")
                except Exception as exc:
                    logger.warning("ephemeral_producer_failed producer_id=%s error=%s", producer.producer_id, exc)
                    degraded_producers.append(producer.producer_id)

        # ---- Assemble belief slices ----
        updated_durable_snapshot = self._store.snapshot()
        ephemeral_snapshot = ephemeral_store.snapshot()

        # Build node_id → anchor map for tier_outcomes aggregation
        node_id_to_anchor: dict[str, str] = {
            nid: n.anchor_scope
            for nid, n in {**updated_durable_snapshot.nodes, **ephemeral_snapshot.nodes}.items()
        }
        tier_outcomes_by_anchor = _aggregate_tier_outcomes(materialization_results, node_id_to_anchor)

        anchor_slices: dict[str, AnchorBeliefSliceV1] = {}
        for anchor in anchors:
            durable_nodes = [n for n in updated_durable_snapshot.nodes.values() if n.anchor_scope == anchor]
            ephemeral_nodes = [n for n in ephemeral_snapshot.nodes.values() if n.anchor_scope == anchor]
            all_nodes = durable_nodes + ephemeral_nodes

            anchor_producers = self._registry.producers_for_anchor(anchor)
            anchor_producer_ids = {p.producer_id for p in anchor_producers}
            is_degraded = bool(anchor_producer_ids & set(degraded_producers))

            anchor_slices[anchor] = AnchorBeliefSliceV1(
                anchor=anchor,
                concepts=[n for n in all_nodes if n.node_kind in ("concept", "hypothesis")],
                tensions=[n for n in all_nodes if n.node_kind == "tension"],
                goals=[n for n in all_nodes if n.node_kind == "goal"],
                drives=[n for n in all_nodes if n.node_kind == "drive"],
                snapshots=[n for n in all_nodes if n.node_kind == "state_snapshot"],
                events=[n for n in all_nodes if n.node_kind == "event"],
                degraded=is_degraded,
                tier_outcomes=tier_outcomes_by_anchor.get(anchor, []),
            )

        return UnifiedRelationalBeliefSetV1(
            anchors=anchor_slices,
            cold_anchors=list(cold_anchors),
            degraded_producers=sorted(set(degraded_producers)),
            lineage=lineage,
        )
