"""Spark state snapshot adapter — concept_induced tier (ctx-sourced).

Reads ``ctx["spark_state_json"]`` (string) or ``ctx["spark_state_snapshot"]`` (dict / model)
and maps via ``map_spark_state_snapshot_to_substrate``.  No network.
"""

from __future__ import annotations

import logging
from typing import Any

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.substrate.adapters.spark import map_spark_state_snapshot_to_substrate

logger = logging.getLogger("orion.substrate.relational.adapters.spark_ctx")

_TIER_RANK = 3  # concept_induced (Spark profile lane)


def map_spark_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map Spark state snapshot from ctx → substrate record (concept_induced tier)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    raw = ctx.get("spark_state_json")
    if raw is None:
        raw = ctx.get("spark_state_snapshot")

    snap: SparkStateSnapshotV1 | None = None
    try:
        if isinstance(raw, SparkStateSnapshotV1):
            snap = raw
        elif isinstance(raw, str) and raw.strip():
            snap = SparkStateSnapshotV1.model_validate_json(raw)
        elif isinstance(raw, dict):
            snap = SparkStateSnapshotV1.model_validate(raw)
    except Exception as exc:
        logger.debug("spark_ctx_adapter_parse_failed error=%s", exc)
        return None

    if snap is None:
        return None

    try:
        record = map_spark_state_snapshot_to_substrate(snapshot=snap)
    except Exception as exc:
        logger.debug("spark_ctx_adapter_map_failed error=%s", exc)
        return None

    patched: list[Any] = []
    for node in record.nodes:
        prov = node.provenance.model_copy(update={"tier_rank": _TIER_RANK})
        patched.append(node.model_copy(update={"provenance": prov}))

    return SubstrateGraphRecordV1(
        graph_id=record.graph_id,
        anchor_scope=record.anchor_scope,
        subject_ref=record.subject_ref,
        nodes=patched,
        edges=list(record.edges),
        created_at=record.created_at,
    )
