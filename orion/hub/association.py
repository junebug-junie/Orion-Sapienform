from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import HubAssociationBundleV1
from orion.substrate.felt_state_reader import (
    SubstrateFeltStateReader,
    _database_url,
    _max_age_sec,
)

logger = logging.getLogger("orion.hub.association")

ReadSource = Literal["felt_state_reader", "hub_sql_fallback"]
_FELT_STATE_READER: ReadSource = "felt_state_reader"

_TRUTHY = {"1", "true", "yes", "on"}


def _broadcast_enabled() -> bool:
    return os.getenv("ORION_ATTENTION_BROADCAST_ENABLED", "false").strip().lower() in _TRUTHY


def _default_reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url=_database_url(),
        max_age_sec=_max_age_sec(),
    )


def _parse_broadcast(raw: Any) -> AttentionBroadcastProjectionV1 | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return AttentionBroadcastProjectionV1.model_validate(raw)
    if isinstance(raw, AttentionBroadcastProjectionV1):
        return raw
    return None


def _broadcast_is_stale(
    broadcast: AttentionBroadcastProjectionV1 | None,
    *,
    max_age_sec: int,
) -> bool:
    if broadcast is None:
        return True
    ts = broadcast.generated_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() > max_age_sec


def _read_association_data(
    *,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> tuple[AttentionBroadcastProjectionV1 | None, dict[str, Any] | None, ReadSource]:
    if not _broadcast_enabled():
        return None, None, _FELT_STATE_READER
    reader = (reader_factory or _default_reader)()
    ctx: dict[str, Any] = {}
    reader.hydrate(ctx)
    broadcast = _parse_broadcast(ctx.get("attention_broadcast"))
    slice_raw = ctx.get("execution_trajectory_projection")
    trajectory = slice_raw if isinstance(slice_raw, dict) else None
    return broadcast, trajectory, _FELT_STATE_READER


def build_hub_association_bundle(
    *,
    correlation_id: str,
    repair_bundle: TurnAppraisalBundleV1 | None,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> HubAssociationBundleV1:
    max_age = _max_age_sec()

    if not _broadcast_enabled():
        return HubAssociationBundleV1(
            correlation_id=correlation_id,
            broadcast=None,
            broadcast_stale=True,
            execution_trajectory_slice=None,
            repair_bundle=repair_bundle,
            read_source=_FELT_STATE_READER,
        )

    broadcast: AttentionBroadcastProjectionV1 | None = None
    trajectory: dict[str, Any] | None = None
    read_source: ReadSource = _FELT_STATE_READER
    try:
        broadcast, trajectory, read_source = _read_association_data(reader_factory=reader_factory)
    except Exception:
        logger.debug("hub association lane read failed", exc_info=True)
        broadcast = None
        trajectory = None

    return HubAssociationBundleV1(
        correlation_id=correlation_id,
        broadcast=broadcast,
        broadcast_stale=_broadcast_is_stale(broadcast, max_age_sec=max_age),
        execution_trajectory_slice=trajectory,
        repair_bundle=repair_bundle,
        read_source=read_source,
    )
