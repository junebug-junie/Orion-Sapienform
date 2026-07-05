from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import HubAssociationBundleV1
from orion.substrate.felt_state_reader import SubstrateFeltStateReader

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_MAX_AGE_SEC = 120


def _broadcast_enabled() -> bool:
    return os.getenv("ORION_ATTENTION_BROADCAST_ENABLED", "false").strip().lower() in _TRUTHY


def _max_age_sec() -> int:
    raw = os.getenv("SUBSTRATE_FELT_STATE_MAX_AGE_SEC", str(_DEFAULT_MAX_AGE_SEC))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SEC


def _read_broadcast(
    *,
    max_age_sec: int = _DEFAULT_MAX_AGE_SEC,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> tuple[AttentionBroadcastProjectionV1 | None, str]:
    if not _broadcast_enabled():
        return None, "felt_state_reader"
    reader = (reader_factory or _default_reader)()
    ctx: dict[str, Any] = {}
    reader.hydrate(ctx)
    raw = ctx.get("attention_broadcast")
    if raw is None:
        return None, "felt_state_reader"
    if isinstance(raw, dict):
        broadcast = AttentionBroadcastProjectionV1.model_validate(raw)
    else:
        broadcast = raw
    ts = broadcast.generated_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age > max_age_sec:
        return broadcast, "felt_state_reader"
    return broadcast, "felt_state_reader"


def _default_reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url=os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL")
        or os.getenv("POSTGRES_URI")
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        max_age_sec=_max_age_sec(),
    )


def _read_execution_trajectory(reader: SubstrateFeltStateReader) -> dict[str, Any] | None:
    ctx: dict[str, Any] = {}
    reader.hydrate(ctx)
    slice_ = ctx.get("execution_trajectory_projection")
    return slice_ if isinstance(slice_, dict) else None


def build_hub_association_bundle(
    *,
    correlation_id: str,
    repair_bundle: TurnAppraisalBundleV1 | None,
    reader_factory: Callable[[], SubstrateFeltStateReader] | None = None,
) -> HubAssociationBundleV1:
    max_age = _max_age_sec()
    broadcast, read_source = _read_broadcast(max_age_sec=max_age, reader_factory=reader_factory)
    broadcast_stale = broadcast is None
    if broadcast is not None:
        ts = broadcast.generated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        broadcast_stale = (datetime.now(timezone.utc) - ts).total_seconds() > max_age

    trajectory: dict[str, Any] | None = None
    if reader_factory or _broadcast_enabled():
        try:
            reader = (reader_factory or _default_reader)()
            trajectory = _read_execution_trajectory(reader)
        except Exception:
            trajectory = None

    return HubAssociationBundleV1(
        correlation_id=correlation_id,
        broadcast=broadcast,
        broadcast_stale=broadcast_stale,
        execution_trajectory_slice=trajectory,
        repair_bundle=repair_bundle,
        read_source=read_source,  # type: ignore[arg-type]
    )
