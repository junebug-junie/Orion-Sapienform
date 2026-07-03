from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger("orion.substrate.felt_state_reader")

_TRUTHY = {"1", "true", "yes", "on"}

_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
_DEFAULT_MAX_AGE_SEC = 120


@dataclass(frozen=True)
class LaneSpec:
    ctx_key: str
    table: str
    payload_col: str
    ts_col: str
    projection_id: str | None
    max_age_sec: int | None = None


_LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        ctx_key="self_state",
        table="substrate_self_state",
        payload_col="self_state_json",
        ts_col="generated_at",
        projection_id=None,
    ),
    LaneSpec(
        ctx_key="execution_trajectory_projection",
        table="substrate_execution_trajectory_projection",
        payload_col="projection_json",
        ts_col="generated_at",
        projection_id="active_execution_trajectory",
    ),
    LaneSpec(
        ctx_key="transport_bus_projection",
        table="substrate_transport_bus_projection",
        payload_col="projection_json",
        ts_col="updated_at",
        projection_id="active_transport_bus_projection",
    ),
    LaneSpec(
        ctx_key="active_node_pressure_projection",
        table="substrate_active_node_pressure_projection",
        payload_col="projection_json",
        ts_col="generated_at",
        projection_id="active_node_pressure_projection",
    ),
    LaneSpec(
        ctx_key="attention_broadcast",
        table="substrate_attention_broadcast_projection",
        payload_col="projection_json",
        ts_col="generated_at",
        projection_id="substrate.attention.broadcast.v1",
    ),
    LaneSpec(
        ctx_key="episode_summary",
        table="substrate_episode_summaries",
        payload_col="episode_json",
        ts_col="created_at",
        projection_id=None,
        max_age_sec=1800,
    ),
    LaneSpec(
        ctx_key="curiosity_signals",
        table="substrate_endogenous_curiosity_candidates",
        payload_col="candidates_json",
        ts_col="generated_at",
        projection_id=None,
        # 2× the substrate-runtime curiosity tick (60s) so the lane can
        # actually observe a fresh candidate row between writes.
        max_age_sec=120,
    ),
)


def _flag_enabled() -> bool:
    return os.getenv("ENABLE_SUBSTRATE_FELT_STATE_CTX", "false").strip().lower() in _TRUTHY


def _max_age_sec() -> int:
    raw = os.getenv("SUBSTRATE_FELT_STATE_MAX_AGE_SEC", str(_DEFAULT_MAX_AGE_SEC))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SEC


def _database_url() -> str:
    return (
        os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL")
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL")
        or _DEFAULT_DATABASE_URL
    )


class SubstrateFeltStateReader:
    def __init__(self, *, enabled: bool, database_url: str, max_age_sec: int) -> None:
        self._enabled = enabled
        self._database_url = database_url
        self._max_age_sec = max_age_sec
        self._engine = create_engine(database_url, pool_pre_ping=True) if enabled else None
        self._cache: dict[str, tuple[Any, float]] = {}

    def _fetch_lane(self, lane: LaneSpec) -> tuple[Any, datetime] | None:
        if self._engine is None:
            return None
        if lane.projection_id is None:
            query = text(
                f"SELECT {lane.payload_col}, {lane.ts_col} "
                f"FROM {lane.table} "
                f"ORDER BY {lane.ts_col} DESC LIMIT 1"
            )
            params: dict[str, Any] = {}
        else:
            query = text(
                f"SELECT {lane.payload_col}, {lane.ts_col} "
                f"FROM {lane.table} "
                f"WHERE projection_id = :pid"
            )
            params = {"pid": lane.projection_id}
        with self._engine.connect() as conn:
            row = conn.execute(query, params).mappings().first()
        if row is None:
            return None
        return (row.get(lane.payload_col), row.get(lane.ts_col))

    def hydrate(self, ctx: dict) -> None:
        if not self._enabled:
            return
        for lane in _LANES:
            try:
                max_age = lane.max_age_sec if lane.max_age_sec is not None else self._max_age_sec
                if ctx.get(lane.ctx_key) is not None:
                    continue
                cached = self._cache.get(lane.ctx_key)
                if cached is not None:
                    payload, fetched_at = cached
                    if (time.monotonic() - fetched_at) <= max_age:
                        ctx[lane.ctx_key] = payload
                        continue
                result = self._fetch_lane(lane)
                if result is None:
                    continue
                payload, ts = result
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > max_age:
                    continue
                ctx[lane.ctx_key] = payload
                self._cache[lane.ctx_key] = (payload, time.monotonic())
            except Exception:
                logger.debug("felt-state lane hydrate failed: %s", lane.ctx_key, exc_info=True)
                continue


_READER: SubstrateFeltStateReader | None = None


def _get_reader() -> SubstrateFeltStateReader:
    global _READER
    if _READER is None:
        _READER = SubstrateFeltStateReader(
            enabled=_flag_enabled(),
            database_url=_database_url(),
            max_age_sec=_max_age_sec(),
        )
    return _READER


def hydrate_felt_state_ctx(ctx: dict) -> None:
    """Public entrypoint. Fail-open: never raises."""
    try:
        if not isinstance(ctx, dict):
            return
        reader = _get_reader()
        reader.hydrate(ctx)
    except Exception:
        logger.debug("hydrate_felt_state_ctx failed", exc_info=True)
        return


def reset_reader_for_tests() -> None:
    global _READER
    _READER = None
