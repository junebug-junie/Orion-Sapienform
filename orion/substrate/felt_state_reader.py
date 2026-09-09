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
    # Optional extra WHERE (no leading `WHERE`/`AND`), for lanes that pick a
    # row by something other than projection_id. Constant SQL from this
    # module only -- never interpolated from ctx.
    where_sql: str | None = None
    # How long a fetched payload is reused before re-querying. Defaults to
    # `max_age_sec`; separate for lanes whose ROW may be days old and still
    # valid but whose cache should still refresh within minutes.
    cache_ttl_sec: int | None = None


_LANES: tuple[LaneSpec, ...] = (
    # self_state lane removed 2026-07-22, SelfStateV1 burn
    # (docs/superpowers/specs/2026-07-22-self-state-phi-endo-origination-burn-
    # spec.md): substrate_self_state no longer has a producer.
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
    # Orion's own current self-definition (orion/curiosity/self_inquiry.py):
    # the latest `self:definition` row Hub mirrored from a self-inquiry run's
    # `:SelfDefinition` node. Read by the `self_definition` stance producer
    # (orion/substrate/relational/adapters/self_definition_ctx.py). A row is
    # valid for 30 days -- a definition Orion wrote last week is still what
    # Orion last said about themself -- but the cache refreshes every 5 min so
    # a new run's definition reaches chat within a few turns.
    LaneSpec(
        ctx_key="orion_self_definition",
        table="self_concept_history",
        payload_col=(
            "json_build_object('entry_id', entry_id, 'content', content, "
            "'version', version, 'evidence_refs', evidence_refs, "
            "'created_at', created_at)"
        ),
        ts_col="created_at",
        projection_id=None,
        max_age_sec=30 * 86400,
        where_sql="concept_id = 'self:definition' AND produced_by = 'curiosity_self_inquiry'",
        cache_ttl_sec=300,
    ),
    LaneSpec(
        ctx_key="latest_reverie_thought",
        table="substrate_reverie_thought",
        payload_col="thought_json",
        ts_col="created_at",
        projection_id=None,
        # 2× the reverie tick interval (90s), same 2x convention as
        # curiosity_signals above.
        max_age_sec=180,
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
            where = f"WHERE {lane.where_sql} " if lane.where_sql else ""
            query = text(
                f"SELECT {lane.payload_col} AS payload, {lane.ts_col} AS ts "
                f"FROM {lane.table} "
                f"{where}"
                f"ORDER BY {lane.ts_col} DESC LIMIT 1"
            )
            params: dict[str, Any] = {}
        else:
            query = text(
                f"SELECT {lane.payload_col} AS payload, {lane.ts_col} AS ts "
                f"FROM {lane.table} "
                f"WHERE projection_id = :pid"
            )
            params = {"pid": lane.projection_id}
        with self._engine.connect() as conn:
            row = conn.execute(query, params).mappings().first()
        if row is None:
            return None
        return (row.get("payload"), row.get("ts"))

    def _remember_miss(self, lane: LaneSpec) -> None:
        """Negative cache, ONLY for lanes that declare their own `cache_ttl_sec`.

        A lane whose expected steady state is "no row yet" (the self-definition
        lane until the first self-inquiry run lands) would otherwise re-query
        on every chat turn and gate tick, because a miss never reached the
        cache. Lanes without an explicit TTL keep the old behaviour on purpose:
        their max_age doubles as their cache TTL, and remembering a miss for
        `curiosity_signals`' 120s would delay a fresh candidate by that long.
        """
        if lane.cache_ttl_sec is None:
            return
        self._cache[lane.ctx_key] = (None, time.monotonic())

    def hydrate(self, ctx: dict, *, lanes: tuple[str, ...] | None = None) -> None:
        """Hydrate every lane, or only the ctx_keys named in `lanes`.

        The subset form exists for callers on the quick lane, which must not
        pay eight blocking SELECTs for one key (review finding on PR #2169)."""
        if not self._enabled:
            return
        for lane in _LANES:
            if lanes is not None and lane.ctx_key not in lanes:
                continue
            try:
                max_age = lane.max_age_sec if lane.max_age_sec is not None else self._max_age_sec
                if ctx.get(lane.ctx_key) is not None:
                    continue
                cached = self._cache.get(lane.ctx_key)
                if cached is not None:
                    payload, fetched_at = cached
                    cache_ttl = lane.cache_ttl_sec if lane.cache_ttl_sec is not None else max_age
                    if (time.monotonic() - fetched_at) <= cache_ttl:
                        # A cached None is a remembered MISS (see below): skip
                        # the query, leave ctx untouched, until the TTL lapses.
                        if payload is not None:
                            ctx[lane.ctx_key] = payload
                        continue
                result = self._fetch_lane(lane)
                if result is None:
                    self._remember_miss(lane)
                    continue
                payload, ts = result
                if ts is None:
                    self._remember_miss(lane)
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > max_age:
                    self._remember_miss(lane)
                    continue
                ctx[lane.ctx_key] = payload
                self._cache[lane.ctx_key] = (payload, time.monotonic())
            except Exception:
                logger.debug("felt-state lane hydrate failed: %s", lane.ctx_key, exc_info=True)
                # A failing query is a miss too: without this an unreachable DB
                # is re-tried on every turn for lanes that opted into the TTL.
                self._remember_miss(lane)
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


def hydrate_felt_state_ctx(ctx: dict, *, lanes: tuple[str, ...] | None = None) -> None:
    """Public entrypoint. Fail-open: never raises. `lanes` limits the pull to
    those ctx_keys (see `SubstrateFeltStateReader.hydrate`)."""
    try:
        if not isinstance(ctx, dict):
            return
        reader = _get_reader()
        reader.hydrate(ctx, lanes=lanes)
    except Exception:
        logger.debug("hydrate_felt_state_ctx failed", exc_info=True)
        return


def substrate_felt_state_database_url() -> str:
    """Public URL resolver for Hub association and other felt-state readers."""
    return _database_url()


def substrate_felt_state_max_age_sec() -> int:
    """Public max-age gate (seconds) for projection freshness checks."""
    return _max_age_sec()


def reset_reader_for_tests() -> None:
    global _READER
    _READER = None
