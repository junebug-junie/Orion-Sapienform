"""Hub read API backing the Cognitive EKG card's Biometrics toggle + deep-inspection modal.

Athena and circe are the only two live host nodes; a third node (`atlas`) was
decommissioned 2026-08-20 and is rejected explicitly rather than silently
dropped or forwarded as a doomed cross-host call (see `biometrics_node_client.py`).

Postgres access follows `cabinet_sensors_routes.py`'s convention (async
`asyncpg` against `DATABASE_URL`, `timestamp::timestamptz` cast because that
column is `text`, not native timestamp) for `/history`, since it is reading
the same `orion_biometrics_summary` table that file already queries. `/induction`
is the one route that needs a sync SQLAlchemy `Engine` because it calls the
existing `orion.substrate.metacog_trend_signals.latest_biometrics_induction_by_node`
helper directly rather than re-deriving that query.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query

from orion.substrate.metacog_trend_signals import latest_biometrics_induction_by_node

from . import biometrics_node_client
from .biometrics_node_client import BiometricsNodeClientError, LIVE_NODES
from .cabinet_ambient_routes import downsample_points, parse_window
from .settings import settings

logger = logging.getLogger("orion-hub.biometrics-preview")

router = APIRouter(prefix="/api/biometrics/preview", tags=["biometrics-preview"])

DECOMMISSIONED_NODES: tuple[str, ...] = ("atlas",)

#: channel name -> which JSONB column on orion_biometrics_summary it lives in.
#: Pressure channel names per orion/telemetry/biometrics_pipeline.py's
#: PRESSURE channel set; composite names per the same module's `composites`
#: dict ("strain", "homeostasis", "stability").
_CHANNEL_COLUMN: dict[str, str] = {
    "strain": "composites",
    "homeostasis": "composites",
    "stability": "composites",
    "cpu": "pressures",
    "gpu_util": "pressures",
    "gpu_mem": "pressures",
    "mem": "pressures",
    "swap": "pressures",
    "disk": "pressures",
    "net": "pressures",
    "thermal": "pressures",
    "power": "pressures",
    "disk_capacity": "pressures",
    "fan": "pressures",
    # Raw watts, not a 0-1 pressure -- lives in the `measurements` JSONB column, not
    # `pressures`. Only athena writes this into its own summary row today (self-reported
    # via iLO); circe's current reading comes from the live cluster read in /snapshot
    # above (cluster_measurements_by_node), not from history, since circe's own PDU proxy
    # value is computed on athena and never persisted into circe's own summary row.
    "chassis_watts": "measurements",
}

# Injectable seam for tests, same pattern as cabinet_sensors_routes._history_query.
HistoryQuery = Callable[..., Awaitable[Sequence[Mapping[str, Any]]]]
_history_query: HistoryQuery | None = None

InductionEngineFactory = Callable[[], Any]
_induction_engine_factory: InductionEngineFactory | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: Any) -> str:
    from .cabinet_ambient_routes import _parse_db_timestamp

    return _parse_db_timestamp(value).isoformat().replace("+00:00", "Z")


def _validate_node(node: str) -> str:
    """Lowercase/validate a node name. Raises 404 for unknown/decommissioned nodes.

    A decommissioned node gets a distinguishable reason from a plain typo/unknown
    node -- "no longer exists" is a different fact than "never existed" -- same
    absent-is-not-a-guess discipline this codebase applies to readings.
    """
    normalized = (node or "").strip().lower()
    if normalized in LIVE_NODES:
        return normalized
    if normalized in DECOMMISSIONED_NODES:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "node_decommissioned",
                "node": normalized,
                "live_nodes": list(LIVE_NODES),
            },
        )
    raise HTTPException(
        status_code=404,
        detail={
            "error": "unknown_node",
            "node": normalized,
            "live_nodes": list(LIVE_NODES),
        },
    )


def _parse_lane_map(node: str) -> Dict[str, str]:
    raw = (
        settings.GPU_LANE_MAP_ATHENA_JSON
        if node == "athena"
        else settings.GPU_LANE_MAP_CIRCE_JSON
    )
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        logger.warning("Invalid GPU_LANE_MAP_%s_JSON; treating as empty.", node.upper())
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


@router.get("/snapshot")
async def api_biometrics_preview_snapshot(node: str = Query(...)) -> Dict[str, Any]:
    nid = _validate_node(node)
    try:
        payload = await biometrics_node_client.fetch_snapshot(nid)
    except BiometricsNodeClientError as exc:
        logger.warning("biometrics preview snapshot unavailable for %s: %s", nid, exc)
        return {"ok": False, "node": nid, "error": "node_unreachable"}
    node_payload = (payload.get("nodes") or {}).get(nid, {}) if isinstance(payload, dict) else {}
    cluster_payload = payload.get("cluster") if isinstance(payload, dict) else None
    return {
        "ok": True,
        "node": nid,
        "as_of": node_payload.get("as_of"),
        "freshness_s": node_payload.get("freshness_s"),
        "status": node_payload.get("status"),
        "reason": node_payload.get("reason"),
        "summary": node_payload.get("summary") or {},
        "induction": node_payload.get("induction") or {},
        # Per-node raw measurements (e.g. chassis_watts) this node's own biometrics hub
        # aggregator computed, keyed by node -- see BiometricsClusterV1.measurements_by_node.
        # Only the node whose PDU-proxy poller is configured (athena, for circe's wattage)
        # will actually have a proxied entry for another node; absent means unmeasured, not
        # zero, same as every other measurement in this module.
        "cluster_measurements_by_node": (cluster_payload or {}).get("measurements_by_node") if isinstance(cluster_payload, dict) else None,
    }


async def query_channel_history_rows(
    *, node: str, channel: str, column: str, hours: int
) -> Sequence[Mapping[str, Any]]:
    """No `::timestamptz` cast on the bound cutoff parameter (confirmed live
    2026-09-02): asyncpg infers a `$n::timestamptz` cast as "this parameter
    must already be a datetime.datetime", and errors on the plain ISO string
    `_iso_utc()` produces -- `invalid input for query argument $2: ...
    expected a datetime.date or datetime.datetime instance, got 'str'`. This
    table's `timestamp` column is TEXT (see BiometricsSummarySQL), and
    cabinet_sensors_routes.py's query_sensor_history_rows already compares it
    as plain text successfully in production against the same table -- match
    that proven pattern rather than re-adding a cast that only unit tests
    (which mock the DB layer) failed to catch.
    """
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    cutoff = _iso_utc(_now_utc() - timedelta(hours=hours))
    connection = await asyncpg.connect(dsn=database_url)
    try:
        return await connection.fetch(
            f"""
            SELECT
              timestamp AS t,
              ({column}->>$3)::double precision AS v
            FROM orion_biometrics_summary
            WHERE node = $1
              AND timestamp >= $2
              AND {column} ? $3
            ORDER BY timestamp ASC
            """,
            node,
            cutoff,
            channel,
        )
    finally:
        await connection.close()


@router.get("/history")
async def api_biometrics_preview_history(
    node: str = Query(...),
    channel: str = Query(...),
    window: str = Query("24h"),
) -> Dict[str, Any]:
    nid = _validate_node(node)
    column = _CHANNEL_COLUMN.get(channel)
    if column is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_channel", "channel": channel, "known_channels": sorted(_CHANNEL_COLUMN)},
        )
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_points = int(settings.CABINET_AMBIENT_HISTORY_MAX_POINTS)
    base = {"node": nid, "channel": channel, "window": window}
    query = _history_query or query_channel_history_rows
    try:
        rows = await query(node=nid, channel=channel, column=column, hours=hours)
        points = [
            {"t": _iso_utc(row["t"]), "v": float(row["v"])}
            for row in rows
            if row.get("t") is not None and row.get("v") is not None
        ]
        # downsample_points bucket-averages a fixed field set (rms/peak/activity);
        # wrap/unwrap through "rms" the same way cabinet_sensors_routes does for
        # its own single-value series, rather than duplicating bucket logic.
        wrapped = [{"t": p["t"], "rms": p["v"]} for p in points]
        sampled = downsample_points(wrapped, max_points)
        series = [{"t": p["t"], "v": p["rms"]} for p in sampled if p.get("rms") is not None]
    except Exception as exc:
        logger.warning("Biometrics preview history unavailable for %s/%s: %s", nid, channel, exc)
        return {"ok": False, **base, "series": [], "error": "history_unavailable"}

    return {"ok": True, **base, "series": series, "n_raw": len(points)}


async def query_multi_channel_history_rows(
    *, node: str, columns_by_channel: dict[str, str], hours: int
) -> Sequence[Mapping[str, Any]]:
    """One connection, one query, N channels -- the multi-channel sibling of
    query_channel_history_rows. Added because the modal's Trended section
    originally called /history once per channel (up to 14 concurrent
    asyncpg.connect() calls per node-detail open, no pooling); this repo has
    live incident history with Postgres connection exhaustion (PR #2010), so
    N short-lived connections opening at once -- exactly when an operator is
    trying to diagnose a problem -- is a real risk, not a theoretical one.

    Same no-`::timestamptz`-cast-on-the-bound-parameter fix as
    query_channel_history_rows above -- see that function's docstring.
    """
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    cutoff = _iso_utc(_now_utc() - timedelta(hours=hours))
    # Channel names are validated against the fixed _CHANNEL_COLUMN whitelist
    # before reaching here (never raw user input as a SQL identifier);
    # aliased positionally (c0, c1, ...) rather than by channel name so the
    # channel string itself never has to be interpolated as an identifier.
    aliases = list(columns_by_channel.items())
    select_cols = ",\n              ".join(
        f"({column}->>'{channel}')::double precision AS c{i}"
        for i, (channel, column) in enumerate(aliases)
    )
    connection = await asyncpg.connect(dsn=database_url)
    try:
        rows = await connection.fetch(
            f"""
            SELECT
              timestamp AS t,
              {select_cols}
            FROM orion_biometrics_summary
            WHERE node = $1
              AND timestamp >= $2
            ORDER BY timestamp ASC
            """,
            node,
            cutoff,
        )
    finally:
        await connection.close()
    # Translate positional c0..cN back to channel names for the caller.
    channel_by_alias = {f"c{i}": channel for i, (channel, _column) in enumerate(aliases)}
    return [
        {"t": row["t"], **{channel_by_alias[alias]: row[alias] for alias in channel_by_alias}}
        for row in rows
    ]


_history_multi_query: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] | None = None


@router.get("/history_multi")
async def api_biometrics_preview_history_multi(
    node: str = Query(...),
    channels: str = Query(...),
    window: str = Query("24h"),
) -> Dict[str, Any]:
    """Same data as N calls to /history, in one request/one Postgres
    connection -- see query_multi_channel_history_rows for why this exists.
    """
    nid = _validate_node(node)
    requested = [c.strip() for c in channels.split(",") if c.strip()]
    unknown = [c for c in requested if c not in _CHANNEL_COLUMN]
    if unknown or not requested:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_channel", "channels": unknown or requested, "known_channels": sorted(_CHANNEL_COLUMN)},
        )
    try:
        hours = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_points = int(settings.CABINET_AMBIENT_HISTORY_MAX_POINTS)
    base = {"node": nid, "channels": requested, "window": window}
    columns_by_channel = {c: _CHANNEL_COLUMN[c] for c in requested}
    query = _history_multi_query or query_multi_channel_history_rows
    try:
        rows = await query(node=nid, columns_by_channel=columns_by_channel, hours=hours)
        series: dict[str, list[dict[str, Any]]] = {c: [] for c in requested}
        raw_counts: dict[str, int] = {c: 0 for c in requested}
        for channel in requested:
            points = [
                {"t": _iso_utc(row["t"]), "v": float(row[channel])}
                for row in rows
                if row.get("t") is not None and row.get(channel) is not None
            ]
            raw_counts[channel] = len(points)
            wrapped = [{"t": p["t"], "rms": p["v"]} for p in points]
            sampled = downsample_points(wrapped, max_points)
            series[channel] = [{"t": p["t"], "v": p["rms"]} for p in sampled if p.get("rms") is not None]
    except Exception as exc:
        logger.warning("Biometrics preview multi-history unavailable for %s/%s: %s", nid, requested, exc)
        return {"ok": False, **base, "series": {c: [] for c in requested}, "error": "history_unavailable"}

    return {"ok": True, **base, "series": series, "n_raw": raw_counts}


def _induction_engine():
    if _induction_engine_factory is not None:
        return _induction_engine_factory()
    from sqlalchemy import create_engine

    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise RuntimeError("POSTGRES_URI is not configured")
    return create_engine(uri, pool_pre_ping=True)


@router.get("/induction")
async def api_biometrics_preview_induction(node: str = Query(...)) -> Dict[str, Any]:
    nid = _validate_node(node)
    try:
        engine = _induction_engine()
        by_node = latest_biometrics_induction_by_node(engine, [nid])
    except Exception as exc:
        logger.warning("Biometrics preview induction unavailable for %s: %s", nid, exc)
        return {"ok": False, "node": nid, "metrics": {}, "error": "induction_unavailable"}
    metrics = by_node.get(nid, {})
    return {"ok": bool(metrics), "node": nid, "metrics": metrics}


@router.get("/gpu")
async def api_biometrics_preview_gpu(node: str = Query(...), limit: int = Query(40, ge=1, le=60)) -> Dict[str, Any]:
    nid = _validate_node(node)
    lane_map = _parse_lane_map(nid)
    try:
        payload = await biometrics_node_client.fetch_raw_recent(nid, limit=limit)
    except BiometricsNodeClientError as exc:
        logger.warning("biometrics preview gpu unavailable for %s: %s", nid, exc)
        return {"ok": False, "node": nid, "gpus": [], "error": "node_unreachable"}

    items = payload.get("items") or [] if isinstance(payload, dict) else []
    # orion-biometrics' /raw/recent iterates reversed(_RAW_RECENT) -- items[0]
    # is the newest sample, items[-1] the oldest.
    latest_gpus: list[dict[str, Any]] = []
    trend_by_index: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        raw = item.get("raw") or {}
        gpu_block = raw.get("gpu") if isinstance(raw, dict) else None
        gpus = gpu_block.get("gpus") if isinstance(gpu_block, dict) else None
        if not gpus:
            continue
        ts = item.get("timestamp")
        for gpu in gpus:
            idx = str(gpu.get("index") if gpu.get("index") is not None else gpu.get("gpu_index", ""))
            trend_by_index.setdefault(idx, []).append(
                {"t": ts, "utilization_gpu": gpu.get("utilization_gpu")}
            )
        if not latest_gpus:
            latest_gpus = gpus  # first item encountered is the most recent

    cards = []
    for gpu in latest_gpus:
        idx = str(gpu.get("index") if gpu.get("index") is not None else gpu.get("gpu_index", ""))
        cards.append(
            {
                "index": idx,
                "name": gpu.get("name") or gpu.get("gpu_name"),
                "lane": lane_map.get(idx, "unassigned"),
                "utilization_gpu": gpu.get("utilization_gpu"),
                "memory_used_mb": gpu.get("memory_used_mb"),
                "memory_total_mb": gpu.get("memory_total_mb"),
                "power_draw_watts": gpu.get("power_draw_watts"),
                "processes": gpu.get("processes") or [],
                "trend": list(reversed(trend_by_index.get(idx, []))),
            }
        )

    return {"ok": bool(cards), "node": nid, "gpus": cards}
