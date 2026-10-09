"""The evidence bundle an urgent curiosity run starts with: the hardware right now.

Plan: docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-3-seeded-urgent-runs.md (Task 5).

Every section reuses a reader Hub already serves its panels from, so the
numbers Orion is shown are the numbers Juniper sees:

- ``cooling``: the cabinet AC plug's latest sample and freshness
  (``cabinet_cooling_routes``). Read-only; nothing here touches the cooler.
- ``cabinet_trend``: the last 60 minutes of ``cabinet_temp_c``
  (``cabinet_sensors_routes``).
- ``hosts``: each node's biometrics snapshot (``biometrics_node_client``).
- ``gpus``: each node's per-GPU util/power/memory cards (``biometrics_preview_routes``).
- ``pool``: active and queued GPU leases from Hub's live pool feed (``gpu_pool_routes``).

Hardware telemetry and pool state only -- never chat, memory, or journal
content. A failing or slow section becomes ``{"error": "<type>: <msg>"}`` and
the rest still arrive; ``collect_evidence`` never raises. The whole bundle is
trimmed to fit ``URGENT_EVIDENCE_MAX_BYTES`` (the seed rejects anything larger).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from orion.schemas.curiosity_urgent import URGENT_EVIDENCE_MAX_BYTES

from . import biometrics_node_client, cabinet_cooling_routes, cabinet_sensors_routes, gpu_pool_routes
from .biometrics_preview_routes import gpu_cards_from_raw_recent, pool_lane_map
from .cabinet_ambient_routes import _iso_utc, _parse_db_timestamp
from .settings import settings

logger = logging.getLogger("orion-hub.urgent_evidence")

TREND_WINDOW_MIN = 60
MAX_PROCESSES_PER_GPU = 8
ACTIVE_LEASE_STATUSES = frozenset({"queued", "backlogged", "granted", "recalling", "retry_wait"})
_LEASE_FIELDS = (
    "lease_id", "holder", "work_class", "priority", "kind", "status", "role",
    "attempt", "created_at", "granted_at", "recall_by",
)
# Replaced whole, largest first, only when trimming the trend was not enough.
_SECTIONS = ("cooling", "cabinet_trend", "hosts", "gpus", "pool")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def encoded_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8"))


def _error(exc: BaseException) -> dict[str, str]:
    return {"error": f"{type(exc).__name__}: {exc}"[:300]}


async def _bounded(make: Callable[[], Awaitable[Any]], timeout: float) -> Any:
    try:
        return await asyncio.wait_for(make(), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except (TimeoutError, asyncio.TimeoutError):
        return {"error": f"TimeoutError: no answer within {timeout:g}s"}
    except Exception as exc:  # noqa: BLE001 -- a broken reader is reported, never raised
        return _error(exc)


async def _cooling(now: datetime) -> dict[str, Any]:
    node = str(settings.CABINET_AMBIENT_HISTORY_NODE)
    row = await cabinet_cooling_routes._latest_query(node=node)
    latest = cabinet_cooling_routes._load_latest(
        row, stale_after_sec=float(settings.CABINET_SENSORS_STALE_AFTER_SEC), now=now
    )
    return {"node": node, **latest}


async def _cabinet_trend(now: datetime) -> dict[str, Any]:
    node = str(settings.CABINET_AMBIENT_HISTORY_NODE)
    query = cabinet_sensors_routes._history_query or cabinet_sensors_routes.query_sensor_history_rows
    rows = await query(node=node, hours=1)
    cutoff = now - timedelta(minutes=TREND_WINDOW_MIN)
    points = []
    for row in rows:
        if row.get("t") is None or row.get("temp_c") is None:
            continue
        ts = _parse_db_timestamp(row["t"])
        if ts < cutoff:
            continue
        points.append({"t": _iso_utc(ts), "temp_c": round(float(row["temp_c"]), 3)})
    return {"node": node, "window_min": TREND_WINDOW_MIN, "points": points}


async def _host(node: str) -> dict[str, Any]:
    payload = await biometrics_node_client.fetch_snapshot(node)
    node_payload = (payload.get("nodes") or {}).get(node) or {}
    summary = node_payload.get("summary") or {}
    return {
        "as_of": node_payload.get("as_of"),
        "freshness_s": node_payload.get("freshness_s"),
        "status": node_payload.get("status"),
        "reason": node_payload.get("reason"),
        # Raw units: temp_c_max, fan_pct_max, chassis_watts, gpu_watts_total, ...
        # A key is absent when the node does not measure it -- never zero.
        "measurements": summary.get("measurements"),
        "peak_pressure": summary.get("peak_pressure"),
        "peak_pressure_channel": summary.get("peak_pressure_channel"),
        "constraint": summary.get("constraint"),
    }


async def _gpu_cards(node: str) -> list[dict[str, Any]]:
    payload = await biometrics_node_client.fetch_raw_recent(node, limit=1)
    cards = gpu_cards_from_raw_recent(payload, *pool_lane_map(node, now=_now_utc()))
    for card in cards:
        card.pop("trend", None)
        card["processes"] = list(card.get("processes") or [])[:MAX_PROCESSES_PER_GPU]
    return cards


async def _pool() -> dict[str, Any]:
    snap = gpu_pool_routes.feed.snapshot()
    state = snap.get("state")
    if not isinstance(state, dict):
        return {"error": "no pool state received on the bus yet"}
    leases = [
        {key: lease.get(key) for key in _LEASE_FIELDS if lease.get(key) is not None}
        for lease in state.get("leases") or []
        if isinstance(lease, dict) and lease.get("status") in ACTIVE_LEASE_STATUSES
    ]
    return {
        "feed_version": snap.get("version"),
        "generated_at": state.get("generated_at"),
        "mode": state.get("mode"),
        "leases": leases,
        "queue_depth": state.get("queue_depth") or {},
        "backlog_depth": state.get("backlog_depth") or {},
        "cards": [
            {k: card.get(k) for k in ("card", "swap_state", "swapped_in", "lent") if k in card}
            for card in state.get("cards") or []
            if isinstance(card, dict)
        ],
        "roles": [
            {k: role.get(k) for k in ("role", "status", "cards") if k in role}
            for role in state.get("roles") or []
            if isinstance(role, dict)
        ],
    }


async def _per_node(nodes: tuple[str, ...], read: Callable[[str], Awaitable[Any]], timeout: float) -> dict[str, Any]:
    results = await asyncio.gather(*(_bounded(lambda n=n: read(n), timeout) for n in nodes))
    return dict(zip(nodes, results))


def trim_to_cap(bundle: dict[str, Any], cap: int = URGENT_EVIDENCE_MAX_BYTES) -> dict[str, Any]:
    """Fit the bundle under ``cap`` bytes: oldest trend points first, then whole sections."""
    out = json.loads(json.dumps(bundle, default=str))
    size = encoded_size(out)
    if size <= cap:
        return out
    trend = out.get("cabinet_trend")
    points = trend.get("points") if isinstance(trend, dict) else None
    if isinstance(points, list) and points:
        dropped = 0
        while size > cap and points:
            per_point = max(1, encoded_size(points) // len(points))
            drop = min(len(points), max(1, math.ceil((size - cap) / per_point)))
            del points[:drop]
            dropped += drop
            trend["trimmed_points"] = dropped
            size = encoded_size(out)
    trimmed = {"error": f"trimmed: bundle exceeded {cap} bytes"}
    while size > cap:
        candidates = [s for s in _SECTIONS if s in out and out[s] != trimmed]
        if not candidates:
            break
        largest = max(candidates, key=lambda s: encoded_size(out[s]))
        if encoded_size(out[largest]) <= encoded_size(trimmed):
            break
        out[largest] = dict(trimmed)
        size = encoded_size(out)
    if size > cap:
        logger.warning("urgent_evidence_over_cap size=%s cap=%s", size, cap)
    return out


async def collect_evidence(
    *, nodes: tuple[str, ...] = ("athena", "circe"), per_section_timeout: float = 5.0
) -> dict[str, Any]:
    """The hardware and GPU-pool readings right now. Never raises."""
    now = _now_utc()
    nodes = tuple(nodes)
    cooling, trend, hosts, gpus, pool = await asyncio.gather(
        _bounded(lambda: _cooling(now), per_section_timeout),
        _bounded(lambda: _cabinet_trend(now), per_section_timeout),
        _per_node(nodes, _host, per_section_timeout),
        _per_node(nodes, _gpu_cards, per_section_timeout),
        _bounded(_pool, per_section_timeout),
    )
    bundle = {
        "cooling": cooling,
        "cabinet_trend": trend,
        "hosts": hosts,
        "gpus": gpus,
        "pool": pool,
        "collected_at": now.isoformat(),
    }
    return trim_to_cap(bundle)
