"""Operator gate for lent lanes: is `chat-burst` open for borrowing right now?

WHAT THIS IS
------------
`chat-burst` (orion/llm/routes.py, OPERATOR_GATED_LLM_ROUTES) is Juniper's own chat worker
(circe-worker-1, the same upstream `chat` and `harness` dispatch to) lent to the durable-runs
burst queue. Unlike `agent-burst`, whose availability is a physical fact (is the worker
resident on GPU2?), this lane's availability is a *decision*: Juniper presses "Lend chat lane"
in the Hub when she is not using chat, and presses it again when she wants the lane back.

This module is the one place that decision lives. One Redis key per gated route on the bus
Redis (ORION_BUS_URL), so the state survives a gateway restart and is the same for every
gateway replica:

    orion:llm_gateway:lane_gate:<route_id> -> {"open": bool, "changed_at": iso, "changed_by": str}

Consumers, all in this service:

- route_catalog.py reports a gated route `operator_closed` while the gate is closed, which is
  what keeps durable admission from widening onto it (admission_runtime.refresh_lanes treats
  anything but `up` as unhealthy).
- main.py / anthropic_passthrough.py / openai_passthrough.py refuse a dispatch on a closed gated
  route with `route_operator_closed`, so nothing reaches the worker by naming the route.
- main.py exposes GET/PUT /routes/{route_id}/gate; the Hub's button is the only producer.

FAIL CLOSED
-----------
A missing key is closed. A Redis error is closed (`is_open` returns False and logs). The lane
is Juniper's reserved capacity: the failure mode where borrowing silently continues because a
cache went stale is the one this gate exists to prevent, so nothing here caches an "open".
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from orion.llm.routes import CHAT_BURST_LENDS_ROUTE, OPERATOR_GATED_LLM_ROUTES

from .settings import settings

logger = logging.getLogger("orion-llm-gateway.lane_gate")

GATE_KEY_PREFIX = "orion:llm_gateway:lane_gate:"

_client: Optional[aioredis.Redis] = None


class RouteNotOperatorGated(KeyError):
    """The route id is real but has no operator gate (or is not a route at all)."""


class LaneGateUnavailable(RuntimeError):
    """Redis could not answer; callers must treat the gate as closed."""


def gate_key(route_id: str) -> str:
    return f"{GATE_KEY_PREFIX}{route_id}"


def _require_gated(route_id: str) -> str:
    rid = str(route_id or "").strip().lower()
    if rid not in OPERATOR_GATED_LLM_ROUTES:
        raise RouteNotOperatorGated(rid)
    return rid


def _lends_route(route_id: str) -> Optional[str]:
    return CHAT_BURST_LENDS_ROUTE if route_id == "chat-burst" else None


async def _redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.orion_bus_url,
            decode_responses=True,
            # Tight on purpose: this read runs inside the catalog refresh that durable-runs
            # fetches with a 5 s timeout; a stalled bus Redis must read as closed quickly,
            # not turn every lane's admission off by timing the whole catalog out.
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
    return _client


def reset_for_tests(client: Any = None) -> None:
    """Swap the Redis client (tests pass a fake); None forces a fresh connection next call."""
    global _client
    _client = client


def _shape(route_id: str, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "route_id": route_id,
        "open": bool(raw.get("open")),
        "changed_at": raw.get("changed_at"),
        "changed_by": raw.get("changed_by"),
        "lends_route": _lends_route(route_id),
    }


async def read_gate(route_id: str) -> Dict[str, Any]:
    """Current gate state. Missing key -> closed. Redis failure -> LaneGateUnavailable."""
    rid = _require_gated(route_id)
    try:
        client = await _redis()
        value = await client.get(gate_key(rid))
    except Exception as exc:  # any redis/connection error
        raise LaneGateUnavailable(f"lane gate redis unavailable: {exc}") from exc
    if not value:
        return _shape(rid, None)
    try:
        parsed = json.loads(value)
    except ValueError:
        logger.warning("lane_gate_corrupt route=%s value=%r -- treating as closed", rid, value[:80])
        return _shape(rid, None)
    return _shape(rid, parsed)


async def set_gate(route_id: str, *, open: bool, changed_by: str) -> Dict[str, Any]:
    rid = _require_gated(route_id)
    record = {
        "open": bool(open),
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "changed_by": str(changed_by or "").strip() or "unknown",
    }
    try:
        client = await _redis()
        await client.set(gate_key(rid), json.dumps(record))
    except Exception as exc:
        raise LaneGateUnavailable(f"lane gate redis unavailable: {exc}") from exc
    logger.info(
        "lane_gate_changed route=%s open=%s changed_by=%s lends_route=%s",
        rid, record["open"], record["changed_by"], _lends_route(rid),
    )
    return _shape(rid, record)


async def is_open(route_id: str) -> bool:
    """True only when the gate is confirmed open. Unavailable Redis reads as closed."""
    try:
        return bool((await read_gate(route_id))["open"])
    except LaneGateUnavailable as exc:
        logger.warning("lane_gate_unavailable route=%s -- treating as closed: %s", route_id, exc)
        return False


def route_operator_closed_error(route_id: str) -> Dict[str, Any]:
    """The one error shape every dispatch surface returns for a closed gate."""
    return {
        "error": {
            "type": "route_operator_closed",
            "message": (
                f"Route '{route_id}' is operator-gated and currently closed; open it from the "
                "Hub (Lend chat lane) before dispatching on it"
            ),
            "route": route_id,
        }
    }
