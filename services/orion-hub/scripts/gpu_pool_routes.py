"""GPU pool operator panel (stage 2 of docs/superpowers/specs/2026-09-24-gpu-pool-design.md).

- ``GET /gpu-pool``: the standalone page (embedded as the Hub "GPU pool" tab).
- ``GET /api/gpu-pool/state``: on-demand pool state over bus RPC (optionally with the parsed YAML
  and one lease's path through the lease graph).
- ``GET /api/gpu-pool/stream``: SSE of live ``orion:gpu_pool:state`` + ``orion:gpu_pool:event``.
- ``GET /api/gpu-pool/history``: historical traffic from ``gpu_pool_events`` (sql-writer), at three
  zoom levels: per role, per class/holder/priority, and individual events, plus a time series.
- ``POST /api/gpu-pool/control``: operator verbs (lend, unlend, hold, release, replay, cancel,
  backfill). No token (Juniper's call); only Hub's own page can send it (see CSRF note below).

Everything to and from the pool rides the bus; only Hub's own browser API is HTTP.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.gpu_pool import (
    GPU_POOL_CONTROL_KIND, GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_REQUEST_CHANNEL,
    GPU_POOL_EVENT_CHANNEL, GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_REPLY_PREFIX,
    GPU_POOL_STATE_REQUEST_CHANNEL, GPU_POOL_STATE_REQUEST_KIND, GpuPoolControlReplyV1, GpuPoolControlV1,
    GpuPoolStateRequestV1,
)

from .settings import settings

logger = logging.getLogger("orion-hub.gpu_pool")

router = APIRouter(prefix="/api/gpu-pool", tags=["gpu-pool"])
page_router = APIRouter(tags=["gpu-pool"])

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
MAX_EVENTS = 500
FAILURE_EVENTS = ("aborted", "expired", "dead_lettered", "unavailable")


# --- live feed ------------------------------------------------------------------------------
class GpuPoolFeed:
    """Keeps the latest pool state and recent lease events from the bus and fans them out to SSE
    listeners. Fails open: a bus hiccup leaves the panel on its last snapshot, marked stale."""

    def __init__(self, max_events: int = MAX_EVENTS):
        self.state: dict[str, Any] | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.version = 0
        self._listeners: set[asyncio.Queue] = set()
        self._tasks: list[asyncio.Task] = []
        self._bus: Any = None

    async def start(self, bus: Any) -> None:
        self._bus = bus
        self._tasks = [asyncio.create_task(self._listen(GPU_POOL_STATE_CHANNEL)),
                       asyncio.create_task(self._listen(GPU_POOL_EVENT_CHANNEL))]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _listen(self, channel: str) -> None:
        while True:
            try:
                async with self._bus.subscribe(channel) as pubsub:
                    async for msg in self._bus.iter_messages(pubsub):
                        decoded = self._bus.codec.decode(msg.get("data"))
                        if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
                            continue
                        self.absorb(channel, decoded.envelope.payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("gpu_pool_feed_listen_failed channel=%s err=%s", channel, exc)
                await asyncio.sleep(5)

    def absorb(self, channel: str, payload: dict[str, Any]) -> None:
        self.version += 1
        if channel == GPU_POOL_STATE_CHANNEL:
            self.state = payload
            update = {"kind": "state", "state": payload}
        else:
            self.events.append(payload)
            update = {"kind": "event", "event": payload}
        update["version"] = self.version
        for queue in list(self._listeners):
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                pass  # a slow browser misses a frame; the next state frame resyncs it

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "state": self.state, "events": list(self.events)}

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._listeners.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._listeners.discard(queue)


feed = GpuPoolFeed()


# --- page -----------------------------------------------------------------------------------
@page_router.get("/gpu-pool")
async def gpu_pool_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "gpu_pool.html").read_text(encoding="utf-8")
    return HTMLResponse(template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()),
                        headers=_NO_CACHE)


# --- bus RPC --------------------------------------------------------------------------------
def _rpc_bus() -> Any:
    from . import main

    bus = getattr(main, "rpc_bus", None) or getattr(main, "bus", None)
    if bus is None or not getattr(bus, "enabled", False):
        raise HTTPException(503, "bus_unavailable")
    return bus


async def _rpc(channel: str, prefix: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    bus = _rpc_bus()
    reply_channel = f"{prefix}{uuid.uuid4().hex}"
    env = BaseEnvelope(kind=kind, source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
                       correlation_id=uuid.uuid4(), reply_to=reply_channel, payload=payload)
    try:
        raw = await bus.rpc_request(channel, env, reply_channel=reply_channel,
                                    timeout_sec=float(settings.HUB_GPU_POOL_RPC_TIMEOUT_SEC), health_label="gpu_pool_panel")
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "gpu_pool_rpc_timeout") from exc
    decoded = bus.codec.decode(raw.get("data"))
    if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
        raise HTTPException(502, "gpu_pool_reply_undecodable")
    return decoded.envelope.payload


def _enabled() -> None:
    if not settings.HUB_GPU_POOL_ENABLED:
        raise HTTPException(404, "gpu_pool_panel_disabled")


@router.get("/state")
async def gpu_pool_state(config: bool = False,
                         history_for: Optional[str] = Query(None, max_length=128)) -> dict[str, Any]:
    _enabled()
    req = GpuPoolStateRequestV1(include_leases=True, include_config=config, history_for=history_for or None)
    # exclude_defaults: a plain state read stays readable by a pool that predates the optional fields
    # (GpuPoolStateRequestV1 is extra="forbid"), so Hub and pool can deploy in either order.
    return await _rpc(GPU_POOL_STATE_REQUEST_CHANNEL, GPU_POOL_STATE_REPLY_PREFIX, GPU_POOL_STATE_REQUEST_KIND,
                      req.model_dump(mode="json", exclude_defaults=True))


# --- live stream ----------------------------------------------------------------------------
def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\nid: {data.get('version', 0)}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream(request: Request, source: GpuPoolFeed):
    queue = source.subscribe()
    try:
        yield _sse("snapshot", source.snapshot())
        while True:
            if await request.is_disconnected():
                return
            try:
                update = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _sse(update["kind"], update)
    finally:
        source.unsubscribe(queue)


@router.get("/stream")
async def gpu_pool_stream(request: Request) -> StreamingResponse:
    _enabled()
    return StreamingResponse(_stream(request, feed), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- history (gpu_pool_events) ---------------------------------------------------------------
def _engine():
    from .pg_engine import get_engine

    engine = get_engine()
    if engine is None:
        raise HTTPException(503, "postgres_not_configured")
    return engine


def _filters(work_class: Optional[str], holder: Optional[str], lease_id: Optional[str]) -> tuple[str, dict[str, Any]]:
    clauses, params = ["generated_at >= NOW() - (:minutes * INTERVAL '1 minute')"], {}
    for col, value in (("work_class", work_class), ("holder", holder), ("lease_id", lease_id)):
        if value:
            clauses.append(f"{col} = :{col}")
            params[col] = value
    return " AND ".join(clauses), params


_AGG = """
    count(*) FILTER (WHERE event = 'granted') AS grants,
    count(*) FILTER (WHERE event = 'released' AND reason = 'ok') AS released_ok,
    count(*) FILTER (WHERE event IN ('aborted','expired','dead_lettered','unavailable')) AS failures,
    count(*) FILTER (WHERE event = 'recalled') AS recalls,
    count(*) FILTER (WHERE event = 'backlogged') AS backlogged,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY waited_ms) FILTER (WHERE event = 'granted') AS wait_p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY waited_ms) FILTER (WHERE event = 'granted') AS wait_p95_ms,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY held_ms) FILTER (WHERE event IN ('released','aborted','expired')) AS held_p50_ms
"""


def history_payload(conn, *, minutes: int, work_class: Optional[str], holder: Optional[str],
                    lease_id: Optional[str], limit: int) -> dict[str, Any]:
    from sqlalchemy import text

    where, params = _filters(work_class, holder, lease_id)
    params.update(minutes=minutes, limit=limit, bucket=max(60, (minutes * 60) // 60))
    # A week of stage-3 traffic is millions of rows; never let one panel click hold a connection.
    conn.execute(text("SET LOCAL statement_timeout = '5s'"))

    def rows(sql: str) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]

    return {
        "minutes": minutes,
        "by_role": rows(f"SELECT role, {_AGG} FROM gpu_pool_events WHERE {where} AND role IS NOT NULL "
                        "GROUP BY role ORDER BY role"),
        "by_class": rows(f"SELECT work_class, holder, priority, {_AGG} FROM gpu_pool_events WHERE {where} "
                         "AND work_class IS NOT NULL GROUP BY work_class, holder, priority "
                         "ORDER BY grants DESC, work_class LIMIT 200"),
        "series": rows("SELECT to_timestamp(floor(extract(epoch FROM generated_at) / :bucket) * :bucket) AS t, "
                       f"work_class, {_AGG} FROM gpu_pool_events WHERE {where} AND work_class IS NOT NULL "
                       "GROUP BY t, work_class ORDER BY t"),
        "bucket_sec": params["bucket"],
        "events": rows("SELECT event_id, generated_at, event, lease_id, holder, work_class, priority, role, cards, "
                       "turn_correlation_id, attempt, waited_ms, held_ms, reason FROM gpu_pool_events "
                       f"WHERE {where} ORDER BY generated_at DESC LIMIT :limit"),
    }


@router.get("/history")
def gpu_pool_history(minutes: int = 60, work_class: Optional[str] = None, holder: Optional[str] = None,
                     lease_id: Optional[str] = None, limit: int = 200) -> dict[str, Any]:
    _enabled()
    minutes = max(5, min(int(minutes), 7 * 24 * 60))
    limit = max(1, min(int(limit), 2000))
    with _engine().connect() as conn:
        return history_payload(conn, minutes=minutes, work_class=work_class, holder=holder,
                               lease_id=lease_id, limit=limit)


# --- operator control -----------------------------------------------------------------------
class ControlBody(BaseModel):
    verb: str = Field(pattern="^(lend|unlend|hold|release|replay|cancel|backfill|clear_fault)$")
    card: Optional[str] = None
    lease_id: Optional[str] = None
    work_class: Optional[str] = None
    backfill: Optional[dict[str, Any]] = None


CSRF_HEADER_VALUE = "orion-hub"


@router.post("/control")
async def gpu_pool_control(body: ControlBody, request: Request,
                           x_requested_with: str | None = Header(default=None)) -> dict[str, Any]:
    _enabled()
    # No token, by design. What stays is cross-site request forgery protection, which is not a
    # credential: any web page could otherwise POST here from the operator's own browser. A custom
    # header forces a CORS preflight that Hub never grants, and a JSON content type rules out the
    # typeless "simple" POST that FastAPI would still parse as JSON.
    if x_requested_with != CSRF_HEADER_VALUE or "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(403, "gpu_pool_control_requires_hub_page")
    ctl = GpuPoolControlV1(verb=body.verb, card=body.card, lease_id=body.lease_id,
                           work_class=body.work_class, backfill=body.backfill, actor="hub-operator")
    reply = await _rpc(GPU_POOL_CONTROL_REQUEST_CHANNEL, GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_KIND,
                       ctl.model_dump(mode="json"))
    out = GpuPoolControlReplyV1.model_validate(reply)
    logger.info("gpu_pool_panel_control verb=%s ok=%s reason=%s", body.verb, out.ok, out.reason)
    return out.model_dump(mode="json")
