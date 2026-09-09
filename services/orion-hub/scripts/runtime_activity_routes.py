"""Runtime activity: the header marquee's data + the two feeds only Hub's
process can run for it.

The reducer itself lives in ``orion/hub/runtime_activity.py`` and is fed
in-process by turn_orchestrator (turn handoffs), curiosity_investigation
(run dispatch + run-state events) and harness_step_relay (steps). This module
adds the two feeds that need I/O -- a poll of the LLM gateway's live lane
gauges and a one-shot cold-start backfill of still-active durable runs from
Postgres -- and serves the result:

- ``GET /api/runtime-activity``          one snapshot
- ``GET /api/runtime-activity/stream``   SSE; a snapshot on connect, then one
                                          per change (coalesced), heartbeat
                                          comments in between.

Both feeds fail open: a gateway that is down shows as ``gateway.error`` next
to the last good gauges, a Postgres that is down logs once and the marquee
starts empty until the next live event.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Optional

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.settings import settings
from orion.hub.runtime_activity import RuntimeActivity, get_runtime_activity

logger = logging.getLogger("orion-hub.runtime_activity")

router = APIRouter(prefix="/api/runtime-activity", tags=["runtime-activity"])

# Latest transition per run in the last 48h, plus each run's first-seen time
# so a backfilled run's duration counts from its real start, not from Hub's
# restart. Only rows whose latest status is still running/resumed are adopted
# (RuntimeActivity.backfill_runs filters) -- terminal rows are history the
# Hub Surface page already covers.
BACKFILL_SQL = """
SELECT DISTINCT ON (s.run_id)
    s.run_id, s.workflow, s.node, s.next_node, s.status, s.resumed_from_node,
    s.correlation_id, s.generated_at,
    (SELECT MIN(f.generated_at) FROM substrate_durable_run_state f WHERE f.run_id = s.run_id) AS first_seen_at
FROM substrate_durable_run_state s
WHERE s.generated_at > NOW() - INTERVAL '48 hours'
ORDER BY s.run_id, s.generated_at DESC
"""

# The SSE writer coalesces bursts: many folds within this window become one
# frame. Steps can arrive several per second on a busy turn.
_COALESCE_SEC = 0.25
_HEARTBEAT_SEC = 15.0


def _routes_by_upstream(routes_payload: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Group the gateway's route catalog by its `upstream` join key so each
    admission gauge can name the lanes it serves. Routes without an upstream
    (not configured) are dropped -- they have no gauge to attach to."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in (routes_payload or {}).get("routes") or []:
        if not isinstance(r, dict):
            continue
        key = r.get("upstream")
        if not key:
            continue
        out.setdefault(str(key), []).append(
            {
                "id": r.get("id"),
                "served_by": r.get("served_by"),
                "status": r.get("status"),
                "priority": r.get("priority"),
                "model": r.get("model"),
            }
        )
    return out


def merge_gateway(admission: dict[str, Any], routes_payload: dict[str, Any] | None) -> dict[str, Any]:
    """One record per upstream: the live gauge plus the route ids that
    dispatch to it. An upstream the catalog does not name still appears
    (`routes: []`) -- a gauge with real traffic is evidence regardless."""
    by_upstream = _routes_by_upstream(routes_payload)
    upstreams = admission.get("upstreams") if isinstance(admission.get("upstreams"), dict) else {}
    lanes = []
    for url, gauge in sorted(upstreams.items()):
        if not isinstance(gauge, dict):
            continue
        lanes.append({"upstream": url, "routes": by_upstream.get(url, []), **gauge})
    ledger = {k: v for k, v in admission.items() if k != "upstreams"}
    return {"lanes": lanes, "ledger": ledger, "default_route": (routes_payload or {}).get("default_route")}


class RuntimeActivityFeeds:
    """The gateway poll loop + the startup backfill. `engine_factory` is
    hub_surface_routes' shared SQLAlchemy engine getter (same POSTGRES_URI,
    same pool), injected so tests can hand in a fake."""

    def __init__(
        self,
        *,
        activity: RuntimeActivity,
        gateway_url: str,
        poll_sec: float,
        timeout_sec: float,
        engine_factory: Optional[Callable[[], Any]] = None,
        session_factory: Optional[Callable[[], aiohttp.ClientSession]] = None,
    ) -> None:
        self._activity = activity
        self._gateway_url = gateway_url.rstrip("/")
        self._poll_sec = float(poll_sec)
        self._timeout_sec = max(0.5, float(timeout_sec))
        self._engine_factory = engine_factory
        self._session_factory = session_factory or (
            lambda: aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout_sec))
        )
        self._task: Optional[asyncio.Task] = None
        self._routes_cache: dict[str, Any] | None = None
        self._routes_cached_at: float = 0.0

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        await self.backfill()
        if self._poll_sec > 0:
            self._task = asyncio.create_task(self._poll_loop(), name="hub-runtime-activity-gateway-poll")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def backfill(self) -> int:
        if self._engine_factory is None:
            return 0

        def _read() -> list[dict[str, Any]]:
            engine = self._engine_factory()
            with engine.connect() as conn:
                return [dict(r) for r in conn.execute(text(BACKFILL_SQL)).mappings().all()]

        try:
            rows = await asyncio.to_thread(_read)
        except Exception as exc:  # noqa: BLE001 -- fail open, marquee starts empty
            logger.warning("runtime_activity_backfill_failed error=%s", exc)
            return 0
        adopted = self._activity.backfill_runs(rows)
        logger.info("runtime_activity_backfill rows=%s adopted=%s", len(rows), adopted)
        return adopted

    async def poll_once(self) -> None:
        """One gateway round trip. Routes are re-read every ~60s (they change
        only on gateway restart); admission every poll."""
        try:
            async with self._session_factory() as session:
                async with session.get(f"{self._gateway_url}/admission", params={"window_s": 300}) as resp:
                    resp.raise_for_status()
                    admission = await resp.json()
                loop_now = asyncio.get_running_loop().time()
                if self._routes_cache is None or (loop_now - self._routes_cached_at) > 60.0:
                    async with session.get(f"{self._gateway_url}/routes") as resp:
                        if resp.status == 200:
                            self._routes_cache = await resp.json()
                            self._routes_cached_at = loop_now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a down gateway is a reported state
            self._activity.gateway_admission(None, error=f"{type(exc).__name__}: {exc}"[:200])
            return
        if not isinstance(admission, dict):
            self._activity.gateway_admission(None, error="admission_not_an_object")
            return
        self._activity.gateway_admission(merge_gateway(admission, self._routes_cache))

    async def _poll_loop(self) -> None:
        try:
            while True:
                await self.poll_once()
                await asyncio.sleep(self._poll_sec)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("runtime_activity_gateway_poll_loop_failed")


def _sse(data: dict[str, Any], *, event: str = "snapshot") -> str:
    return f"event: {event}\nid: {data.get('version', 0)}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def _stream(activity: RuntimeActivity, request: Request) -> AsyncIterator[str]:
    q = activity.subscribe()
    try:
        yield _sse(activity.snapshot())
        while True:
            if await request.is_disconnected():
                return
            try:
                await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SEC)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            # Coalesce: drain whatever else landed in the window, send once.
            await asyncio.sleep(_COALESCE_SEC)
            while not q.empty():
                q.get_nowait()
            yield _sse(activity.snapshot())
    finally:
        activity.unsubscribe(q)


def _require_enabled() -> RuntimeActivity:
    if not settings.HUB_RUNTIME_ACTIVITY_ENABLED:
        raise HTTPException(status_code=503, detail="runtime_activity_disabled")
    return get_runtime_activity()


@router.get("")
async def runtime_activity_snapshot() -> dict[str, Any]:
    return _require_enabled().snapshot()


@router.get("/stream")
async def runtime_activity_stream(request: Request) -> StreamingResponse:
    activity = _require_enabled()
    return StreamingResponse(
        _stream(activity, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
