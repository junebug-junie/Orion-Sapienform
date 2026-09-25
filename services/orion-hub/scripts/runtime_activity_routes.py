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
    s.correlation_id, s.generated_at, s.detail,
    (SELECT MIN(f.generated_at) FROM substrate_durable_run_state f WHERE f.run_id = s.run_id) AS first_seen_at
FROM substrate_durable_run_state s
WHERE s.generated_at > NOW() - INTERVAL '48 hours'
ORDER BY s.run_id, s.generated_at DESC
"""

# The SSE writer coalesces bursts: many folds within this window become one
# frame. Steps can arrive several per second on a busy turn.
_COALESCE_SEC = 0.25
_HEARTBEAT_SEC = 15.0


LEDGER_WINDOW_SEC = 300.0
WAIT_THRESHOLD_MS = 500.0  # same "made to wait" line as cortex-exec's admission cue


def pool_lanes(state: dict[str, Any] | None, events: list[dict[str, Any]], *, now_ts: float) -> dict[str, Any]:
    """The "what's running" gateway section, built from orion-gpu-pool instead of the gateway's
    deleted /admission ledger. One record per pool role that runs work:
      inflight / max_inflight -- leases held on it now / its discovered slots
      waiting                 -- leases queued for the class that calls it home (role name == class)
    plus a 5-minute ledger from the pool's lease events (the same shape the page already reads)."""
    from datetime import datetime

    state = state or {}
    held: dict[str, int] = {}
    waiting: dict[str, int] = {}
    for lease in state.get("leases") or []:
        if lease.get("status") in ("granted", "recalling") and lease.get("role"):
            held[lease["role"]] = held.get(lease["role"], 0) + 1
        elif lease.get("status") == "queued":
            waiting[lease.get("work_class") or "?"] = waiting.get(lease.get("work_class") or "?", 0) + 1
    lanes = []
    role_names = set()
    for role in state.get("roles") or []:
        name = role.get("role")
        role_names.add(name)
        if not role.get("slots") and not held.get(name) and not waiting.get(name):
            continue  # unloaded seats / evicted residents: nothing to show
        lanes.append({
            "upstream": role.get("url"),
            "routes": [{"id": name, "served_by": f"circe-worker-{name}", "status": role.get("status"),
                        "priority": None, "model": role.get("model_file")}],
            "inflight": held.get(name, 0), "waiting": waiting.get(name, 0), "max_inflight": role.get("slots"),
        })
    for cls, n in sorted(waiting.items()):
        if cls not in role_names:
            lanes.append({"upstream": f"queue:{cls}", "routes": [{"id": cls}], "inflight": 0, "waiting": n,
                          "max_inflight": None})

    def ts(event: dict[str, Any]) -> float:
        try:
            return datetime.fromisoformat(str(event.get("generated_at")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    recent = [e for e in events if now_ts - ts(e) <= LEDGER_WINDOW_SEC]
    waits = [float(e.get("waited_ms") or 0) for e in recent if e.get("event") == "granted"]
    deferred = [w for w in waits if w >= WAIT_THRESHOLD_MS]
    ledger = {
        "checked": sum(1 for e in recent if e.get("event") == "admitted"),
        "queued": sum(waiting.values()),
        "deferrals": len(deferred) + sum(1 for e in recent if e.get("event") == "unavailable"),
        "longest_wait_s": (max(deferred) / 1000.0) if deferred else 0.0,
    }
    return {"lanes": lanes, "ledger": ledger, "default_route": None, "source": "gpu_pool"}


class RuntimeActivityFeeds:
    """The gateway poll loop + the startup backfill. `engine_factory` is
    hub_surface_routes' shared SQLAlchemy engine getter (same POSTGRES_URI,
    same pool), injected so tests can hand in a fake."""

    def __init__(
        self,
        *,
        activity: RuntimeActivity,
        poll_sec: float,
        engine_factory: Optional[Callable[[], Any]] = None,
        pool_feed: Any = None,
    ) -> None:
        self._activity = activity
        self._poll_sec = float(poll_sec)
        self._engine_factory = engine_factory
        # Hub's live orion-gpu-pool feed (scripts/gpu_pool_routes.feed): state + recent lease events.
        self._pool_feed = pool_feed
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        await self.backfill()
        if self._poll_sec > 0:
            self._task = asyncio.create_task(self._poll_loop(), name="hub-runtime-activity-pool-poll")

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
        """Read the pool feed Hub already keeps (no network hop). No state yet -> reported as a
        down source, never as idle."""
        feed = self._pool_feed
        state = getattr(feed, "state", None) if feed is not None else None
        if not state:
            self._activity.gateway_admission(None, error="gpu_pool_state_unavailable")
            return
        import time

        self._activity.gateway_admission(pool_lanes(state, list(getattr(feed, "events", []) or []), now_ts=time.time()))

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
