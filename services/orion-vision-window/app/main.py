import asyncio
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import (
    VisionArtifactPayload,
    VisionWindowPayload,
    VisionWindowRequestPayload,
    VisionWindowResultPayload,
)

from .projection import build_window_payload, envelope_to_http_dict, stream_key_from_artifact
from .recovery_store import RecoveryStore
from .settings import Settings

settings = Settings()


def _source_ref() -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)


def _corr_uuid(value: Optional[Union[str, UUID]]) -> UUID:
    if value is None:
        return uuid.uuid4()
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class WindowService:
    def __init__(self) -> None:
        self.bus = OrionBusAsync(
            url=settings.ORION_BUS_URL,
            enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self._consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
        self._emitter_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Per-stream ingest buffers: list of {artifact, ts, env}
        self._buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._buffer_lock = asyncio.Lock()

        self._live_lock = asyncio.Lock()
        self._live_by_stream: Dict[str, VisionWindowPayload] = {}
        self._live_global: Optional[VisionWindowPayload] = None
        self._recent_by_stream: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.VISION_WINDOW_RECOVERY_MAX_N)
        )
        self._recent_global: deque = deque(maxlen=settings.VISION_WINDOW_RECOVERY_MAX_N)

        self._recovery: Optional[RecoveryStore] = None
        self._recovery_ok = False
        self._bus_ready = False

        self._cursor_i = 0

        # Metrics counters (§12)
        self._m_ingest = 0
        self._m_snapshots = 0
        self._m_recovery_ok = 0
        self._m_recovery_fail = 0
        self._m_catchup_expired = 0

    def _recovery_url(self) -> str:
        return (settings.VISION_WINDOW_RECOVERY_REDIS_URL or settings.ORION_BUS_URL).strip()

    def _next_cursor(self) -> str:
        self._cursor_i += 1
        return f"vw:{self._cursor_i:012d}:{uuid.uuid4().hex[:6]}"

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        logger.info(
            f"[WINDOW] Startup config: recovery_enabled={settings.VISION_WINDOW_RECOVERY_ENABLED} "
            f"redis_url_redacted=yes max_n={settings.VISION_WINDOW_RECOVERY_MAX_N} "
            f"ttl_sec={settings.VISION_WINDOW_RECOVERY_TTL_SEC} "
            f"flush_interval_ms={settings.FLUSH_INTERVAL_MS} "
            f"ready_requires_recovery={settings.VISION_WINDOW_READY_REQUIRES_RECOVERY}"
        )

        await self.bus.connect()
        self._bus_ready = True

        if settings.VISION_WINDOW_RECOVERY_ENABLED:
            self._recovery = RecoveryStore(
                self._recovery_url(),
                ttl_sec=settings.VISION_WINDOW_RECOVERY_TTL_SEC,
                max_n=settings.VISION_WINDOW_RECOVERY_MAX_N,
            )
            self._recovery_ok = await self._recovery.connect()
            if not self._recovery_ok:
                logger.warning("[WINDOW] Recovery disabled at runtime (connection failed)")
        else:
            self._recovery = None
            self._recovery_ok = False

        self._consumer_task = asyncio.create_task(self._consume())
        self._rpc_task = asyncio.create_task(self._consume_rpc())
        self._emitter_task = asyncio.create_task(self._emit_loop())
        logger.info(
            f"[WINDOW] Started. intake={settings.CHANNEL_WINDOW_INTAKE} "
            f"pub={settings.CHANNEL_WINDOW_PUB} rpc={settings.CHANNEL_WINDOW_REQUEST}"
        )

    async def stop(self) -> None:
        self._shutdown_event.set()
        for t in [self._consumer_task, self._rpc_task, self._emitter_task]:
            if t:
                try:
                    t.cancel()
                    await t
                except asyncio.CancelledError:
                    pass
        await self.bus.close()
        self._bus_ready = False
        if self._recovery:
            await self._recovery.close()
            self._recovery = None

    def _should_flush_stream(self, stream_id: str, now: float) -> bool:
        buf = self._buffers.get(stream_id) or []
        if not buf:
            return False
        first_ts = buf[0]["ts"]
        age_ms = (now - first_ts) * 1000.0
        if len(buf) >= settings.MAX_ARTIFACTS_PER_WINDOW:
            return True
        if age_ms >= float(settings.FLUSH_INTERVAL_MS):
            return True
        if age_ms >= float(settings.MAX_WINDOW_AGE_MS):
            return True
        if age_ms >= settings.WINDOW_SIZE_SEC * 1000.0 and len(buf) > 0:
            return True
        return False

    async def _consume(self) -> None:
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break
                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue
                env = decoded.envelope
                try:
                    if isinstance(env.payload, dict):
                        payload = VisionArtifactPayload(**env.payload)
                    else:
                        payload = env.payload
                except Exception as e:
                    logger.warning(f"[WINDOW] Invalid artifact payload: {e}")
                    continue
                sk = stream_key_from_artifact(payload)
                logger.debug(f"[WINDOW] ingest accepted stream={sk} artifact={payload.artifact_id}")
                self._m_ingest += 1
                async with self._buffer_lock:
                    self._buffers[sk].append({"artifact": payload, "ts": time.time(), "env": env})

    async def _consume_rpc(self) -> None:
        async with self.bus.subscribe(settings.CHANNEL_WINDOW_REQUEST) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break
                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue
                env = decoded.envelope
                asyncio.create_task(self._handle_rpc(env))

    async def _handle_rpc(self, env: BaseEnvelope) -> None:
        try:
            if isinstance(env.payload, dict):
                req = VisionWindowRequestPayload(**env.payload)
            else:
                req = env.payload
        except Exception as e:
            logger.error(f"[WINDOW] RPC invalid payload: {e}")
            return
        art = req.artifact
        sk = stream_key_from_artifact(art)
        now = time.time()
        entry = {"artifact": art, "ts": now, "env": env}
        # Single projection path: same flush/materialize as streaming (§3.3)
        await self._flush_and_publish(
            stream_id=sk,
            buffered=[entry],
            correlation_id=env.correlation_id,
            causality_chain=list(env.causality_chain),
        )
        async with self._live_lock:
            window_payload = self._live_by_stream.get(sk) or self._live_global
        if window_payload is None:
            return
        res_payload = VisionWindowResultPayload(window=window_payload)
        reply_env = BaseEnvelope(
            kind="vision.window.result",
            source=_source_ref(),
            correlation_id=env.correlation_id,
            causality_chain=[*env.causality_chain],
            payload=res_payload.model_dump(mode="json"),
            reply_to=None,
        )
        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

    async def _emit_loop(self) -> None:
        while not self._shutdown_event.is_set():
            now = time.time()
            async with self._buffer_lock:
                stream_ids = list(self._buffers.keys())
            for sid in stream_ids:
                flush = False
                async with self._buffer_lock:
                    if self._should_flush_stream(sid, now):
                        flush = True
                if flush:
                    await self._drain_stream(sid)
            await asyncio.sleep(0.25)

    async def _drain_stream(self, stream_id: str) -> None:
        async with self._buffer_lock:
            buf = self._buffers.get(stream_id)
            if not buf:
                return
            items = list(buf)
            buf.clear()
        await self._flush_and_publish(
            stream_id=stream_id,
            buffered=items,
            correlation_id=uuid.uuid4(),
            causality_chain=[],
        )

    async def _flush_and_publish(
        self,
        *,
        stream_id: str,
        buffered: List[Dict[str, Any]],
        correlation_id: Optional[Union[str, UUID]],
        causality_chain: List[Any],
    ) -> None:
        if not buffered:
            return
        items: List[Tuple[VisionArtifactPayload, float]] = [(b["artifact"], b["ts"]) for b in buffered]
        envs: List[BaseEnvelope] = [b["env"] for b in buffered]
        window_start = min(b["ts"] for b in buffered)
        window_end = time.time()
        cursor = self._next_cursor()
        payload = build_window_payload(
            stream_id=stream_id,
            items=items,
            envs=envs,
            window_start=window_start,
            window_end=window_end,
            cursor=cursor,
            stale_after_ms=settings.STALE_AFTER_MS,
        )
        env_dump = payload.model_dump(mode="json")

        async with self._live_lock:
            self._live_by_stream[stream_id] = payload
            self._live_global = payload
            self._recent_by_stream[stream_id].appendleft(env_dump)
            self._recent_global.appendleft(env_dump)

        if self._recovery and self._recovery.enabled:
            ok = await self._recovery.persist_snapshot(stream_id, env_dump, cursor)
            if ok:
                self._m_recovery_ok += 1
            else:
                self._m_recovery_fail += 1

        cid = _corr_uuid(correlation_id)
        envelope = BaseEnvelope(
            kind="vision.window",
            source=_source_ref(),
            correlation_id=cid,
            causality_chain=[*causality_chain],
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(settings.CHANNEL_WINDOW_PUB, envelope)
        self._m_snapshots += 1
        logger.info(
            f"[WINDOW] flush snapshot_id={payload.window_id} stream={stream_id} "
            f"artifacts={len(buffered)} cursor={cursor}"
        )

    async def health_live(self) -> Dict[str, Any]:
        return {"status": "ok", "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}

    async def ready_probe(self) -> Tuple[bool, str]:
        if not self._bus_ready:
            return False, "bus_not_connected"
        if settings.VISION_WINDOW_READY_REQUIRES_RECOVERY:
            if not self._recovery or not self._recovery.enabled:
                return False, "recovery_required_unavailable"
            if not await self._recovery.ping():
                return False, "recovery_ping_failed"
        elif self._recovery and self._recovery.enabled:
            if not await self._recovery.ping():
                return True, "degraded_recovery_unavailable"
        return True, "ok"

    def _cap_limit(self, limit: Optional[int]) -> int:
        raw = limit or 20
        return max(1, min(int(raw), settings.VISION_WINDOW_HTTP_MAX_LIMIT))

    async def http_current(self, stream_id: Optional[str]) -> Dict[str, Any]:
        async with self._live_lock:
            if stream_id:
                live = self._live_by_stream.get(stream_id)
            else:
                live = self._live_global
        if live:
            body = envelope_to_http_dict(live, source="live_state")
            body["status"] = "ok"
            return body
        if self._recovery and self._recovery.enabled:
            data = await self._recovery.read_latest(stream_id)
            if data:
                try:
                    p = VisionWindowPayload(**data)
                    body = envelope_to_http_dict(p, source="recovery_state")
                    body["status"] = "ok"
                    return body
                except ValidationError:
                    pass
        return {
            "status": "empty",
            "source": "none",
            "snapshot_id": None,
            "stream_id": stream_id,
            "generated_at": None,
            "cursor": None,
            "age_ms": None,
            "envelope": None,
        }

    async def http_current_stale_check(self, stream_id: Optional[str]) -> Dict[str, Any]:
        body = await self.http_current(stream_id)
        if body.get("status") == "empty":
            return body
        env = body.get("envelope") or {}
        end_ts = float(env.get("end_ts") or 0)
        age_ms = int(max(0.0, (time.time() - end_ts) * 1000))
        stale_after = int((env.get("freshness") or {}).get("stale_after_ms") or settings.STALE_AFTER_MS)
        if age_ms > stale_after:
            body["status"] = "stale"
        return body

    async def http_recent(self, stream_id: Optional[str], limit: int) -> Dict[str, Any]:
        lim = self._cap_limit(limit)
        recovery_ok = bool(self._recovery and self._recovery.enabled and await self._recovery.ping())
        rows: List[Dict[str, Any]] = []
        if recovery_ok:
            rows = await self._recovery.read_last_n(stream_id, lim)
        if not rows:
            async with self._live_lock:
                dq = self._recent_by_stream[stream_id] if stream_id else self._recent_global
                rows = list(dq)[:lim]
        degraded = settings.VISION_WINDOW_RECOVERY_ENABLED and not recovery_ok
        return {"items": rows, "recovery_degraded": degraded, "limit": lim}

    async def http_catchup(
        self, stream_id: Optional[str], after_cursor: Optional[str], limit: int
    ) -> Any:
        lim = self._cap_limit(limit)
        recovery_ok = bool(self._recovery and self._recovery.enabled and await self._recovery.ping())
        rows: List[Dict[str, Any]] = []
        if recovery_ok:
            rows = await self._recovery.read_last_n(stream_id, settings.VISION_WINDOW_RECOVERY_MAX_N)
        if not rows:
            async with self._live_lock:
                dq = self._recent_by_stream[stream_id] if stream_id else self._recent_global
                rows = list(dq)
        degraded = settings.VISION_WINDOW_RECOVERY_ENABLED and not recovery_ok
        if not after_cursor:
            return {"items": rows[:lim], "recovery_degraded": degraded}
        sorted_rows = sorted(rows, key=lambda r: str(r.get("cursor") or ""))
        cursors = [str(r.get("cursor") or "") for r in sorted_rows if r.get("cursor")]
        if not cursors:
            return {"items": [], "recovery_degraded": degraded}
        earliest = min(cursors)
        latest = max(cursors)
        if after_cursor < earliest:
            self._m_catchup_expired += 1
            logger.info(f"[WINDOW] catch-up cursor_expired after={after_cursor}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "cursor_expired",
                    "message": "Requested cursor is outside the bounded recovery window.",
                    "latest_cursor": latest,
                    "earliest_available_cursor": earliest,
                },
            )
        out: List[Dict[str, Any]] = []
        for r in sorted_rows:
            c = str(r.get("cursor") or "")
            if c and c > after_cursor:
                out.append(r)
            if len(out) >= lim:
                break
        return {"items": out[:lim], "recovery_degraded": degraded}


service = WindowService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()


app = FastAPI(title="Orion Vision Window", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return await service.health_live()


@app.get("/readyz")
async def readyz() -> JSONResponse:
    ok, reason = await service.ready_probe()
    code = 200 if ok else 503
    return JSONResponse(status_code=code, content={"status": "ready" if ok else "not_ready", "detail": reason})


@app.get("/api/vision-window/current")
async def api_current() -> Dict[str, Any]:
    return await service.http_current_stale_check(None)


@app.get("/api/vision-window/streams/{stream_id}/current")
async def api_current_stream(stream_id: str) -> Dict[str, Any]:
    return await service.http_current_stale_check(stream_id)


@app.get("/api/vision-window/recent")
async def api_recent(limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
    return await service.http_recent(None, limit)


@app.get("/api/vision-window/streams/{stream_id}/recent")
async def api_recent_stream(stream_id: str, limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
    return await service.http_recent(stream_id, limit)


@app.get("/api/vision-window/catch-up")
async def api_catchup(
    after_cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
):
    return await service.http_catchup(None, after_cursor, limit)


@app.get("/api/vision-window/streams/{stream_id}/catch-up")
async def api_catchup_stream(
    stream_id: str,
    after_cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
):
    return await service.http_catchup(stream_id, after_cursor, limit)


@app.get("/api/vision-window/metrics")
async def api_metrics() -> Dict[str, Any]:
    return {
        "vision_window_ingest_events_total": service._m_ingest,
        "vision_window_snapshots_published_total": service._m_snapshots,
        "vision_window_recovery_writes_total": service._m_recovery_ok,
        "vision_window_recovery_write_failures_total": service._m_recovery_fail,
        "vision_window_cursor_expired_total": service._m_catchup_expired,
    }
