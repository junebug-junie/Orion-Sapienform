from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx
from orion.core.bus.bus_schemas import BaseEnvelope

from app import rdf_store as rdf_store_mod
from app.rdf_builder import build_triples_from_envelope
from app.rdf_store import RdfStoreClient, RdfWriteResult, build_rdf_store_client, _strip_credentials
from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

_rdf_bus_publish: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]] = None


def register_rdf_write_publisher(fn: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]]) -> None:
    global _rdf_bus_publish
    _rdf_bus_publish = fn


_DEDUP_WINDOW_SEC = 2.0
_DEDUP_MAX_SIZE = 512
_dedupe_cache: dict[str, float] = {}


def _payload_fingerprint(payload: object) -> str:
    dumped = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _evict_dedupe_cache() -> None:
    if len(_dedupe_cache) <= _DEDUP_MAX_SIZE:
        return
    overflow = len(_dedupe_cache) - _DEDUP_MAX_SIZE
    for old_key, _ in sorted(_dedupe_cache.items(), key=lambda item: item[1])[:overflow]:
        _dedupe_cache.pop(old_key, None)


def _should_dedupe(env: BaseEnvelope) -> bool:
    now = time.monotonic()
    expired = [key for key, ts in _dedupe_cache.items() if now - ts > _DEDUP_WINDOW_SEC]
    for key in expired:
        _dedupe_cache.pop(key, None)

    correlation_id = str(env.correlation_id or "")
    payload_hash = _payload_fingerprint(env.payload)
    key = f"{env.kind}|{correlation_id}|{payload_hash}"
    last_seen = _dedupe_cache.get(key)
    if last_seen is not None and now - last_seen <= _DEDUP_WINDOW_SEC:
        return True

    _dedupe_cache[key] = now
    _evict_dedupe_cache()
    return False


@dataclass
class RdfWriteJob:
    kind: str
    graph_name: str | None
    content: str
    correlation_id: str | None
    source: str | None
    created_at: float
    payload_fingerprint: str | None
    attempt: int = 0


_write_queue: asyncio.Queue[RdfWriteJob | None] | None = None
_worker_tasks: list[asyncio.Task[None]] = []
_http_client: httpx.AsyncClient | None = None
_store: RdfStoreClient | None = None
_inflight_sem: asyncio.Semaphore | None = None
_in_flight_estimate: int = 0


def _job_digest(job: RdfWriteJob) -> dict[str, Any]:
    body = job.content.encode("utf-8")
    return {
        "kind": job.kind,
        "graph_name": job.graph_name,
        "correlation_id": job.correlation_id,
        "source": job.source,
        "created_at": job.created_at,
        "payload_fingerprint": job.payload_fingerprint,
        "attempt": job.attempt,
        "content_bytes": len(body),
        "content_sha256": hashlib.sha256(body).hexdigest(),
    }


def _append_deadletter(record: dict[str, Any]) -> None:
    if not settings.RDF_WRITE_DEAD_LETTER_ENABLED:
        return
    path = settings.RDF_WRITE_DEAD_LETTER_PATH
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


async def _maybe_publish_bus(reason: str, job: RdfWriteJob, err: str | None) -> None:
    pub = _rdf_bus_publish
    if pub is None:
        return
    payload: dict[str, Any] = {
        "kind": "rdf.write.error",
        "reason": reason,
        "rdf_kind": job.kind,
        "correlation_id": job.correlation_id,
        "graph_name": job.graph_name,
        "content_sha256": hashlib.sha256(job.content.encode("utf-8")).hexdigest(),
    }
    if err:
        payload["error"] = err[:2000]
    try:
        await pub(settings.CHANNEL_RDF_ERROR, payload)
    except Exception:
        logger.exception("rdf_write_bus_notify_failed channel=%s", settings.CHANNEL_RDF_ERROR)


async def _retrying_write(store: RdfStoreClient, job: RdfWriteJob) -> RdfWriteResult:
    last_exc: BaseException | None = None
    for attempt in range(settings.RDF_WRITE_RETRY_ATTEMPTS):
        try:
            return await store.write_graph(job.content, job.graph_name)
        except OSError as e:
            last_exc = e
        except httpx.TimeoutException as e:
            last_exc = e
        except httpx.HTTPStatusError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 400:
                raise
            last_exc = e
        except httpx.HTTPError as e:
            last_exc = e
        if attempt + 1 < settings.RDF_WRITE_RETRY_ATTEMPTS:
            delay = min(
                settings.RDF_WRITE_RETRY_MAX_DELAY_SEC,
                settings.RDF_WRITE_RETRY_BASE_DELAY_SEC * (2**attempt),
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


async def _handle_write_final_failure(job: RdfWriteJob, exc: BaseException) -> None:
    err = f"{type(exc).__name__}: {exc}"
    record = {"kind": "rdf_write_failed", "reason": "write_failed", "error": err, "job": _job_digest(job)}
    await asyncio.to_thread(_append_deadletter, record)
    await _maybe_publish_bus("write_failed", job, err)
    logger.error("rdf_write_failed kind=%s correlation_id=%s: %s", job.kind, job.correlation_id, err)


@asynccontextmanager
async def _inflight_slot():
    global _in_flight_estimate
    if _inflight_sem is None:
        yield
        return
    async with _inflight_sem:
        _in_flight_estimate += 1
        try:
            yield
        finally:
            _in_flight_estimate -= 1


async def _execute_write_job(job: RdfWriteJob) -> None:
    assert _store is not None
    async with _inflight_slot():
        try:
            res = await _retrying_write(_store, job)
            logger.info(
                "rdf_write_committed status_code=%s elapsed_ms=%s endpoint=%s kind=%s graph=%s",
                res.status_code,
                round(res.elapsed_ms or 0.0, 3),
                _strip_credentials(res.endpoint) or "",
                job.kind,
                job.graph_name,
            )
        except BaseException as exc:
            await _handle_write_final_failure(job, exc)
            raise


async def rdf_write_worker_loop(worker_id: int) -> None:
    assert _write_queue is not None
    while True:
        job = await _write_queue.get()
        try:
            if job is None:
                return
            try:
                await _execute_write_job(job)
            except BaseException:
                pass
        finally:
            _write_queue.task_done()


async def init_rdf_write_pipeline() -> None:
    global _http_client, _store, _write_queue, _worker_tasks, _inflight_sem
    if _http_client is not None:
        return
    _http_client = httpx.AsyncClient(
        timeout=settings.RDF_STORE_TIMEOUT_SEC,
        limits=rdf_store_mod.httpx_limits_for_settings(settings),
    )
    _store = build_rdf_store_client(settings, _http_client)
    if settings.RDF_WRITE_ASYNC_ENABLED:
        _write_queue = asyncio.Queue(maxsize=settings.RDF_WRITE_QUEUE_MAXSIZE)
        _inflight_sem = asyncio.Semaphore(settings.RDF_WRITE_MAX_IN_FLIGHT)
        for i in range(settings.RDF_WRITE_WORKERS):
            _worker_tasks.append(asyncio.create_task(rdf_write_worker_loop(i), name=f"rdf-write-{i}"))
    else:
        _write_queue = None
        _inflight_sem = None

    assert _store is not None
    try:
        h = await _store.health()
        graph_url = h.get("graph_store_url") or h.get("endpoint")
        logger.info(
            "rdf_store_backend_selected backend=%s graph_store_url=%s dataset=%s",
            (settings.RDF_STORE_BACKEND or "").strip().lower(),
            _strip_credentials(str(graph_url)) if graph_url else "",
            settings.RDF_STORE_DATASET,
        )
    except Exception as exc:
        logger.warning(
            "rdf_store_backend_selected backend=%s dataset=%s (health snapshot failed: %s)",
            (settings.RDF_STORE_BACKEND or "").strip().lower(),
            settings.RDF_STORE_DATASET,
            exc,
        )


async def shutdown_rdf_write_pipeline(*, drain_timeout_sec: float = 8.0) -> None:
    global _http_client, _store, _write_queue, _worker_tasks, _inflight_sem
    tasks = list(_worker_tasks)
    _worker_tasks = []
    if _write_queue is not None and tasks:
        for _ in tasks:
            await _write_queue.put(None)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=drain_timeout_sec)
        except TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _store = None
    _write_queue = None
    _inflight_sem = None


async def _push_to_rdf_store(content: str, graph_name: str | None, *, env: BaseEnvelope) -> None:
    assert _store is not None
    src = getattr(env.source, "name", None) if env.source is not None else None
    job = RdfWriteJob(
        kind=env.kind,
        graph_name=graph_name,
        content=content,
        correlation_id=str(env.correlation_id) if env.correlation_id is not None else None,
        source=str(src) if src else None,
        created_at=time.time(),
        payload_fingerprint=_payload_fingerprint(env.payload),
        attempt=0,
    )
    if not settings.RDF_WRITE_ASYNC_ENABLED:
        await _execute_write_job(job)
        return
    assert _write_queue is not None
    try:
        _write_queue.put_nowait(job)
    except asyncio.QueueFull:
        record = {"kind": "rdf_write_failed", "reason": "queue_full", "job": _job_digest(job)}
        await asyncio.to_thread(_append_deadletter, record)
        await _maybe_publish_bus("queue_full", job, None)
        logger.error(
            "rdf_write_backpressure kind=%s correlation_id=%s graph=%s",
            job.kind,
            job.correlation_id,
            job.graph_name,
        )
        raise asyncio.QueueFull()
    logger.info(
        "rdf_write_enqueued kind=%s correlation_id=%s graph=%s bytes=%s",
        job.kind,
        job.correlation_id,
        job.graph_name,
        len(content.encode("utf-8")),
    )


def rdf_write_health_snapshot() -> dict[str, Any]:
    q = _write_queue
    return {
        "rdf_store_backend": settings.RDF_STORE_BACKEND,
        "rdf_store_base_url": _strip_credentials(settings.RDF_STORE_BASE_URL or settings.GRAPHDB_URL),
        "rdf_store_dataset": settings.RDF_STORE_DATASET,
        "rdf_write_async_enabled": settings.RDF_WRITE_ASYNC_ENABLED,
        "rdf_write_queue_size": q.qsize() if q is not None else 0,
        "rdf_write_queue_maxsize": q.maxsize if q is not None else 0,
        "rdf_write_workers": settings.RDF_WRITE_WORKERS if settings.RDF_WRITE_ASYNC_ENABLED else 0,
        "rdf_write_in_flight_estimate": _in_flight_estimate,
        "rdf_write_dead_letter_enabled": settings.RDF_WRITE_DEAD_LETTER_ENABLED,
        "rdf_write_dead_letter_path": settings.RDF_WRITE_DEAD_LETTER_PATH,
    }


async def handle_envelope(env: BaseEnvelope) -> None:
    payload = env.payload if isinstance(env.payload, dict) else {}
    memory_status = str(payload.get("memory_status") or "").lower()
    memory_tier = str(payload.get("memory_tier") or "").lower()
    if settings.RDF_SKIP_REJECTED and memory_status == "rejected":
        return
    if settings.RDF_DURABLE_ONLY and memory_tier != "durable":
        return
    if env.kind in settings.get_skip_kinds():
        logger.info("skip rdf kind=%s correlation_id=%s", env.kind, env.correlation_id)
        return
    logger.debug("Received %s from %s", env.kind, env.source)
    if _should_dedupe(env):
        logger.info("dedupe skip kind=%s correlation_id=%s", env.kind, env.correlation_id)
        return

    try:
        content, graph = build_triples_from_envelope(env.kind, env.payload)
        if content:
            try:
                await _push_to_rdf_store(content, graph, env=env)
            except asyncio.QueueFull:
                logger.error(
                    "rdf_write_queue_full_dropped kind=%s correlation_id=%s",
                    env.kind,
                    env.correlation_id,
                )
                return
        else:
            logger.debug("No triples generated for %s", env.kind)
    except Exception as e:
        logger.error("Failed to process RDF for %s: %s", env.kind, e, exc_info=True)
