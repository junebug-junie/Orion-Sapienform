# services/orion-spark-introspector/app/queue_worker.py
from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.codec import OrionCodec
from orion.core.bus.work_queue import RedisStreamWorkQueue, _parse_ts, extract_work_metadata

from .queue_jobs import SparkIntrospectionJobV1, extract_spark_introspection_job
from .settings import settings
from .worker import SparkCandidatePayload, run_heavy_spark_introspection

logger = logging.getLogger("orion-spark-introspector")


def _job_to_candidate(job: SparkIntrospectionJobV1) -> SparkCandidatePayload:
    return SparkCandidatePayload(
        trace_id=job.trace_id,
        source=job.source,
        prompt=job.prompt,
        response=job.response,
        spark_meta=dict(job.spark_meta or {}),
        introspection=job.introspection,
    )


async def get_spark_queue_status() -> dict[str, Any]:
    """Lightweight Redis stream inspection for operators/debug."""
    out: dict[str, Any] = {
        "enabled": bool(settings.spark_introspection_queue_enabled),
        "stream": settings.spark_introspection_queue_stream,
        "group": settings.spark_introspection_queue_group,
        "dlq_stream": settings.spark_introspection_queue_dlq,
        "pending": None,
        "length": None,
        "dlq_length": None,
    }
    if not settings.spark_introspection_queue_enabled:
        return out
    url = settings.spark_introspection_redis_url or settings.orion_bus_url
    wq = RedisStreamWorkQueue(url, codec=OrionCodec())
    await wq.connect()
    try:
        try:
            out["length"] = int(await wq.client.xlen(settings.spark_introspection_queue_stream))
        except Exception:
            out["length"] = None
        try:
            out["pending"] = await wq.pending_summary(
                settings.spark_introspection_queue_stream,
                settings.spark_introspection_queue_group,
            )
        except Exception:
            out["pending"] = None
        try:
            out["dlq_length"] = int(await wq.client.xlen(settings.spark_introspection_queue_dlq))
        except Exception:
            out["dlq_length"] = None
    finally:
        await wq.close()
    return out


async def handle_spark_introspection_job(env: BaseEnvelope) -> BaseEnvelope | None:
    t0 = perf_counter()
    try:
        job = extract_spark_introspection_job(env)
    except Exception as exc:
        logger.error(
            "spark_queue_failed trace_id=unknown job_id=unknown error=%s retryable=False",
            exc,
        )
        return None

    wm = extract_work_metadata(env)
    job_id = str(wm.get("job_id", ""))
    try:
        attempt = max(1, int(wm.get("attempt", 1)))
    except (TypeError, ValueError):
        attempt = 1
    trace_id = str(job.trace_id)

    logger.info(
        "spark_queue_claim trace_id=%s job_id=%s attempt=%s",
        trace_id,
        job_id,
        attempt,
    )

    exp = _parse_ts(wm.get("expires_at"))
    now = datetime.now(timezone.utc)
    if exp is not None and now > exp:
        age_sec = (now - exp).total_seconds()
        logger.info(
            "spark_queue_skip_stale trace_id=%s job_id=%s age_sec=%.1f max_age_sec=%s",
            trace_id,
            job_id,
            max(0.0, age_sec),
            settings.spark_introspection_queue_max_age_sec,
        )
        return None

    candidate = _job_to_candidate(job)
    corr = job.correlation_id

    try:
        result = await run_heavy_spark_introspection(
            candidate=candidate,
            source_env=env,
            correlation_id=corr,
            bus=None,
            from_queue=True,
            job=job,
        )
    except (ConnectionError, OSError) as exc:
        logger.error(
            "spark_queue_failed trace_id=%s job_id=%s error=%s retryable=True",
            trace_id,
            job_id,
            exc,
        )
        raise

    status = str(result.get("status", "failed"))
    reason = str(result.get("reason", ""))
    ms = (perf_counter() - t0) * 1000.0

    logger.info(
        "spark_queue_complete trace_id=%s job_id=%s status=%s runtime_ms=%.1f",
        trace_id,
        job_id,
        status,
        ms,
    )

    if status == "degraded":
        logger.info(
            "spark_queue_degraded trace_id=%s job_id=%s reason=%s",
            trace_id,
            job_id,
            reason,
        )
        return None

    if status in ("complete", "skipped", "redis_unavailable"):
        return None

    if status == "failed":
        logger.error(
            "spark_queue_failed trace_id=%s job_id=%s error=%s retryable=True",
            trace_id,
            job_id,
            reason,
        )
        raise RuntimeError(reason or "heavy_failed")

    logger.error(
        "spark_queue_failed trace_id=%s job_id=%s error=unexpected_status retryable=False",
        trace_id,
        job_id,
    )
    return None
