# services/orion-spark-introspector/app/queue_jobs.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from .settings import Settings


class SparkIntrospectionJobV1(BaseModel):
    schema_version: Literal["spark.introspection.job.v1"] = "spark.introspection.job.v1"
    trace_id: str
    candidate_trace_id: str
    correlation_id: str | None = None
    source: str
    prompt: str
    response: str
    spark_meta: dict[str, Any] = Field(default_factory=dict)
    introspection: str | None = None
    candidate_created_at: str
    enqueued_at: str
    session_id: str | None = None
    message_id: str | None = None
    workflow_id: str | None = None
    execution_lane: str = "spark"
    llm_lane: str = "spark"
    priority: str = "low"
    allow_degrade: bool = True
    allow_chat_fallback: bool = False
    max_tokens: int | None = None
    timeout_sec: float | None = None


def build_idempotency_key(trace_id: str) -> str:
    return f"spark:introspect:{trace_id}"


def _coerce_epoch_ts(v: Any) -> float:
    import time
    from datetime import datetime, timezone

    if v is None:
        return time.time()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            pass
        try:
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return time.time()
    return time.time()


def _candidate_created_epoch(candidate: Any) -> float:
    meta = getattr(candidate, "spark_meta", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    ts = meta.get("as_of_ts") or meta.get("timestamp")
    return float(_coerce_epoch_ts(ts))


def envelope_correlation_uuid(env: BaseEnvelope, trace_id: str) -> UUID:
    """
    Stable correlation UUID for the job envelope: prefer parseable envelope id,
    else UUIDv5 from ``trace_id`` (never random) so queue + cortex stay joinable.
    """
    cid = env.correlation_id
    if isinstance(cid, UUID):
        return cid
    try:
        return UUID(str(cid).strip())
    except Exception:
        return uuid5(NAMESPACE_URL, f"orion:spark:introspect:correlation:{trace_id}")


def build_spark_introspection_job_envelope(
    candidate: Any,
    env: BaseEnvelope,
    settings: Settings,
    source: ServiceRef,
) -> BaseEnvelope:
    """
    Build a queue job envelope. Does not mutate ``candidate`` or ``env``.
    """
    now = datetime.now(timezone.utc)
    cand_ts = _candidate_created_epoch(candidate)
    cand_iso = datetime.fromtimestamp(cand_ts, tz=timezone.utc).isoformat()
    expires = datetime.fromtimestamp(cand_ts, tz=timezone.utc) + timedelta(
        seconds=float(settings.spark_introspection_queue_max_age_sec)
    )
    job_id = str(uuid4())
    tid = str(candidate.trace_id)
    corr_u = envelope_correlation_uuid(env, tid)
    lane = str(settings.spark_introspection_execution_lane or "spark").strip().lower() or "spark"
    llm_lane = str(settings.spark_introspection_llm_lane or "spark").strip().lower() or "spark"

    continuity_meta = candidate.spark_meta or {}
    session_id = (
        continuity_meta.get("session_id")
        or continuity_meta.get("conversation_id")
        or continuity_meta.get("thread_id")
    )
    if session_id is not None:
        session_id = str(session_id)
    workflow_id = continuity_meta.get("workflow_id")
    if workflow_id is not None:
        workflow_id = str(workflow_id)

    job = SparkIntrospectionJobV1(
        trace_id=tid,
        candidate_trace_id=tid,
        correlation_id=str(corr_u),
        source=str(candidate.source or "brain"),
        prompt=candidate.prompt,
        response=candidate.response,
        spark_meta=dict(candidate.spark_meta or {}),
        introspection=candidate.introspection,
        candidate_created_at=cand_iso,
        enqueued_at=now.isoformat(),
        session_id=session_id,
        message_id=None,
        workflow_id=workflow_id,
        execution_lane=lane,
        llm_lane=llm_lane,
        priority="low",
        allow_degrade=True,
        allow_chat_fallback=bool(settings.spark_introspection_allow_chat_fallback),
        max_tokens=int(settings.spark_introspection_max_tokens),
        timeout_sec=float(settings.spark_introspection_timeout_sec),
    )

    work: dict[str, Any] = {
        "job_id": job_id,
        "idempotency_key": build_idempotency_key(tid),
        "lane": "spark",
        "priority": "low",
        "attempt": 1,
        "max_attempts": int(settings.spark_introspection_queue_max_attempts),
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "producer": "spark-introspector",
        "reply_required": False,
        "reply_optional": True,
    }

    trace: dict[str, Any] = {
        "trace_id": tid,
        "candidate_trace_id": tid,
        "execution_lane": lane,
        "llm_lane": llm_lane,
        "work": work,
    }

    return BaseEnvelope(
        kind="spark.introspection.job.v1",
        source=source,
        correlation_id=corr_u,
        causality_chain=list(env.causality_chain),
        trace=trace,
        payload=job.model_dump(mode="json"),
    )


def extract_spark_introspection_job(env: BaseEnvelope) -> SparkIntrospectionJobV1:
    if env.kind != "spark.introspection.job.v1":
        raise ValueError(f"unexpected_job_kind:{env.kind}")
    raw = env.payload if isinstance(env.payload, dict) else {}
    if raw.get("schema_version") != "spark.introspection.job.v1":
        raise ValueError("invalid_spark_introspection_job_schema")
    return SparkIntrospectionJobV1.model_validate(raw)
