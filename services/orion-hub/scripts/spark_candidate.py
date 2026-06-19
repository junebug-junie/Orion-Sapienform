from __future__ import annotations

from typing import Any, Dict, Optional

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from .settings import settings


def build_spark_introspect_candidate_envelope(
    *,
    trace_id: str,
    prompt: str,
    response: str,
    spark_meta: Optional[Dict[str, Any]] = None,
    source: str = "hub",
    correlation_id: str | None = None,
) -> BaseEnvelope:
    """Build a spark.candidate envelope for the introspector EKG / turn-effect UI."""
    candidate_payload = {
        "trace_id": trace_id,
        "source": source,
        "prompt": prompt,
        "response": response,
        "spark_meta": dict(spark_meta or {}),
    }
    return BaseEnvelope(
        kind="spark.candidate",
        correlation_id=correlation_id or trace_id,
        source=ServiceRef(name=settings.SERVICE_NAME, node=settings.NODE_NAME),
        payload=candidate_payload,
    )


async def publish_spark_introspect_candidate(
    bus,
    *,
    trace_id: str,
    prompt: str,
    response: str,
    spark_meta: Optional[Dict[str, Any]] = None,
    source: str = "hub",
    correlation_id: str | None = None,
) -> None:
    """Publish spark.candidate so spark-introspector refreshes tissue/phi on chat turns."""
    env = build_spark_introspect_candidate_envelope(
        trace_id=trace_id,
        prompt=prompt,
        response=response,
        spark_meta=spark_meta,
        source=source,
        correlation_id=correlation_id,
    )
    await bus.publish(settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE, env)
