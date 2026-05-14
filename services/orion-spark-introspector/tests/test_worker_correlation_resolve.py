from __future__ import annotations

from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app import worker


def _env() -> BaseEnvelope:
    return BaseEnvelope(
        kind="spark.candidate",
        source=ServiceRef(name="hub", node="n1"),
        correlation_id=uuid4(),
        payload={
            "trace_id": "corr-t1",
            "prompt": "p",
            "response": "r",
            "spark_meta": {"phi_before": {}},
        },
    )


def test_resolve_prefers_job_correlation_uuid_string() -> None:
    env = _env()
    u = uuid4()
    r = worker._resolve_cortex_correlation_uuid(correlation_id=str(u), source_env=env, trace_id="t1")
    assert r == u


def test_resolve_uses_env_when_job_correlation_invalid() -> None:
    env = _env()
    r = worker._resolve_cortex_correlation_uuid(correlation_id="not-a-uuid", source_env=env, trace_id="t1")
    assert r == env.correlation_id


def test_resolve_uuid5_fallback_when_env_correlation_unparseable() -> None:
    from unittest.mock import MagicMock

    m = MagicMock()
    m.correlation_id = "not-a-uuid-either"
    r = worker._resolve_cortex_correlation_uuid(correlation_id=None, source_env=m, trace_id="stable-tid")
    r2 = worker._resolve_cortex_correlation_uuid(correlation_id=None, source_env=m, trace_id="stable-tid")
    assert r == r2
