from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app.queue_jobs import (
    SparkIntrospectionJobV1,
    build_idempotency_key,
    build_spark_introspection_job_envelope,
    extract_spark_introspection_job,
)
from app.settings import Settings
from app.worker import SparkCandidatePayload


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_build_idempotency_key() -> None:
    assert build_idempotency_key("abc") == "spark:introspect:abc"


def test_build_and_extract_job_envelope(settings: Settings) -> None:
    cand = SparkCandidatePayload(
        trace_id="tid-1",
        source="brain",
        prompt="hello",
        response="world",
        spark_meta={"phi_before": {}, "session_id": "sess-9", "workflow_id": "wf-1"},
    )
    env = BaseEnvelope(
        kind="spark.candidate",
        source=ServiceRef(name="hub", node="n1"),
        correlation_id=uuid4(),
        payload=cand.model_dump(mode="json"),
    )
    src = ServiceRef(name="spark-introspector", node="athena", version="0.2.0")
    job_env = build_spark_introspection_job_envelope(cand, env, settings, src)
    assert job_env.kind == "spark.introspection.job.v1"
    assert job_env.source.name == "spark-introspector"
    job = extract_spark_introspection_job(job_env)
    assert isinstance(job, SparkIntrospectionJobV1)
    assert job.trace_id == "tid-1"
    assert job.candidate_trace_id == "tid-1"
    assert job.prompt == "hello"
    assert job.response == "world"
    assert job.spark_meta.get("session_id") == "sess-9"
    assert job.workflow_id == "wf-1"
    assert job.execution_lane == "spark"
    assert job.llm_lane == settings.spark_introspection_llm_lane
    assert job.allow_chat_fallback is False
    assert job.allow_degrade is True
    assert job.priority == "low"

    tr = job_env.trace or {}
    assert tr.get("trace_id") == "tid-1"
    work = tr.get("work") or {}
    assert work.get("idempotency_key") == "spark:introspect:tid-1"
    assert work.get("job_id")
    assert work.get("expires_at")
    exp = datetime.fromisoformat(str(work["expires_at"]).replace("Z", "+00:00"))
    assert exp.tzinfo == timezone.utc
