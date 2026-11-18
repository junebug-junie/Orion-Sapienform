from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentCreateResponseV1,
    SelfExperimentListResponseV1,
    SelfExperimentRecordV1,
)

from .experiment_registry import (
    ExperimentValidationError,
    compute_dedupe_key,
    normalize_create_request,
)
from .settings import settings
from .store import get_record, init_db, insert_record_dedupe_safe, list_records, update_record

logger = logging.getLogger("orion-self-experiments")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LegacyExperimentCreateRequest(BaseModel):
    skill_id: str = Field(min_length=1, max_length=120)
    provenance: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)


heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. This is the service's only bus usage: the old per-request
    dispatch to orion-context-exec was removed with that service (retired 2026-10-10). See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    level_name = (settings.log_level or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))
    init_db()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None
    try:
        yield
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            heartbeat_chassis = None


app = FastAPI(title="orion-self-experiments", version=settings.service_version, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
    }


def _build_record_from_spec(spec, *, status: str, reason: str | None) -> SelfExperimentRecordV1:
    now = _now_utc()
    dedupe_key = compute_dedupe_key(
        experiment_type=spec.experiment_type,
        question=spec.question,
        source=spec.source,
        source_ref=spec.source_ref,
    )
    return SelfExperimentRecordV1(
        experiment_id=spec.experiment_id,
        spec=spec,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        dedupe_key=dedupe_key,
        created_at_utc=now,
        updated_at_utc=now,
    )


@app.post("/v1/experiments", response_model=SelfExperimentCreateResponseV1)
def create_experiment(body: dict[str, Any]) -> SelfExperimentCreateResponseV1:
    if "skill_id" in body and "experiment_type" not in body and "question" not in body:
        legacy = LegacyExperimentCreateRequest.model_validate(body)
        req = SelfExperimentCreateRequestV1(
            skill_id=legacy.skill_id,
            provenance=legacy.provenance,
            args=legacy.args,
        )
    else:
        req = SelfExperimentCreateRequestV1.model_validate(body)

    experiment_id = str(uuid4())
    now = _now_utc()
    try:
        spec, _ = normalize_create_request(
            req,
            experiment_id=experiment_id,
            created_at_utc=now,
            allow_non_read_only=settings.experiments_allow_non_read_only,
        )
    except ExperimentValidationError as exc:
        logger.info("self_experiment_rejected reason=%s", exc.reason)
        return SelfExperimentCreateResponseV1(
            ok=False,
            experiment_id=experiment_id,
            status="rejected",
            message=exc.reason,
        )

    record = _build_record_from_spec(spec, status="validated", reason=None)
    stored, outcome = insert_record_dedupe_safe(record)
    if outcome == "dedupe_hit":
        logger.info(
            "self_experiment_dedupe_hit experiment_id=%s dedupe=%s",
            stored.experiment_id,
            stored.dedupe_key,
        )
        return SelfExperimentCreateResponseV1(
            ok=True,
            experiment_id=stored.experiment_id,
            status=stored.status,
            message="dedupe_hit",
        )

    logger.info(
        "self_experiment_created experiment_id=%s type=%s source=%s",
        stored.experiment_id,
        spec.experiment_type,
        spec.source,
    )
    return SelfExperimentCreateResponseV1(
        ok=True,
        experiment_id=stored.experiment_id,
        status=stored.status,
        message=None,
    )


@app.get("/v1/experiments/{experiment_id}", response_model=SelfExperimentRecordV1)
def get_experiment(experiment_id: str) -> SelfExperimentRecordV1:
    record = get_record(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    return record


@app.get("/v1/experiments", response_model=SelfExperimentListResponseV1)
def list_experiments(
    limit: int = Query(default=25, ge=1, le=200),
    status: str | None = Query(default=None),
    experiment_type: str | None = Query(default=None),
    source: str | None = Query(default=None),
    date: str | None = Query(default=None),
    correlation_id: str | None = Query(default=None),
    attention_required: bool | None = Query(default=None),
    skill_id: str | None = Query(default=None),
) -> SelfExperimentListResponseV1:
    items = list_records(
        limit=limit,
        status=status,
        experiment_type=experiment_type,
        source=source,
        date=date,
        correlation_id=correlation_id,
        attention_required=attention_required,
    )
    if skill_id:
        items = [item for item in items if item.spec.requested_skill_id == skill_id]
    return SelfExperimentListResponseV1(total=len(items), items=items)


@app.post("/v1/experiments/{experiment_id}/discard", response_model=SelfExperimentCreateResponseV1)
def discard_experiment(experiment_id: str) -> SelfExperimentCreateResponseV1:
    record = get_record(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    record.status = "discarded"
    record.updated_at_utc = _now_utc()
    update_record(record)
    return SelfExperimentCreateResponseV1(
        ok=True,
        experiment_id=experiment_id,
        status="discarded",
        message=None,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port)
