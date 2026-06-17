from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentCreateResponseV1,
    SelfExperimentDispatchResponseV1,
    SelfExperimentListResponseV1,
    SelfExperimentRecordV1,
)

from .context_exec_client import dispatch_context_exec
from .experiment_registry import (
    ExperimentValidationError,
    compile_experiment_to_context_exec_request,
    compute_dedupe_key,
    normalize_create_request,
    parse_context_exec_result,
    registry_config_for_type,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    level_name = (settings.log_level or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))
    init_db()
    yield


app = FastAPI(title="orion-self-experiments", version=settings.service_version, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "dispatch_enabled": settings.self_experiments_dispatch_enabled,
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


def _apply_context_exec_result(record: SelfExperimentRecordV1, run_payload: dict[str, Any]) -> SelfExperimentRecordV1:
    parsed = parse_context_exec_result(run_payload)
    record.context_exec_run_id = str(parsed.get("context_exec_run_id") or "")
    record.context_exec_status = str(parsed.get("status") or "")
    record.artifact_type = parsed.get("artifact_type")
    record.artifact_summary = parsed.get("operator_summary")
    record.artifact_payload = run_payload
    record.proposal_id = str(parsed["proposal_id"]) if parsed.get("proposal_id") else None
    record.proposal_status = str(parsed["ledger_status"]) if parsed.get("ledger_status") else None
    record.attention_required = bool(parsed.get("attention_required"))

    run_status = str(parsed.get("status") or "error")
    artifact_type = str(parsed.get("artifact_type") or "")
    if run_status != "ok":
        record.status = "failed"
        record.reason = f"context_exec_{run_status}"
    elif artifact_type == "ProposalEnvelopeV1":
        record.status = "proposal_stored"
        if record.attention_required:
            record.status = "pending_review"
    else:
        record.status = "completed"
    record.completed_at_utc = _now_utc()
    record.updated_at_utc = record.completed_at_utc
    return record


@app.post("/v1/experiments/{experiment_id}/dispatch", response_model=SelfExperimentDispatchResponseV1)
async def dispatch_experiment(experiment_id: str) -> SelfExperimentDispatchResponseV1:
    record = get_record(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")

    if record.status in ("rejected", "discarded", "expired", "completed", "proposal_stored", "pending_review"):
        raise HTTPException(status_code=409, detail=f"cannot_dispatch_status:{record.status}")

    if not settings.self_experiments_dispatch_enabled:
        record.status = "queued"
        record.reason = "dispatch_disabled"
        record.updated_at_utc = _now_utc()
        update_record(record)
        logger.info("self_experiment_queued dispatch_disabled experiment_id=%s", experiment_id)
        config = registry_config_for_type(record.spec.experiment_type)
        return SelfExperimentDispatchResponseV1(
            ok=True,
            experiment_id=experiment_id,
            status="queued",
            context_exec_mode=config["context_exec_mode"],
            expected_artifact_type=config["expected_artifact_type"],
            message="dispatch_disabled",
        )

    if record.dispatch_attempts >= settings.self_experiments_max_dispatch_attempts:
        raise HTTPException(status_code=409, detail="max_dispatch_attempts_exceeded")

    try:
        ctx_req = compile_experiment_to_context_exec_request(record)
    except ExperimentValidationError as exc:
        record.status = "rejected"
        record.reason = exc.reason
        record.updated_at_utc = _now_utc()
        update_record(record)
        raise HTTPException(status_code=400, detail=exc.reason) from exc

    record.status = "dispatching"
    record.dispatch_attempts += 1
    record.context_exec_request = ctx_req.model_dump(mode="json")
    record.updated_at_utc = _now_utc()
    update_record(record)
    logger.info(
        "self_experiment_dispatch_started experiment_id=%s mode=%s",
        experiment_id,
        ctx_req.mode,
    )

    record.status = "running"
    record.updated_at_utc = _now_utc()
    update_record(record)

    try:
        run = await dispatch_context_exec(ctx_req)
        record = _apply_context_exec_result(record, run.model_dump(mode="json"))
        logger.info(
            "self_experiment_context_exec_result_received experiment_id=%s status=%s",
            experiment_id,
            record.status,
        )
    except Exception as exc:
        record.status = "failed"
        record.reason = f"dispatch_error:{exc.__class__.__name__}"
        record.updated_at_utc = _now_utc()
        logger.exception("self_experiment_failed experiment_id=%s", experiment_id)

    update_record(record)
    config = registry_config_for_type(record.spec.experiment_type)
    return SelfExperimentDispatchResponseV1(
        ok=record.status in ("completed", "proposal_stored", "pending_review"),
        experiment_id=experiment_id,
        status=record.status,
        context_exec_mode=config["context_exec_mode"],
        expected_artifact_type=config["expected_artifact_type"],
        message=record.reason,
    )


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


@app.post("/v1/experiments/{experiment_id}/retry", response_model=SelfExperimentDispatchResponseV1)
async def retry_experiment(experiment_id: str) -> SelfExperimentDispatchResponseV1:
    record = get_record(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    if record.status not in ("failed", "queued") and not (
        record.status == "validated" and record.reason == "dispatch_disabled"
    ):
        raise HTTPException(status_code=409, detail=f"cannot_retry_status:{record.status}")
    if record.status == "failed":
        record.status = "validated"
        record.reason = None
        record.updated_at_utc = _now_utc()
        update_record(record)
    return await dispatch_experiment(experiment_id)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port)
