from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException

from app.models import (
    ModelCreateRequest,
    ModelCreateResponse,
    ModelListResponse,
    ModelPromoteRequest,
    ModelPromoteResponse,
    ModelSummary,
    ModelVersionEntry,
    ModelVersionsResponse,
)
from app.services.metrics import normalize_metric, validate_metric
from app.settings import settings
from app.storage.repository import (
    create_model,
    fetch_active_model_by_name,
    fetch_dataset,
    fetch_model,
    fetch_model_versions,
    list_models,
    promote_model_stage,
    utc_now,
)


logger = logging.getLogger("topic-foundry.models")

router = APIRouter()


@router.post("/models", response_model=ModelCreateResponse)
def create_model_endpoint(payload: ModelCreateRequest) -> ModelCreateResponse:
    dataset = fetch_dataset(payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        metric = normalize_metric(payload.model_spec.metric)
        validate_metric(metric)
        payload.model_spec.metric = metric
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not payload.model_spec.embedding_source_url:
        payload.model_spec.embedding_source_url = settings.topic_foundry_embedding_url
    model_id = uuid4()
    created_at = utc_now()
    create_model(model_id, payload, created_at)
    return ModelCreateResponse(model_id=model_id, created_at=created_at)


@router.get("/models", response_model=ModelListResponse)
def list_models_endpoint() -> ModelListResponse:
    rows = list_models()
    models = [
        ModelSummary(
            model_id=UUID(row["model_id"]),
            name=row["name"],
            version=row["version"],
            stage=row.get("stage"),
            dataset_id=UUID(row["dataset_id"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return ModelListResponse(models=models)


@router.get("/models/{name}/versions", response_model=ModelVersionsResponse)
def list_model_versions(name: str) -> ModelVersionsResponse:
    rows = fetch_model_versions(name)
    versions = [
        ModelVersionEntry(
            model_id=UUID(row["model_id"]),
            name=row["name"],
            version=row["version"],
            stage=row.get("stage"),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return ModelVersionsResponse(name=name, versions=versions)


@router.get("/models/{name}/active", response_model=ModelSummary)
def get_active_model(name: str) -> ModelSummary:
    row = fetch_active_model_by_name(name)
    if not row:
        raise HTTPException(status_code=404, detail="Active model not found")
    return ModelSummary(
        model_id=UUID(row["model_id"]),
        name=row["name"],
        version=row["version"],
        stage=row.get("stage"),
        dataset_id=UUID(row["dataset_id"]),
        created_at=row["created_at"],
    )


@router.post("/models/{model_id}/promote", response_model=ModelPromoteResponse)
def promote_model(model_id: UUID, payload: ModelPromoteRequest) -> ModelPromoteResponse:
    model = fetch_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    updated = promote_model_stage(model_id=model_id, to_stage=payload.stage, reason=payload.reason)
    if not updated:
        raise HTTPException(status_code=404, detail="Model not found")
    summary = ModelSummary(
        model_id=UUID(updated["model_id"]),
        name=updated["name"],
        version=updated["version"],
        stage=updated.get("stage"),
        dataset_id=UUID(updated["dataset_id"]),
        created_at=updated["created_at"],
    )
    return ModelPromoteResponse(model=summary)
