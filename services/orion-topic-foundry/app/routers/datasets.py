from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from uuid import uuid4

from app.models import (
    DatasetCreateRequest,
    DatasetCreateResponse,
    DatasetListResponse,
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DatasetSpec,
)
from app.services.preview import preview_dataset
from app.storage.repository import create_dataset, fetch_dataset, list_datasets, utc_now


logger = logging.getLogger("topic-foundry.datasets")

router = APIRouter()


@router.post("/datasets", response_model=DatasetCreateResponse)
def create_dataset_endpoint(payload: DatasetCreateRequest) -> DatasetCreateResponse:
    dataset_id = uuid4()
    created_at = utc_now()
    dataset = DatasetSpec(
        dataset_id=dataset_id,
        name=payload.name,
        source_table=payload.source_table,
        id_column=payload.id_column,
        time_column=payload.time_column,
        text_columns=payload.text_columns,
        where_sql=payload.where_sql,
        where_params=payload.where_params,
        timezone=payload.timezone,
        created_at=created_at,
    )
    create_dataset(dataset)
    return DatasetCreateResponse(dataset_id=dataset_id, created_at=created_at)


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets_endpoint() -> DatasetListResponse:
    datasets = list_datasets()
    return DatasetListResponse(datasets=datasets)


@router.post("/datasets/preview", response_model=DatasetPreviewResponse)
def preview_dataset_endpoint(payload: DatasetPreviewRequest) -> DatasetPreviewResponse:
    dataset_spec = payload.dataset
    if dataset_spec is None and payload.dataset_id:
        dataset_spec = fetch_dataset(payload.dataset_id)
    if dataset_spec is None:
        raise HTTPException(status_code=422, detail="dataset or dataset_id required")
    windowing_spec = payload.windowing or payload.windowing_spec
    if windowing_spec is None:
        raise HTTPException(status_code=422, detail="windowing or windowing_spec required")
    resolved = DatasetPreviewRequest(
        dataset=dataset_spec,
        windowing=windowing_spec,
        start_at=payload.start_at,
        end_at=payload.end_at,
        limit=payload.limit,
    )
    result = preview_dataset(resolved)
    return result
