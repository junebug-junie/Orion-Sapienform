from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from psycopg2 import OperationalError
from psycopg2 import errors as pg_errors
from uuid import UUID, uuid4

from app.models import (
    DatasetCreateRequest,
    DatasetCreateResponse,
    DatasetListResponse,
    DatasetPreviewRequest,
    DatasetPreviewResponse,
    DatasetSpec,
    DatasetUpdateRequest,
    WindowingSpec,
)
from app.services.data_access import InvalidSourceTableError, validate_dataset_columns, validate_dataset_source_table
from app.services.preview import preview_dataset
from app.storage.repository import create_dataset, fetch_dataset, list_datasets, update_dataset, utc_now


logger = logging.getLogger("topic-foundry.datasets")

router = APIRouter()


@router.post("/datasets", response_model=DatasetCreateResponse)
def create_dataset_endpoint(payload: DatasetCreateRequest) -> DatasetCreateResponse:
    logger.info(
        "Create dataset request received",
        extra={
            "payload_keys": sorted(payload.model_dump(exclude_none=True).keys()),
            "source_table": payload.source_table,
        },
    )
    if payload.boundary_column and not payload.boundary_strategy:
        payload.boundary_strategy = "column"
    dataset_id = uuid4()
    created_at = utc_now()
    dataset = DatasetSpec(
        dataset_id=dataset_id,
        name=payload.name,
        source_table=payload.source_table,
        id_column=payload.id_column,
        time_column=payload.time_column,
        text_columns=payload.text_columns,
        timezone=(payload.timezone or "UTC"),
        boundary_column=payload.boundary_column,
        boundary_strategy=payload.boundary_strategy,
        created_at=created_at,
    )
    try:
        validate_dataset_source_table(dataset)
        validate_dataset_columns(dataset)
    except (InvalidSourceTableError, ValueError) as exc:
        detail = {"ok": False, "error": "invalid_source_table", "detail": str(exc) or "Invalid source_table"}
        logger.warning("Create dataset failed due to invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except (pg_errors.UndefinedTable, pg_errors.InvalidSchemaName, pg_errors.InvalidName) as exc:
        detail = {"ok": False, "error": "invalid_source_table", "detail": str(exc) or "Invalid source_table"}
        logger.warning("Create dataset failed due to missing/invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except OperationalError as exc:
        detail = {"ok": False, "error": "db_unavailable", "detail": "Topic Foundry database unavailable"}
        logger.exception("Create dataset failed due to database connectivity error")
        raise HTTPException(status_code=503, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Create dataset failed unexpectedly")
        raise HTTPException(status_code=500, detail="Create dataset failed") from exc
    create_dataset(dataset)
    return DatasetCreateResponse(dataset_id=dataset_id, created_at=created_at)


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets_endpoint() -> DatasetListResponse:
    datasets = list_datasets()
    return DatasetListResponse(datasets=datasets)


@router.patch("/datasets/{dataset_id}", response_model=DatasetSpec)
def update_dataset_endpoint(dataset_id: UUID, payload: DatasetUpdateRequest) -> DatasetSpec:
    dataset = fetch_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    updates = payload.model_dump(exclude_unset=True)
    if "boundary_column" in updates and updates.get("boundary_column") and not updates.get("boundary_strategy"):
        updates["boundary_strategy"] = "column"
    if "boundary_column" in updates and not updates.get("boundary_column"):
        updates["boundary_strategy"] = None
    updated = dataset.copy(update=updates)
    try:
        validate_dataset_source_table(updated)
        validate_dataset_columns(updated)
    except (InvalidSourceTableError, ValueError) as exc:
        detail = {"ok": False, "error": "invalid_source_table", "detail": str(exc) or "Invalid source_table"}
        logger.warning("Update dataset failed due to invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except (pg_errors.UndefinedTable, pg_errors.InvalidSchemaName, pg_errors.InvalidName) as exc:
        detail = {"ok": False, "error": "invalid_source_table", "detail": str(exc) or "Invalid source_table"}
        logger.warning("Update dataset failed due to missing/invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except OperationalError as exc:
        detail = {"ok": False, "error": "db_unavailable", "detail": "Topic Foundry database unavailable"}
        logger.exception("Update dataset failed due to database connectivity error")
        raise HTTPException(status_code=503, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Update dataset failed unexpectedly")
        raise HTTPException(status_code=500, detail="Update dataset failed") from exc
    update_dataset(updated)
    return updated


@router.post("/datasets/preview", response_model=DatasetPreviewResponse)
def preview_dataset_endpoint(payload: DatasetPreviewRequest) -> DatasetPreviewResponse:
    dataset_spec = payload.dataset
    if dataset_spec is None and payload.dataset_id:
        dataset_spec = fetch_dataset(payload.dataset_id)
    if dataset_spec is None:
        raise HTTPException(status_code=422, detail="dataset or dataset_id required")
    windowing_spec = payload.windowing or payload.windowing_spec or WindowingSpec()
    effective_boundary = windowing_spec.boundary_column or dataset_spec.boundary_column
    dataset_for_preview = dataset_spec
    if effective_boundary and dataset_spec.boundary_column != effective_boundary:
        dataset_for_preview = dataset_spec.copy(update={"boundary_column": effective_boundary})
    resolved = DatasetPreviewRequest(
        dataset=dataset_for_preview,
        windowing=windowing_spec,
        start_at=payload.start_at,
        end_at=payload.end_at,
        limit=payload.limit,
    )
    try:
        validate_dataset_source_table(dataset_for_preview)
        validate_dataset_columns(dataset_for_preview)
        if windowing_spec.windowing_mode.startswith("conversation") and not effective_boundary:
            detail = {
                "ok": False,
                "error": "invalid_windowing",
                "detail": f"boundary_column required for windowing_mode={windowing_spec.windowing_mode} dataset_id={dataset_spec.dataset_id}",
            }
            raise HTTPException(status_code=400, detail=detail)
        result = preview_dataset(resolved)
        return result
    except (InvalidSourceTableError, ValueError) as exc:
        detail = {
            "ok": False,
            "error": "invalid_request",
            "detail": str(exc) or "Invalid request",
        }
        logger.warning("Preview failed due to invalid request", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except (pg_errors.UndefinedTable, pg_errors.InvalidSchemaName, pg_errors.InvalidName) as exc:
        detail = {
            "ok": False,
            "error": "invalid_source_table",
            "detail": str(exc) or "Invalid source_table",
        }
        logger.warning("Preview failed due to missing/invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preview failed unexpectedly")
        raise HTTPException(status_code=500, detail="Preview failed") from exc
