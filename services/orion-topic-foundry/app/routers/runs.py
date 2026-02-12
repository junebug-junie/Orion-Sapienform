from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from psycopg2 import errors as pg_errors

from app.models import (
    EnrichmentSpec,
    ModelSpec,
    RunListPage,
    RunListResponse,
    RunRecord,
    RunListItem,
    RunSummary,
    RunTrainRequest,
    RunTrainResponse,
    RunSpecSnapshot,
    WindowingSpec,
)
from app.services.data_access import InvalidSourceTableError, validate_dataset_columns, validate_dataset_source_table
from app.services.spec_hash import compute_spec_hash
from app.services.training import enqueue_training
from app.storage.repository import (
    create_run,
    fetch_dataset,
    fetch_model,
    fetch_run,
    fetch_run_by_spec_hash,
    list_runs,
    list_runs_paginated,
    utc_now,
)


logger = logging.getLogger("topic-foundry.runs")

router = APIRouter()


@router.post("/runs/train", response_model=RunTrainResponse)
def train_run_endpoint(payload: RunTrainRequest, background_tasks: BackgroundTasks) -> RunTrainResponse:
    model_row = fetch_model(payload.model_id)
    if not model_row:
        raise HTTPException(status_code=404, detail="Model not found")
    dataset = fetch_dataset(payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    windowing_spec = payload.windowing_spec or WindowingSpec(**model_row["windowing_spec"])
    effective_boundary = windowing_spec.boundary_column or dataset.boundary_column
    dataset_for_validation = dataset
    if effective_boundary and dataset.boundary_column != effective_boundary:
        dataset_for_validation = dataset.copy(update={"boundary_column": effective_boundary})
    try:
        validate_dataset_source_table(dataset_for_validation)
        validate_dataset_columns(dataset_for_validation)
    except (InvalidSourceTableError, ValueError) as exc:
        detail = {
            "ok": False,
            "error": "invalid_source_table",
            "detail": str(exc) or "Invalid source_table",
        }
        logger.warning("Training failed due to invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except (pg_errors.UndefinedTable, pg_errors.InvalidSchemaName, pg_errors.InvalidName) as exc:
        detail = {
            "ok": False,
            "error": "invalid_source_table",
            "detail": str(exc) or "Invalid source_table",
        }
        logger.warning("Training failed due to missing/invalid source_table", exc_info=True)
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training validation failed unexpectedly")
        raise HTTPException(status_code=500, detail="Training validation failed") from exc

    run_id = uuid4()
    created_at = utc_now()
    dataset_for_spec = dataset
    if effective_boundary and dataset.boundary_column != effective_boundary:
        dataset_for_spec = dataset.copy(update={"boundary_column": effective_boundary})
    specs = RunSpecSnapshot(
        dataset=dataset_for_spec,
        windowing=windowing_spec,
        model=ModelSpec(**model_row["model_spec"]),
        enrichment=EnrichmentSpec(**model_row["enrichment_spec"]) if model_row.get("enrichment_spec") else EnrichmentSpec(),
        run_scope=payload.run_scope,
    )
    if specs.run_scope is None:
        specs.run_scope = "micro" if specs.windowing.windowing_mode.startswith("conversation") else "macro"
    if specs.windowing.windowing_mode.startswith("conversation") and not effective_boundary:
        detail = {
            "ok": False,
            "error": "invalid_windowing",
            "detail": f"boundary_column required for windowing_mode={specs.windowing.windowing_mode} dataset_id={dataset.dataset_id}",
        }
        raise HTTPException(status_code=400, detail=detail)
    spec_hash = compute_spec_hash(
        dataset_id=payload.dataset_id,
        model_id=payload.model_id,
        start_at=payload.start_at,
        end_at=payload.end_at,
        windowing_spec=specs.windowing,
        model_spec=specs.model,
        enrichment_spec=specs.enrichment,
        run_scope=specs.run_scope,
        topic_mode=payload.topic_mode,
        topic_mode_params=payload.topic_mode_params,
    )
    existing = fetch_run_by_spec_hash(spec_hash)
    if existing:
        return RunTrainResponse(run_id=UUID(existing["run_id"]), status=existing["status"], topic_mode=payload.topic_mode, topic_mode_params=payload.topic_mode_params)
    run = RunRecord(
        run_id=run_id,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        specs=specs,
        spec_hash=spec_hash,
        status="queued",
        stage="training",
        run_scope=payload.run_scope,
        stats={},
        artifact_paths={},
        created_at=created_at,
    )
    create_run(run)
    enqueue_training(background_tasks, run_id, payload, model_row, dataset, spec_hash)
    return RunTrainResponse(run_id=run_id, status=run.status, topic_mode=payload.topic_mode, topic_mode_params=payload.topic_mode_params, model_meta_used=(model_row.get("model_meta") or {}))


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: UUID):
    row = fetch_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


@router.get("/runs")
def list_runs_endpoint(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    format: str | None = Query(default=None),
):
    use_wrapped = format == "wrapped" or offset > 0 or status or stage or model_name
    if not use_wrapped:
        rows = list_runs(limit=limit)
        runs = [
            RunSummary(
                run_id=UUID(row["run_id"]),
                model_id=UUID(row["model_id"]),
                dataset_id=UUID(row["dataset_id"]),
                status=row["status"],
                stage=row.get("stage"),
                created_at=row["created_at"],
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at"),
            )
            for row in rows
        ]
        return RunListResponse(runs=runs)
    rows, total = list_runs_paginated(
        limit=limit,
        offset=offset,
        status=status,
        stage=stage,
        model_name=model_name,
    )
    items = []
    for row in rows:
        specs = row.get("specs") or {}
        window = {
            "start_at": specs.get("start_at"),
            "end_at": specs.get("end_at"),
        }
        stats = row.get("stats") or {}
        items.append(
            RunListItem(
                run_id=UUID(row["run_id"]),
                status=row["status"],
                stage=row.get("stage"),
                created_at=row["created_at"],
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at"),
                model={
                    "model_id": UUID(row["model_id"]),
                    "name": row.get("model_name"),
                    "version": row.get("model_version"),
                    "stage": row.get("model_stage"),
                },
                dataset={
                    "dataset_id": UUID(row["dataset_id"]),
                    "name": row.get("dataset_name"),
                    "source_table": row.get("dataset_source_table"),
                },
                window=window,
                stats_summary={
                    "docs_generated": stats.get("docs_generated"),
                    "segments_generated": stats.get("segments_generated"),
                    "cluster_count": stats.get("cluster_count"),
                    "outlier_pct": stats.get("outlier_pct"),
                    "segments_enriched": stats.get("segments_enriched"),
                },
            )
        )
    return RunListPage(items=items, limit=limit, offset=offset, total=total)
