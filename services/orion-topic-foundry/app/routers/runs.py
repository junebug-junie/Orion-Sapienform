from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from psycopg2 import errors as pg_errors

from app.models import (
    EnrichmentSpec,
    ModelSpec,
    RunEnrichRequest,
    RunEnrichResponse,
    RunListPage,
    RunListResponse,
    RunRecord,
    RunListItem,
    RunCompareResponse,
    RunSummary,
    RunTrainRequest,
    RunTrainResponse,
    RunSpecSnapshot,
    WindowingSpec,
)
from app.services.data_access import InvalidSourceTableError, validate_dataset_columns, validate_dataset_source_table
from app.services.enrichment import enqueue_enrichment
from app.services.spec_hash import compute_spec_hash
from app.services.training import enqueue_training
from app.settings import settings
from app.storage.repository import (
    create_run,
    fetch_dataset,
    fetch_model,
    fetch_run,
    fetch_run_by_spec_hash,
    list_runs,
    list_runs_paginated,
    list_aspect_counts,
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
    try:
        validate_dataset_source_table(dataset)
        validate_dataset_columns(dataset)
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
    specs = RunSpecSnapshot(
        dataset=dataset,
        windowing=WindowingSpec(**model_row["windowing_spec"]),
        model=ModelSpec(**model_row["model_spec"]),
        enrichment=EnrichmentSpec(**model_row["enrichment_spec"]) if model_row.get("enrichment_spec") else EnrichmentSpec(),
        run_scope=payload.run_scope,
    )
    if specs.run_scope is None:
        specs.run_scope = "micro" if specs.windowing.windowing_mode.startswith("conversation") else "macro"
    if specs.windowing.windowing_mode.startswith("conversation") and not dataset.boundary_column:
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
    )
    existing = fetch_run_by_spec_hash(spec_hash)
    if existing:
        return RunTrainResponse(run_id=UUID(existing["run_id"]), status=existing["status"])
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
    return RunTrainResponse(run_id=run_id, status=run.status)


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


@router.post("/runs/{run_id}/enrich", response_model=RunEnrichResponse)
def enrich_run_endpoint(run_id: UUID, payload: RunEnrichRequest, background_tasks: BackgroundTasks) -> RunEnrichResponse:
    row = fetch_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    if payload.enricher == "llm" and not settings.topic_foundry_llm_enable:
        raise HTTPException(
            status_code=409,
            detail="LLM enrichment is disabled. Enable TOPIC_FOUNDRY_LLM_ENABLE to use llm enricher.",
        )
    enqueue_enrichment(
        background_tasks,
        run_id,
        force=payload.force,
        enricher=payload.enricher,
        limit=payload.limit,
        target=payload.target,
        fields=payload.fields,
        llm_backend=payload.llm_backend,
        prompt_template=payload.prompt_template,
    )
    stats = row.get("stats") or {}
    return RunEnrichResponse(
        run_id=run_id,
        status=row.get("status", "running"),
        enriched_count=stats.get("segments_enriched", 0),
        failed_count=stats.get("enrichment_failed", 0),
    )


@router.get("/runs/compare", response_model=RunCompareResponse)
def compare_runs(left_run_id: UUID, right_run_id: UUID):
    left = fetch_run(left_run_id)
    right = fetch_run(right_run_id)
    if not left or not right:
        raise HTTPException(status_code=404, detail="Run not found")
    left_stats = left.get("stats") or {}
    right_stats = right.get("stats") or {}
    diffs = {
        "docs_generated": (left_stats.get("docs_generated", 0) or 0) - (right_stats.get("docs_generated", 0) or 0),
        "segments_generated": (left_stats.get("segments_generated", 0) or 0) - (right_stats.get("segments_generated", 0) or 0),
        "cluster_count": (left_stats.get("cluster_count", 0) or 0) - (right_stats.get("cluster_count", 0) or 0),
        "outlier_pct": (left_stats.get("outlier_pct", 0.0) or 0.0) - (right_stats.get("outlier_pct", 0.0) or 0.0),
        "segments_enriched": (left_stats.get("segments_enriched", 0) or 0) - (right_stats.get("segments_enriched", 0) or 0),
    }
    left_aspects = {row["key"]: row["count"] for row in list_aspect_counts(left_run_id)}
    right_aspects = {row["key"]: row["count"] for row in list_aspect_counts(right_run_id)}
    aspect_keys = set(left_aspects) | set(right_aspects)
    aspect_diffs = []
    for key in aspect_keys:
        left_count = left_aspects.get(key, 0)
        right_count = right_aspects.get(key, 0)
        aspect_diffs.append(
            {
                "aspect": key,
                "left_count": left_count,
                "right_count": right_count,
                "delta": left_count - right_count,
            }
        )
    aspect_diffs.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return RunCompareResponse(
        left_run_id=left_run_id,
        right_run_id=right_run_id,
        left_stats=left_stats,
        right_stats=right_stats,
        diffs=diffs,
        aspect_diffs=aspect_diffs[:20],
    )
