from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Query

from app.models import DriftListResponse, DriftRunRequest, DriftRunResponse
from app.services.drift import fetch_drift_records, run_drift_check


logger = logging.getLogger("topic-foundry.drift")

router = APIRouter()


@router.post("/drift/run", response_model=DriftRunResponse)
def run_drift_endpoint(payload: DriftRunRequest) -> DriftRunResponse:
    try:
        drift_id, status = run_drift_check(
            model_name=payload.model_name,
            window_days=payload.window_days,
            window_hours=payload.window_hours,
            threshold_js=payload.threshold_js,
            threshold_outlier=payload.threshold_outlier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Drift run failed model=%s", payload.model_name)
        raise HTTPException(status_code=500, detail="Drift run failed") from exc
    return DriftRunResponse(drift_id=drift_id, status=status)


@router.get("/drift", response_model=DriftListResponse)
def list_drift_endpoint(model_name: str, limit: int = Query(default=50, ge=1, le=200)) -> DriftListResponse:
    records = fetch_drift_records(model_name, limit=limit)
    return DriftListResponse(model_name=model_name, records=records)
