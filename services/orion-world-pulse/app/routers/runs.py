from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models import RunRequest
from app.state import RUN_RESULTS
from app.services.emit_runtime import emit_sql_runtime
from app.services.pipeline import run_world_pulse
from app.services.publish_hub import publish_hub_message
from app.services.renderers import render_hub_digest
from app.settings import settings

router = APIRouter()


@router.get("/api/world-pulse/latest")
def world_pulse_latest():
    if not RUN_RESULTS:
        raise HTTPException(status_code=404, detail="No runs")
    latest = sorted(RUN_RESULTS.values(), key=lambda x: x.run.started_at)[-1]
    return latest.model_dump(mode="json")


@router.get("/api/world-pulse/runs")
def world_pulse_runs():
    return [v.run.model_dump(mode="json") for v in sorted(RUN_RESULTS.values(), key=lambda x: x.run.started_at, reverse=True)]


@router.get("/api/world-pulse/runs/{run_id}")
def world_pulse_run_detail(run_id: str):
    result = RUN_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result.model_dump(mode="json")


@router.post("/api/world-pulse/run")
def world_pulse_run(payload: RunRequest):
    fixture_items = None
    if payload.fixtures:
        fixture_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rss_items.json"
        if not fixture_path.exists():
            raise HTTPException(status_code=500, detail="Fixture data unavailable")
        fixture_items = json.loads(fixture_path.read_text(encoding="utf-8"))
        now_iso = datetime.now(timezone.utc).isoformat()
        for row in fixture_items:
            row["published_at"] = row.get("published_at") or now_iso
    result = run_world_pulse(
        run_id=None,
        date=payload.date,
        requested_by=payload.requested_by,
        dry_run=payload.dry_run,
        fixture_items=fixture_items,
    )
    sql_emit = emit_sql_runtime(result)
    result.run.sql_emit_status = str(sql_emit.get("status", "failed"))
    result.publish_status["sql_emit"] = sql_emit
    if result.digest and settings.world_pulse_hub_messages_enabled and not result.run.dry_run:
        hub_message = render_hub_digest(result.digest)
        hub_result = publish_hub_message(message=hub_message, dry_run=result.run.dry_run)
        result.run.hub_publish_status = str(hub_result.get("status", "failed"))
        result.publish_status["hub_publish"] = hub_result
    RUN_RESULTS[result.run.run_id] = result
    return result.model_dump(mode="json")
