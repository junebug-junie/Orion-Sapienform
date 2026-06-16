from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query

from .cursor_reset import (
    parse_timestamp_at,
    record_cursor_reset,
    require_operator_token,
    validate_cursor_name,
    validate_mode,
)
from .grammar_truth import build_substrate_grammar_truth
from .quarantine_ack import (
    record_quarantine_ack,
    reducer_key_for_cursor,
    validate_cursor_name as validate_quarantine_cursor_name,
    require_operator_token as require_quarantine_operator_token,
)
from .settings import get_settings
from .store import GRAMMAR_CURSOR_REGISTRY
from .worker import BiometricsSubstrateWorker

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))

worker = BiometricsSubstrateWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="orion-substrate-runtime", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    snap = build_substrate_grammar_truth(worker._store)
    return {
        "status": "ok" if snap["ok"] else "degraded",
        "ok": snap["ok"],
        "degraded": snap["degraded"],
        "service": get_settings().service_name,
    }


@app.get("/grammar/truth")
async def grammar_truth() -> dict:
    return build_substrate_grammar_truth(worker._store)


@app.post("/grammar/cursor/reset")
async def reset_grammar_cursor(
    cursor_name: str = Query(...),
    mode: str = Query(...),
    at: str | None = Query(None, description="Timezone-aware ISO timestamp for mode=timestamp"),
    actor: Annotated[str, Depends(require_operator_token)] = "",
) -> dict:
    """Internal operator endpoint. Not exposed via hub/Caddy. Requires X-Orion-Operator-Token."""
    try:
        valid_name = validate_cursor_name(cursor_name)
        valid_mode = validate_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    at_timestamp = None
    if valid_mode == "timestamp":
        if not at:
            raise HTTPException(status_code=400, detail="timestamp mode requires at parameter")
        try:
            at_timestamp = parse_timestamp_at(at)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = worker._store.reset_grammar_cursor(
            cursor_name=valid_name,
            mode=valid_mode,
            at_timestamp=at_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_cursor_reset(
        cursor_name=valid_name,
        mode=valid_mode,
        requested_timestamp=at,
        prior_created_at=result.get("prior_created_at"),
        prior_event_id=result.get("prior_event_id"),
        new_created_at=result["new_created_at"],
        new_event_id=result["new_event_id"],
        actor=actor or "operator",
        history_may_be_skipped=bool(result.get("history_may_be_skipped")),
    )
    return result


@app.post("/grammar/quarantine/ack")
async def acknowledge_quarantine(
    cursor_name: str = Query(...),
    event_id: str | None = Query(None),
    ack_all: bool = Query(False, description="Acknowledge all unacked quarantine for cursor"),
    actor: Annotated[str, Depends(require_quarantine_operator_token)] = "",
) -> dict:
    """Internal operator endpoint. Acknowledges durable poison quarantine without erasing audit trail."""
    try:
        valid_name = validate_quarantine_cursor_name(cursor_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if ack_all and event_id:
        raise HTTPException(status_code=400, detail="ack_all cannot be combined with event_id")
    if not ack_all and not event_id:
        raise HTTPException(status_code=400, detail="event_id required unless ack_all=true")

    try:
        acknowledged_count = worker._store.acknowledge_quarantine(
            cursor_name=valid_name,
            event_id=event_id,
            ack_all=ack_all,
            actor=actor or "operator",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_quarantine_ack(
        cursor_name=valid_name,
        reducer_key=reducer_key_for_cursor(valid_name),
        event_id=event_id,
        ack_all=ack_all,
        actor=actor or "operator",
        acknowledged_count=acknowledged_count,
    )
    return {
        "cursor_name": valid_name,
        "event_id": event_id,
        "ack_all": ack_all,
        "acknowledged_count": acknowledged_count,
    }
