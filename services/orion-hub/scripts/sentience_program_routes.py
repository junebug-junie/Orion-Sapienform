"""Sentience Striving Program board -- operator view of the program's instruments.

Renders, per instrument: what it is doing now, how far its history actually goes
back, what bounds that history, what it affects, which outcome (O1-O4) it ladders
to, and whether the claims the program recorded against it still hold.

Read-only. The join logic lives in `orion.sentience_striving_program.instruments`
and is shared verbatim with `scripts/check_sentience_instruments.py`, so the page
and the CI gate cannot disagree about the program's state -- which is the whole
point of having a board at all.

Postgres access deliberately uses SQLAlchemy's `raw_connection()` rather than
`orion.metrics.liveness.open_readonly_connection`: that helper imports `psycopg`,
which is NOT installed in this container (verified live 2026-09-02 -- psycopg2,
asyncpg and sqlalchemy are present, psycopg is not). The reducer only ever calls
`.cursor()`, so any DB-API connection satisfies it.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_ENGINE = None


def _engine():
    """Lazily build a read-only-intent engine, mirroring attention_organ_routes."""
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine

        uri = (
            os.getenv("POSTGRES_URI")
            or os.getenv("DATABASE_URL")
            or os.getenv("HUB_POSTGRES_URI")
        )
        if not uri:
            return None
        _ENGINE = create_engine(uri, pool_pre_ping=True)
    return _ENGINE


def _state(with_consumers: bool):
    """Build board state, degrading to manifest-only if the database is down.

    A database outage must NOT blank this page. The board's job is to say what
    the program knows and how it knows it; with no connection it can still
    render every instrument, its code presence, its retention ceiling and its
    unlock narrative, and mark the SQL-backed claims ERROR -- which
    `evaluate_claim` already does for a None connection.

    Returning 500 here instead would remove the operator's only view of the
    program at exactly the moment something is wrong -- the same shape as the
    curiosity-graph deadlock in README.md section 15c, where a "graceful"
    degradation removed the one mechanism that could have ended it.
    """
    from orion.sentience_striving_program.instruments import build_state, load_manifest

    manifest = load_manifest()
    conn = None
    db_error = ""
    try:
        engine = _engine()
        if engine is None:
            db_error = "no POSTGRES_URI/DATABASE_URL configured"
        else:
            conn = engine.raw_connection()
    except Exception as exc:  # noqa: BLE001
        db_error = f"{type(exc).__name__}: {exc}"
        logger.warning("sentience_program_db_unavailable", exc_info=True)

    try:
        states = build_state(manifest, conn=conn, with_consumers=with_consumers)
    finally:
        if conn is not None:
            conn.close()
    return manifest, states, db_error


@router.get("/api/sentience-program")
async def sentience_program_state(consumers: bool = False) -> JSONResponse:
    """Machine-readable board state.

    `consumers=true` resolves blast radius through the metric semantic layer.
    Off by default: that scan walks ~4,300 source files and takes tens of
    seconds, which is fine for a CLI gate and far too slow for a page load.
    """
    try:
        manifest, states, db_error = _state(with_consumers=consumers)
    except Exception as exc:  # noqa: BLE001
        # Only a manifest-level failure reaches here -- a database outage is
        # handled above and still renders. This is an authoring error (bad YAML,
        # unknown outcome id) and should be loud.
        logger.exception("sentience_program_state_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "outcomes": manifest["outcomes"],
            "review_max_age_days": manifest["review_max_age_days"],
            "consumers_resolved": consumers,
            "db_error": db_error,
            "instruments": [
                {
                    "id": s.instrument.id,
                    "title": s.instrument.title,
                    "theory": s.instrument.theory,
                    "outcome": s.instrument.outcome,
                    "program_ref": s.instrument.program_ref,
                    "module": s.instrument.module,
                    "entrypoint": s.instrument.entrypoint,
                    "module_exists": s.module_exists,
                    "entrypoint_exists": s.entrypoint_exists,
                    "unlock": s.instrument.unlock,
                    "last_reviewed": s.instrument.last_reviewed,
                    "review_age_days": s.review_age_days,
                    "review_stale": s.review_stale,
                    "storage_kind": s.instrument.storage.kind,
                    "table": s.instrument.storage.table,
                    "row_count": s.row_count,
                    "history_hours": s.history_hours,
                    "retention_hours": s.retention_hours,
                    "retention_source": s.retention_source,
                    "retention_setting": s.instrument.storage.retention_setting,
                    "storage_note": s.storage_note,
                    "last_seen": s.last_seen.isoformat() if s.last_seen else None,
                    "first_seen": s.first_seen.isoformat() if s.first_seen else None,
                    "consumers": s.consumers,
                    "consumer_note": s.consumer_note,
                    "claims": [
                        {
                            "id": c.claim.id,
                            "question": c.claim.question,
                            "status": c.status,
                            "recorded": c.claim.recorded,
                            "observed": c.observed,
                            "recorded_at": c.claim.recorded_at,
                            "blocks": c.claim.blocks,
                            "detail": c.detail,
                            "note": c.claim.note,
                        }
                        for c in s.claims
                    ],
                }
                for s in states
            ],
        }
    )


@router.get("/sentience-program")
async def sentience_program_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template_path = TEMPLATES_DIR / "sentience_program.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="sentience_program_template_missing")
    rendered = template_path.read_text(encoding="utf-8").replace(
        "{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()
    )
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
