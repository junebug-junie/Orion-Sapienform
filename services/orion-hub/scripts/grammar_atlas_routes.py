"""Substrate Grammar Atlas read API (trace/graph introspection)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("orion-hub.grammar-atlas")

router = APIRouter(prefix="/api/substrate/atlas", tags=["substrate-atlas"])

_SQL_WRITER_ROOT = Path(__file__).resolve().parents[2] / "orion-sql-writer"
_engine = None
_sessionmaker = None


def _ensure_sql_writer_on_path() -> None:
    if not _SQL_WRITER_ROOT.is_dir():
        return
    root = str(_SQL_WRITER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _grammar_query():
    _ensure_sql_writer_on_path()
    from orion.grammar import query as grammar_query

    return grammar_query


def _postgres_uri() -> str:
    from app.settings import get_settings

    settings = get_settings()
    return (settings.GRAMMAR_ATLAS_POSTGRES_URI or os.getenv("DATABASE_URL", "")).strip()


def _session_factory():
    global _engine, _sessionmaker
    uri = os.getenv("GRAMMAR_ATLAS_POSTGRES_URI") or os.getenv("DATABASE_URL", "") or _postgres_uri()
    if _sessionmaker is None:
        _engine = create_engine(uri, pool_pre_ping=True)
        _sessionmaker = sessionmaker(bind=_engine)
    return _sessionmaker()


async def _with_session(fn: Callable[[Any], Any]) -> Any:
    def run() -> Any:
        sess = _session_factory()
        try:
            return fn(sess)
        finally:
            sess.close()

    return await asyncio.to_thread(run)


def _require_atlas_available() -> None:
    from app.settings import get_settings

    settings = get_settings()
    if not settings.GRAMMAR_ATLAS_ENABLED:
        raise HTTPException(status_code=503, detail="grammar_atlas_disabled")
    if not _postgres_uri():
        raise HTTPException(status_code=503, detail="grammar_atlas_database_unconfigured")


@router.get("/traces")
async def list_traces_api(
    session_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    items = await _with_session(lambda sess: q.list_traces(sess, session_id=session_id, limit=limit))
    return {"items": items}


@router.get("/traces/{trace_id}")
async def get_trace_api(trace_id: str) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    payload = await _with_session(lambda sess: q.get_trace(sess, trace_id))
    if payload is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return payload


@router.get("/traces/{trace_id}/graph")
async def get_trace_graph_api(
    trace_id: str,
    layout: str = Query("layered"),
    depth: int = Query(2, ge=0, le=32),
) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    payload = await _with_session(
        lambda sess: q.get_trace_graph(sess, trace_id, layout=layout, depth=depth)
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return payload


@router.get("/atoms/{atom_id}/neighborhood")
async def atom_neighborhood_api(
    atom_id: str,
    depth: int = Query(2, ge=0, le=32),
    direction: str = Query("both"),
) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    payload = await _with_session(
        lambda sess: q.get_atom_neighborhood(sess, atom_id, depth=depth, direction=direction)
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="atom_not_found")
    return payload


@router.get("/atoms/{atom_id}/provenance")
async def atom_provenance_api(atom_id: str) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    payload = await _with_session(lambda sess: q.get_atom_provenance(sess, atom_id))
    if payload is None:
        raise HTTPException(status_code=404, detail="atom_not_found")
    return payload


@router.get("/atoms/{atom_id}/temporal-path")
async def atom_temporal_path_api(
    atom_id: str,
    direction: str = Query("backward"),
    limit: int = Query(25, ge=1, le=500),
) -> dict[str, Any]:
    _require_atlas_available()
    q = _grammar_query()
    payload = await _with_session(
        lambda sess: q.get_temporal_path(sess, atom_id, direction=direction, limit=limit)
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="atom_not_found")
    return payload
