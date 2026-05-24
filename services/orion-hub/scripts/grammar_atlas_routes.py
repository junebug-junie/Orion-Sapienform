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

_engine = None
_sessionmaker = None


def _resolve_sql_writer_root() -> Path:
    """Locate orion-sql-writer for `app.models` imports used by orion.grammar.query."""
    repo_root = os.getenv("ORION_REPO_ROOT", "").strip()
    if repo_root:
        mounted = Path(repo_root) / "services" / "orion-sql-writer"
        if mounted.is_dir():
            return mounted

    # Dev layout: services/orion-hub/scripts/grammar_atlas_routes.py -> services/orion-sql-writer
    dev_root = Path(__file__).resolve().parents[2] / "orion-sql-writer"
    if dev_root.is_dir():
        return dev_root

    # Hub Docker image flattens to /app; repo is mounted at /repo in compose.
    docker_mounted = Path("/repo/services/orion-sql-writer")
    if docker_mounted.is_dir():
        return docker_mounted

    return dev_root


def _ensure_sql_writer_on_path() -> None:
    root_path = _resolve_sql_writer_root()
    if not root_path.is_dir():
        logger.warning("orion-sql-writer not found at %s; grammar atlas queries unavailable", root_path)
        return
    root = str(root_path)
    if root not in sys.path:
        sys.path.insert(0, root)


def _register_grammar_orm_models() -> None:
    """Load sql-writer grammar ORM tables while Hub's top-level ``app`` package is active."""
    import importlib.util
    import types

    from sqlalchemy.orm import declarative_base

    existing = sys.modules.get("app.models.grammar_trace")
    if existing is not None and hasattr(existing, "GrammarTraceSQL"):
        return

    writer_root = _resolve_sql_writer_root()
    if not writer_root.is_dir():
        raise HTTPException(status_code=503, detail="grammar_atlas_sql_writer_not_found")

    if "app.db" not in sys.modules or not hasattr(sys.modules.get("app.db"), "Base"):
        db_mod = types.ModuleType("app.db")
        db_mod.Base = declarative_base()
        sys.modules["app.db"] = db_mod
        import app as hub_app

        hub_app.db = db_mod

    models_pkg = sys.modules.get("app.models")
    if models_pkg is None:
        import app as hub_app

        models_pkg = types.ModuleType("app.models")
        sys.modules["app.models"] = models_pkg
        hub_app.models = models_pkg

    gt_path = writer_root / "app" / "models" / "grammar_trace.py"
    if not gt_path.is_file():
        raise HTTPException(status_code=503, detail="grammar_atlas_models_missing")

    spec = importlib.util.spec_from_file_location("app.models.grammar_trace", gt_path)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=503, detail="grammar_atlas_models_load_failed")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["app.models.grammar_trace"] = mod
    spec.loader.exec_module(mod)
    setattr(models_pkg, "grammar_trace", mod)


def _grammar_query():
    _ensure_sql_writer_on_path()
    _register_grammar_orm_models()
    try:
        from orion.grammar import query as grammar_query
    except ModuleNotFoundError as exc:
        logger.exception("grammar atlas query import failed")
        raise HTTPException(
            status_code=503,
            detail=f"grammar_atlas_query_unavailable: {exc}",
        ) from exc

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
