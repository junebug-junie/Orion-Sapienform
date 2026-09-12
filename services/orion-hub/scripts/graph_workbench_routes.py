"""Hub-native graph launcher, read-only exports, and local Gephi Lite assets."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .settings import settings
from .graph_workbench import (
    MAX_NODES, SOURCE_LABELS, FalkorReader, caption,
    crystal_search, crystal_snapshot, graph_search, graph_snapshot,
)

logger = logging.getLogger("orion-hub.graph_workbench")
SERVICE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = SERVICE_ROOT / "gephi-lite"
Source = Literal["worldview", "substrate", "crystallizations"]
PRIVATE_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
# All graph processing stays in the browser and on Hub. External GitHub/sample
# integrations are intentionally unavailable for this private-memory deployment.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "font-src 'self' data:; connect-src 'self' blob:; worker-src 'self' blob:; "
       "frame-ancestors 'self'; object-src 'none'; base-uri 'self'; form-action 'self'")
# The upstream bundle's graph libraries use Function constructors. Confine that
# allowance to Gephi; the launcher keeps the stricter policy above.
GEPHI_CSP = CSP.replace("'wasm-unsafe-eval'", "'unsafe-eval'")


router = APIRouter(tags=["graph-workbench"])


def pool(request: Request):
    value = getattr(request.app.state, "memory_pg_pool", None)
    if value is None:
        raise HTTPException(503, "Memory database is unavailable.", headers=PRIVATE_HEADERS)
    return value


def _with_graph(source: str, fn, *args):
    uri = settings.FALKORDB_URI
    if not uri:
        raise HTTPException(503, "FalkorDB is not configured.", headers=PRIVATE_HEADERS)
    graph_name = settings.FALKORDB_SUBSTRATE_GRAPH if source == "substrate" else "orion_worldview"
    reader = FalkorReader(uri, graph_name)
    try:
        return fn(reader, *args)
    finally:
        reader.close()


@router.get("/graph-workbench", response_class=HTMLResponse)
async def launcher():
    return HTMLResponse((SERVICE_ROOT / "templates/graph_workbench.html").read_text(), headers={**PRIVATE_HEADERS, "Content-Security-Policy": CSP})


@router.get("/api/graph-workbench/sources")
async def sources():
    return Response(json.dumps({"sources": [{"id": k, "label": v} for k, v in SOURCE_LABELS.items()], "max_nodes": MAX_NODES}), media_type="application/json", headers=PRIVATE_HEADERS)


@router.get("/api/graph-workbench/search/{source}")
async def search(request: Request, source: Source, q: str = Query("", max_length=200)):
    try:
        if source == "crystallizations":
            rows = await crystal_search(pool(request), q)
            matches = [{**r, "id": str(r["id"])} for r in rows]
        else:
            rows = await asyncio.to_thread(_with_graph, source, graph_search, q)
            matches = [{"id": str(r["id"]), "label": caption(r["properties"], str(r["id"])), "kind": ", ".join(r["labels"])} for r in rows]
        from fastapi.responses import JSONResponse
        return JSONResponse({"matches": matches}, headers=PRIVATE_HEADERS)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("graph_workbench_search_failed source=%s error_type=%s", source, type(exc).__name__)
        raise HTTPException(503, "Graph search unavailable.", headers=PRIVATE_HEADERS) from exc


@router.get("/api/graph-workbench/export/{source}.gexf")
async def export(request: Request, source: Source, seed: str = Query("", max_length=100),
                 depth: int = Query(1, ge=1, le=3), limit: int = Query(300, ge=1, le=MAX_NODES),
                 lineage: bool = True):
    try:
        if source == "crystallizations":
            result = await crystal_snapshot(pool(request), seed, depth, limit, lineage)
        else:
            if seed and (not seed.isascii() or not seed.isdecimal() or int(seed) > 2**63 - 1):
                raise ValueError("Invalid graph node id")
            result = await asyncio.to_thread(_with_graph, source, graph_snapshot, source, seed, depth, limit)
        content = result.gexf()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, "Use a node id from search results.", headers=PRIVATE_HEADERS) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc), headers=PRIVATE_HEADERS) from exc
    except Exception as exc:
        logger.warning("graph_workbench_export_failed source=%s error_type=%s", source, type(exc).__name__)
        raise HTTPException(503, "Graph export unavailable.", headers=PRIVATE_HEADERS) from exc
    logger.info("graph_workbench_export source=%s nodes=%s edges=%s truncated=%s", source, len(result.nodes), len(result.edges), result.truncated)
    return Response(content, media_type="application/gexf+xml", headers={
        **PRIVATE_HEADERS,
        "Content-Disposition": f'inline; filename="{source}.gexf"',
        "X-Graph-Nodes": str(len(result.nodes)), "X-Graph-Edges": str(len(result.edges)),
        "X-Graph-Truncated": str(result.truncated).lower(),
    })


@router.get("/gephi-lite/{asset:path}")
async def gephi(request: Request, asset: str):
    if not ASSET_ROOT.is_dir():
        raise HTTPException(503, "Gephi Lite assets are missing; build the Hub Docker image.", headers=PRIVATE_HEADERS)
    response = await StaticFiles(directory=ASSET_ROOT).get_response(asset or "index.html", request.scope)
    if asset.endswith(".css") and response.status_code == 200:
        css = await asyncio.to_thread(Path(response.path).read_text)
        # Upstream imports Google fonts. Keep local fallback fonts without making
        # external requests or weakening the private workbench's network policy.
        css = re.sub(r'''@import\s*(?:url\()?['"]https://fonts\.googleapis\.com/[^'"]+['"]\)?\s*;''', "", css)
        response = Response(css, media_type="text/css")
    response.headers.update({**PRIVATE_HEADERS, "Content-Security-Policy": GEPHI_CSP})
    return response
