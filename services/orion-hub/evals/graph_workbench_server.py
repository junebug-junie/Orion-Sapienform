"""Isolated read-only browser-eval server; never starts Hub's cognition workers.

Run from any directory with --env-file, --credentials-file, and --assets.
Only SELECT/GRAPH.RO_QUERY paths are exposed, bound to loopback.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--credentials-file", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18089)
    args = parser.parse_args()
    hub = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(hub.parents[1]))
    sys.path.insert(0, str(hub))
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    load_dotenv(args.credentials_file, override=True)
    import asyncpg
    import uvicorn
    from fastapi import Depends, FastAPI
    from fastapi.responses import Response
    from fastapi.staticfiles import StaticFiles
    from scripts import graph_workbench_routes as routes
    if not routes.settings.HUB_GRAPH_WORKBENCH_PASSWORD:
        parser.error("Credentials file must set HUB_GRAPH_WORKBENCH_PASSWORD.")
    routes.ASSET_ROOT = args.assets

    @asynccontextmanager
    async def lifespan(app):
        # No DDL/bootstrap: this pool can only read the canonical tables.
        app.state.memory_pg_pool = await asyncpg.create_pool(
            routes.settings.RECALL_PG_DSN, min_size=1, max_size=2,
            server_settings={"default_transaction_read_only": "on", "statement_timeout": "3000"},
        )
        try:
            yield
        finally:
            await app.state.memory_pg_pool.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(routes.router)
    app.mount("/static", StaticFiles(directory=hub / "static"))

    @app.get("/eval/parallel.gexf", dependencies=[Depends(routes.operator)])
    async def parallel_fixture():
        from scripts.graph_workbench import Snapshot
        snap = Snapshot("eval-parallel-fixture")
        snap.node("a", {"label": "Alpha", "score": 0.25}, 2)
        snap.node("b", {"label": "Beta", "score": 0.75}, 2)
        snap.edge("first", "a", "b", {"relation": "supports"})
        snap.edge("second", "a", "b", {"relation": "refines"})
        snap.edge("reverse", "b", "a", {"relation": "contradicts"})
        return Response(snap.gexf(), media_type="application/gexf+xml", headers=routes.PRIVATE_HEADERS)

    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
