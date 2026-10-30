"""Live eval: do the substrate debug routes hold a bounded number of connections?

Before 2026-10-30, nine `substrate_*_routes.py` modules built a brand-new
SQLAlchemy engine (its own connection pool) on every request and never
disposed it. Each request left an idle Postgres connection open until the
garbage collector reached the abandoned pool; the hub held ~59 connections.

This eval calls each module's `_engine()` the way a route does
(`with _engine().connect()`), REQUESTS_PER_MODULE times, then counts the live
backends Postgres holds under a unique `application_name` (libpq reads
PGAPPNAME, so it is set before anything connects).

PASS: every module reused one engine, and the connections left open are at
most one per module. Exit 0 PASS, 1 FAIL, 2 UNKNOWN (database unreachable).

Run against the live database from the hub image (host network):
    python services/orion-hub/evals/run_route_engine_pool_eval.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

APP_NAME = f"route-pool-eval-{os.getpid()}"
os.environ["PGAPPNAME"] = APP_NAME

_HUB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HUB_ROOT))
sys.path.append(str(_HUB_ROOT.parents[1]))

MODULES = (
    "substrate_attention_routes",
    "substrate_biometrics_routes",
    "substrate_consolidation_routes",
    "substrate_execution_dispatch_routes",
    "substrate_feedback_routes",
    "substrate_field_routes",
    "substrate_observability_routes",
    "substrate_policy_routes",
    "substrate_proposal_routes",
)
REQUESTS_PER_MODULE = 25


def _held_connections(uri: str) -> int:
    import psycopg2

    conn = psycopg2.connect(uri, application_name=f"{APP_NAME}-probe")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = %s",
                (APP_NAME,),
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def main() -> int:
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        print("UNKNOWN: POSTGRES_URI is not set", file=sys.stderr)
        return 2
    from sqlalchemy import text

    engines_per_module: dict[str, int] = {}
    try:
        for name in MODULES:
            module = importlib.import_module(f"scripts.{name}")
            seen: set[int] = set()
            for _ in range(REQUESTS_PER_MODULE):
                engine = module._engine()
                seen.add(id(engine))
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            engines_per_module[name] = len(seen)
        # The consolidation endpoints reach Postgres through
        # orion/consolidation/repository.py, which builds its own engine unless
        # one is passed in -- call the real endpoints, not just `_engine()`.
        import asyncio

        routes = importlib.import_module("scripts.substrate_consolidation_routes")
        for _ in range(REQUESTS_PER_MODULE):
            asyncio.run(routes.consolidation_motifs(limit=5))
            asyncio.run(routes.consolidation_schema_candidates(limit=5))
            asyncio.run(routes.consolidation_tensor_slices_latest(limit=5))
            asyncio.run(routes.consolidation_expectations(trigger_motif_id="route-pool-eval"))
        held = _held_connections(uri)
    except Exception as exc:  # noqa: BLE001
        print(f"UNKNOWN: database unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for name, count in engines_per_module.items():
        print(f"{name}: {count} distinct engine(s) over {REQUESTS_PER_MODULE} requests")
    limit = len(MODULES)
    print(f"connections held after {REQUESTS_PER_MODULE * (limit + 4)} requests: {held} (limit {limit})")

    reused = all(count == 1 for count in engines_per_module.values())
    if reused and held <= limit:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
