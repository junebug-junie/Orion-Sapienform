"""Background retention scheduler for orion-rdf-writer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.graph.rdf_retention import parse_retention_policies, run_retention_pass

from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

_retention_task: asyncio.Task[None] | None = None


async def _retention_loop() -> None:
    interval_hours = max(1, int(settings.RDF_RETENTION_INTERVAL_HOURS))
    interval_sec = interval_hours * 3600
    while True:
        await asyncio.sleep(interval_sec)
        if not settings.RDF_RETENTION_ENABLED:
            continue
        try:
            policies = parse_retention_policies(settings.RDF_RETENTION_POLICIES)
            query_url = settings.RDF_STORE_QUERY_URL
            update_url = settings.RDF_STORE_UPDATE_URL
            if not query_url or not update_url:
                base = (settings.RDF_STORE_BASE_URL or "http://orion-athena-fuseki:3030").rstrip("/")
                ds = settings.RDF_STORE_DATASET.strip("/")
                query_url = query_url or f"{base}/{ds}/query"
                update_url = update_url or f"{base}/{ds}/update"
            results = await asyncio.to_thread(
                run_retention_pass,
                policies=policies,
                query_url=query_url,
                update_url=update_url,
                user=settings.RDF_STORE_USER,
                password=settings.RDF_STORE_PASS,
                dry_run=settings.RDF_RETENTION_DRY_RUN,
                timeout_sec=settings.RDF_RETENTION_TIMEOUT_SEC,
            )
            for result in results:
                logger.info(
                    "rdf_retention_pass graph=%s age_batches=%s cap_artifacts=%s dry_run=%s errors=%s",
                    result.graph,
                    result.deleted_by_age,
                    result.deleted_by_cap,
                    result.dry_run,
                    len(result.errors),
                )
        except Exception:
            logger.exception("rdf_retention_pass_failed")


async def start_retention_scheduler() -> None:
    global _retention_task
    if not settings.RDF_RETENTION_ENABLED or settings.RDF_RETENTION_INTERVAL_HOURS <= 0:
        return
    if _retention_task is not None:
        return
    _retention_task = asyncio.create_task(_retention_loop(), name="rdf-retention")


async def stop_retention_scheduler() -> None:
    global _retention_task
    task = _retention_task
    _retention_task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def retention_health_snapshot() -> dict[str, Any]:
    return {
        "rdf_retention_enabled": settings.RDF_RETENTION_ENABLED,
        "rdf_retention_interval_hours": settings.RDF_RETENTION_INTERVAL_HOURS,
        "rdf_retention_dry_run": settings.RDF_RETENTION_DRY_RUN,
        "rdf_retention_scheduler_running": _retention_task is not None and not _retention_task.done(),
    }
