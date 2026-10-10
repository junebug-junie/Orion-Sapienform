"""Readiness check for cross-window concept-relation resolution.

Why this exists: from 2026-09-07 to 2026-10-10 the concept-relation writer wrote
nothing while CONCEPT_RELATION_RESOLUTION_ENABLED=true, because the embed host and
Chroma host were empty in the live env. fetch_similar_candidates() degrades to []
on that, so the whole feature became a silent no-op. This module makes that state
loud: a WARNING at boot and a `degraded` status on /health.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SEC = 5.0

# Last readiness result, surfaced by /health. Written at boot.
READINESS_STATUS: dict[str, Any] = {"status": "unchecked", "problems": []}


def config_problems(settings: Any) -> list[str]:
    """Config-only problems (no network). Empty list means config is complete."""
    problems: list[str] = []
    if not (getattr(settings, "CRYSTALLIZER_EMBED_HOST_URL", "") or "").strip():
        problems.append("embed_host_url_empty")
    if not (getattr(settings, "CHROMA_HOST", "") or "").strip():
        problems.append("chroma_host_empty")
    return problems


async def _probe_embed(url: str) -> str | None:
    """POST a tiny text to the real embed endpoint. Returns a problem string or None."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC) as client:
            resp = await client.post(
                url.rstrip("/"),
                json={"doc_id": "concept_relation_readiness_probe", "text": "readiness probe", "embedding_profile": "default"},
            )
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or not data.get("embedding"):
            return "embed_host_returned_no_embedding"
    except Exception as exc:
        return f"embed_host_unreachable:{type(exc).__name__}"
    return None


def _probe_chroma_sync(host: str, port: int) -> None:
    import chromadb  # type: ignore

    chromadb.HttpClient(host=host, port=port).heartbeat()


async def _probe_chroma(host: str, port: int) -> str | None:
    try:
        await asyncio.wait_for(asyncio.to_thread(_probe_chroma_sync, host, port), timeout=_PROBE_TIMEOUT_SEC)
    except Exception as exc:
        return f"chroma_unreachable:{type(exc).__name__}"
    return None


async def check_concept_relation_readiness(settings: Any, *, probe: bool = True) -> dict[str, Any]:
    """Return {status, enabled, problems, checked_at}; status is ok|disabled|degraded.

    Logs a WARNING whenever resolution is enabled but cannot actually find candidates.
    Never raises.
    """
    enabled = bool(getattr(settings, "CONCEPT_RELATION_RESOLUTION_ENABLED", False))
    result: dict[str, Any] = {
        "enabled": enabled,
        "status": "disabled",
        "problems": [],
        "embed_host_url": getattr(settings, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        "chroma_host": getattr(settings, "CHROMA_HOST", "") or "",
        "chroma_port": int(getattr(settings, "CHROMA_PORT", 8000) or 8000),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if not enabled:
        READINESS_STATUS.clear()
        READINESS_STATUS.update(result)
        return result

    problems = config_problems(settings)
    if probe:
        if "embed_host_url_empty" not in problems:
            p = await _probe_embed(result["embed_host_url"])
            if p:
                problems.append(p)
        if "chroma_host_empty" not in problems:
            p = await _probe_chroma(result["chroma_host"], result["chroma_port"])
            if p:
                problems.append(p)

    result["problems"] = problems
    result["status"] = "degraded" if problems else "ok"
    if problems:
        logger.warning(
            "concept_relation_resolution_degraded enabled=true problems=%s embed_host_url=%r chroma_host=%r "
            "-- no candidates will be found and no relation decisions will be written",
            ",".join(problems),
            result["embed_host_url"],
            result["chroma_host"],
        )
    else:
        logger.info("concept_relation_resolution_ready embed_host_url=%s chroma_host=%s", result["embed_host_url"], result["chroma_host"])

    READINESS_STATUS.clear()
    READINESS_STATUS.update(result)
    return result
