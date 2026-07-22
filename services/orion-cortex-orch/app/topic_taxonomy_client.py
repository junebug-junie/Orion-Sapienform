from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from .settings import get_settings

logger = logging.getLogger("orion.cortex.topic_taxonomy")

_cache: Optional[list[str]] = None
_cache_at: float = 0.0


async def _fetch_active_run_id(client: httpx.AsyncClient, base: str) -> Optional[str]:
    """Wrapped /runs items already carry model.stage (see
    orion-topic-foundry/app/routers/runs.py's RunListItem), so the first
    complete run for the active model can be found in one call -- no
    separate /models/{name}/active round trip needed. Rows are ordered
    created_at DESC (list_runs_paginated), so the first match is the most
    recent."""
    resp = await client.get(
        f"{base}/runs",
        params={"model_name": "topic-foundry", "status": "complete", "format": "wrapped", "limit": 20},
    )
    resp.raise_for_status()
    data = resp.json()
    for item in data.get("items") or []:
        model = item.get("model") or {}
        if model.get("stage") == "active":
            return item.get("run_id")
    return None


async def _fetch_topic_labels(client: httpx.AsyncClient, base: str, run_id: str) -> list[str]:
    resp = await client.get(f"{base}/topics", params={"run_id": run_id, "limit": 200})
    resp.raise_for_status()
    data = resp.json()
    labels = [item.get("label") for item in data.get("items") or [] if item.get("label")]
    # Dedupe, preserve order.
    return list(dict.fromkeys(labels))


async def fetch_active_topic_labels() -> list[str]:
    """Current orion-topic-foundry active-model topic labels, for grounding
    write-time card annotation's types/tags in real, data-driven categories
    instead of the LLM inventing them fresh every call. In-process cache
    with a TTL (default 1h, topics.py::TOPIC_FOUNDRY_LABELS_CACHE_TTL_SEC)
    -- topics don't change turn to turn, and this must not add an extra
    bus/HTTP round trip to every single chat turn's annotation latency
    beyond a cache miss. Fails open to [] on any error (empty base_url,
    timeout, HTTP error, malformed response) -- annotation proceeds exactly
    as it did before this feature existed, never blocks card creation."""
    global _cache, _cache_at

    settings = get_settings()
    ttl = float(settings.topic_foundry_labels_cache_ttl_sec)
    if _cache is not None and (time.monotonic() - _cache_at) < ttl:
        return _cache

    base = (settings.topic_foundry_base_url or "").rstrip("/")
    if not base:
        return []

    timeout_sec = max(0.5, min(10.0, float(settings.topic_foundry_timeout_sec)))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec)) as client:
            run_id = await _fetch_active_run_id(client, base)
            if not run_id:
                return []
            labels = await _fetch_topic_labels(client, base, run_id)
    except httpx.HTTPError as exc:
        logger.warning("topic_taxonomy_fetch_http_err err=%s", exc)
        return _cache or []
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("topic_taxonomy_fetch_parse_err err=%s", exc)
        return _cache or []

    _cache = labels
    _cache_at = time.monotonic()
    return labels
