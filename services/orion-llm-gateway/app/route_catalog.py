"""Route catalog and upstream health cache for GET /routes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from orion.llm.routes import BACKGROUND_LLM_ROUTES, LLM_ROUTE_DISPLAY_ORDER, SYSTEM_LLM_ROUTES

from .llm_backend import RouteTarget, get_route_targets
from .settings import settings

# Derived, not re-typed. This was a hardcoded ("chat", "quick", "agent", "metacog") until
# 2026-08-19 -- one of four independent copies that between them made `quick_background`
# invisible to `GET /routes`, the Hub UI and the routes smoke, months after Orion's own
# journalling had been moved onto it. `orion/llm/routes.py` raises at import if a route is
# accepted but absent from the display order, so the next route to exist cannot be omitted here
# by forgetting.
CATALOG_ROUTE_IDS = LLM_ROUTE_DISPLAY_ORDER
_CACHE_TTL_SEC = 15.0


@dataclass(frozen=True)
class RouteHealthEntry:
    route_id: str
    served_by: Optional[str]
    backend: Optional[str]
    status: str
    latency_ms: Optional[int]
    last_checked_at: Optional[str]
    model: Optional[str] = None
    vision: Optional[bool] = None
    # Why two routes on the same worker behave differently. Without this, `quick` and
    # `quick_background` appear in the catalog as duplicate rows pointing at one URL with no
    # visible reason -- and the Hub has no property to filter its picker on, forcing it back
    # to a hardcoded name list.
    priority: Optional[str] = None
    reserved_free_slots: Optional[int] = None


_cache: Dict[str, RouteHealthEntry] = {}
_cache_lock = asyncio.Lock()
_last_refresh_mono: float = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _probe_model(target: RouteTarget) -> Optional[str]:
    """Best-effort read of the model actually loaded behind a route.

    `target.model` (the route table's configured label, e.g.
    "Active-GGUF-Model") names a route, not necessarily a specific weights
    file -- confirmed live 2026-08-14: llama.cpp's OpenAI-compat
    `/v1/models` echoes the real served model
    ("Qwen3.6-35B-A3B-UD-Q5_K_M.gguf") regardless of what alias was
    requested, the same fact `_served_model()` in llm_backend.py uses for
    per-call responses. This is the equivalent for the route catalog: a
    point-in-time "what's actually loaded right now" read, not a per-request
    value. Fails open to None on any error, timeout, or unexpected shape --
    a route health check must never fail because a model name couldn't be
    read.
    """
    url = f"{target.url.rstrip('/')}/v1/models"
    try:
        timeout = float(settings.llm_route_health_timeout_sec or 1.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return None
            payload = response.json()
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    model_id = first.get("id") if isinstance(first, dict) else None
    return model_id if isinstance(model_id, str) and model_id.strip() else None


async def _probe_vision(target: RouteTarget) -> Optional[bool]:
    """Best-effort read of whether this route's worker can actually see.

    llama.cpp reports `modalities.vision` on `/props`, and it is the only
    trustworthy answer: it reflects whether the server was started with
    `--mmproj`, which the profile registry's `supports_vision` flag does not.
    Confirmed live 2026-08-14 that the two disagree -- the chat lane's weights
    and chat template are VL-capable and its HF repo ships an mmproj, but the
    worker was launched without one, so it reports false while a config reader
    would have said otherwise (and nothing read the config flag at all).

    Publishing this on /routes is what lets the Hub enable or grey out its
    attach button against the truth rather than against a YAML claim.

    Fails open to None ("unknown") on any error -- a route health check must
    never fail because a modality couldn't be read. None is rendered distinctly
    from False so a probe failure is never mistaken for a confirmed blindness.
    """
    url = f"{target.url.rstrip('/')}/props"
    try:
        timeout = float(settings.llm_route_health_timeout_sec or 1.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return None
            payload = response.json()
    except Exception:
        return None
    modalities = payload.get("modalities") if isinstance(payload, dict) else None
    if not isinstance(modalities, dict):
        return None
    return bool(modalities.get("vision"))


async def _probe_health(target: RouteTarget) -> tuple[str, Optional[int]]:
    url = f"{target.url.rstrip('/')}/health"
    start = time.monotonic()
    status = "down"
    latency_ms: Optional[int] = None
    try:
        timeout = float(settings.llm_route_health_timeout_sec or 1.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            latency_ms = int((time.monotonic() - start) * 1000)
            status = "up" if response.status_code < 400 else "down"
    except Exception:
        if latency_ms is None:
            latency_ms = int((time.monotonic() - start) * 1000)
        status = "down"
    return status, latency_ms


def _probe_key(target: RouteTarget) -> str:
    """Dedup key for one upstream. Matches the normalisation the probe helpers apply."""
    return str(target.url or "").rstrip("/")


async def _probe_backend(target: RouteTarget) -> tuple[str, Optional[int], Optional[str], Optional[bool]]:
    """Health + model + vision for ONE upstream URL, concurrently.

    Routes are not one-to-one with workers: `quick` and
    `quick_background` are the same llama.cpp process under two admission policies, so probing
    per *route* would send three identical HTTP calls to `atlas-worker-fast-1` every refresh
    for no new information. `refresh_route_health_cache` probes each distinct URL once and
    fans the result out.
    """
    (status, latency_ms), model, vision = await asyncio.gather(
        _probe_health(target), _probe_model(target), _probe_vision(target)
    )
    if status != "up":
        model = None
        vision = None
    return status, latency_ms, model, vision


def _definitional_priority(route_id: str) -> Optional[str]:
    """What this route IS, independent of whether the route table configures it.

    A background lane missing from LLM_GATEWAY_ROUTE_TABLE_JSON has no RouteTarget and so no
    configured priority -- but it is still a background lane, and a consumer filtering on
    `priority` would otherwise offer it to a human as an ordinary one. Fail-safe, not fail-open.

    `system` (harness, 2026-08-20) is the same fail-safe for a different reason: not a yielding
    lane (it must dispatch immediately, not wait for slot slack), just never a human's Compute
    choice. Checked after `background` so the two stay mutually exclusive, matching the
    RuntimeError orion.llm.routes raises if a route is ever placed in both sets.
    """
    if route_id in BACKGROUND_LLM_ROUTES:
        return "background"
    if route_id in SYSTEM_LLM_ROUTES:
        return "system"
    return None


def _entry_from_probe(
    route_id: str,
    target: RouteTarget,
    status: str,
    latency_ms: Optional[int],
    model: Optional[str],
    vision: Optional[bool],
) -> RouteHealthEntry:
    return RouteHealthEntry(
        route_id=route_id,
        served_by=target.served_by,
        backend=target.backend,
        status=status,
        latency_ms=latency_ms,
        last_checked_at=_utc_now_iso(),
        model=model,
        vision=vision,
        priority=getattr(target, "priority", None) or _definitional_priority(route_id),
        reserved_free_slots=getattr(target, "reserved_free_slots", None),
    )


async def refresh_route_health_cache(*, force: bool = False) -> None:
    global _last_refresh_mono
    async with _cache_lock:
        now = time.monotonic()
        if not force and _cache and (now - _last_refresh_mono) < _CACHE_TTL_SEC:
            return
        targets = get_route_targets()
        # One probe per distinct upstream URL, all URLs concurrently. Previously this awaited
        # each route in sequence, so refresh latency grew linearly with the catalog -- and the
        # catalog just grew, with a route that shares a URL with one already in it.
        # Key on the rstripped URL: the three probe helpers all normalise trailing slashes, so
        # keying on the raw string would treat "http://atlas:8013" and "http://atlas:8013/" as
        # two workers and send six calls where three would do -- reintroducing the exact
        # duplicate-probe cost this dedup exists to remove.
        by_url: Dict[str, RouteTarget] = {}
        for route_id in CATALOG_ROUTE_IDS:
            target = targets.get(route_id)
            if target is not None and target.url:
                by_url.setdefault(_probe_key(target), target)
        urls = list(by_url)
        results = await asyncio.gather(*(_probe_backend(by_url[u]) for u in urls))
        probes = dict(zip(urls, results))

        entries: Dict[str, RouteHealthEntry] = {}
        for route_id in CATALOG_ROUTE_IDS:
            target = targets.get(route_id)
            if target is None:
                entries[route_id] = RouteHealthEntry(
                    route_id=route_id,
                    served_by=None,
                    backend=None,
                    status="not_configured",
                    latency_ms=None,
                    last_checked_at=_utc_now_iso(),
                    priority=_definitional_priority(route_id),
                )
                continue
            probe = probes.get(_probe_key(target))
            if probe is None:
                # Configured but unprobeable -- an empty or malformed URL. That is `down` with
                # its identity intact, NOT `not_configured`: a misconfigured route and a route
                # that was never in the table are different problems, and before the URL dedup
                # this case was probed, failed, and correctly reported `down`.
                entries[route_id] = _entry_from_probe(route_id, target, "down", None, None, None)
            else:
                entries[route_id] = _entry_from_probe(route_id, target, *probe)
        _cache.clear()
        _cache.update(entries)
        _last_refresh_mono = now


def _entry_to_dict(entry: RouteHealthEntry) -> Dict[str, Any]:
    return {
        "id": entry.route_id,
        "served_by": entry.served_by,
        "backend": entry.backend,
        "status": entry.status,
        "latency_ms": entry.latency_ms,
        "last_checked_at": entry.last_checked_at,
        "model": entry.model,
        "vision": entry.vision,
        "priority": entry.priority,
        "reserved_free_slots": entry.reserved_free_slots,
    }


def build_routes_response() -> Dict[str, Any]:
    targets = get_route_targets()
    routes: List[Dict[str, Any]] = []
    for route_id in CATALOG_ROUTE_IDS:
        cached = _cache.get(route_id)
        if cached is not None:
            routes.append(_entry_to_dict(cached))
            continue
        target = targets.get(route_id)
        if target is None:
            routes.append(
                {
                    "id": route_id,
                    "served_by": None,
                    "backend": None,
                    "status": "not_configured",
                    "latency_ms": None,
                    "last_checked_at": None,
                    "model": None,
                    "vision": None,
                    "priority": _definitional_priority(route_id),
                    "reserved_free_slots": None,
                }
            )
        else:
            routes.append(
                {
                    "id": route_id,
                    "served_by": target.served_by,
                    "backend": target.backend,
                    "status": "unknown",
                    "latency_ms": None,
                    "last_checked_at": None,
                    "model": None,
                    "vision": None,
                    # Config, not probe result -- so these are known even before the first
                    # health refresh. Reporting None here would make a background lane look
                    # pickable to the Hub (which filters on `priority`) for the first 15s of
                    # gateway uptime, which is exactly when someone is most likely looking.
                    "priority": getattr(target, "priority", None) or _definitional_priority(route_id),
                    "reserved_free_slots": getattr(target, "reserved_free_slots", None),
                }
            )
    return {
        "default_route": str(settings.llm_route_default or "chat"),
        "routes": routes,
    }


async def get_routes_payload() -> Dict[str, Any]:
    await refresh_route_health_cache()
    return build_routes_response()


def reset_route_health_cache_for_tests() -> None:
    global _last_refresh_mono
    _cache.clear()
    _last_refresh_mono = 0.0
