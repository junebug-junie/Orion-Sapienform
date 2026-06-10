from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from .schemas import TraceHit
from .settings import settings

logger = logging.getLogger("orion-context-exec.trace_tools")

_MAX_STORE = 2000
_store: list[TraceHit] = []
_payloads: dict[str, dict[str, Any]] = {}
_lock = Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_handle(source: str, kind: str) -> str:
    return f"trace:{source}:{kind}:{uuid4().hex[:10]}"


def register_trace_hit(
    *,
    source: str,
    kind: str,
    corr_id: str | None = None,
    run_id: str | None = None,
    snippet: str = "",
    payload: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> TraceHit:
    handle = _make_handle(source, kind)
    hit = TraceHit(
        handle=handle,
        source=source,
        corr_id=corr_id,
        run_id=run_id,
        kind=kind,
        timestamp=timestamp or _now_iso(),
        snippet=snippet[:500],
        payload_ref=handle,
    )
    with _lock:
        _store.append(hit)
        if payload is not None:
            _payloads[handle] = payload
        if len(_store) > _MAX_STORE:
            drop = _store[: len(_store) - _MAX_STORE]
            _store[:] = _store[-_MAX_STORE:]
            for old in drop:
                _payloads.pop(old.handle, None)
    return hit


def clear_trace_store() -> None:
    """Test helper."""
    with _lock:
        _store.clear()
        _payloads.clear()


def _matches_query(hit: TraceHit, query: str | None) -> bool:
    if not query:
        return True
    q = query.lower()
    hay = " ".join(
        filter(
            None,
            [hit.snippet, hit.kind, hit.corr_id or "", hit.run_id or "", hit.source],
        )
    ).lower()
    return q in hay


def traces_search(
    *,
    query: str | None = None,
    corr_id: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> list[TraceHit]:
    cap = limit if limit is not None else settings.context_exec_trace_limit
    hits: list[TraceHit] = []

    with _lock:
        candidates = list(reversed(_store))

    for hit in candidates:
        if corr_id and hit.corr_id != corr_id and corr_id not in (hit.snippet or ""):
            if not _corr_in_payload(hit, corr_id):
                continue
        if run_id and hit.run_id != run_id and run_id not in (hit.snippet or ""):
            continue
        if not _matches_query(hit, query):
            continue
        hits.append(hit)
        if len(hits) >= cap:
            break

    if hits or not settings.orion_bus_enabled:
        return hits

    redis_hits = _search_redis_traces(
        query=query,
        corr_id=corr_id,
        run_id=run_id,
        limit=max(0, cap - len(hits)),
    )
    hits.extend(redis_hits)
    return hits[:cap]


def _corr_in_payload(hit: TraceHit, corr_id: str) -> bool:
    payload = _payloads.get(hit.handle or "")
    if not payload:
        return False
    blob = json.dumps(payload, default=str).lower()
    return corr_id.lower() in blob


def traces_read(handle: str) -> dict[str, Any]:
    with _lock:
        payload = _payloads.get(handle)
        for hit in _store:
            if hit.handle == handle:
                base = hit.model_dump(mode="json")
                if payload is not None:
                    base["payload"] = payload
                return base
    return {"handle": handle, "error": "not_found"}


def _search_redis_traces(
    *,
    query: str | None,
    corr_id: str | None,
    run_id: str | None,
    limit: int,
) -> list[TraceHit]:
    if limit <= 0:
        return []
    try:
        import redis  # type: ignore[import-untyped]

        client = redis.Redis.from_url(settings.orion_bus_url, decode_responses=True)
        patterns: list[str] = []
        if corr_id:
            patterns.append(f"*{corr_id}*")
        if run_id:
            patterns.append(f"*{run_id}*")
        if not patterns and query:
            token = query.strip()[:64]
            if token:
                patterns.append(f"*{token}*")
        if not patterns:
            return []

        seen: set[str] = set()
        out: list[TraceHit] = []
        for pattern in patterns:
            for key in client.scan_iter(match=pattern, count=200):
                if key in seen:
                    continue
                seen.add(key)
                try:
                    raw = client.get(key)
                except Exception:
                    continue
                if not raw:
                    continue
                snippet = raw[:500] if isinstance(raw, str) else str(raw)[:500]
                if query and query.lower() not in snippet.lower():
                    continue
                hit = register_trace_hit(
                    source="redis",
                    kind="redis_key",
                    corr_id=corr_id,
                    run_id=run_id,
                    snippet=snippet,
                    payload={"redis_key": key},
                )
                out.append(hit)
                if len(out) >= limit:
                    return out
        return out
    except Exception as exc:
        logger.debug("redis trace search skipped: %s", exc)
        return []
