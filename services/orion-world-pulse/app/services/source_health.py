"""Per-source freshness window and stuck-source detection.

Live 2026-10-01..13 (``world_pulse_read_seed``, kind digest_item): who.int
served 26 items over 2 distinct URLs, one a 2026-01-24 statement re-served
daily; usgs.gov and cisa.gov served the same single index/data URL 13 days in
a row. Nothing noticed, because nothing compared one digest's picks to the
last one's. Two fixes live here:

1. ``filter_fresh``: drop feed items older than N days by the item's own
   published date (undated items are dropped too -- freshness that cannot be
   checked is not freshness).
2. ``assess_source_health``: compare each source's accepted URLs to the
   previous real (non-dry) run's, persisted in ``world_pulse_run`` under
   ``run.metrics.source_health``. A source with no new URL for K consecutive
   digests is ``stuck``: a ``source_stuck:<id>`` run warning + a log warning.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Iterable, TypeVar

logger = logging.getLogger("orion-world-pulse.source_health")

# Cap persisted URLs per source; max_articles_per_day is <= 10 in practice.
MAX_URLS_RECORDED = 20

T = TypeVar("T")


def candidate_field(candidate: Any, name: str) -> Any:
    """Attribute of an ArticleCandidate, or the same key of its model_dump()
    (callers and tests also pass duck-typed candidates exposing only that)."""
    if hasattr(candidate, name):
        return getattr(candidate, name)
    dump = getattr(candidate, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="python").get(name)
        except Exception:
            return None
    return None


def filter_fresh(
    candidates: Iterable[T],
    *,
    now: datetime,
    max_age_days: float,
) -> tuple[list[T], dict[str, int]]:
    """Keep items published within ``max_age_days`` of ``now``, newest first.

    ``max_age_days <= 0`` disables the window (all items kept, input order).
    Returns ``(kept, {"stale": n, "undated": n})``.
    """
    items = list(candidates)
    if max_age_days <= 0:
        return items, {"stale": 0, "undated": 0}
    cutoff = now - timedelta(days=max_age_days)
    kept: list[tuple[datetime, T]] = []
    stale = undated = 0
    for item in items:
        published = candidate_field(item, "published_at")
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                published = None
        if isinstance(published, datetime) and published.tzinfo is None:
            published = published.replace(tzinfo=now.tzinfo)
        if not isinstance(published, datetime):
            undated += 1
            continue
        if published < cutoff:
            stale += 1
            continue
        kept.append((published, item))
    kept.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in kept], {"stale": stale, "undated": undated}


def assess_source_health(
    current: dict[str, dict[str, Any]],
    previous: dict[str, Any] | None,
    *,
    stuck_after: int,
) -> dict[str, dict[str, Any]]:
    """Per-source health for this run.

    ``current[source_id]`` carries ``urls`` (accepted article URLs this run)
    plus counters (``fetched``, ``stale_dropped``, ``undated_dropped``,
    ``fetch_error``). ``previous`` is the last real run's ``source_health``
    map (or None when unavailable). A run "has something new" when at least
    one accepted URL was not accepted last run; ``consecutive_no_new`` counts
    runs in a row without that, and ``stuck`` is ``consecutive_no_new >=
    stuck_after`` (``stuck_after <= 0`` disables the flag).
    """
    out: dict[str, dict[str, Any]] = {}
    prev_map = previous if isinstance(previous, dict) else {}
    for source_id, row in current.items():
        urls = [str(u) for u in (row.get("urls") or [])][:MAX_URLS_RECORDED]
        prev = prev_map.get(source_id) if isinstance(prev_map.get(source_id), dict) else None
        prev_urls = set(prev.get("urls") or []) if prev else set()
        new_urls = [u for u in urls if u not in prev_urls]
        if new_urls:
            consecutive = 0
        else:
            try:
                prev_count = int(prev.get("consecutive_no_new", 0)) if prev else 0
            except (TypeError, ValueError):
                prev_count = 0
            consecutive = prev_count + 1
        entry = {
            "fetched": int(row.get("fetched", 0)),
            "fresh": len(urls),
            "new": len(new_urls),
            "stale_dropped": int(row.get("stale_dropped", 0)),
            "undated_dropped": int(row.get("undated_dropped", 0)),
            "fetch_error": bool(row.get("fetch_error", False)),
            "consecutive_no_new": consecutive,
            "stuck": stuck_after > 0 and consecutive >= stuck_after,
            "urls": urls,
        }
        out[source_id] = entry
    return out


def stuck_reason(entry: dict[str, Any]) -> str:
    if entry.get("fetch_error"):
        return "fetch_error"
    if entry.get("fresh", 0) == 0:
        return "zero_fresh_items"
    return "same_urls"


# --- history read (fail-open) ---------------------------------------------

_engine: object | None = None
_engine_url: str | None = None
_engine_lock = threading.Lock()

_PREVIOUS_SQL = """
SELECT payload_json->'run'->'metrics'->'source_health'
FROM world_pulse_run
WHERE lower(coalesce(dry_run, '')) = 'false'
  AND run_id <> :run_id
  AND payload_json->'run'->'metrics'->'source_health' IS NOT NULL
ORDER BY created_at DESC
LIMIT 1
"""


def _get_engine(db_url: str):
    global _engine, _engine_url
    if not db_url:
        return None
    if _engine is None or _engine_url != db_url:
        with _engine_lock:
            if _engine is None or _engine_url != db_url:
                from sqlalchemy import create_engine

                _engine = create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=1,
                    max_overflow=0,
                    connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"},
                )
                _engine_url = db_url
    return _engine


def load_previous_source_health(db_url: str, *, run_id: str) -> dict[str, Any] | None:
    """Last real run's ``source_health`` map, or None (no URL, no row, DB down)."""
    if not db_url:
        return None
    try:
        from sqlalchemy import text

        engine = _get_engine(db_url)

        with engine.connect() as conn:  # type: ignore[attr-defined]
            row = conn.execute(text(_PREVIOUS_SQL), {"run_id": run_id}).first()
    except Exception:
        logger.warning("world_pulse_source_health_history_unavailable", exc_info=True)
        return None
    if row is None or not isinstance(row[0], dict):
        return None
    return row[0]
