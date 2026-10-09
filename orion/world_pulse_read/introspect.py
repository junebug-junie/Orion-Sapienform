"""Read-only "what did I learn" lookups for the orion-introspect reading_results tool.

Plain SELECTs over world_pulse_read_seed and journal_entries; never enqueues,
retries, or charges a wallet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from orion.schemas.introspect import (
    DEFAULT_LIMIT,
    DEFAULT_TEXT_CAP,
    SHORT_FIELD_CAP,
    URL_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    clip_text,
)
from orion.world_pulse_read.operator import _journal_entries
from orion.world_pulse_read.queue import derive_reading_status, reading_status
from orion.world_pulse_read.read_evidence import parse_source_fetches, source_read_evidence

JOURNAL_EXCERPT_CHARS = 600

_COLUMNS = """
s.seed_id, s.request_id, s.url, s.title, s.status, s.stage2_status,
s.created_at, s.handoff_at, s.stage2_completed_at, s.landing_at,
s.handoff_json, s.stage2_result_json, s.trace_id, s.stage2_trace_id,
s.request_json->>'why_now' AS why_now
"""
_OCCURRED = "COALESCE(s.landing_at, s.stage2_completed_at, s.handoff_at, s.created_at)"

_ROW_SQL = f"SELECT {_COLUMNS} FROM world_pulse_read_seed s WHERE s.seed_id = $1"

# Rows whose Stage 1 handoff carries no tool-trace fetch are not readings (see
# read_evidence.py); a Stage 2 summary built on one is no better. CASE, not AND,
# because SQL does not guarantee short-circuit evaluation.
_VERIFIED_WHERE = """s.duplicate_of IS NULL
  AND s.status = 'done'
  AND CASE WHEN jsonb_typeof(s.handoff_json->'read_evidence') = 'array'
           THEN jsonb_array_length(s.handoff_json->'read_evidence') > 0
           ELSE false END"""

# count(*) OVER () is evaluated before LIMIT, so ``total`` is the full match count.
_RECENT_SQL = f"""
SELECT {_COLUMNS}, count(*) OVER () AS total
FROM world_pulse_read_seed s
WHERE {_VERIFIED_WHERE}
  AND ($2::timestamptz IS NULL OR {_OCCURRED} >= $2)
ORDER BY {_OCCURRED} DESC, s.seed_id DESC
LIMIT $1
"""

# Half-open window [$1, $2) over the same verified rows, oldest first; the daily
# letter (orion/orion_day/gather.py) reads a whole day this way.
_WINDOW_SQL = f"""
SELECT {_COLUMNS}
FROM world_pulse_read_seed s
WHERE {_VERIFIED_WHERE}
  AND {_OCCURRED} >= $1 AND {_OCCURRED} < $2
ORDER BY {_OCCURRED}, s.seed_id
LIMIT $3
"""


def _obj(raw: Any) -> dict[str, Any]:
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _source_read(row: Any) -> bool:
    fetches = parse_source_fetches(_obj(row["handoff_json"]).get("read_evidence")) or []
    return bool(source_read_evidence(row["url"], fetches))


def _learned(row: Any, source_read: bool) -> str:
    # A handoff on a row that did not finish Stage 1 was rejected, and one with
    # no read of its source is model prose -- neither is "learned".
    if row["status"] != "done" or not source_read:
        return ""
    result = _obj(row["stage2_result_json"])
    return str(result.get("summary") or _obj(row["handoff_json"]).get("what_i_learned") or "")


def _item(
    row: Any,
    *,
    request_id: str | None,
    journal_excerpt: str | None = None,
    text_cap: int | None = DEFAULT_TEXT_CAP,
) -> IntrospectItemV1:
    """``text_cap=None`` keeps the full learned text (the daily letter's material)."""
    source_read = _source_read(row)
    learned = _learned(row, source_read)
    text, truncated = (learned.strip(), False) if text_cap is None else clip_text(learned, text_cap)
    url, url_truncated = clip_text(row["url"], URL_CAP)
    extra: dict[str, Any] = {
        "url": url,
        "title": clip_text(row["title"], SHORT_FIELD_CAP)[0],
        "why_now": clip_text(row["why_now"], SHORT_FIELD_CAP)[0],
        "reading_status": derive_reading_status(row["status"], row["stage2_status"], row["landing_at"]),
        "source_read": source_read,
        "learned": bool(text),
        "request_id": request_id,
    }
    if url_truncated:
        # A clipped URL matches nothing on lookup; say so rather than let a
        # re-query read as "never read".
        extra["url_truncated"] = True
    if journal_excerpt:
        extra["journal_excerpt"] = journal_excerpt
    return IntrospectItemV1(
        id=row["seed_id"],
        occurred_at=row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"],
        kind="reading_result",
        epistemic_status="unsettled",
        text=text,
        truncated=truncated,
        extra=extra,
    )


async def reading_results(
    conn: Any,
    *,
    request_id: UUID | None = None,
    url: str | None = None,
    limit: int = DEFAULT_LIMIT,
    since: datetime | None = None,
    text_cap: int | None = DEFAULT_TEXT_CAP,
) -> IntrospectResultV1:
    as_of = datetime.now(timezone.utc)
    if request_id is None and url is None:
        rows = await conn.fetch(_RECENT_SQL, limit, since)
        return IntrospectResultV1(
            ok=True, operation="reading_result", as_of=as_of,
            total_available=int(rows[0]["total"]) if rows else 0,
            items=[
                _item(r, request_id=str(r["request_id"]) if r["request_id"] else None, text_cap=text_cap)
                for r in rows
            ],
        )
    status = await (reading_status(conn, url=url) if url is not None else reading_status(conn, request_id))
    if status["status"] == "not_found":
        return IntrospectResultV1(ok=True, operation="reading_result", as_of=as_of, total_available=0)
    row = await conn.fetchrow(_ROW_SQL, status.get("duplicate_of") or status["seed_id"])
    if row is None:
        raise RuntimeError("reading row vanished between status and detail lookup")
    refs = []
    if row["trace_id"]:
        refs.append(f"world_pulse_read:{row['trace_id']}")
    if row["stage2_trace_id"]:
        refs.append(f"world_pulse_read_stage2:{row['stage2_trace_id']}")
    journal = await _journal_entries(conn, refs)
    excerpt = clip_text(journal[-1]["body"], JOURNAL_EXCERPT_CHARS)[0] if journal else None
    return IntrospectResultV1(
        ok=True, operation="reading_result", as_of=as_of,
        total_available=int(status.get("matched_request_count") or 1),
        items=[_item(row, request_id=status.get("request_id"), journal_excerpt=excerpt, text_cap=text_cap)],
    )


async def reading_items_between(
    conn: Any,
    *,
    since: datetime,
    until: datetime,
    limit: int = 500,
    text_cap: int | None = None,
) -> list[IntrospectItemV1]:
    """Verified readings that landed in ``[since, until)``, oldest first, as items.

    Not wrapped in ``IntrospectResultV1`` (that caps items at MAX_ITEMS for the tool's
    result budget); a whole day can hold more. ``text_cap=None`` keeps full text."""
    rows = await conn.fetch(_WINDOW_SQL, since, until, limit)
    return [
        _item(r, request_id=str(r["request_id"]) if r["request_id"] else None, text_cap=text_cap)
        for r in rows
    ]
