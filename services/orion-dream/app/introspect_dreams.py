"""Read-only dream lookups for the orion-introspect ``dreams`` tool.

Two record kinds: narrative dreams (``dreams``, written by orion-sql-writer)
and sleep-cycle hypotheses (``dream_hypothesis``, written by the v2 cycle).
Hypotheses are a blind experiment (orion/dream/hypotheses.py): curiosity shows
each one once with the arm hidden. So only hypotheses already offered to Orion
are ever returned, from both arms, and ``arm`` / ``ref_a`` / ``ref_b`` are
never selected. Every statement here is a SELECT.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from sqlalchemy import text

from orion.schemas.introspect import (
    DEFAULT_TEXT_CAP,
    FULL_TEXT_CAP,
    SHORT_FIELD_CAP,
    THEME_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    clip_text,
)

Kind = Literal["narrative", "hypothesis"]
NARRATIVE_PREFIX = "dream:"
THEME_CHARS = 80
INDEX_SCAN_LIMIT = 1000

# dreams.created_at is timestamp without time zone written by now() on a UTC
# server; AT TIME ZONE 'UTC' turns it into the timestamptz it always meant.
_N_OCCURRED = "(d.created_at AT TIME ZONE 'UTC')"
_N_COLS = f"d.id, d.dream_date, d.tldr, d.themes, d.narrative, {_N_OCCURRED} AS occurred_at"
_H_COLS = "h.hypothesis_id, h.cycle_id, h.claim, h.why, h.offered_at AS occurred_at, h.expires_at"
_SINCE = "(CAST(:since AS timestamptz) IS NULL OR {col} >= CAST(:since AS timestamptz))"

NARRATIVE_RECENT_SQL = f"""
SELECT {_N_COLS}, count(*) OVER () AS total FROM dreams d
WHERE d.created_at IS NOT NULL AND {_SINCE.format(col=_N_OCCURRED)}
ORDER BY occurred_at DESC, d.id DESC
LIMIT :limit
"""

NARRATIVE_BY_IDS_SQL = f"""
SELECT {_N_COLS} FROM dreams d
WHERE d.id = ANY(:ids) AND d.created_at IS NOT NULL AND {_SINCE.format(col=_N_OCCURRED)}
"""

HYPOTHESIS_RECENT_SQL = f"""
SELECT {_H_COLS}, count(*) OVER () AS total FROM dream_hypothesis h
WHERE h.offered_at IS NOT NULL AND {_SINCE.format(col="h.offered_at")}
ORDER BY h.offered_at DESC, h.hypothesis_id DESC
LIMIT :limit
"""

HYPOTHESIS_BY_IDS_SQL = f"""
SELECT {_H_COLS} FROM dream_hypothesis h
WHERE h.hypothesis_id = ANY(:ids) AND h.offered_at IS NOT NULL AND {_SINCE.format(col="h.offered_at")}
"""

HYPOTHESIS_SQL = (HYPOTHESIS_RECENT_SQL, HYPOTHESIS_BY_IDS_SQL)


def _rows(conn: Any, sql: str, **params: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def _aware(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def doc_id(kind: Kind, row: dict[str, Any]) -> str:
    return f"{NARRATIVE_PREFIX}{row['id']}" if kind == "narrative" else str(row["hypothesis_id"])


def split_ids(ids: Iterable[str]) -> tuple[list[int], list[str]]:
    narratives, hypotheses = [], []
    for item_id in ids:
        if item_id.startswith(NARRATIVE_PREFIX):
            tail = item_id[len(NARRATIVE_PREFIX):]
            if tail.isascii() and tail.isdigit():
                narratives.append(int(tail))
        elif item_id.startswith("dh-"):
            hypotheses.append(item_id)
    return narratives, hypotheses


def narrative_item(row: dict[str, Any], *, text_cap: int = DEFAULT_TEXT_CAP) -> IntrospectItemV1:
    body = "\n\n".join(p for p in (str(row["tldr"] or "").strip(), str(row["narrative"] or "").strip()) if p)
    body_text, truncated = clip_text(body, text_cap)
    themes = row["themes"] if isinstance(row["themes"], list) else []
    return IntrospectItemV1(
        id=doc_id("narrative", row), occurred_at=_aware(row["occurred_at"]), kind="dream_narrative",
        epistemic_status="unsettled", text=body_text, truncated=truncated,
        extra={
            "dream_date": row["dream_date"].isoformat() if row["dream_date"] else None,
            "themes": [clip_text(str(t), THEME_CHARS)[0] for t in themes if str(t).strip()][:THEME_CAP],
        },
    )


def hypothesis_item(
    row: dict[str, Any], *, now: datetime, text_cap: int = DEFAULT_TEXT_CAP,
) -> IntrospectItemV1:
    claim, why = str(row["claim"] or "").strip(), str(row["why"] or "").strip()
    body_text, truncated = clip_text(f"{claim}\nWhy: {why}" if why else claim, text_cap)
    expires = row["expires_at"]
    return IntrospectItemV1(
        id=doc_id("hypothesis", row), occurred_at=_aware(row["occurred_at"]), kind="dream_hypothesis",
        epistemic_status="unsettled", text=body_text, truncated=truncated,
        extra={
            "cycle_id": clip_text(str(row["cycle_id"] or ""), SHORT_FIELD_CAP)[0],
            "expired": bool(expires is not None and _aware(expires) <= now),
        },
    )


def _result(items: list[IntrospectItemV1], total: int, now: datetime) -> IntrospectResultV1:
    return IntrospectResultV1(ok=True, operation="dreams", as_of=now, total_available=total, items=items)


def recent(
    conn: Any, *, kind: Kind | None, since: datetime | None, limit: int, now: datetime,
) -> IntrospectResultV1:
    items: list[IntrospectItemV1] = []
    total = 0
    if kind in (None, "narrative"):
        rows = _rows(conn, NARRATIVE_RECENT_SQL, since=since, limit=limit)
        total += int(rows[0]["total"]) if rows else 0
        items += [narrative_item(r) for r in rows]
    if kind in (None, "hypothesis"):
        rows = _rows(conn, HYPOTHESIS_RECENT_SQL, since=since, limit=limit)
        total += int(rows[0]["total"]) if rows else 0
        items += [hypothesis_item(r, now=now) for r in rows]
    items.sort(key=lambda i: (i.occurred_at, i.id), reverse=True)
    return _result(items[:limit], total, now)


def by_ids(
    conn: Any, scored: list[tuple[str, float]], *,
    kind: Kind | None, since: datetime | None, limit: int, now: datetime,
) -> IntrospectResultV1:
    """Re-read ranked hits from Postgres (the index is not the record); rank order kept."""
    narrative_ids, hypothesis_ids = split_ids(sid for sid, _ in scored)
    found: dict[str, IntrospectItemV1] = {}
    if narrative_ids and kind in (None, "narrative"):
        for r in _rows(conn, NARRATIVE_BY_IDS_SQL, ids=narrative_ids, since=since):
            found[doc_id("narrative", r)] = narrative_item(r)
    if hypothesis_ids and kind in (None, "hypothesis"):
        for r in _rows(conn, HYPOTHESIS_BY_IDS_SQL, ids=hypothesis_ids, since=since):
            found[doc_id("hypothesis", r)] = hypothesis_item(r, now=now)
    hits = [
        found[sid].model_copy(update={"extra": {**found[sid].extra, "similarity": round(sim, 3)}})
        for sid, sim in scored if sid in found
    ]
    return _result(hits[:limit], len(hits), now)


def one(conn: Any, dream_id: str, *, now: datetime) -> IntrospectResultV1:
    narrative_ids, hypothesis_ids = split_ids([dream_id])
    items: list[IntrospectItemV1] = []
    if narrative_ids:
        items = [narrative_item(r, text_cap=FULL_TEXT_CAP)
                 for r in _rows(conn, NARRATIVE_BY_IDS_SQL, ids=narrative_ids, since=None)]
    elif hypothesis_ids:
        items = [hypothesis_item(r, now=now, text_cap=FULL_TEXT_CAP)
                 for r in _rows(conn, HYPOTHESIS_BY_IDS_SQL, ids=hypothesis_ids, since=None)]
    return _result(items[:1], min(len(items), 1), now)


def index_rows(conn: Any) -> list[tuple[Kind, dict[str, Any]]]:
    pairs: list[tuple[Kind, dict[str, Any]]] = []
    pairs += [("narrative", r) for r in _rows(conn, NARRATIVE_RECENT_SQL, since=None, limit=INDEX_SCAN_LIMIT)]
    pairs += [("hypothesis", r) for r in _rows(conn, HYPOTHESIS_RECENT_SQL, since=None, limit=INDEX_SCAN_LIMIT)]
    return pairs
