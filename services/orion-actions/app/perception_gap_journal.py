"""Fold "things I could not name today" into the daily journal seed.

Sibling of `capability_gap_journal.py`, same rule: not a new journal entry, one
key on the seed of the entry Orion already writes, omitted entirely on a day
with nothing in it -- so a quiet day (or a host without the walkway migration)
produces a seed byte-identical to the one before this existed.

Source: `vision_unresolved` (walkway spec idea 4,
docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md), written
when the vision council returned uncertainties, no label cleared threshold, or
embedding surprise spiked. Curiosity sees the most recent three as study
material; this block makes sure they get written down even on a day curiosity
does not pick them up.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from .vision_pg import fetch_rows

logger = logging.getLogger("orion-actions.perception_gap_journal")

# Same cap and same truncation discipline as capability_gaps: producer text is
# bounded before it reaches a prompt.
MAX_GAPS_IN_SEED = 12
MAX_DETAIL_CHARS = 320
# Read a little past the cap so the seed can say how many were left out.
_READ_LIMIT = 200

UNRESOLVED_WINDOW_SQL = """
SELECT unresolved_id, stream_id, camera_id, observed_at, reason, description,
       what_was_tried, count(*) OVER () AS window_total
FROM vision_unresolved
WHERE observed_at >= %(start)s AND observed_at < %(end)s
ORDER BY observed_at DESC
LIMIT %(limit)s
"""


def _truncate(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _ts(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=timezone.utc) if raw.tzinfo is None else raw.astimezone(timezone.utc)
    if isinstance(raw, str) and raw.strip():
        try:
            return _ts(datetime.fromisoformat(raw.strip().replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def summarize_perception_gaps(
    rows: Iterable[dict[str, Any]], *, cap: int | None = MAX_GAPS_IN_SEED
) -> list[dict[str, Any]]:
    """Pure: `vision_unresolved` rows in, seed dicts out.

    Keeps the NEWEST `MAX_GAPS_IN_SEED`, displayed chronologically. A row with
    no id, no time, and no description is skipped rather than rendered as a
    hollow entry.
    """
    items: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _ts(row.get("observed_at"))
        uid = str(row.get("unresolved_id") or "").strip()
        detail = _truncate(str(row.get("description") or ""))
        if ts is None or not uid or not detail:
            continue
        items.append(
            (
                ts,
                {
                    "unresolved_id": uid,
                    "observed_at": ts.isoformat(),
                    "stream_id": row.get("stream_id"),
                    "reason": str(row.get("reason") or ""),
                    "detail": detail,
                    "what_was_tried": [_truncate(t, 80) for t in _as_list(row.get("what_was_tried"))][:6],
                },
            )
        )
    items.sort(key=lambda it: it[0], reverse=True)
    kept = items if cap is None else items[:cap]
    kept.sort(key=lambda it: it[0])
    return [d for _, d in kept]


async def collect_perception_gaps(
    *, dsn: str | None, window_start_utc: str, window_end_utc: str
) -> tuple[list[dict[str, Any]], int]:
    """Window in, (seed dicts, total rows in window) out. ([], 0) on any failure."""
    start, end = _ts(window_start_utc), _ts(window_end_utc)
    if start is None or end is None:
        logger.warning(
            "perception_gap_window_unparseable start=%r end=%r", window_start_utc, window_end_utc
        )
        return [], 0
    rows = await fetch_rows(
        dsn,
        UNRESOLVED_WINDOW_SQL,
        {"start": start, "end": end, "limit": _READ_LIMIT},
        label="perception_gaps",
    )
    if not rows:
        return [], 0
    gaps = summarize_perception_gaps(rows)
    # True window total (not capped at _READ_LIMIT) minus hollow rows we
    # read but could not render, so `perception_gaps_omitted` is honest.
    try:
        total = int(rows[0].get("window_total") or len(rows))
    except (TypeError, ValueError):
        total = len(rows)
    hollow = len(rows) - len(summarize_perception_gaps(rows, cap=None))
    return gaps, max(len(gaps), total - hollow)
