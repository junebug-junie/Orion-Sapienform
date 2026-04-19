"""Read bounded discussion windows from chat_history_log (Postgres via SQLAlchemy)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Sequence

from orion.schemas.discussion_window import (
    DiscussionWindowRequestV1,
    DiscussionWindowResultV1,
    DiscussionWindowTurnV1,
)

logger = logging.getLogger("orion.discussion_window.sql_fetch")

# Break contiguity when adjacent turns are farther apart than this (seconds).
_DEFAULT_CONTIGUITY_MAX_GAP_SEC = 5400


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trim_contiguous_suffix(
    rows: Sequence[dict[str, Any]],
    *,
    max_gap_sec: int = _DEFAULT_CONTIGUITY_MAX_GAP_SEC,
    max_turns: int,
) -> List[dict[str, Any]]:
    """
    Rows must be sorted ascending by created_at.
    Keep the most recent contiguous cluster: walk backward from the last row
    while each gap to the previous row is <= max_gap_sec.
    """
    if not rows:
        return []
    cluster: List[dict[str, Any]] = [rows[-1]]
    for i in range(len(rows) - 2, -1, -1):
        cur = rows[i]["created_at"]
        nxt = rows[i + 1]["created_at"]
        if not isinstance(cur, datetime):
            cur = datetime.fromisoformat(str(cur))
        if not isinstance(nxt, datetime):
            nxt = datetime.fromisoformat(str(nxt))
        cur = _ensure_utc(cur)
        nxt = _ensure_utc(nxt)
        gap = (nxt - cur).total_seconds()
        if gap > max_gap_sec:
            break
        cluster.append(rows[i])
    cluster.reverse()
    if len(cluster) > max_turns:
        cluster = cluster[-max_turns:]
    return cluster


def _format_transcript(turns: List[DiscussionWindowTurnV1]) -> str:
    lines: List[str] = []
    for t in turns:
        ts = _ensure_utc(t.created_at).strftime("%Y-%m-%d %H:%M:%S UTC")
        src = (t.source or "").strip()
        corr = (t.correlation_id or "").strip()
        head = f"[{ts}]"
        if src:
            head += f" source={src}"
        if corr:
            head += f" correlation_id={corr}"
        lines.append(f"{head}\nUser: {t.prompt.strip()}\nAssistant: {t.response.strip()}\n")
    return "\n".join(lines).strip()


def fetch_discussion_window(database_url: str, request: DiscussionWindowRequestV1) -> DiscussionWindowResultV1:
    from sqlalchemy import create_engine, text

    end = request.end_time_utc or datetime.now(timezone.utc)
    end = _ensure_utc(end)
    start = end - timedelta(seconds=int(request.lookback_seconds))

    sql = text(
        """
        SELECT created_at, source, user_id, correlation_id, prompt, response
        FROM chat_history_log
        WHERE created_at >= :start_ts AND created_at <= :end_ts
          AND (:user_id IS NULL OR user_id = :user_id)
          AND (:source IS NULL OR source = :source)
        ORDER BY created_at ASC
        LIMIT 500
        """
    )
    params = {
        "start_ts": start,
        "end_ts": end,
        "user_id": request.user_id,
        "source": request.source,
    }

    engine = create_engine(database_url, pool_pre_ping=True)
    raw_rows: List[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, params)
            for row in result.mappings():
                raw_rows.append(dict(row))
    except Exception as exc:
        logger.exception("discussion_window_sql_failed: %s", exc)
        raise

    filtered: List[dict[str, Any]] = []
    for r in raw_rows:
        p = str(r.get("prompt") or "").strip()
        resp = str(r.get("response") or "").strip()
        if request.require_prompt_and_response and (not p or not resp):
            continue
        filtered.append(r)

    clustered = _trim_contiguous_suffix(filtered, max_turns=request.max_turns)
    turns: List[DiscussionWindowTurnV1] = []
    for r in clustered:
        ca = r.get("created_at")
        if isinstance(ca, datetime):
            cat = _ensure_utc(ca)
        else:
            cat = _ensure_utc(datetime.fromisoformat(str(ca)))
        turns.append(
            DiscussionWindowTurnV1(
                created_at=cat,
                source=str(r.get("source") or "") or None,
                user_id=str(r.get("user_id") or "") or None,
                correlation_id=str(r.get("correlation_id") or "") or None,
                prompt=str(r.get("prompt") or ""),
                response=str(r.get("response") or ""),
            )
        )

    w_start = start
    w_end = end
    if turns:
        w_start = turns[0].created_at
        w_end = turns[-1].created_at

    transcript = _format_transcript(turns)
    return DiscussionWindowResultV1(
        window_start_utc=w_start,
        window_end_utc=w_end,
        turn_count=len(turns),
        source=request.source,
        user_id=request.user_id,
        turns=turns,
        transcript_text=transcript,
        selection_strategy="time_bound_then_contiguous_suffix",
    )
