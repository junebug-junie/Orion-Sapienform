from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    asyncpg = None

try:
    from .settings import settings
except ImportError:  # pragma: no cover - test harness path
    from settings import settings  # type: ignore

try:
    from .chat_source_tagging import render_quoted_chat_text
except ImportError:  # pragma: no cover - test harness path
    from chat_source_tagging import render_quoted_chat_text  # type: ignore


@dataclass
class ChatItem:
    id: str
    ts: float
    text: str
    source_ref: str


def _to_epoch(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


async def _fetch_rows_from_table(
    conn: Any, table: str, select_cols: str, id_col: str, ids: List[str], extra_where: str = ""
) -> List[Any]:
    query = f"""
        SELECT {select_cols}
        FROM {table}
        WHERE {id_col} = ANY($1::text[])
          {extra_where}
    """
    return await conn.fetch(query, ids)


async def fetch_chat_turn_timestamps(
    turn_ids: List[str],
    since_minutes: int,
) -> Dict[str, float]:
    """Resolve created_at (epoch) for chat turns by id, bounded to the last ``since_minutes``.

    Used to window RDF chat-turn recall, which carries no usable timestamp in the graph
    (turns are joined back to ``chat_history_log`` on the turn id). Ids outside the window,
    or not present in either chat table, are simply absent from the returned map so callers
    can drop them.

    AI Town table split (docs/superpowers/specs/2026-08-19-aitown-table-
    split-phase2-recall-migration-design.md): queries
    ``RECALL_SQL_AITOWN_CHAT_TABLE`` separately and merges in Python
    (mirror table applied last, so it wins any conflict) rather than a
    single ``UNION ALL`` -- deliberately NOT relying on "an id lives in
    exactly one table" as a given fact. As of 2026-08-19 that invariant is
    NOT yet true of what's merged to ``main``: orion-sql-writer's Phase 1
    dual-write (``SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED``, additive, both
    tables) is still the live write path there; a separate cutover PR
    retires it in favor of routing (one table, never both) but had not
    merged as of this PR (see that PR's own description for the explicit
    merge-order note). This two-query merge is correct regardless of which
    write-side state is actually live -- during any window where an id
    exists in both tables, the mirror's value deterministically wins,
    matching this file's own documented Shape 1 recommendation; once the
    cutover lands, the second query simply finds nothing extra and this
    degrades to the single-table result with no behavior change.
    """
    if asyncpg is None:
        return {}
    ids = [str(t).strip() for t in (turn_ids or []) if str(t).strip()]
    if not ids:
        return {}
    id_col = settings.RECALL_SQL_CHAT_ID_COL
    created_at_col = settings.RECALL_SQL_CHAT_CREATED_AT_COL
    select_cols = f"{id_col} AS id, {created_at_col} AS created_at"
    time_clause = f"AND {created_at_col} >= NOW() - INTERVAL '{int(since_minutes)} minutes'"
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            primary_rows = await _fetch_rows_from_table(
                conn, settings.RECALL_SQL_CHAT_TABLE, select_cols, id_col, ids, time_clause
            )
            mirror_rows = await _fetch_rows_from_table(
                conn, settings.RECALL_SQL_AITOWN_CHAT_TABLE, select_cols, id_col, ids, time_clause
            )
        finally:
            await conn.close()
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for row in (*primary_rows, *mirror_rows):  # mirror processed last -> wins on conflict
        rid = str(row.get("id") or "").strip()
        if rid:
            out[rid] = _to_epoch(row.get("created_at"))
    return out


async def fetch_chat_turns_by_id(turn_ids: List[str]) -> Dict[str, tuple[str, str, Any]]:
    """Resolve (prompt, response, client_meta) text for chat turns by id.

    Used by storage/falkor_chat_adapter.py, storage/falkor_neighborhood_adapter.py,
    and worker.py -- the Falkor ChatTurn node is deliberately thin
    (turn_id/source_kind/session_id/ts/correlation_id, no prompt/response
    text; Postgres owns that, see
    services/orion-meta-tags/README.md's Falkor writer section) so a
    Falkor-backed chatturn fragment needs this join for the actual quoted
    text. Ids not present in either chat table are simply absent from the
    returned map so callers can drop them, same contract as
    fetch_chat_turn_timestamps above.

    Same two-query-merge, mirror-wins-on-conflict approach as
    fetch_chat_turn_timestamps -- see that function's docstring for why
    this deliberately does not assume "an id lives in exactly one table"
    via a plain UNION ALL.

    client_meta is the third tuple element (not folded into pre-rendered
    text here) so all three callers can share render_quoted_chat_text --
    each independently built the same unlabeled 'ExactUserText: "..."'
    format before this, none aware of client_meta.external_room.platform.
    """
    if asyncpg is None:
        return {}
    ids = [str(t).strip() for t in (turn_ids or []) if str(t).strip()]
    if not ids:
        return {}
    id_col = settings.RECALL_SQL_CHAT_ID_COL
    select_cols = (
        f"{id_col} AS id, {settings.RECALL_SQL_CHAT_TEXT_COL} AS prompt, "
        f"{settings.RECALL_SQL_CHAT_RESPONSE_COL} AS response, client_meta"
    )
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            primary_rows = await _fetch_rows_from_table(
                conn, settings.RECALL_SQL_CHAT_TABLE, select_cols, id_col, ids
            )
            mirror_rows = await _fetch_rows_from_table(
                conn, settings.RECALL_SQL_AITOWN_CHAT_TABLE, select_cols, id_col, ids
            )
        finally:
            await conn.close()
    except Exception:
        return {}

    out: Dict[str, tuple[str, str, Any]] = {}
    for row in (*primary_rows, *mirror_rows):  # mirror processed last -> wins on conflict
        rid = str(row.get("id") or "").strip()
        if rid:
            out[rid] = (str(row.get("prompt") or ""), str(row.get("response") or ""), row.get("client_meta"))
    return out


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _contains_active_prompt(record_text: str, active_text: str) -> bool:
    normalized_active = _normalize_text(active_text)
    if not normalized_active:
        return False
    normalized_record = _normalize_text(record_text)
    if not normalized_record:
        return False
    return normalized_active in normalized_record


async def fetch_chat_history_pairs(
    limit: int,
    since_minutes: int,
    *,
    exclude_text: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ChatItem]:
    """AI Town chat-history table split: unions ``RECALL_SQL_AITOWN_CHAT_TABLE``
    in alongside the primary table. Unlike fetch_chat_turn_timestamps/
    fetch_chat_turns_by_id above, this is a recency scan, not an id-batch
    lookup -- there is no known id set to query the two tables separately
    against and merge, so this keeps ``UNION ALL`` (matching this file's
    own accepted shape-2/3 tradeoff): during any window where the same
    turn genuinely exists in both tables (Phase 1's dual-write, still the
    live write path on ``main`` as of 2026-08-19 -- see
    fetch_chat_turn_timestamps' docstring), that turn could occupy two of
    the ``limit`` slots instead of one. Accepted as a bounded, self-
    resolving cosmetic risk (a duplicate quoted fragment, not lost data),
    not a correctness bug -- per the design doc's own recommendation for
    this query shape. ORDER BY/LIMIT applied over the combined UNION ALL
    result rather than per-table (a per-table LIMIT would risk starving
    one table's rows out of the top-N entirely). ``id``/``source_ref`` are
    now real per-row values (id/correlation_id genuinely selected, source
    table via a per-branch literal) instead of the previous behavior,
    which never selected either column and always fell back to a
    ``{table}:{idx}`` placeholder id mislabeled with the primary table's
    name regardless of which table a row actually came from.
    """
    if asyncpg is None:
        return []
    # Postgres names a UNION's output columns from the first SELECT, so
    # ORDER BY/LIMIT below can reference the aliases (prompt/response/
    # created_at) directly with no wrapping subquery needed.
    id_col = settings.RECALL_SQL_CHAT_ID_COL
    query = f"""
        SELECT {id_col} AS row_id,
               correlation_id,
               {settings.RECALL_SQL_CHAT_TEXT_COL} AS prompt,
               {settings.RECALL_SQL_CHAT_RESPONSE_COL} AS response,
               {settings.RECALL_SQL_CHAT_CREATED_AT_COL} AS created_at,
               client_meta,
               '{settings.RECALL_SQL_CHAT_TABLE}' AS source_ref
        FROM {settings.RECALL_SQL_CHAT_TABLE}
        WHERE {settings.RECALL_SQL_CHAT_CREATED_AT_COL} >= NOW() - INTERVAL '{since_minutes} minutes'
        UNION ALL
        SELECT {id_col} AS row_id,
               correlation_id,
               {settings.RECALL_SQL_CHAT_TEXT_COL} AS prompt,
               {settings.RECALL_SQL_CHAT_RESPONSE_COL} AS response,
               {settings.RECALL_SQL_CHAT_CREATED_AT_COL} AS created_at,
               client_meta,
               '{settings.RECALL_SQL_AITOWN_CHAT_TABLE}' AS source_ref
        FROM {settings.RECALL_SQL_AITOWN_CHAT_TABLE}
        WHERE {settings.RECALL_SQL_CHAT_CREATED_AT_COL} >= NOW() - INTERVAL '{since_minutes} minutes'
        ORDER BY created_at DESC
        LIMIT {limit}
    """
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    except Exception:
        return []

    items: List[ChatItem] = []
    excluded_id_set = {str(v).strip() for v in (exclude_ids or []) if str(v).strip()}
    suppressed = 0
    for idx, row in enumerate(rows):
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        created_at = row.get("created_at")
        source_ref = row.get("source_ref") or settings.RECALL_SQL_CHAT_TABLE
        row_id = str(row.get("correlation_id") or row.get("row_id") or f"{source_ref}:{idx}")
        if row_id in excluded_id_set:
            suppressed += 1
            continue
        if _contains_active_prompt(prompt, exclude_text or ""):
            suppressed += 1
            continue
        text = render_quoted_chat_text(prompt, response, row.get("client_meta"))
        items.append(
            ChatItem(
                id=row_id,
                ts=_to_epoch(created_at),
                text=text,
                source_ref=source_ref,
            )
        )
    if suppressed:
        logger.info("sql_chat self-hit suppression backend=chat_pairs suppressed=%s", suppressed)
    return items


async def fetch_chat_messages(
    limit: int,
    since_minutes: int,
    *,
    exclude_text: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ChatItem]:
    table = settings.RECALL_SQL_MESSAGE_TABLE
    if not table:
        return []
    if asyncpg is None:
        return []
    query = f"""
        SELECT {settings.RECALL_SQL_MESSAGE_ROLE_COL} AS role,
               {settings.RECALL_SQL_MESSAGE_TEXT_COL} AS text,
               {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} AS created_at
        FROM {table}
        WHERE {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} >= NOW() - INTERVAL '{since_minutes} minutes'
        ORDER BY {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} DESC
        LIMIT {limit}
    """
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    except Exception:
        return []

    items: List[ChatItem] = []
    excluded_id_set = {str(v).strip() for v in (exclude_ids or []) if str(v).strip()}
    suppressed = 0
    for idx, row in enumerate(rows):
        role = row.get("role") or "unknown"
        text = row.get("text") or ""
        created_at = row.get("created_at")
        row_id = str(row.get("correlation_id") or row.get("id") or f"{table}:{idx}")
        if row_id in excluded_id_set:
            suppressed += 1
            continue
        if role.lower() == "user" and _contains_active_prompt(text, exclude_text or ""):
            suppressed += 1
            continue
        items.append(
            ChatItem(
                id=row_id,
                ts=_to_epoch(created_at),
                text=f"{role}: {text}",
                source_ref=table,
            )
        )
    if suppressed:
        logger.info("sql_chat self-hit suppression backend=chat_messages suppressed=%s", suppressed)
    return items
