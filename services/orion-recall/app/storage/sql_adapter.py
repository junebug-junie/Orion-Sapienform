# services/orion-recall/app/storage/sql_adapter.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Any, Dict, Optional

import json
import psycopg2

from ..settings import settings
from ..types import Fragment


def _connect():
    return psycopg2.connect(settings.RECALL_PG_DSN)


def _safe_tags(raw: Any) -> List[str]:
    """
    Normalize tags into a list[str].
    Accepts:
      - JSONB/list from DB
      - JSON-encoded string
      - bare string
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        # try to parse as JSON list first
        s = raw.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            # fallback: treat as single tag-ish string
            return [s]
    return [str(raw).strip()]


def _epoch(ts_val: Any, default: Optional[float] = None) -> float:
    """
    Best-effort to get epoch seconds from a DB timestamp.
    """
    if ts_val is None:
        return default or datetime.utcnow().timestamp()
    if isinstance(ts_val, datetime):
        return ts_val.timestamp()
    # string → try parse
    try:
        return datetime.fromisoformat(str(ts_val)).timestamp()
    except Exception:
        return default or datetime.utcnow().timestamp()


def fetch_sql_fragments(
    time_window_days: int = 30,
    include_chat: bool = True,
    include_mirrors: bool = True,
) -> List[Fragment]:
    """
    Pulls recent fragments from Postgres:

      - collapse_mirror          → kind='collapse'
      - collapse_enrichment      → kind='enrichment'
      - chat_history_log         → kind='chat'

    NOTE: This intentionally only selects columns we *know* exist in your
    current schema, mirroring the working Dream service:
      collapse_mirror:
        id, summary, trigger, observer, field_resonance, type, mantra, timestamp

      collapse_enrichment:
        id, collapse_id, tags, salience, ts

      chat_history_log:
        trace_id, prompt, response, created_at
    """
    time_window_days = max(1, min(int(time_window_days), 365))
    since_dt = datetime.utcnow() - timedelta(days=time_window_days)

    conn = _connect()
    cur = conn.cursor()
    frags: List[Fragment] = []

    # ---------------------------------------------------------
    # 1) Collapse Mirror base table
    # ---------------------------------------------------------
    if include_mirrors:
        try:
            cur.execute(
                f"""
                SELECT
                    id,
                    {settings.RECALL_SQL_MIRROR_SUMMARY_COL}    AS summary,
                    {settings.RECALL_SQL_MIRROR_TRIGGER_COL}    AS trigger,
                    {settings.RECALL_SQL_MIRROR_OBSERVER_COL}   AS observer,
                    {settings.RECALL_SQL_MIRROR_FIELD_RESONANCE_COL} AS field_resonance,
                    {settings.RECALL_SQL_MIRROR_TYPE_COL}       AS type,
                    {settings.RECALL_SQL_MIRROR_MANTRA_COL}     AS mantra,
                    {settings.RECALL_SQL_MIRROR_TS_COL}         AS ts_raw
                FROM {settings.RECALL_SQL_MIRROR_TABLE}
                WHERE NULLIF({settings.RECALL_SQL_MIRROR_TS_COL}, '')::timestamptz >= %s
                ORDER BY NULLIF({settings.RECALL_SQL_MIRROR_TS_COL}, '')::timestamptz DESC
                """,
                (since_dt,),
            )
            rows = cur.fetchall()
            for row in rows:
                (
                    cm_id,
                    summary,
                    trigger,
                    observer,
                    field_res,
                    cm_type,
                    mantra,
                    ts_raw,
                ) = row

                text_bits = []
                if summary:
                    text_bits.append(str(summary))
                elif trigger:
                    text_bits.append(str(trigger))

                # simple context glue; we can get fancier later
                context_bits = []
                if observer:
                    context_bits.append(f"observer={observer}")
                if field_res:
                    context_bits.append(f"field_resonance={field_res}")
                if cm_type:
                    context_bits.append(f"type={cm_type}")
                if mantra:
                    context_bits.append(f"mantra={mantra}")

                txt = " ".join(text_bits).strip()
                if context_bits:
                    ctx = " | ".join(context_bits)
                    txt = f"{txt}\n[{ctx}]" if txt else ctx

                frags.append(
                    Fragment(
                        id=str(cm_id),
                        source="sql",
                        kind="collapse",
                        text=txt or "(empty collapse)",
                        ts=_epoch(ts_raw),
                        salience=0.0,
                        tags=[],
                        meta={
                            "observer": observer,
                            "field_resonance": field_res,
                            "type": cm_type,
                            "mantra": mantra,
                        },
                    )
                )
        except Exception as e:
            print(f"⚠️ RECALL SQL: error fetching collapse_mirror rows: {e}")

        # ---------------------------------------------------------
        # 2) Collapse Enrichment (tags, salience)
        # ---------------------------------------------------------
        try:
            cur.execute(
                f"""
                SELECT
                    id,
                    {settings.RECALL_SQL_ENRICH_COLLAPSE_ID_COL} AS collapse_id,
                    {settings.RECALL_SQL_ENRICH_TAGS_COL}        AS tags,
                    {settings.RECALL_SQL_ENRICH_SALIENCE_COL}    AS salience,
                    {settings.RECALL_SQL_ENRICH_TS_COL}          AS ts_raw
                FROM {settings.RECALL_SQL_ENRICH_TABLE}
                WHERE {settings.RECALL_SQL_ENRICH_TS_COL}::timestamptz >= %s::timestamptz
                ORDER BY {settings.RECALL_SQL_ENRICH_TS_COL} DESC
                """,
                (since_dt,),
            )
            rows = cur.fetchall()
            for row in rows:
                (
                    enr_id,
                    collapse_id,
                    tags_raw,
                    salience,
                    ts_raw,
                ) = row

                tags = _safe_tags(tags_raw)
                txt = " ".join(tags) or "enrichment"

                frags.append(
                    Fragment(
                        id=str(enr_id),
                        source="sql",
                        kind="enrichment",
                        text=txt,
                        ts=_epoch(ts_raw),
                        salience=0,
                        tags=tags,
                        meta={
                            "collapse_id": collapse_id,
                        },
                    )
                )
        except Exception as e:
            print(f"⚠️ RECALL SQL: error fetching collapse_enrichment rows: {e}")

    # ---------------------------------------------------------
    # 3) Chat history (random-ish slice within window)
    # ---------------------------------------------------------
    if include_chat:
        try:
            cur.execute(
                f"""
                SELECT
                    correlation_id,
                    {settings.RECALL_SQL_CHAT_TEXT_COL}      AS prompt,
                    {settings.RECALL_SQL_CHAT_RESPONSE_COL}  AS response,
                    {settings.RECALL_SQL_CHAT_CREATED_AT_COL} AS created_at
                FROM {settings.RECALL_SQL_CHAT_TABLE}
                WHERE {settings.RECALL_SQL_CHAT_CREATED_AT_COL} >= %s
                ORDER BY {settings.RECALL_SQL_CHAT_CREATED_AT_COL} DESC
                LIMIT 300
                """,
                (since_dt,),
            )
            rows = cur.fetchall()
            for row in rows:
                trace_id, prompt, response, created_at = row
                prompt = (prompt or "").strip()
                response = (response or "").strip()
                if not (prompt or response):
                    continue

                text = f"User: {prompt}\nOrion: {response}".strip()

                frags.append(
                    Fragment(
                        id=str(trace_id or f"chat_{_epoch(created_at)}"),
                        source="sql",
                        kind="chat",
                        text=text,
                        ts=_epoch(created_at),
                        salience=1,
                        tags=["dialogue"],
                        meta={},
                    )
                )
        except Exception as e:
            print(f"⚠️ RECALL SQL: error fetching chat_history_log rows: {e}")

    cur.close()
    conn.close()
    return frags
