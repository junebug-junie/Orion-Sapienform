"""Read/format endogenous outreach provenance for follow-up unified turns.

Two consumers share one selection rule ("the latest unsolicited assistant row
in this session that no later solicited reply has cleared, within a window"):

- **Provenance injection** (`fetch_latest_outreach_provenance`) -- needs the
  row's `outreach_provenance` capsule, so it only considers unsolicited rows
  that carry one.
- **Reply stamp** (`fetch_reply_target` / `reply_stamp_for_session`,
  2026-09-22) -- needs only the row's correlation id, so it considers every
  unsolicited row. Juniper's next inbound turn gets
  `client_meta.in_reply_to = <that row's id>` so a run story can show her
  answer next to the outreach that prompted it. Heuristic ("next message in
  that session within 12h"), labelled as such by the UI, never a fact.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Mapping, Optional, Sequence

logger = logging.getLogger("orion-hub.outreach_provenance")

OUTREACH_PROVENANCE_SCHEMA = "outreach_provenance.v1"

# Hard ceiling on rows one lookup may load. A 12h window of ONE session is
# far below this in practice; the bound exists so a runaway session cannot
# turn the inbound-chat path's best-effort lookup into a full-table load.
SESSION_ROWS_LIMIT = 500

# The inbound chat turn waits at most this long for the stamp lookup. The
# thread keeps running if it times out; the turn simply proceeds unstamped.
REPLY_STAMP_TIMEOUT_SEC = 1.5

_BLOCK_HEADER = (
    "Your last message to Juniper was unsolicited endogenous outreach. "
    "Here is the exact generation prompt that produced it. If they ask where "
    "that came from, answer from this block — do not invent collapse-mirror "
    "or any other frame that is not present here."
)


def _valid_outreach_capsule(capsule: Any) -> Optional[Dict[str, Any]]:
    """Return capsule only when schema + prompt_text are well-formed; else None."""
    if not isinstance(capsule, dict):
        return None
    if capsule.get("schema") != OUTREACH_PROVENANCE_SCHEMA:
        return None
    prompt = capsule.get("prompt_text")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    return capsule


def _meta_unsolicited(meta: Mapping[str, Any]) -> bool:
    """True for the live jsonb boolean AND the string forms.

    `endogenous_outreach._publish_history` writes `{"unsolicited": True}`;
    sql-writer stores it as a jsonb boolean and SQLAlchemy hands it back as
    Python `True`. Before 2026-09-22 this compared `str(True)` (`"True"`)
    against `"true"` and so never matched a real row -- confirmed against the
    live table (`jsonb_typeof(client_meta->'unsolicited') = 'boolean'` on
    every unsolicited row), which means the provenance injection this file
    was built for had been silently selecting nothing.
    """
    value = meta.get("unsolicited")
    if value is True:
        return True
    return str(value or "").strip().lower() == "true"


def select_active_unsolicited_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_capsule: bool,
) -> Optional[Mapping[str, Any]]:
    """Pure clearing contract over ordered session history rows.

    ``rows`` must be oldest-first (``created_at`` ascending), matching the
    thin SQL reader in ``_fetch_session_rows``.

    Returns the latest unsolicited row (when ``require_capsule``, only rows
    carrying an ``outreach_provenance`` key count), provided no later
    non-unsolicited assistant response (non-empty ``response``) exists after
    it. A normal reply clears it — same semantics as the prior SQL
    ``NOT EXISTS`` gate.
    """
    latest_idx: Optional[int] = None

    for i, row in enumerate(rows):
        meta = row.get("client_meta") or {}
        if not isinstance(meta, dict):
            continue
        if not _meta_unsolicited(meta):
            continue
        if require_capsule and "outreach_provenance" not in meta:
            continue
        latest_idx = i

    if latest_idx is None:
        return None

    for later in rows[latest_idx + 1 :]:
        response = later.get("response")
        if response is None or str(response) == "":
            continue
        meta = later.get("client_meta") or {}
        if isinstance(meta, dict) and _meta_unsolicited(meta):
            continue
        # Later non-unsolicited assistant response clears it.
        return None

    return rows[latest_idx]


def select_active_outreach_provenance(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validated capsule from the latest still-active unsolicited row that
    carries ``outreach_provenance``; see ``select_active_unsolicited_row``."""
    row = select_active_unsolicited_row(rows, require_capsule=True)
    if row is None:
        return None
    meta = row.get("client_meta") or {}
    capsule = meta.get("outreach_provenance") if isinstance(meta, dict) else None
    if isinstance(capsule, dict):
        return _valid_outreach_capsule(dict(capsule))
    return None


def select_reply_target(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The unsolicited row Juniper's next message is most plausibly answering.

    Same clearing rule as provenance, without the capsule requirement (a
    curiosity reach-out carries no capsule). Returns
    ``{"correlation_id": <row correlation_id>, "source": <client_meta.source
    or None>}`` or ``None``. The join key the run story uses is the
    ``correlation_id`` column (the run-derived uuid5); ``id`` is only a
    fallback for a row that somehow lacks one (sql-writer sets both to the
    same value today, but that is its invariant, not this reader's). A row
    with neither cannot be pointed at: ``None`` rather than an empty stamp.
    """
    row = select_active_unsolicited_row(rows, require_capsule=False)
    if row is None:
        return None
    row_id = str(row.get("correlation_id") or row.get("id") or "").strip()
    if not row_id:
        return None
    meta = row.get("client_meta") or {}
    source = meta.get("source") if isinstance(meta, dict) else None
    return {
        "correlation_id": row_id,
        "source": str(source).strip() if source else None,
    }


def reply_stamp_from_target(target: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """`client_meta` keys to merge into an inbound turn, or `{}`."""
    if not target:
        return {}
    corr = str(target.get("correlation_id") or "").strip()
    if not corr:
        return {}
    stamp: Dict[str, Any] = {"in_reply_to": corr}
    source = target.get("source")
    if source:
        stamp["in_reply_to_source"] = str(source)
    return stamp


def format_outreach_provenance_block(capsule: Dict[str, Any] | None) -> str:
    validated = _valid_outreach_capsule(capsule)
    if validated is None:
        return ""
    prompt = validated["prompt_text"].strip()
    summary = str(validated.get("summary_line") or "").strip()
    corr = str(validated.get("correlation_id") or "").strip()
    lines = [_BLOCK_HEADER, ""]
    if summary:
        lines.append(f"Summary: {summary}")
    if corr:
        lines.append(f"correlation_id: {corr}")
    lines.extend(["", "Generation prompt:", prompt])
    return "\n".join(lines).strip()


def merge_situation_with_outreach_provenance(
    situation: Optional[str],
    block: Optional[str],
) -> Optional[str]:
    sit = str(situation or "").strip()
    blk = str(block or "").strip()
    if sit and blk:
        return f"{sit}\n\n{blk}"
    if sit:
        return sit
    if blk:
        return blk
    return None


def _fetch_session_rows(
    session_id: Optional[str],
    *,
    max_age_hours: float,
    what: str,
) -> Optional[list[Dict[str, Any]]]:
    """Age-windowed `chat_history_log` rows for one session, oldest first.

    ``None`` (not ``[]``) on any failure or when there is nothing to query
    with, so callers can tell "no rows" from "could not look". Bounded to the
    newest ``SESSION_ROWS_LIMIT`` rows of the window, then re-sorted ascending
    for the pure selectors. No index covers (session_id, created_at) today
    (checked live 2026-09-22: 472 rows, seq scan); the window + limit keep
    this cheap regardless.
    """
    sid = str(session_id or "").strip()
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not sid or not uri:
        return None
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s_sqlalchemy_import_failed err=%s", what, exc)
        return None
    engine = None
    try:
        engine = create_engine(uri, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT id, correlation_id, created_at, client_meta, response
                    FROM (
                        SELECT id, correlation_id, created_at, client_meta, response
                        FROM chat_history_log
                        WHERE session_id = :sid
                          AND created_at >= now() - make_interval(secs => :max_age_secs)
                        ORDER BY created_at DESC
                        LIMIT :lim
                    ) newest
                    ORDER BY created_at ASC
                    """
                ),
                {
                    "sid": sid,
                    "max_age_secs": float(max_age_hours) * 3600.0,
                    "lim": int(SESSION_ROWS_LIMIT),
                },
            )
            return [dict(r) for r in result.mappings().all()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s_fetch_failed sid=%s err=%s", what, sid, exc)
        return None
    finally:
        if engine is not None:
            engine.dispose()


def fetch_latest_outreach_provenance(
    session_id: Optional[str],
    *,
    max_age_hours: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Latest still-relevant outreach capsule for this session, or None.

    Thin SQL reader: loads age-windowed chat_history_log rows for the session,
    then applies ``select_active_outreach_provenance`` (unsolicited+provenance
    only if no later non-unsolicited assistant response clears it).
    """
    rows = _fetch_session_rows(
        session_id, max_age_hours=max_age_hours, what="outreach_provenance"
    )
    if rows is None:
        return None
    return select_active_outreach_provenance(rows)


def fetch_reply_target(
    session_id: Optional[str],
    *,
    max_age_hours: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """The unsolicited row an inbound message in this session is answering.

    Sync (SQLAlchemy); run it through ``reply_stamp_for_session`` from the
    event loop. ``None`` when nothing qualifies OR when the lookup failed --
    both mean "no stamp", which is the only safe answer on this path.
    """
    rows = _fetch_session_rows(session_id, max_age_hours=max_age_hours, what="reply_stamp")
    if rows is None:
        return None
    return select_reply_target(rows)


async def reply_stamp_for_session(
    session_id: Optional[str],
    *,
    max_age_hours: float = 12.0,
    timeout_sec: float = REPLY_STAMP_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """`client_meta` keys for an inbound turn (`in_reply_to`,
    `in_reply_to_source`) or `{}`. NEVER raises and never waits longer than
    ``timeout_sec``: a DB outage or a slow scan must not delay Juniper's turn,
    it only costs the stamp. The lookup runs on a worker thread so Hub's
    event loop is not blocked by the sync SQLAlchemy call.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return {}
    try:
        target = await asyncio.wait_for(
            asyncio.to_thread(fetch_reply_target, sid, max_age_hours=max_age_hours),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("reply_stamp_timeout sid=%s timeout_sec=%s", sid, timeout_sec)
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("reply_stamp_failed sid=%s err=%s", sid, exc)
        return {}
    return reply_stamp_from_target(target)
