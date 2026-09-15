"""Read/format endogenous outreach provenance for follow-up unified turns."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("orion-hub.outreach_provenance")

_BLOCK_HEADER = (
    "Your last message to Juniper was unsolicited endogenous outreach. "
    "Here is the exact generation prompt that produced it. If they ask where "
    "that came from, answer from this block — do not invent collapse-mirror "
    "or any other frame that is not present here."
)


def format_outreach_provenance_block(capsule: Dict[str, Any] | None) -> str:
    if not isinstance(capsule, dict):
        return ""
    prompt = str(capsule.get("prompt_text") or "").strip()
    if not prompt:
        return ""
    summary = str(capsule.get("summary_line") or "").strip()
    corr = str(capsule.get("correlation_id") or "").strip()
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


def fetch_latest_outreach_provenance(
    session_id: Optional[str],
    *,
    max_age_hours: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Latest still-relevant outreach capsule for this session, or None.

    Selects the newest chat_history_log row for the session that carries
    client_meta.unsolicited + outreach_provenance, only if no later
    non-unsolicited assistant response exists after it (so a normal reply
    clears the injection).
    """
    sid = str(session_id or "").strip()
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not sid or not uri:
        return None
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_provenance_sqlalchemy_import_failed err=%s", exc)
        return None
    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    WITH latest AS (
                      SELECT created_at,
                             client_meta,
                             response
                      FROM chat_history_log
                      WHERE session_id = :sid
                        AND created_at >= now() - make_interval(secs => :max_age_secs)
                        AND coalesce(client_meta->>'unsolicited', '') = 'true'
                        AND client_meta ? 'outreach_provenance'
                      ORDER BY created_at DESC
                      LIMIT 1
                    )
                    SELECT l.client_meta
                    FROM latest l
                    WHERE NOT EXISTS (
                      SELECT 1
                      FROM chat_history_log later
                      WHERE later.session_id = :sid
                        AND later.created_at > l.created_at
                        AND coalesce(later.response, '') <> ''
                        AND coalesce(later.client_meta->>'unsolicited', '') <> 'true'
                    )
                    """
                ),
                {"sid": sid, "max_age_secs": float(max_age_hours) * 3600.0},
            ).mappings().first()
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_provenance_fetch_failed sid=%s err=%s", sid, exc)
        return None
    finally:
        engine.dispose()
    if not row:
        return None
    meta = row.get("client_meta") or {}
    if not isinstance(meta, dict):
        return None
    capsule = meta.get("outreach_provenance")
    return dict(capsule) if isinstance(capsule, dict) else None
