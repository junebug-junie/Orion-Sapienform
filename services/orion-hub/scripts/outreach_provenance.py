"""Read/format endogenous outreach provenance for follow-up unified turns."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional, Sequence

logger = logging.getLogger("orion-hub.outreach_provenance")

OUTREACH_PROVENANCE_SCHEMA = "outreach_provenance.v1"

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
    return str(meta.get("unsolicited") or "").strip() == "true"


def select_active_outreach_provenance(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pure clearing contract over ordered session history rows.

    ``rows`` must be oldest-first (``created_at`` ascending), matching the
    thin SQL reader in ``fetch_latest_outreach_provenance``.

    Returns the validated capsule from the latest unsolicited row that carries
    ``outreach_provenance``, only when no later non-unsolicited assistant
    response (non-empty ``response``) exists after it. A normal reply clears
    injection — same semantics as the prior SQL ``NOT EXISTS`` gate.
    """
    latest_idx: Optional[int] = None
    latest_capsule: Optional[Dict[str, Any]] = None

    for i, row in enumerate(rows):
        meta = row.get("client_meta") or {}
        if not isinstance(meta, dict):
            continue
        if not _meta_unsolicited(meta):
            continue
        if "outreach_provenance" not in meta:
            continue
        capsule = meta.get("outreach_provenance")
        latest_idx = i
        if isinstance(capsule, dict):
            latest_capsule = _valid_outreach_capsule(dict(capsule))
        else:
            latest_capsule = None

    if latest_idx is None:
        return None

    for later in rows[latest_idx + 1 :]:
        response = later.get("response")
        if response is None or str(response) == "":
            continue
        meta = later.get("client_meta") or {}
        if isinstance(meta, dict) and _meta_unsolicited(meta):
            continue
        # Later non-unsolicited assistant response clears injection.
        return None

    return latest_capsule


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
            result = conn.execute(
                text(
                    """
                    SELECT created_at,
                           client_meta,
                           response
                    FROM chat_history_log
                    WHERE session_id = :sid
                      AND created_at >= now() - make_interval(secs => :max_age_secs)
                    ORDER BY created_at ASC
                    """
                ),
                {"sid": sid, "max_age_secs": float(max_age_hours) * 3600.0},
            )
            rows = [dict(r) for r in result.mappings().all()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_provenance_fetch_failed sid=%s err=%s", sid, exc)
        return None
    finally:
        engine.dispose()
    return select_active_outreach_provenance(rows)
