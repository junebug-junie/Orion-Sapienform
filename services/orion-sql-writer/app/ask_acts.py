"""Open one ``vision.ask`` act per camera ask Orion made (attention patch 4a).

Design: docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md, "4a". Rules and grade:
orion/grammar/camera_ask_acts.py. Every opener writes its card to ``orion_ask`` (app/vision_individuals.py,
vision_handoff_loop.py, vision_office_sessions.py), so the acts are opened here from those rows rather
than in each opener: one place, every opener covered, no change to how or when a card opens.

Runs at the start of each act-resolver tick (app/act_resolver_loop.py, every
SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC, 60 s): an ask made in the last ``LIVE_LOOKBACK`` that has no act gets
one, opened at the ask's own ``created_at``; the same tick's resolver then closes the acts whose ask is
already answered. Juniper's own story-naming cards are excluded (``ORION_MADE_CAMERA_ASK_SQL``). Older asks
are the back-fill's (scripts/backfill_camera_ask_acts.py), never this loop's.

Writes ``grammar_acts`` directly through the act ledger's ``open_act`` (sql-writer is the ledger); the open is
not published on ``orion:grammar:event``. Env: SQL_WRITER_ACT_CAMERA_ASKS_ENABLED (default true).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from orion.grammar.camera_ask_acts import ORION_MADE_CAMERA_ASK_SQL, ask_act_ids, ask_opened_event, ask_rule

logger = logging.getLogger("sql-writer.ask_acts")

LIVE_LOOKBACK = timedelta(hours=24)

ASKS_WITHOUT_ACTS_SQL = f"""
SELECT a.ask_id, a.source_kind, a.evidence_refs, a.created_at, a.expires_at, a.status, i.kind AS individual_kind
  FROM orion_ask a
  LEFT JOIN vision_individual i ON i.individual_id = split_part(a.source_ref, '|', 1)
 WHERE {ORION_MADE_CAMERA_ASK_SQL}
   AND a.created_at >= :since AND a.created_at <= :until
 ORDER BY a.created_at, a.ask_id
"""


def asks_needing_acts(conn: Any, *, since: datetime, until: datetime) -> list[dict[str, Any]]:
    """Orion-made camera asks created in [since, until] with no act yet, each with its rule and act id."""
    from sqlalchemy import text

    rows = [dict(r) for r in conn.execute(text(ASKS_WITHOUT_ACTS_SQL), {"since": since, "until": until}).mappings()]
    if not rows:
        return []
    ids = ask_act_ids([r["ask_id"] for r in rows])
    have = {str(r[0]) for r in conn.execute(text("SELECT act_id FROM grammar_acts WHERE act_id = ANY(:ids)"),
                                             {"ids": list(ids.values())})}
    out = []
    for r in rows:
        if ids[r["ask_id"]] in have:
            continue
        r["act_id"] = ids[r["ask_id"]]
        r["rule"] = ask_rule(str(r["source_kind"]), r["evidence_refs"], r.get("individual_kind"))
        out.append(r)
    return out


def open_ask_acts(conn: Any, *, now: Optional[datetime] = None, since: Optional[datetime] = None) -> list[str]:
    """Open the missing acts; returns their act ids. The caller owns the transaction."""
    from orion.grammar.act_ledger import open_act

    now = now or datetime.now(timezone.utc)
    opened = []
    for r in asks_needing_acts(conn, since=since or now - LIVE_LOOKBACK, until=now):
        open_act(conn, ask_opened_event(r["ask_id"], rule=r["rule"], created_at=r["created_at"],
                                        expires_at=r["expires_at"]))
        opened.append(r["act_id"])
        logger.info("ask_act_opened ask_id=%s rule=%s act_id=%s", r["ask_id"], r["rule"], r["act_id"])
    return opened


__all__ = ["ASKS_WITHOUT_ACTS_SQL", "LIVE_LOOKBACK", "asks_needing_acts", "open_ask_acts"]
