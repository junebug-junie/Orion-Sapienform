"""The market's funded event wants, for Orion Prime's concerns (attention as the want competition, patch 2).

``GET /physiology/event-wants`` (app.main). Read-only on ``want_ledger`` and the two v1 source tables:

1. the LATEST decision of every event want (``decision->'want' ? 'event_ref'``, real producers only) in
   the last ``hours``;
2. kept when that decision is ``funded`` and the event has not expired: the want's own window end, else
   ``opened_at`` + the family's ``deferrable_window_s``; a camera ask that is no longer ``open`` has
   expired too (Juniper answered it);
3. ordered by the market's net (highest first), then ``limit``.

This is the same market's record, not a second menu: nothing here scores or re-decides a want. The
label is the source row's own words (a read's title, an ask's question, an office visit's label), so Prime can say what the
event is; a cooling-act want's label comes from its own ref (the cabinet reading's node and time).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from orion.physiology.choice import SYNTHETIC_PRODUCERS
from orion.physiology.families import FamiliesConfig
from orion.schemas.physiology_want import FundedEventWantsV1, FundedEventWantV1

READ_PREFIX = "world_pulse_read_seed:"
ASK_PREFIX = "orion_ask:"
# The cooling world act (ladder retirement PR 5, orion/autonomy/shed_act.py): cabinet_reading:<node>:<UTC second>.
CABINET_PREFIX = "cabinet_reading:"
# An office visit (office camera design step C, services/orion-hub/scripts/office_reach_out.py).
OFFICE_PREFIX = "office_session:"
MAX_HOURS = 72.0
MAX_LIMIT = 50

LATEST_EVENT_DECISIONS_SQL = """
SELECT DISTINCT ON (want_id) want_id, family, outcome, decided_at, net,
       (decision->>'value_rank')::float AS value_rank,
       decision->'want'->>'event_ref' AS event_ref,
       (decision->'want'->>'salience')::float AS salience,
       decision->'want'->>'opened_at' AS opened_at,
       decision->'want'->'window'->>'latest' AS latest
  FROM want_ledger
 WHERE decided_at >= %(since)s AND decision->'want' ? 'event_ref' AND NOT (producer = ANY(%(synthetic)s))
 ORDER BY want_id, decided_at DESC
"""

READ_LABELS_SQL = "SELECT seed_id, title, url FROM world_pulse_read_seed WHERE seed_id = ANY(%(ids)s)"
ASK_LABELS_SQL = "SELECT ask_id, question, status FROM orion_ask WHERE ask_id = ANY(%(ids)s)"
OFFICE_TABLE_SQL = "SELECT to_regclass('vision_office_session') IS NOT NULL AS present"
OFFICE_LABELS_SQL = "SELECT session_id, label, who_text FROM vision_office_session WHERE session_id = ANY(%(ids)s)"


def _ts(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def expires_at(row: Mapping[str, Any], families: Optional[FamiliesConfig]) -> Optional[datetime]:
    """The want's window end: its own ``latest``, else opened_at + the family window (as Choice reads it)."""
    latest = _ts(row.get("latest"))
    if latest is not None:
        return latest
    opened = _ts(row.get("opened_at"))
    cfg = families.get(str(row.get("family"))) if families is not None else None
    if opened is not None and cfg is not None and cfg.deferrable_window_s:
        return opened + timedelta(seconds=cfg.deferrable_window_s)
    return None


def menu_from_rows(rows: Sequence[Mapping[str, Any]], *, labels: Mapping[str, str], gone: set[str],
                   families: Optional[FamiliesConfig], now: datetime, limit: int) -> FundedEventWantsV1:
    """Pure: the funded, unexpired event wants, highest net first. ``gone``: event refs whose source row
    says the event is over (an answered ask)."""
    keep: list[FundedEventWantV1] = []
    for r in rows:
        ref = str(r.get("event_ref") or "")
        if r.get("outcome") != "funded" or not ref or ref in gone:
            continue
        end = expires_at(r, families)
        if end is not None and end <= now:
            continue
        keep.append(FundedEventWantV1(
            want_id=str(r["want_id"]), family=str(r["family"]), event_ref=ref,
            salience=r.get("salience"), value_rank=r.get("value_rank"), net=r.get("net"),
            decided_at=_ts(r["decided_at"]), expires_at=end, label=labels.get(ref, "")))
    keep.sort(key=lambda w: (-(w.net if w.net is not None else float("-inf")), w.event_ref))
    return FundedEventWantsV1(as_of=now, considered=len(rows), funded_open=len(keep), wants=keep[:limit])


async def read_menu(pool: Any, families: Optional[FamiliesConfig], *, now: datetime, hours: float,
                    limit: int) -> FundedEventWantsV1:
    hours = min(MAX_HOURS, max(0.1, float(hours)))
    limit = min(MAX_LIMIT, max(1, int(limit)))
    async with pool.connection() as conn:
        rows = await (await conn.execute(LATEST_EVENT_DECISIONS_SQL, {
            "since": now - timedelta(hours=hours), "synthetic": sorted(SYNTHETIC_PRODUCERS)})).fetchall()
        refs = [str(r["event_ref"] or "") for r in rows if r["outcome"] == "funded"]
        read_ids = sorted({x[len(READ_PREFIX):] for x in refs if x.startswith(READ_PREFIX)})
        ask_ids = sorted({x[len(ASK_PREFIX):] for x in refs if x.startswith(ASK_PREFIX)})
        office_ids = sorted({x[len(OFFICE_PREFIX):] for x in refs if x.startswith(OFFICE_PREFIX)})
        labels: dict[str, str] = {}
        gone: set[str] = set()
        if read_ids:
            for r in await (await conn.execute(READ_LABELS_SQL, {"ids": read_ids})).fetchall():
                labels[f"{READ_PREFIX}{r['seed_id']}"] = f"a read finished: {r['title'] or r['url']}"[:300]
        for ref in refs:
            if ref.startswith(CABINET_PREFIX):
                # The ref is the reading itself (node and time), so the label needs no source-row read.
                node, _, at = ref[len(CABINET_PREFIX):].partition(":")
                labels[ref] = f"the {node} cabinet was warming at {at} (Orion's cooling act)"[:300]
        if ask_ids:
            for r in await (await conn.execute(ASK_LABELS_SQL, {"ids": ask_ids})).fetchall():
                ref = f"{ASK_PREFIX}{r['ask_id']}"
                labels[ref] = f"an open camera question to Juniper: {r['question']}"[:300]
                if r["status"] != "open":
                    gone.add(ref)
        present = (await (await conn.execute(OFFICE_TABLE_SQL)).fetchall()) if office_ids else []
        if present and present[0]["present"]:
            # The visit row's own label: "Juniper (probably) in the office: sitting at a desk, ...". A
            # visit that ended is still a concern until the want's window closes (Juniper may answer later).
            for r in await (await conn.execute(OFFICE_LABELS_SQL, {"ids": office_ids})).fetchall():
                labels[f"{OFFICE_PREFIX}{r['session_id']}"] = (
                    r["label"] or f"{r['who_text'] or 'someone'} in the office")[:300]
    return menu_from_rows(rows, labels=labels, gone=gone, families=families, now=now, limit=limit)


__all__ = ["LATEST_EVENT_DECISIONS_SQL", "OFFICE_LABELS_SQL", "OFFICE_TABLE_SQL", "expires_at", "menu_from_rows", "read_menu"]
