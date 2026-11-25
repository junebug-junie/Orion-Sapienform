"""Undo one "same person" link Juniper made on a walkway story card (who-card redesign Patch D).

docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md, section 3 ("Undo, mirroring the split
undo"). The story card asks "Orion has another Sasha group. Same person?"; her yes is an already-answered
``vision_same_individual`` ask (``answer_detail.via = 'motion_story'``) that ``_apply_pair_answer`` turns
into a ``same`` row in ``vision_individual_pair_verdict``. This module removes that row again.

Scope: only links that card made. A typed-name link already has its own undo (retyping the name drops
it, ``_unlink_old_name``); this never reaches it.

Refused, naming what is in the way (nothing changes):
- not a link this card made, already undone, or the row is no longer hers (changed since);
- older than ``LINK_UNDO_WINDOW`` (7 days from the row's ``decided_at``);
- a split recorded the pair in its ``linked_pair``;
- a name typed on any individual of the linked group after the link was made (a name she gave
  believing they were one person);
- **still joined** (Orion's question 3): another ``same`` row (a split's, a typed name's, another
  answer's) joins the same two individuals, so removing this one would change nothing she could see.

Labels are re-derived, never reverted: every copied label left in either group follows the newest
name Juniper typed in what is left of its group, or is cleared (``rederive_copied_labels``). A typed
name is never touched. A pair joined by ``same`` links AND carrying a ``different`` row between two
members of its group is a contradiction: it is reported in ``contradictions`` and logged.

Two doors, one function: ``scripts/undo_vision_link.py`` (dry run by default) and the Hub's
"Undo" button, which only marks the ask ``answer_detail.undo_requested_at``; the individuals cycle
applies it under its own lock (``apply_link_undos``), so this service stays the single writer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.vision_individuals import (
    INDIVIDUALS_LOCK_KEY,
    SOURCE_KIND_SAME,
    _detail,
    parse_pair_ref,
    parse_yes_no,
    rederive_copied_labels,
    same_groups,
)

logger = logging.getLogger("sql-writer.vision_link_undo")

LINK_UNDO_WINDOW = timedelta(days=7)
CARD_VIA = "motion_story"


def _same_edges(conn, *, skip: Optional[tuple] = None) -> List[tuple]:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT individual_a, individual_b, ask_id FROM vision_individual_pair_verdict WHERE verdict = 'same'")).fetchall()
    return [(r[0], r[1], r[2]) for r in rows if skip is None or (r[0], r[1]) != skip]


def _path(edges: List[tuple], a: str, b: str) -> List[tuple]:
    """The 'same' rows on one path from a to b (breadth first), [] when none."""
    adj: Dict[str, List[tuple]] = {}
    for x, y, ask in edges:
        adj.setdefault(x, []).append((y, (x, y, ask)))
        adj.setdefault(y, []).append((x, (x, y, ask)))
    prev: Dict[str, Optional[tuple]] = {a: None}
    queue = [a]
    while queue:
        cur = queue.pop(0)
        if cur == b:
            break
        for nxt, edge in adj.get(cur, []):
            if nxt not in prev:
                prev[nxt] = (cur, edge)
                queue.append(nxt)
    if b not in prev:
        return []
    out = []
    cur = b
    while prev[cur] is not None:
        cur, edge = prev[cur]
        out.append(edge)
    return list(reversed(out))


def contradictions(conn, group: List[str]) -> List[str]:
    """'different' rows between two members of one 'same' group: she said both."""
    from sqlalchemy import text

    if len(group) < 2:
        return []
    return [f"{r[0]}|{r[1]}:different(ask {r[2]})" for r in conn.execute(text(
        "SELECT individual_a, individual_b, ask_id FROM vision_individual_pair_verdict WHERE verdict = 'different' "
        "AND individual_a = ANY(:g) AND individual_b = ANY(:g) ORDER BY 1, 2"), {"g": list(group)}).fetchall()]


def undo_link(conn, ask_id: str, *, now: Optional[datetime] = None, lock: bool = True) -> Dict[str, Any]:
    """Undo one card-made link inside ``conn``'s transaction (the caller commits or rolls back).
    ``lock``: take the individuals cycle's lock (False when the cycle itself is calling)."""
    from sqlalchemy import text

    now = now or datetime.now(timezone.utc)
    if lock:
        conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})
    ask = conn.execute(text(
        "SELECT source_kind, source_ref, status, answer, answer_detail FROM orion_ask WHERE ask_id = :a FOR UPDATE"),
        {"a": ask_id}).fetchone()
    if ask is None:
        return {"ok": False, "reason": "no such ask"}
    detail = _detail(ask[4])
    if ask[0] != SOURCE_KIND_SAME or detail.get("via") != CARD_VIA or parse_yes_no(ask[3]) is not True:
        return {"ok": False, "reason": "not a 'same person' link made on a walkway story card; only those undo here"}
    if detail.get("undone_at"):
        return {"ok": False, "reason": f"already undone at {detail['undone_at']}"}
    pair = parse_pair_ref(ask[1])
    if pair is None:
        return {"ok": False, "reason": f"bad pair ref {ask[1]!r}"}
    a, b = pair
    row = conn.execute(text(
        "SELECT verdict, ask_id, decided_at FROM vision_individual_pair_verdict WHERE individual_a = :a "
        "AND individual_b = :b FOR UPDATE"), {"a": a, "b": b}).fetchone()
    if row is None or row[0] != "same" or row[1] != ask_id:
        return {"ok": False, "reason": "the link is no longer this answer's (removed or answered again since)",
                "pair_now": None if row is None else {"verdict": row[0], "ask_id": row[1]}}
    decided = row[2]
    if decided is not None and decided.tzinfo is None:
        decided = decided.replace(tzinfo=timezone.utc)
    if decided is not None and now - decided > LINK_UNDO_WINDOW:
        return {"ok": False, "reason": f"older than {LINK_UNDO_WINDOW.days} days (made {decided.isoformat()})"}
    edges = _same_edges(conn)
    groups = same_groups((x, y) for x, y, _ in edges)
    root = groups.get(a, a)
    group = sorted(i for i, r in groups.items() if r == root)
    report: Dict[str, Any] = {"ask_id": ask_id, "pair": f"{a}|{b}", "group": group,
                              "contradictions": contradictions(conn, group)}
    if report["contradictions"]:
        logger.warning("[VISION_LINK_CONTRADICTION] ask_id=%s group=%s different=%s", ask_id, ",".join(group),
                       ";".join(report["contradictions"]))
    # Dependents are found BEFORE anything is deleted.
    splits = [r[0] for r in conn.execute(text(
        "SELECT split_id FROM vision_individual_split WHERE undone_at IS NULL AND linked_pair @> "
        "CAST(:p AS jsonb)"), {"p": json.dumps([{"individual_a": a, "individual_b": b}])}).fetchall()]
    if splits:
        return {**report, "ok": False, "reason": "a split recorded this link; undo the split first",
                "splits": splits}
    # Any name she typed on the linked person since, whichever card wrote it (a who-is-this, a story
    # card, an "Is this <name>?", a name box on a "same person?" card): read from the labels themselves.
    names = [r[0] for r in conn.execute(text(
        "SELECT coalesce(label_ask_id, individual_id) FROM vision_individual WHERE individual_id = ANY(:g) "
        "AND label IS NOT NULL AND NOT label_copied AND labeled_at > :d AND coalesce(label_ask_id, '') <> :k "
        "ORDER BY labeled_at"), {"g": group, "d": decided or now, "k": ask_id}).fetchall()]
    if names:
        return {**report, "ok": False, "reason": "a name was typed on this linked person after the link; Juniper decides",
                "name_asks": names}
    # Orion's question 3: another 'same' row still joining the two would make the undo look like it did nothing.
    still = _path(_same_edges(conn, skip=(a, b)), a, b)
    if still:
        return {**report, "ok": False,
                "reason": "still joined through other 'same' links (a split's, a typed name's or another answer's); "
                          "undoing this one would change nothing",
                "still_joined_by": [f"{x}|{y}(ask {k})" for x, y, k in still]}
    conn.execute(text(
        "DELETE FROM vision_individual_pair_verdict WHERE individual_a = :a AND individual_b = :b AND verdict = 'same' "
        "AND ask_id = :k"), {"a": a, "b": b, "k": ask_id})
    relabelled = rederive_copied_labels(conn, [a, b], ask_id, now)
    conn.execute(text(
        "UPDATE orion_ask SET answer_detail = coalesce(answer_detail, '{}'::jsonb) || "
        "jsonb_build_object('undone_at', CAST(:t AS text)) WHERE ask_id = :a"), {"t": now.isoformat(), "a": ask_id})
    if report["contradictions"]:
        conn.execute(text(
            "UPDATE orion_ask SET answer_detail = answer_detail || jsonb_build_object('contradictions', "
            "CAST(:c AS jsonb)) WHERE ask_id = :a"), {"c": json.dumps(report["contradictions"]), "a": ask_id})
    logger.info("[VISION_LINK_UNDONE] ask_id=%s pair=%s|%s relabelled=%d", ask_id, a, b, len(relabelled))
    return {**report, "ok": True, "relabelled": relabelled}


def apply_link_undos(conn, now: datetime) -> Dict[str, int]:
    """Undos the Hub requested (``answer_detail.undo_requested_at``), each in its own savepoint. A refusal is
    written back as ``answer_detail.undo_refused`` (the card shows it), never retried."""
    from sqlalchemy import text

    out = {"links_undone": 0, "link_undos_refused": 0}
    ids = [r[0] for r in conn.execute(text(
        "SELECT ask_id FROM orion_ask WHERE source_kind = :k AND status = 'answered' "
        "AND answer_detail ? 'undo_requested_at' AND NOT (answer_detail ? 'undone_at') "
        "AND NOT (answer_detail ? 'undo_refused') ORDER BY answered_at"), {"k": SOURCE_KIND_SAME}).fetchall()]
    for ask_id in ids:
        try:
            with conn.begin_nested():
                rep = undo_link(conn, ask_id, now=now, lock=False)
                if not rep["ok"]:
                    raise _Refused(rep)
            out["links_undone"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad undo must not block the others, nor loop
            if isinstance(exc, _Refused):
                reason = exc.report.get("reason") or "refused"
                extra = {k: exc.report[k] for k in ("still_joined_by", "splits", "name_asks", "contradictions")
                         if exc.report.get(k)}
            else:
                logger.warning("vision_link_undo_error ask_id=%s error=%s", ask_id, exc)
                reason, extra = f"error: {type(exc).__name__}: {exc}"[:200], {}
            conn.execute(text(
                "UPDATE orion_ask SET answer_detail = answer_detail || jsonb_build_object('undo_refused', "
                "CAST(:r AS text), 'undo_refused_detail', CAST(:d AS jsonb)) WHERE ask_id = :a"),
                {"r": reason, "d": json.dumps(extra), "a": ask_id})
            logger.info("[VISION_LINK_UNDO_REFUSED] ask_id=%s reason=%s %s", ask_id, reason, json.dumps(extra))
            out["link_undos_refused"] += 1
    return out


class _Refused(Exception):
    def __init__(self, report: Dict[str, Any]):
        super().__init__(report.get("reason"))
        self.report = report


__all__ = ["CARD_VIA", "LINK_UNDO_WINDOW", "apply_link_undos", "contradictions", "undo_link"]
