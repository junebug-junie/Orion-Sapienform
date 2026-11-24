"""Juniper's engagement with a game line, graded for the family weight (game company patch 4).

Spec: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md section 4 ("Did the player
like it"): "The economy's weight for a family already has a term for Juniper's engagement. ... The build wires
lines to the same record. A guest's thumb is stored but is not counted as Juniper's."

The same record: the family weight's engagement source (b) is a grammar act with signal ``juniper_replied``
closed by the horizon resolver (orion/physiology/weights.py ``evidence_for``). So a line said while the page
was told "Juniper" opens its act (actor ``hub.retro_commentary``, verb ``say``) with signal ``juniper_replied``
and horizon ``LINE_ENGAGE_HORIZON_SEC`` (services/orion-hub/scripts/act_emit.py ``publish_line_said``), and the
resolver (orion/grammar/act_ledger.py ``resolve_due_acts``) grades it here once the horizon has passed:

1. Juniper's thumb on the line inside the horizon decides: up 1, down 0 (her own verdict on THIS line).
2. Else 1 when she answered it within ``LINE_REPLY_WINDOW_SEC``: the line row's ``replied_by = 'juniper'``
   (a chat-box message, retro_commentary.py REPLIED_SQL), or any Juniper chat turn that started in that window
   (the resolver's own reply read, which never counts a guest's row).
3. Else 0.

A line said to a guest, or with nobody stated, keeps the patch 3 act (``artifact_produced``, closed 1 at
delivery): it is no record of Juniper's engagement, so a guest's thumb never reaches the weight. A line row
that cannot be found closes ``expired`` (no grade: the weight counts only resolver closes).

Pure: the resolver passes the row and the clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

LINE_ACTOR = "hub.retro_commentary"
LINE_REF_PREFIX = "retro_line:"
# How long Juniper's verdict on a line can arrive. The line strip shows the newest line with its thumbs until the
# next line or a dismissal, and lines are at least 2 minutes apart (HUB_RETRO_LINE_MIN_GAP_SEC): 5 minutes covers
# a line and the gap after it without grading a thumb given to a later line's neighbour.
LINE_ENGAGE_HORIZON_SEC = 300.0
# "A chat-box reply within 2 minutes of a line also counts as a response" (spec section 4).
LINE_REPLY_WINDOW_SEC = 120.0
JUNIPER = "juniper"


@dataclass(frozen=True)
class LineGrade:
    value: float
    observed_at: datetime
    source: str      # resolver | expired
    note: str


def _aware(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, datetime):
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def line_id_of(test_before: Optional[str]) -> Optional[str]:
    t = str(test_before or "")
    if not t.startswith(LINE_REF_PREFIX):
        return None
    return t[len(LINE_REF_PREFIX):] or None


def line_grade(line: Optional[Mapping[str, Any]], *, opened_at: datetime, horizon_at: datetime, now: datetime,
               juniper_turn_at: Optional[datetime]) -> Optional[LineGrade]:
    """The grade of one Juniper-stated line's act, or None while its horizon has not passed."""
    opened_at, horizon_at, now = _aware(opened_at), _aware(horizon_at), _aware(now)
    if now < horizon_at:
        return None
    if line is None:
        return LineGrade(0.0, horizon_at, "expired", "line row not found: no engagement grade")
    thumb_at = _aware(line.get("thumb_at"))
    if (line.get("thumb_by") == JUNIPER and line.get("thumb") in ("up", "down") and thumb_at is not None
            and opened_at < thumb_at <= horizon_at):
        up = line.get("thumb") == "up"
        return LineGrade(1.0 if up else 0.0, thumb_at, "resolver", f"Juniper's thumb {'up' if up else 'down'}")
    reply_end = opened_at + timedelta(seconds=LINE_REPLY_WINDOW_SEC)
    replied_at = _aware(line.get("replied_at"))
    if line.get("replied_by") == JUNIPER and replied_at is not None and opened_at < replied_at <= reply_end:
        return LineGrade(1.0, replied_at, "resolver", "Juniper answered in the game chat box within 2 min")
    turn = _aware(juniper_turn_at)
    if turn is not None and opened_at < turn <= reply_end:
        return LineGrade(1.0, turn, "resolver", "juniper chat turn started within 2 min of the line")
    return LineGrade(0.0, horizon_at, "resolver", "no thumb from Juniper and no reply within 2 min")


__all__ = ["JUNIPER", "LINE_ACTOR", "LINE_ENGAGE_HORIZON_SEC", "LINE_REF_PREFIX", "LINE_REPLY_WINDOW_SEC",
           "LineGrade", "line_grade", "line_id_of"]
