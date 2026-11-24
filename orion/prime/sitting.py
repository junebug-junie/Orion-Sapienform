"""A game sitting wakes Prime, twice at most (game company patch 4, the one change to Prime's own rules).

Spec: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md section 6 part 7. Funded event
wants and attention moves do not wake Prime (both made Prime churn), so without this a four-hour sitting could pass
with no Prime thought at all. Two facts can be shown to Prime, each at most once per sitting:

* ``sitting_opened``: the open sitting has run ``OPEN_WAKE_AFTER_SEC`` (3 minutes);
* ``sitting_closed``: a sitting Prime saw open (past 3 minutes) is no longer the open one.

Each is a pending change whose text names ``retro_sitting:<id>``, so a thought woken by it names it in its act's
close note (``woke: ...``, orion/mindloop). ``note_sitting_thought`` counts every thought that opens while a change
naming the sitting is pending; once ``MAX_THOUGHTS_PER_SITTING`` (2) have, any change still naming it is dropped and
no new one is added (a try that never reached the model keeps its pending changes: without the cap the next try
would name the sitting a third time). Prime still bids for every thought, and its refractory rest still holds.

Where it is read: the gateway's Redis copy of the open sitting (``orion:retro:sitting``, orion/retro/sitting.py
SITTING_KEY; deleted at close) and its watch state (``orion:retro:watch``, services/orion-hub/scripts/retro_watch.py
WATCH_KEY), on the bus Redis orion-mind already holds (services/orion-mind/app/prime_loop.py ``_read_sitting``). A
failed read changes nothing: an unread sitting is never "closed". ``sitting_frame_line`` is the one line Prime's
frame shows while a sitting is open.

Pure: no I/O, no clock of its own.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.retro.sitting import WATCH_KEY  # noqa: F401 -- the gateway's watch-state key, read by prime_loop

SITTING_REF_PREFIX = "retro_sitting:"
OPEN_WAKE_AFTER_SEC = 180.0
MAX_THOUGHTS_PER_SITTING = 2
MAX_TRACKED = 20
NOTE_CHARS = 200


class PrimeSittingV1(BaseModel):
    """The open sitting as Prime reads it (a subset of ``orion.retro.sitting.Sitting.as_dict``)."""

    model_config = ConfigDict(extra="ignore")

    sitting_id: str = Field(min_length=1)
    started_at: float
    last_input_at: float
    game_id: Optional[str] = None
    game_title: Optional[str] = None
    player_stated: Optional[str] = None
    ended_at: Optional[float] = None


class PrimeSittingSeenV1(BaseModel):
    """What Prime remembers of one sitting, persisted inside ``PrimeStateV1.sittings``."""

    model_config = ConfigDict(extra="forbid")

    who: str
    title: str
    minutes: float = Field(ge=0.0)
    last_seen_at: AwareDatetime
    opened_woke: bool = False
    closed_woke: bool = False
    thoughts: int = Field(default=0, ge=0)


def parse_sitting(raw: Any) -> Optional[PrimeSittingV1]:
    """The Redis copy (a dict) as a sitting, or None when there is none, it is closed, or it is malformed."""
    if not isinstance(raw, Mapping):
        return None
    try:
        s = PrimeSittingV1.model_validate(dict(raw))
    except Exception:  # noqa: BLE001 -- a malformed copy reads as no sitting, never as a closed one
        return None
    return None if s.ended_at is not None else s


def _who(s: PrimeSittingV1) -> str:
    return "Juniper" if str(s.player_stated or "") == "juniper" else "Someone"


def _title(s: PrimeSittingV1) -> str:
    return " ".join(str(s.game_title or s.game_id or "a game").split())[:80]


def ref(sitting_id: str) -> str:
    return f"{SITTING_REF_PREFIX}{sitting_id}"


def names_sitting(change: str, sitting_id: str) -> bool:
    return f"({ref(sitting_id)})" in change


def observe_sitting(state: Any, sitting: Optional[PrimeSittingV1], *, read_ok: bool, now: datetime) -> Any:
    """One tick of the sitting wakes over ``state`` (a ``PrimeStateV1``). ``read_ok`` False: the read failed, nothing
    changes (an unread sitting is not a closed one)."""
    from orion.prime.state import note_change

    if not read_ok:
        return state
    seen = dict(state.sittings)
    if sitting is not None:
        minutes = max(0.0, (sitting.last_input_at - sitting.started_at) / 60.0)
        rec = seen.get(sitting.sitting_id) or PrimeSittingSeenV1(
            who=_who(sitting), title=_title(sitting), minutes=minutes, last_seen_at=now)
        rec = rec.model_copy(update={"who": _who(sitting), "title": _title(sitting), "minutes": minutes,
                                     "last_seen_at": now})
        if not rec.opened_woke and minutes * 60.0 >= OPEN_WAKE_AFTER_SEC:
            rec = rec.model_copy(update={"opened_woke": True})
            if rec.thoughts < MAX_THOUGHTS_PER_SITTING:
                state = note_change(state, f"a game sitting is open: {rec.who} has played {rec.title} for "
                                           f"{int(minutes)} min on your Retro service ({ref(sitting.sitting_id)})")
        seen[sitting.sitting_id] = rec
    open_id = sitting.sitting_id if sitting is not None else None
    for sid, rec in list(seen.items()):
        if sid == open_id or not rec.opened_woke or rec.closed_woke:
            continue
        seen[sid] = rec.model_copy(update={"closed_woke": True})
        if rec.thoughts < MAX_THOUGHTS_PER_SITTING:
            state = note_change(state, f"a game sitting ended: {rec.who} played {rec.title} for about "
                                       f"{int(round(rec.minutes))} min ({ref(sid)})")
    state = _drop_capped(state, seen)
    if len(seen) > MAX_TRACKED:
        keep = sorted(seen.items(), key=lambda kv: kv[1].last_seen_at)[-MAX_TRACKED:]
        seen = dict(keep)
    return state.model_copy(update={"sittings": seen})


def _drop_capped(state: Any, seen: Mapping[str, PrimeSittingSeenV1]) -> Any:
    capped = [sid for sid, rec in seen.items() if rec.thoughts >= MAX_THOUGHTS_PER_SITTING]
    if not capped:
        return state
    kept = [c for c in state.pending_changes if not any(names_sitting(c, sid) for sid in capped)]
    if len(kept) == len(state.pending_changes):
        return state
    return state.model_copy(update={"pending_changes": kept})


def note_sitting_thought(state: Any) -> Any:
    """A thought opened: every sitting a pending change names counts one more thought. At the cap, the next tick
    (``observe_sitting``) drops any change still naming it, so no further thought can name that sitting."""
    seen = dict(state.sittings)
    hit = False
    for sid, rec in seen.items():
        if any(names_sitting(c, sid) for c in state.pending_changes):
            seen[sid] = rec.model_copy(update={"thoughts": rec.thoughts + 1})
            hit = True
    if not hit:
        return state
    return state.model_copy(update={"sittings": seen})


def _ago(now: datetime, epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    sec = max(0.0, now.timestamp() - float(epoch))
    return f"{int(sec // 60)} min ago" if sec >= 60 else f"{int(sec)} s ago"


def sitting_frame_line(sitting: Optional[PrimeSittingV1], watch: Optional[Mapping[str, Any]], *, now: datetime,
                       read_error: Optional[str] = None) -> Optional[str]:
    """The one line Prime's frame shows while a sitting is open (spec section 6 part 7): who plays what as stated, for
    how long, the watch mode and who set it, the look counts, the newest note. None when nobody plays."""
    if read_error:
        return f"(the game sitting could not be read: {read_error})"
    if sitting is None:
        return None
    minutes = int(max(0.0, sitting.last_input_at - sitting.started_at) // 60)
    known = "" if sitting.player_stated else " (nobody said who holds the controller)"
    parts = [f"{_who(sitting)} is playing {_title(sitting)} on your Retro service, {minutes} min so far{known}; "
             f"last button press {_ago(now, sitting.last_input_at)}."]
    w = dict(watch) if isinstance(watch, Mapping) and watch.get("sitting_id") == sitting.sitting_id else None
    if w is None:
        parts.append("Your watch: unknown (the game page's watch state was not read).")
    else:
        set_by = {"orion": "you", "juniper": "Juniper's switch"}.get(str(w.get("set_by") or ""), "nobody (both on)")
        why = f": {w['why']}" if w.get("why") else ""
        mode = str(w.get("mode") or "on")
        word = {"off": "off", "quiet": "quiet (looks, no lines)", "on": "on (looks and lines)"}.get(mode, mode)
        counts = w.get("counts") if isinstance(w.get("counts"), Mapping) else {}
        parts.append(f"Your watch: {word}, set by {set_by}{why}. Looks: {int(counts.get('notes_stored', 0))} notes "
                     f"stored of {int(counts.get('model_calls', 0))} model calls, "
                     f"{int(counts.get('failed_looks', 0))} failed.")
        note = " ".join(str(w.get("last_note") or "").split())[:NOTE_CHARS]
        if note:
            parts.append(f"Newest note ({_ago(now, w.get('last_note_at')) or 'time unknown'}; a look, not a fact): "
                         f"{note}")
    return " ".join(parts)


__all__ = ["MAX_THOUGHTS_PER_SITTING", "OPEN_WAKE_AFTER_SEC", "PrimeSittingSeenV1", "PrimeSittingV1",
           "SITTING_REF_PREFIX", "WATCH_KEY", "names_sitting", "note_sitting_thought", "observe_sitting", "parse_sitting",
           "ref", "sitting_frame_line"]

