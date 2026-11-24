"""A sitting: one stretch of play by a person on the Retro service.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, section 1
("The sitting"). Pure: no I/O, no clock of its own. Hub's gateway feeds it the browser's input
messages (services/orion-hub/scripts/retro_sitting.py) and the backfill replays the evidence
journals through the same code (scripts/backfill_retro_sittings.py), so the live count and the
baseline come from one instrument.

What counts as input: a CHANGE of the button mask on one port of one Retro session, or of an
analog stick's coarse position (N64). The page repeats the held state every 50 ms, so a message
alone proves only that the page has focus; a press or a release proves a person. A stick is read
as pushed or not pushed per direction (``STICK_DEAD_ZONE``), so a resting stick's noise is not a hand. Measured on the sitting of 2026-11-15 19:57Z (step 0): 268,628
journal rows, 37,586 mask changes; 243 minutes held a row, 204 held a change.

A sitting opens on the first mask change of a new stretch and closes when no mask changed for
``gap_sec`` (its end is the last change, not the moment the gap was noticed), or when the Retro
session stops, or when input arrives for another Retro session. A sitting shorter than ``min_sec``
is not worth a record (``worth_recording``): 14 Super Mario Bros. sessions in the journals lasted
seconds each.

Who is playing is never guessed. ``player_stated`` is what the page was told ("juniper" or
"guest"), or None.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Optional

SITTING_KEY = "orion:retro:sitting"       # Redis: the open sitting as JSON, for both Hub processes
WATCH_KEY = "orion:retro:watch"           # Redis: the open sitting's watch state, for Prime's frame (patch 4)
SITTING_SUBJECT_PREFIX = "retro_sitting:" # Temporal Self subject of a person's sitting (never Orion's play)
GAP_SEC_DEFAULT = 300.0
MIN_SEC_DEFAULT = 120.0
PLAYERS = ("juniper", "guest")
ACTIONS = ("save", "load", "pause")
STICK_DEAD_ZONE = 8000                    # of 32767: under this a stick axis reads as centred
CENTRED = (0, 0, 0, 0)
END_IDLE = "idle"
END_SESSION_STOPPED = "session_stopped"
END_SESSION_CHANGED = "session_changed"
END_SHUTDOWN = "shutdown"


@dataclass
class Sitting:
    sitting_id: str
    retro_session_id: str
    started_at: float                 # epoch seconds of the first mask change
    last_input_at: float              # epoch seconds of the newest mask change
    game_id: Optional[str] = None
    game_title: Optional[str] = None
    mask_changes: int = 0
    saves: int = 0
    loads: int = 0
    pauses: int = 0
    player_stated: Optional[str] = None
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None
    backfilled: bool = False

    @property
    def open(self) -> bool:
        return self.ended_at is None

    @property
    def duration_sec(self) -> float:
        return max(0.0, (self.ended_at if self.ended_at is not None else self.last_input_at) - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"duration_sec": round(self.duration_sec, 1)}


def worth_recording(sitting: Sitting, min_sec: float = MIN_SEC_DEFAULT) -> bool:
    return sitting.duration_sec >= float(min_sec)


def who_words(player_stated: Optional[str]) -> str:
    """The only names this organ ever writes: "Juniper" when the page was told so, else "Someone"."""
    return "Juniper" if player_stated == "juniper" else "Someone"


@dataclass
class SittingTracker:
    """One open sitting at most (Retro runs one session at a time). Every method returns the
    sittings it closed, oldest first; the caller records them."""

    gap_sec: float = GAP_SEC_DEFAULT
    new_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex)
    current: Optional[Sitting] = None
    _masks: dict[tuple[str, int], tuple[int, tuple[int, ...]]] = field(default_factory=dict)

    def on_input(self, session_id: str, port: int, mask: int, now: float,
                 axes: Optional[Iterable[int]] = None) -> list[Sitting]:
        """One browser input message. ``axes`` are the RAW stick values of the message (-32767 to
        32767); this is the one place that makes them coarse. Returns closed sittings (an idle one
        that this input ended)."""
        key = (str(session_id), int(port))
        state = (int(mask), coarse_axes(axes))
        prev = self._masks.get(key)
        self._masks[key] = state
        # The first message of a port is a change only when something is held: a neutral first
        # message is the page announcing itself, not a hand.
        changed = (prev is None and state != (0, CENTRED)) or (prev is not None and state != prev)
        if not changed:
            return []
        closed: list[Sitting] = []
        cur = self.current
        if cur is not None and cur.retro_session_id != str(session_id):
            closed.append(self._close(cur.last_input_at, END_SESSION_CHANGED))
        elif cur is not None and now - cur.last_input_at > self.gap_sec:
            closed.append(self._close(cur.last_input_at, END_IDLE))
        if self.current is None:
            self.current = Sitting(sitting_id=self.new_id(), retro_session_id=str(session_id),
                                   started_at=float(now), last_input_at=float(now))
        self.current.last_input_at = float(now)
        self.current.mask_changes += 1
        return closed

    def on_action(self, session_id: str, kind: str) -> None:
        """A save, a load or a pause the player asked for, counted on the open sitting of that session."""
        cur = self.current
        if cur is None or cur.retro_session_id != str(session_id) or kind not in ACTIONS:
            return
        setattr(cur, kind + "s", getattr(cur, kind + "s") + 1)

    def state_player(self, player: Optional[str]) -> bool:
        """What the page was told. False when there is no open sitting or the word is not known."""
        if self.current is None or player not in PLAYERS:
            return False
        self.current.player_stated = player
        return True

    def tick(self, now: float) -> list[Sitting]:
        cur = self.current
        if cur is not None and now - cur.last_input_at > self.gap_sec:
            return [self._close(cur.last_input_at, END_IDLE)]
        return []

    def end_session(self, session_id: str, reason: str = END_SESSION_STOPPED) -> list[Sitting]:
        """The Retro session stopped or failed. The sitting ends at its last input."""
        self.forget_masks(session_id)
        cur = self.current
        if cur is not None and cur.retro_session_id == str(session_id):
            return [self._close(cur.last_input_at, reason)]
        return []

    def held(self, session_id: str) -> bool:
        """A button or a pushed stick is held right now on some port of this Retro session. A held
        direction while walking changes no mask, yet a hand is on the pad: the look's calibration counts
        that window as input (orion/retro/watch.py)."""
        return any(k[0] == str(session_id) and state != (0, CENTRED) for k, state in self._masks.items())

    def forget_masks(self, session_id: str) -> None:
        """Drop the remembered button state of one Retro session (its input socket closed, or it
        stopped). Without this a socket that dropped while a button was held would leave that state
        behind, and the neutral message of a later reconnect would count as a release by a hand."""
        for key in [k for k in self._masks if k[0] == str(session_id)]:
            del self._masks[key]

    def close_all(self, reason: str = END_SHUTDOWN) -> list[Sitting]:
        self._masks.clear()
        if self.current is not None:
            return [self._close(self.current.last_input_at, reason)]
        return []

    def _close(self, ended_at: float, reason: str) -> Sitting:
        cur = self.current
        assert cur is not None
        cur.ended_at, cur.end_reason = float(ended_at), reason
        self.current = None
        return cur


def coarse_axes(axes: Optional[Iterable[int]]) -> tuple[int, ...]:
    """Each stick axis as -1, 0 or 1. Anything that is not four integers reads as centred."""
    try:
        values = tuple(axes) if axes is not None else ()
    except TypeError:
        return CENTRED
    if len(values) != 4 or any(type(v) is not int for v in values):
        return CENTRED
    return tuple((v > STICK_DEAD_ZONE) - (v < -STICK_DEAD_ZONE) for v in values)


def parse_input_message(raw: str) -> Optional[tuple[int, int, Any]]:
    """(port, mask, raw axes) from one browser input message (wire v1: version, session_id,
    generation, lease, epoch, seq, port, mask, and axes on N64). The axes are passed on as sent;
    ``SittingTracker.on_input`` makes them coarse, once. None for anything that is not an input
    message: the relay forwards it untouched either way."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    port, mask = data.get("port"), data.get("mask")
    if type(port) is not int or type(mask) is not int or port not in (0, 1) or not 0 <= mask <= 0xFFFF:
        return None
    return port, mask, data.get("axes")


def _epoch(utc: Any) -> Optional[float]:
    try:
        ts = datetime.fromisoformat(str(utc))
    except ValueError:
        return None
    return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).timestamp()


def sittings_from_journal(lines: Iterable[str], *, gap_sec: float = GAP_SEC_DEFAULT,
                          new_id: Optional[Callable[[], str]] = None) -> Iterator[Sitting]:
    """Replay one Retro evidence journal (JSONL) through the tracker.

    Only rows the journal itself attributes to a human count (``kind=applied``, ``controller=human``);
    rows from before controller attribution existed (no ``controller`` field) are not claimed as
    human play. ``save``/``load`` rows and ``transition`` rows to ``paused`` with reason ``pause``
    are counted on the open sitting. Yields every sitting, closed, in order; ``backfilled`` is True.
    """
    tracker = SittingTracker(gap_sec=gap_sec, **({"new_id": new_id} if new_id else {}))
    game_id: Optional[str] = None
    last_ts: Optional[float] = None

    def done(items: list[Sitting]) -> Iterator[Sitting]:
        for s in items:
            s.game_id, s.backfilled = game_id, True
            yield s

    for line in lines:
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        sid = str(row.get("session_id") or "")
        ts = _epoch(row.get("utc"))
        if kind == "start":
            game_id = row.get("game_id") or game_id
        if ts is None or not sid:
            continue
        last_ts = ts
        yield from done(tracker.tick(ts))
        if kind == "applied" and row.get("controller") == "human":
            port, mask = row.get("port"), row.get("mask")
            if type(port) is int and type(mask) is int:
                yield from done(tracker.on_input(sid, port, mask, ts, row.get("axes")))
        elif kind in ("save", "load"):
            tracker.on_action(sid, kind)
        elif kind == "transition" and row.get("state") == "paused" and row.get("reason") == "pause":
            tracker.on_action(sid, "pause")
        elif kind == "transition" and row.get("state") in ("stopped", "failed"):
            yield from done(tracker.end_session(sid))
    if last_ts is not None:
        yield from done(tracker.close_all(END_IDLE))


__all__ = [
    "ACTIONS", "END_IDLE", "END_SESSION_CHANGED", "END_SESSION_STOPPED", "END_SHUTDOWN", "GAP_SEC_DEFAULT",
    "MIN_SEC_DEFAULT", "PLAYERS", "SITTING_KEY", "SITTING_SUBJECT_PREFIX", "STICK_DEAD_ZONE", "Sitting",
    "SittingTracker", "coarse_axes", "parse_input_message", "sittings_from_journal", "who_words", "worth_recording",
]
