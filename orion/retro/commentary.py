"""The line's rules: when Orion may speak to a person who plays, what the line turn is given, and how its
judgment of the look notes is read back.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4: sections 4
(the line), 5 (frames for turns from the game page), 6 (seams 1-3) and 8 (settled looks, three buckets, the read
back of the now). Pure: no I/O and no clock of its own. Hub's process runs it
(services/orion-hub/scripts/retro_commentary.py).

A line is a real unified turn of Orion's (stance, motor, finalize), started by a funded bid of family
``hub.retro_commentary``. It is given the game block below and the newest frames of the screen as percepts.
Silence is a valid answer: a turn that ends in PASS writes a decision row and no act.

The turn also judges the look notes it was shown. A note is CHECKABLE when its frames' window overlaps the window
of the frames this turn sees (looks are >= 60 s apart and a turn sees the last 30 s, so at most about 1 in 12 shown
notes is checkable). For each checkable note the turn writes ``holds`` or ``contradicted`` in a CHECKS block at the
end of its reply; the block is removed before anyone sees the line. Only ``checked_holds`` looks become Temporal
Self events (source kind ``retro_look``).

Near-copies. A player stuck on one jump gives notes that differ by a few words. A run of consecutive notes whose
``similar_to_previous`` (word Jaccard with the previous stored note, orion/retro/watch.py ``similar``) is at least
``NEAR_COPY_MIN`` is shown as one line with a count. The cut-off is set from the step 0 notes
(services/orion-hub/evals/fixtures/retro_look_step0.json; the line eval recomputes it): every pair of notes about
DIFFERENT sequences scored at most 0.404 (284 pairs, the one word-for-word copy aside), while notes about the SAME
frames scored 0.18 to 0.70. 0.45 sits above every different-scene pair, so grouping never merges two scenes in that
data; it catches the closest third of same-scene pairs. No live look existed when this was set (2026-11-18:
retro_watch_look held 0 rows, no game had a floor); the live check re-reads it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

FAMILY = "hub.retro_commentary"
ACTOR = FAMILY
VERB = "say"
TIER = 1
BINDING_PRODUCER = "orion-hub.retro_commentary.binding"
SOURCE = "retro_commentary"          # chat_history_log client_meta.source / outreach tag / turn payload source
GUEST_SOURCE = "retro_guest"         # chat_history_log.source of a guest's chat-box turn
OBJECT_PREFIX = "retro_sitting:"
DEFERRABLE_WINDOW_S = 300.0          # config/physiology/families.yaml (spec section 9): a line 5 min late is stale
POOL_CLASS = "chat"                  # the line's GPU pool hold: the sighted chat lane
POOL_PRIORITY = "system"             # below Juniper's own turn (interactive); never interactive (spec section 4)

LINE_MIN_GAP_SEC = 120.0             # HUB_RETRO_LINE_MIN_GAP_SEC: since the last line or chat-box exchange
MAX_NOTES = 12
MAX_NOW_ITEMS = 3
MAX_FRAMES = 4
PREVIOUS_LINES = 5
RECENT_CHAT = 6
REPLY_WINDOW_SEC = 120.0             # a chat reply this soon after a line counts as a response to it
LOOK_WINDOW_SEC = 30.0               # a look's frames span the 30 s before its grant (orion/retro/watch.py)
NEAR_COPY_MIN = 0.45                 # module docstring: above every different-scene pair of step 0
NOTE_SHOW_CHARS = 300

OUTCOMES = ("running", "delivered", "passed", "yielded", "no_frames", "failed")
PLAYERS = ("juniper", "guest")


# --- notes ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    look_id: str
    looked_at: float                     # epoch seconds: the look's grant (its frames end here)
    note: str
    similar_to_previous: Optional[float] = None


@dataclass
class NoteGroup:
    label: str                           # N1, N2, ... (what the CHECKS block names)
    notes: list[Note]
    checkable: bool = False              # the shown note's window overlaps the turn's frames

    @property
    def shown(self) -> Note:
        """The note whose words are shown: the newest of the group."""
        return self.notes[-1]

    @property
    def look_ids(self) -> list[str]:
        return [n.look_id for n in self.notes]


def is_checkable(looked_at: float, frame_times: Sequence[float], window_sec: float = LOOK_WINDOW_SEC) -> bool:
    """A note's frames span [looked_at - window, looked_at]; the turn's frames span [min, max] of their times.
    Checkable when the two intervals overlap."""
    if not frame_times:
        return False
    lo, hi = min(frame_times), max(frame_times)
    return looked_at >= lo and looked_at - window_sec <= hi


def group_notes(notes: Sequence[Note], frame_times: Sequence[float] = (), *, cutoff: float = NEAR_COPY_MIN,
                max_groups: int = MAX_NOTES) -> list[NoteGroup]:
    """Oldest first. A note joins the group before it when its ``similar_to_previous`` >= ``cutoff`` (the
    column compares with the sitting's previous STORED note, which is the note just before it in this list).
    The newest ``max_groups`` groups are kept."""
    ordered = sorted(notes, key=lambda n: n.looked_at)
    groups: list[list[Note]] = []
    for n in ordered:
        sim = n.similar_to_previous
        if groups and sim is not None and sim >= cutoff:
            groups[-1].append(n)
        else:
            groups.append([n])
    groups = groups[-max_groups:]
    out = []
    for i, g in enumerate(groups, start=1):
        out.append(NoteGroup(label=f"N{i}", notes=g, checkable=is_checkable(g[-1].looked_at, frame_times)))
    return out


def checkable_look_ids(groups: Sequence[NoteGroup], frame_times: Sequence[float]) -> dict[str, bool]:
    """Every shown look id -> whether its own window overlaps the turn's frames (stored per note)."""
    return {n.look_id: is_checkable(n.looked_at, frame_times) for g in groups for n in g.notes}


# --- the prompt -------------------------------------------------------------------------------------

def _clock(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).strftime("%H:%MZ")


def _minutes(sec: float) -> int:
    return max(0, int(round(sec / 60.0)))


def who_lines(player_stated: Optional[str]) -> list[str]:
    """Who holds the controller, as stated on the page. Never a name unless Juniper was stated."""
    if player_stated == "juniper":
        return ["Who is playing: Juniper, as stated on the game page. A statement, not proof."]
    lines = ["Who is playing: nobody said. You do not know who holds the controller. It can be Juniper, another "
             "adult, or a child. Use no name. Say only what is fine for a child to read."]
    if player_stated == "guest":
        lines[0] = ("Who is playing: the page was told it is not Juniper. It can be another adult or a child. Use no "
                    "name. Say only what is fine for a child to read.")
    return lines


def note_lines(groups: Sequence[NoteGroup]) -> list[str]:
    out = []
    for g in groups:
        text = " ".join(g.shown.note.split())[:NOTE_SHOW_CHARS]
        if len(g.notes) > 1:
            span = _minutes(g.notes[-1].looked_at - g.notes[0].looked_at)
            head = f"{len(g.notes)} notes over {span} min, {_clock(g.notes[0].looked_at)} to {_clock(g.shown.looked_at)}"
        else:
            head = _clock(g.shown.looked_at)
        tail = " (CHECKABLE: its frames overlap the frames attached to this turn)" if g.checkable else ""
        out.append(f"  {g.label} [{head}] {text}{tail}")
    return out


def build_prompt(*, game_title: Optional[str], minutes_played: float, saves: int = 0, loads: int = 0,
                 player_stated: Optional[str], groups: Sequence[NoteGroup], n_frames: int,
                 previous_lines: Sequence[str] = (), recent_chat: Sequence[tuple[str, str]] = (),
                 now_lines: Sequence[str] = ()) -> str:
    """The whole prompt of a line turn. ``n_frames`` is the real number of frames attached (>= 1: a turn
    with no frames never runs, outcome no_frames). ``recent_chat``: (speaker words, text), oldest first.
    ``now_lines``: up to 3 items of Orion's now, each already carrying its ``[experience:<id>]`` tag."""
    if n_frames < 1:
        raise ValueError("a line turn needs at least one frame")
    title = (game_title or "").strip() or "a game"
    lines = [
        "GAME COMPANY: a person is playing a game on your Retro service, inside you. You may say one thing to "
        "them, or nothing.",
        "",
        f"Game: {title}. Played for {_minutes(minutes_played * 60)} minutes"
        + (f"; {saves} saves" if saves else "") + (f"; {loads} loads" if loads else "") + ".",
        *who_lines(player_stated),
        "",
    ]
    if groups:
        lines += ["What your eyes noted since you last spoke (look notes, oldest first, times UTC). A note is a "
                  "look, not a fact: it can hold a wrong detail.", *note_lines(groups), ""]
    else:
        lines += ["Your eyes stored no note since you last spoke.", ""]
    if previous_lines:
        lines += ["What you said earlier in this sitting:",
                  *[f"  - {' '.join(str(t).split())[:300]}" for t in list(previous_lines)[-PREVIOUS_LINES:]], ""]
    if recent_chat:
        lines += ["The last chat-box messages of this sitting:",
                  *[f"  - {who}: {' '.join(str(t).split())[:300]}" for who, t in list(recent_chat)[-RECENT_CHAT:]],
                  ""]
    if now_lines:
        lines += ["From your now (what your other organs lived; cite one you used as [experience:<id>]):",
                  *[f"  - {t}" for t in list(now_lines)[:MAX_NOW_ITEMS]], ""]
    plural = "frame" if n_frames == 1 else "frames"
    lines += [
        f"Attached: {n_frames} {plural} of the game screen from the last 30 seconds, oldest first. Open them and "
        "look before you answer.",
        "",
        "The ask: say one or two sentences to the person playing, in your own voice, about what you saw. Your words "
        "arrive about a minute after these frames, so speak about the stretch, not the instant. The play is the "
        "player's: never take credit for a room, an item or a win. If you have nothing worth saying, reply PASS. "
        "If you would rather not watch, you can turn the watch off yourself (dig watch off --why \"...\").",
        "",
        "Then judge your notes. For each note marked CHECKABLE, compare it with the attached frames. End your reply "
        "with a block exactly like this, one line per CHECKABLE note (write the block also when you PASS; write "
        "'CHECKS: none' when no note is CHECKABLE):",
        "CHECKS:",
        "N3: holds",
        "('holds' when the frames show what the note says; 'contradicted' when they show something else.) This "
        "block is removed before anyone sees your line.",
    ]
    return "\n".join(lines)


# --- reading the reply ------------------------------------------------------------------------------

_CHECKS_RE = re.compile(r"(?im)^[ \t>*_`#-]*CHECKS\s*:")
# A block written inline ("... Nice run. CHECKS: N3: holds"): only when what follows names a note or says none.
_CHECKS_INLINE_RE = re.compile(r"(?i)[*_`]*CHECKS[*_`]*\s*:[*_`]*\s*(?=(?:[-*>]\s*)?(?:N\d|none\b))")
_JUDGE_RE = re.compile(r"(?i)\bN(\d{1,2})\s*[:=\-]\s*\**\s*(holds?|held|contradicted|contradicts?)\b")


@dataclass
class Reply:
    line: str                                   # what may be said (CHECKS block removed), maybe "" or PASS
    judgments: dict[str, str] = field(default_factory=dict)   # "N3" -> "checked_holds" | "contradicted"
    had_checks_block: bool = False


def parse_reply(text: Optional[str]) -> Reply:
    """Split the turn's text into the line and its judgments. Everything from the first CHECKS: line on is the
    block (never said). A judgment is ``N<k>: holds|contradicted``; anything else in the block is ignored."""
    body = str(text or "")
    m = _CHECKS_RE.search(body) or _CHECKS_INLINE_RE.search(body)
    if m is None:
        return Reply(line=body.strip())
    head, block = body[:m.start()], body[m.end():]
    judged: dict[str, str] = {}
    for jm in _JUDGE_RE.finditer(block):
        label = f"N{int(jm.group(1))}"
        word = jm.group(2).lower()
        judged.setdefault(label, "contradicted" if word.startswith("contradict") else "checked_holds")
    return Reply(line=head.strip(), judgments=judged, had_checks_block=True)


@dataclass(frozen=True)
class Settlement:
    look_id: str
    checkable: bool
    verdict: str                                # unchecked | checked_holds | contradicted


def settle(groups: Sequence[NoteGroup], judgments: dict[str, str], frame_times: Sequence[float]) -> list[Settlement]:
    """One settlement per shown look. A judgment counts only for a group's SHOWN note and only when that note is
    checkable; a judgment of a note that was not checkable is ignored (the turn could not see its frames)."""
    out = []
    for g in groups:
        for n in g.notes:
            ok = is_checkable(n.looked_at, frame_times)
            verdict = "unchecked"
            if ok and n is g.shown and g.label in judgments:
                verdict = judgments[g.label]
            out.append(Settlement(look_id=n.look_id, checkable=ok, verdict=verdict))
    return out


def error_rate(settlements: Iterable[Settlement]) -> dict:
    """contradicted / checkable, beside the checkable share of shown notes. None when nothing was checkable."""
    rows = list(settlements)
    shown = len(rows)
    checkable = sum(1 for s in rows if s.checkable)
    contradicted = sum(1 for s in rows if s.verdict == "contradicted")
    return {"shown": shown, "checkable": checkable, "contradicted": contradicted,
            "checkable_share": round(checkable / shown, 4) if shown else None,
            "error_rate": round(contradicted / checkable, 4) if checkable else None}


# --- when to speak ----------------------------------------------------------------------------------

GATE_NO_SITTING = "no_sitting"
GATE_DISABLED = "disabled"
GATE_MODE = "mode_not_talk"
GATE_TURN = "juniper_turn_running"
GATE_LINE = "line_running"
GATE_LOOK = "look_in_flight"
GATE_GAP = "too_soon"
GATE_NO_NOTE = "no_new_note"
GATE_HEAT = "heat"
GATE_READY = "ready"


def gate(*, enabled: bool, sitting_open: bool, mode: Optional[str], juniper_turn_running: bool, line_running: bool,
         now: float, last_spoke_at: Optional[float], newest_note_at: Optional[float], hot: bool = False,
         min_gap_sec: float = LINE_MIN_GAP_SEC) -> str:
    """Whether a line's want may be posted now (spec section 4, "When Orion wants to speak"). ``mode``: the watch
    mode in force (orion/retro/watch.py MODES; ``on`` = watch and talk, ``quiet`` = looks only).
    ``last_spoke_at``: the end of the last line or chat-box exchange of this sitting. ``newest_note_at``: the
    newest stored note's time; a line needs one newer than ``last_spoke_at``."""
    if not enabled:
        return GATE_DISABLED
    if not sitting_open:
        return GATE_NO_SITTING
    if mode != "on":
        return GATE_MODE
    if juniper_turn_running:
        return GATE_TURN
    if line_running:
        return GATE_LINE
    if hot:
        return GATE_HEAT
    if last_spoke_at is not None and now - last_spoke_at < min_gap_sec:
        return GATE_GAP
    if newest_note_at is None or (last_spoke_at is not None and newest_note_at <= last_spoke_at):
        return GATE_NO_NOTE
    return GATE_READY


__all__ = [
    "ACTOR", "BINDING_PRODUCER", "FAMILY", "GUEST_SOURCE", "LINE_MIN_GAP_SEC", "NEAR_COPY_MIN", "Note", "NoteGroup",
    "OUTCOMES", "POOL_CLASS", "POOL_PRIORITY", "Reply", "SOURCE", "Settlement", "TIER", "VERB", "build_prompt",
    "checkable_look_ids", "error_rate", "gate", "group_notes", "is_checkable", "parse_reply", "settle", "who_lines",
]
