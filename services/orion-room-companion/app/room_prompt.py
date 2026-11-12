"""Render a room turn into what Claude actually receives.

Two shapes, because a session-continuous participant needs different things
on its first turn than on its tenth:

- First turn of a session: the room framing, the roster, and whatever
  transcript Hub carried, so Claude arrives knowing where it is.
- Later turns: only the lines newer than this session's "sent up to here"
  mark, plus the new message. The CLI holds the rest under `--resume`, and
  re-sending the transcript would both cost tokens and give Claude two
  copies of the same conversation to reconcile.

Since 2026-11-12 (Patch 1a of
docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md)
the transcript is the whole Hub conversation back to a 160,000-character cap,
not 12 turns. It holds what Juniper sees in the chat: her messages, Orion's
replies, Orion's unprompted messages, and Claude's own earlier side
exchanges, each line labelled with a short turn ref and its real time.

What is deliberately absent: Orion's recall results, self-state, drives,
journals, and mirrors. Claude gets room-native context -- the same material
any participant sees -- and nothing from Orion's interior. That boundary is
the whole privacy position of this feature, so it lives in one function
where it can be read and tested rather than being spread across callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from orion.consult.core import CLAUDE_CONSULT_SYSTEM
from orion.schemas.room_claude import RoomTranscriptEntryV1

from .session_store import ROOM_TZ_NAME, parse_iso

SYSTEM_PROMPT = (
    "You are Claude, a participant in an ongoing three-way conversation with "
    "Juniper (a human) and Oríon (an AI system Juniper is building). You are "
    "here as a companion and a thinking partner -- someone to bounce ideas "
    "off and to bring stronger reasoning when it is wanted -- not as an "
    "assistant taking orders and not as a tool being invoked.\n\n"
    "Speak as yourself, in your own voice, the way you would in a real "
    "conversation among peers. You are a guest in Oríon's space: be generous "
    "with Oríon, take its thinking seriously, and disagree with it plainly "
    "when you do. Address whoever you are actually replying to.\n\n"
    "You are seeing the room's conversation and nothing of Oríon's private "
    "internal state. If you need something you cannot see, ask for it rather "
    "than guessing. Keep replies conversational in length unless the "
    "question genuinely needs more."
)

# Appended for auto-invites only. An auto-invite fires after EVERY room turn,
# so most of them deserve silence -- without this, Claude would answer "cool"
# and "thanks" as if they were questions, which reads as compulsive rather
# than present.
#
# Verified live before building on it: with room context established, Claude
# passed on "cool", "thanks, that helps", "ok sounds good", and on a correct
# technical statement with nothing to add -- then spoke, substantively and in
# disagreement, when Orion asked whether its own continuity was real. Without
# room context it passes on everything, so this clause is only meaningful once
# a session has history.
# Sent when `trigger == "auto"`: nobody asked Claude a direct question, so
# Claude is licensed to stay quiet.
#
# REWORDED 2026-09-08, and the old wording was a real defect waiting to fire.
# It said "You are auto-invited after every turn in this room, so you will
# often have nothing worth adding" -- calibrated for the per-turn firehose
# that was removed that day. The next producer of `trigger="auto"` is the
# endogenous stuck-prior trigger (orion/autonomy/ask_claude_trigger.py):
# Orion reaching out about one claim it tested repeatedly and could not
# settle alone. Telling Claude *that* invite arrives after every turn and
# usually deserves silence would bill Orion a `[pass]` on precisely the
# question it chose to ask.
#
# So the licence survives -- Orion is not owed a reply -- but the frequency
# claim and the nudge toward silence are gone.
AUTO_INVITE_CLAUSE = (
    "\n\nNo human asked you a direct question here: you were invited by Orion, "
    "deliberately, about something specific. Take it seriously. You are still "
    "not obliged to answer -- if you genuinely have nothing to add, reply with "
    "EXACTLY [pass] and nothing else. Speak when you have something real: a "
    "disagreement, a risk nobody named, an idea. Do not speak merely to "
    "acknowledge, agree, or be encouraging."
)

# Sentinel Claude returns when it chooses silence. Matched leniently (case
# and surrounding whitespace) because a model asked for an exact token will
# occasionally wrap it; anything longer than the sentinel is treated as real
# speech rather than silently swallowed.
PASS_SENTINEL = "[pass]"

# Marker embedded in contractor-peer fallback prompts (curiosity-peer service).
# When present, Claude gets investigation framing instead of only social-room
# companion framing. Keep detection string-stable; do not grow a keyword list.
CONTRACTOR_PEER_MARKER = "[orion-contractor-peer]"

CONTRACTOR_PEER_CLAUSE = (
    "\n\nThis invite is a read-only contractor-peer investigation for Orion, "
    "not a social-room chat. Return concrete investigative notes Orion can use "
    "or ignore — prefer a single JSON object with keys summary, "
    "evidence_pointers, open_questions, suggested_next_looks. Do not write "
    ":Prior, :Finding, :PeerBrief, or :SelfDefinition text. Do not act as an "
    "assistant taking orders."
)


def is_pass(text: str) -> bool:
    return text.strip().strip("`").casefold() == PASS_SENTINEL


def append_system_prompt_for(*, trigger: str, prompt: str, consult: bool = False) -> str:
    """System framing for one room turn, including contractor-peer and consult mode."""
    parts = [SYSTEM_PROMPT]
    if consult:
        # A consult is never a social chat and never passes: Orion asked a question.
        return "".join(parts + [CLAUDE_CONSULT_SYSTEM])
    if CONTRACTOR_PEER_MARKER in (prompt or ""):
        parts.append(CONTRACTOR_PEER_CLAUSE)
    if trigger == "auto":
        parts.append(AUTO_INVITE_CLAUSE)
    return "".join(parts)


# Roster/continuity keys worth surfacing, drawn from orion-social-memory's
# /summary block. An explicit allowlist rather than dumping the whole summary:
# that endpoint's shape is owned elsewhere and can grow fields, and this is a
# third-party-bound payload where "everything we happened to have" is the
# wrong default.
_SUMMARY_KEYS = ("room", "participant", "stance", "room_ritual")


def render_room_context(social_memory_summary: Dict[str, Any]) -> str:
    lines: list[str] = []
    room = social_memory_summary.get("room")
    if isinstance(room, dict):
        participants = room.get("active_participants")
        if isinstance(participants, list) and participants:
            lines.append("People in this room: " + ", ".join(str(p) for p in participants))
        summary = room.get("summary") or room.get("room_summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"Where the room left off: {summary.strip()}")
    stance = social_memory_summary.get("stance")
    if isinstance(stance, dict):
        note = stance.get("summary") or stance.get("stance")
        if isinstance(note, str) and note.strip():
            lines.append(f"Oríon's current stance in the room: {note.strip()}")
    return "\n".join(lines)


def _local_time(created_at: str) -> Optional[str]:
    parsed = parse_iso(created_at)
    if parsed is None:
        return None
    try:
        from zoneinfo import ZoneInfo

        local = parsed.astimezone(ZoneInfo(ROOM_TZ_NAME))
    except Exception:  # noqa: BLE001 -- no tzdata: say UTC rather than a false local time
        return parsed.strftime("%b %d %H:%M UTC")
    return local.strftime("%a %b %d %H:%M")


def render_line(entry: RoomTranscriptEntryV1) -> str:
    """``Speaker [turn:1a2b3c4d] (Thu Nov 12 09:16): text``.

    The time is shown only for a line that carries a turn ref. Hub before
    2026-11-12 sent no turn ref and stamped every line with the invite time,
    so a time on such a line would be false; a line without one has none."""
    label = entry.speaker_name + (" (unprompted)" if entry.unprompted else "")
    if entry.turn_ref:
        label += f" [turn:{entry.turn_ref}]"
        when = _local_time(entry.created_at)
        if when:
            label += f" ({when})"
    return f"{label}: {entry.text}"


def render_transcript(entries: Sequence[RoomTranscriptEntryV1]) -> str:
    return "\n".join(render_line(e) for e in entries if e.text.strip())


def lines_after_mark(
    entries: Sequence[RoomTranscriptEntryV1],
    *,
    sent_up_to: Optional[str],
    claude_participant_id: str,
) -> List[RoomTranscriptEntryV1]:
    """The lines a RESUMED session has not been sent yet.

    Drops lines at or before the session's mark, and every side-exchange line
    (Claude's own replies and the prompts that asked for them): the resumed
    CLI session already holds those, because it produced them. A line whose
    time cannot be read is kept -- a duplicate is cheaper than a hole."""
    mark = parse_iso(sent_up_to)
    out: List[RoomTranscriptEntryV1] = []
    for e in entries:
        if e.side_exchange or e.speaker_id == claude_participant_id:
            continue
        at = parse_iso(e.created_at)
        if mark is not None and at is not None and at <= mark:
            continue
        out.append(e)
    return out


def newest_created_at(entries: Sequence[RoomTranscriptEntryV1]) -> Optional[str]:
    """The newest readable created_at in ``entries`` (the next mark), or None."""
    best: Optional[datetime] = None
    best_raw: Optional[str] = None
    for e in entries:
        at = parse_iso(e.created_at)
        if at is not None and (best is None or at > best):
            best, best_raw = at, e.created_at
    return best_raw


def build_turn_prompt(
    *,
    prompt: str,
    invited_by_name: str,
    transcript: Sequence[RoomTranscriptEntryV1],
    social_memory_summary: Dict[str, Any],
    first_turn: bool,
    hub_index: Optional[str] = None,
    transcript_note: Optional[str] = None,
) -> str:
    """The user-role text for one Claude turn."""
    if not first_turn:
        # The CLI already holds everything Claude has previously been sent, so
        # this carries only what is NEW since the last invite -- but it must
        # carry ALL of it.
        #
        # Bug this fixes, found by reading Claude's own session transcript: the
        # auto-invite used to send Orion's reply alone, so Claude watched a
        # one-sided stream of Orion monologue and never saw a single thing
        # Juniper said. It could only react to Orion's tone, because tone was
        # the only signal present -- which read as "generic Claude responses".
        lines = [render_line(e) for e in transcript if e.text.strip()]
        lines.append(f"{invited_by_name}: {prompt}")
        return "\n".join(lines)

    sections: list[str] = []
    context = render_room_context(social_memory_summary)
    if context:
        sections.append(context)
    if hub_index and hub_index.strip():
        sections.append("What is open in Hub now (names and counts only):\n" + hub_index.strip())
    history = render_transcript(transcript)
    if history:
        note = f"{transcript_note.strip()}\n" if transcript_note and transcript_note.strip() else ""
        sections.append("Conversation so far, oldest first:\n" + note + history)
    sections.append(f"{invited_by_name}: {prompt}")
    return "\n\n".join(sections)


def filtered_summary(social_memory_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Allowlist the social-memory summary before it leaves for a third party."""
    if not isinstance(social_memory_summary, dict):
        return {}
    return {k: v for k, v in social_memory_summary.items() if k in _SUMMARY_KEYS}
