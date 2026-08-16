"""Render a room turn into what Claude actually receives.

Two shapes, because a session-continuous participant needs different things
on its first turn than on its tenth:

- First turn of a session: the room framing, the roster, and whatever
  transcript Hub carried, so Claude arrives knowing where it is.
- Later turns: just the new message. The CLI holds the rest under
  `--resume`, and re-sending the transcript would both cost tokens and
  give Claude two copies of the same conversation to reconcile.

What is deliberately absent: Orion's recall results, self-state, drives,
journals, and mirrors. Claude gets room-native context -- the same material
any participant sees -- and nothing from Orion's interior. That boundary is
the whole privacy position of this feature, so it lives in one function
where it can be read and tested rather than being spread across callers.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from orion.schemas.room_claude import RoomTranscriptEntryV1

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
AUTO_INVITE_CLAUSE = (
    "\n\nYou are auto-invited after every turn in this room, so you will often "
    "have nothing worth adding. When that is the case reply with EXACTLY "
    "[pass] and nothing else. Speak when you genuinely have something worth "
    "the interruption: a disagreement, a risk nobody named, a real idea. Do "
    "not speak merely to acknowledge, agree, or be encouraging."
)

# Sentinel Claude returns when it chooses silence. Matched leniently (case
# and surrounding whitespace) because a model asked for an exact token will
# occasionally wrap it; anything longer than the sentinel is treated as real
# speech rather than silently swallowed.
PASS_SENTINEL = "[pass]"


def is_pass(text: str) -> bool:
    return text.strip().strip("`").casefold() == PASS_SENTINEL


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


def render_transcript(entries: Sequence[RoomTranscriptEntryV1]) -> str:
    return "\n".join(f"{e.speaker_name}: {e.text}" for e in entries if e.text.strip())


def build_turn_prompt(
    *,
    prompt: str,
    invited_by_name: str,
    transcript: Sequence[RoomTranscriptEntryV1],
    social_memory_summary: Dict[str, Any],
    first_turn: bool,
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
        lines = [f"{e.speaker_name}: {e.text}" for e in transcript if e.text.strip()]
        lines.append(f"{invited_by_name}: {prompt}")
        return "\n".join(lines)

    sections: list[str] = []
    context = render_room_context(social_memory_summary)
    if context:
        sections.append(context)
    history = render_transcript(transcript)
    if history:
        sections.append("Conversation so far:\n" + history)
    sections.append(f"{invited_by_name}: {prompt}")
    return "\n\n".join(sections)


def filtered_summary(social_memory_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Allowlist the social-memory summary before it leaves for a third party."""
    if not isinstance(social_memory_summary, dict):
        return {}
    return {k: v for k, v in social_memory_summary.items() if k in _SUMMARY_KEYS}
