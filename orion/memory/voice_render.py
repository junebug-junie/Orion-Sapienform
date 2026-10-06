"""One voice renderer for recalled memory (memory redesign rev 3, section 7).

Turns a remembered item into the line Orion reads, saying whose thought it
is. Pure function; no I/O.

Source monitoring is one-directional, as in the Stage 1 validator
(``orion/memory/episode/validate.py``):

* Only two paths may put words in Juniper's mouth or claim the two of them
  worked something out: ``juniper_said`` and ``worked_out_together``, and
  only on the ``chat`` channel (``worked_out_together`` also on
  ``confirmation``). ``juniper_said`` also needs a verified quote from one of
  Juniper's own prompts.
* Anything on an internal channel (reverie, curiosity, dream, journal,
  topic_model) renders as Orion's own private thought, whatever its voice
  says. It informs Orion as a prior; it is never something Juniper said and
  never something the two of them discussed.
* Every other mismatch (unknown voice, ``juniper_said`` without a verified
  quote, a voice on the wrong channel) falls back to Orion's own voice, never
  toward Juniper's.

Callers today: the chat stance's reverie glimpse
(``services/orion-cortex-exec/app/chat_stance.py``) and the Stage 1 episode
report (``orion/memory/episode/report.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from orion.memory.episode.validate import INTERNAL_CHANNELS

# Channels on which Juniper and Orion actually exchanged words.
_SHARED_CHANNELS = frozenset({"chat"})
_WORKED_OUT_CHANNELS = frozenset({"chat", "confirmation"})


@dataclass(frozen=True)
class VoicedMemory:
    """What the renderer needs; producers map their own rows onto this."""

    voice: str
    channel: str
    statement: str
    when: date | datetime | None = None
    # juniper_said only renders as Juniper's words with a verified quote from
    # one of her own chat prompts (episode_memory_evidence: source_kind
    # chat_prompt, verified).
    has_verified_juniper_quote: bool = False
    confirmation_state: str = "auto"  # auto | pending_confirmation | confirmed | corrected | rejected
    # Not here yet, on purpose (no producer today): a reading's title and claim
    # status, a graphify build date, and the "faded" marker. They arrive with
    # the producers that can fill them (Stage 2 PR F; fading is Stage 4).


def _day(value: date | datetime | None) -> str:
    if value is None:
        return "undated"
    return value.strftime("%m-%d")


def speaker(item: VoicedMemory) -> str:
    """Whose words the rendered line attributes the statement to.

    One of ``juniper``, ``together``, ``orion_to_juniper``, ``orion_private``
    (internal channel), ``orion_note`` (any other mismatch), ``orion_read``,
    ``orion_self_knowledge``. Tests and evals key on this, not on wording.
    """
    voice, channel = item.voice, item.channel
    if channel in INTERNAL_CHANNELS:
        return "orion_private"
    if voice == "juniper_said" and channel in _SHARED_CHANNELS and item.has_verified_juniper_quote:
        return "juniper"
    if voice == "worked_out_together" and channel in _WORKED_OUT_CHANNELS:
        return "together"
    if voice == "orion_thought" and channel in _SHARED_CHANNELS:
        return "orion_to_juniper"
    if voice == "orion_read" and channel == "reading":
        return "orion_read"
    if voice == "orion_self_knowledge" and channel == "graphify":
        return "orion_self_knowledge"
    return "orion_note"


def render_memory(item: VoicedMemory) -> str:
    """The single line Orion reads for one remembered item."""
    statement = " ".join(str(item.statement or "").split())
    when = _day(item.when)
    who = speaker(item)
    if who == "juniper":
        line = f"Juniper told me ({when}): {statement}"
    elif who == "together":
        line = f"Juniper and I worked out ({when}): {statement}"
    elif who == "orion_to_juniper":
        line = f"I told Juniper ({when}): {statement}"
    elif who == "orion_read":
        line = f"I read ({when}): {statement}"
    elif who == "orion_self_knowledge":
        line = f"From my own code and docs ({when}): {statement}"
    elif who == "orion_private":
        line = (f"Something I was turning over on my own ({item.channel}, {when}), "
                f"not something Juniper and I discussed: {statement}")
    else:
        line = f"My own note ({item.channel or 'unknown source'}, {when}), not Juniper's words: {statement}"
    if item.confirmation_state == "pending_confirmation":
        line = "Unconfirmed, check with Juniper if natural: " + line
    return line


__all__ = ["VoicedMemory", "render_memory", "speaker"]
