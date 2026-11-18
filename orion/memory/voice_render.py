"""One voice renderer for remembered items (memory redesign rev 3, section 7).

Turns a remembered item into the line Orion reads, saying whose thought it
is. Pure function; no I/O.

Source monitoring reuses the Stage 1 validator's own rules
(``orion/memory/episode/validate.py``), so the renderer can never be looser
than what the writer would have kept:

* Juniper's voices only arrive through ``chat`` (``validate.py`` rejects
  them on any other channel).
* The voice is first re-checked against the verified evidence with the
  validator's ``_supported_voice``: ``worked_out_together`` needs a verified
  quote from Juniper's prompt AND from Orion's reply; ``juniper_said`` needs
  a verified prompt quote; ``orion_thought`` needs any verified quote. Missing
  evidence moves the voice only AWAY from Juniper, never toward her.
* Anything on an internal channel (reverie, curiosity, dream, journal,
  topic_model) renders as Orion's own private thought, whatever its voice
  says: an informed prior, never something Juniper said or discussed.
* Everything else falls back to "My own note…, not Juniper's words".
* A memory Juniper rejected or corrected never renders as current truth.

Callers: the daily episode report (``orion/memory/episode/report.py``, live)
and the legacy ``chat_general`` stance's reverie glimpse
(``services/orion-cortex-exec/app/chat_stance.py``; rendered only by
``chat_stance_brief.j2``, which had no live traffic on 2026-10-06).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from orion.memory.episode.validate import INTERNAL_CHANNELS, _supported_voice


@dataclass(frozen=True)
class VoicedMemory:
    """What the renderer needs; producers map their own rows onto this."""

    voice: str
    channel: str
    statement: str
    when: date | datetime | None = None
    # Verified quotes behind the memory (episode_memory_evidence, verified):
    # from one of Juniper's chat prompts, and from one of Orion's replies.
    has_verified_juniper_quote: bool = False
    has_verified_orion_quote: bool = False
    # auto | pending_confirmation | unconfirmed (asked, no answer in 7 days; set by
    # orion/memory/episode/confirmation.py) | confirmed | corrected | rejected
    confirmation_state: str = "auto"


def _day(value: date | datetime | None) -> str:
    if value is None:
        return "undated"
    return value.strftime("%m-%d")


def speaker(item: VoicedMemory) -> str:
    """Whose words the rendered line attributes the statement to.

    One of ``juniper``, ``together``, ``orion_to_juniper``, ``orion_private``
    (internal channel) or ``orion_note`` (anything else). Tests key on this.
    """
    if item.channel in INTERNAL_CHANNELS:
        return "orion_private"
    if item.channel != "chat":
        return "orion_note"
    voice = _supported_voice(item.voice, item.has_verified_juniper_quote, item.has_verified_orion_quote)
    if voice == "worked_out_together":
        return "together"
    if voice == "juniper_said":
        return "juniper"
    if voice == "orion_thought" and (item.has_verified_juniper_quote or item.has_verified_orion_quote):
        return "orion_to_juniper"
    return "orion_note"


def render_memory(item: VoicedMemory) -> str:
    """The single line Orion reads for one remembered item."""
    statement = " ".join(str(item.statement or "").split())
    when = _day(item.when)
    if item.confirmation_state == "rejected":
        return f"Something I had remembered ({when}) that Juniper rejected, not true: {statement}"
    if item.confirmation_state == "corrected":
        return f"Something I had remembered ({when}) that Juniper corrected, superseded: {statement}"
    who = speaker(item)
    if who == "juniper":
        line = f"Juniper told me ({when}): {statement}"
    elif who == "together":
        line = f"Juniper and I worked out ({when}): {statement}"
    elif who == "orion_to_juniper":
        line = f"I told Juniper ({when}): {statement}"
    elif who == "orion_private":
        line = (f"Something I was turning over on my own ({item.channel}, {when}), "
                f"not something Juniper and I discussed: {statement}")
    else:
        line = f"My own note ({item.channel or 'unknown source'}, {when}), not Juniper's words: {statement}"
    if item.confirmation_state in ("pending_confirmation", "unconfirmed"):
        # An expired ask is not a resolution: the memory keeps its Unconfirmed label for good.
        line = "Unconfirmed, check with Juniper if natural: " + line
    return line


__all__ = ["VoicedMemory", "render_memory", "speaker"]
