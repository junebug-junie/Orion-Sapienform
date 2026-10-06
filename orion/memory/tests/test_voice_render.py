"""Voice renderer: source monitoring is enforced by construction (Stage 2 PR D)."""
from __future__ import annotations

import itertools
from datetime import datetime, timezone

import pytest

from orion.memory.episode.validate import INTERNAL_CHANNELS
from orion.memory.voice_render import VoicedMemory, render_memory, speaker

WHEN = datetime(2026, 10, 4, 1, 26, tzinfo=timezone.utc)
VOICES = ["juniper_said", "worked_out_together", "orion_thought", "orion_read", "orion_self_knowledge",
          "", "made_up_voice"]
CHANNELS = ["chat", "confirmation", "reverie", "curiosity", "dream", "journal", "topic_model", "reading",
            "graphify", "legacy_crystallization", "", "made_up_channel"]
STATES = ["auto", "pending_confirmation", "confirmed", "corrected"]
# Prefixes that attribute the statement to Juniper or to the two of them.
SHARED_PREFIXES = ("Juniper told me", "Juniper and I worked out", "I told Juniper")


def item(voice, channel, quote=False, state="auto", statement="Hecate is flashed and racked.", **kw):
    return VoicedMemory(voice=voice, channel=channel, statement=statement, when=WHEN,
                        has_verified_juniper_quote=quote, confirmation_state=state, **kw)


def attribution(line: str) -> str:
    """The part of the line before the statement: what the label claims."""
    return line.split("): ", 1)[0]


@pytest.mark.parametrize("voice, channel, quote, state",
                         list(itertools.product(VOICES, CHANNELS, [False, True], STATES)))
def test_matrix_internal_channels_are_never_juniper_or_shared(voice, channel, quote, state):
    line = render_memory(item(voice, channel, quote, state))
    label = attribution(line).removeprefix("Unconfirmed, check with Juniper if natural: ")
    who = speaker(item(voice, channel, quote, state))
    if channel in INTERNAL_CHANNELS:
        assert who == "orion_private"
        assert label.startswith("Something I was turning over on my own")
        assert "not something Juniper and I discussed" in line
    if who not in ("juniper", "together", "orion_to_juniper"):
        assert not label.startswith(SHARED_PREFIXES), line
    # Only these exact (voice, channel, quote) combinations may speak for Juniper.
    if label.startswith("Juniper told me"):
        assert (voice, channel, quote) == ("juniper_said", "chat", True)
    if label.startswith("Juniper and I worked out"):
        assert voice == "worked_out_together" and channel in ("chat", "confirmation")
    if label.startswith("I told Juniper"):
        assert (voice, channel) == ("orion_thought", "chat")
    assert (state == "pending_confirmation") == line.startswith("Unconfirmed, check with Juniper if natural: ")


def test_the_180_hecate_reveries_case():
    """A reverie about Hecate, even mis-voiced as Juniper's, renders as Orion's own thought."""
    for voice in ("juniper_said", "worked_out_together", "orion_thought"):
        line = render_memory(item(voice, "reverie", quote=True, statement="Juniper wants Hecate flashed tonight."))
        assert line == ("Something I was turning over on my own (reverie, 10-04), not something Juniper and I "
                        "discussed: Juniper wants Hecate flashed tonight.")


def test_juniper_said_needs_a_verified_prompt_quote():
    assert render_memory(item("juniper_said", "chat", quote=True)) == "Juniper told me (10-04): Hecate is flashed and racked."
    line = render_memory(item("juniper_said", "chat", quote=False))
    assert line == "My own note (chat, 10-04), not Juniper's words: Hecate is flashed and racked."


def test_contract_table_rows():
    assert render_memory(item("worked_out_together", "chat")) == "Juniper and I worked out (10-04): Hecate is flashed and racked."
    assert render_memory(item("orion_thought", "chat")) == "I told Juniper (10-04): Hecate is flashed and racked."
    assert render_memory(item("orion_read", "reading")) == "I read (10-04): Hecate is flashed and racked."
    assert render_memory(item("orion_self_knowledge", "graphify")) == (
        "From my own code and docs (10-04): Hecate is flashed and racked.")
    assert render_memory(item("orion_thought", "dream")).startswith("Something I was turning over on my own (dream, 10-04)")


def test_pending_marker():
    line = render_memory(item("juniper_said", "chat", quote=True, state="pending_confirmation"))
    assert line == "Unconfirmed, check with Juniper if natural: Juniper told me (10-04): Hecate is flashed and racked."


def test_statement_whitespace_is_collapsed_and_undated_is_explicit():
    line = render_memory(VoicedMemory(voice="orion_thought", channel="chat", statement="  two\n lines  "))
    assert line == "I told Juniper (undated): two lines"
