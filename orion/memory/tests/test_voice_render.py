"""Voice renderer: source monitoring enforced by construction (Stage 2 PR D)."""
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
STATES = ["auto", "pending_confirmation", "unconfirmed", "confirmed", "corrected", "rejected"]
EVIDENCE = [(False, False), (True, False), (False, True), (True, True)]  # (prompt quote, reply quote)
SHARED_PREFIXES = ("Juniper told me", "Juniper and I worked out", "I told Juniper")
PENDING = "Unconfirmed, check with Juniper if natural: "


def item(voice, channel, prompt=False, reply=False, state="auto", statement="Hecate is flashed and racked."):
    return VoicedMemory(voice=voice, channel=channel, statement=statement, when=WHEN,
                        has_verified_juniper_quote=prompt, has_verified_orion_quote=reply,
                        confirmation_state=state)


@pytest.mark.parametrize("voice, channel, evidence, state",
                         list(itertools.product(VOICES, CHANNELS, EVIDENCE, STATES)))
def test_matrix_never_speaks_for_juniper_without_her_words(voice, channel, evidence, state):
    prompt, reply = evidence
    it = item(voice, channel, prompt, reply, state)
    line = render_memory(it)
    label = line.removeprefix(PENDING).split("): ", 1)[0]
    if state in ("rejected", "corrected"):
        # Never rendered as current truth, never attributed to anyone as fact.
        assert line.startswith("Something I had remembered (") and f"that Juniper {state}" in line
        assert not label.startswith(SHARED_PREFIXES)
        return
    who = speaker(it)
    if channel in INTERNAL_CHANNELS:
        assert who == "orion_private"
        assert label.startswith("Something I was turning over on my own")
        assert "not something Juniper and I discussed" in line
    # The only ways to speak for Juniper, mirroring validate.py's _supported_voice and channel rule.
    if label.startswith("Juniper told me"):
        assert channel == "chat" and prompt and voice in ("juniper_said", "worked_out_together")
    if label.startswith("Juniper and I worked out"):
        assert (voice, channel, prompt, reply) == ("worked_out_together", "chat", True, True)
    if label.startswith("I told Juniper"):
        assert channel == "chat" and (prompt or reply)
        assert voice not in ("juniper_said", "worked_out_together") or not prompt
    assert (state in ("pending_confirmation", "unconfirmed")) == line.startswith(PENDING)


def test_the_180_hecate_reveries_case():
    """A reverie about Hecate, even mis-voiced as Juniper's and fully quoted, is Orion's own thought."""
    for voice in ("juniper_said", "worked_out_together", "orion_thought"):
        line = render_memory(item(voice, "reverie", True, True, statement="Juniper wants Hecate flashed tonight."))
        assert line == ("Something I was turning over on my own (reverie, 10-04), not something Juniper and I "
                        "discussed: Juniper wants Hecate flashed tonight.")


def test_worked_out_together_needs_both_quotes_and_chat():
    assert render_memory(item("worked_out_together", "chat", True, True)) == (
        "Juniper and I worked out (10-04): Hecate is flashed and racked.")
    assert speaker(item("worked_out_together", "chat", True, False)) == "juniper"
    assert speaker(item("worked_out_together", "chat", False, True)) == "orion_to_juniper"
    assert speaker(item("worked_out_together", "chat", False, False)) == "orion_note"
    assert speaker(item("worked_out_together", "confirmation", True, True)) == "orion_note"


def test_juniper_said_needs_a_verified_prompt_quote():
    assert render_memory(item("juniper_said", "chat", True)) == "Juniper told me (10-04): Hecate is flashed and racked."
    assert speaker(item("juniper_said", "chat", False, True)) == "orion_to_juniper"
    assert render_memory(item("juniper_said", "chat")) == (
        "My own note (chat, 10-04), not Juniper's words: Hecate is flashed and racked.")


def test_rejected_and_corrected_are_never_plain_truth():
    assert render_memory(item("juniper_said", "chat", True, True, state="rejected")) == (
        "Something I had remembered (10-04) that Juniper rejected, not true: Hecate is flashed and racked.")
    assert render_memory(item("juniper_said", "chat", True, True, state="corrected")) == (
        "Something I had remembered (10-04) that Juniper corrected, superseded: Hecate is flashed and racked.")


def test_pending_marker_and_whitespace():
    for state in ("pending_confirmation", "unconfirmed"):  # an expired ask stays unconfirmed
        assert render_memory(item("juniper_said", "chat", True, state=state)) == (
            PENDING + "Juniper told me (10-04): Hecate is flashed and racked.")
    line = render_memory(VoicedMemory(voice="orion_thought", channel="chat", statement="  two\n lines  ",
                                      has_verified_orion_quote=True))
    assert line == "I told Juniper (undated): two lines"
