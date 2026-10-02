"""Review of Stage 1 (2026-10-02): source monitoring must be one-directional.

The validator may only move a memory AWAY from Juniper's voice, never into it, and the internal
channel check runs on the FINAL voice and channel. Quotes have a minimum length, are folded
(NFKC, curly quotes, dashes, case, whitespace) before matching, and stakes are the distiller's own.
"""

from __future__ import annotations

import itertools

import pytest

from orion.memory.episode.tests.test_validate import TURNS, _run
from orion.memory.episode.validate import JUNIPER_VOICES, quote_in_text

VOICES = ["juniper_said", "worked_out_together", "orion_thought", "orion_read", "orion_self_knowledge"]
CHANNELS = ["chat", "reverie", "curiosity", "dream", "journal", "topic_model", "reading", "graphify"]
EVIDENCE = {
    "prompt_only": [{"turn": "t2", "field": "prompt", "quote": "Headed to Austin and"}],
    "response_only": [{"turn": "t2", "field": "response", "quote": "I hope the trip goes smoothly"}],
    "both": [{"turn": "t2", "field": "prompt", "quote": "Headed to Austin and"},
             {"turn": "t2", "field": "response", "quote": "I hope the trip goes smoothly"}],
}
STATEMENT = "Something about the Austin trip that is worth remembering later."


def test_reported_repro_reverie_with_a_one_letter_quote_is_not_kept_as_juniper():
    r = _run({"purpose": "happened", "voice": "orion_read", "channel": "reverie",
              "statement": "I dreamed that Juniper secretly wants to leave her job soon.",
              "evidence": [{"turn": "t3", "field": "prompt", "quote": "I"}]})
    assert r.memories == []
    assert [x.reason for x in r.rejections] == ["no_verified_quote"]


@pytest.mark.parametrize("voice, channel, evidence", list(itertools.product(VOICES, CHANNELS, EVIDENCE)))
def test_matrix_never_moves_into_juniper_voice_and_never_internal_as_juniper(voice, channel, evidence):
    r = _run({"purpose": "happened", "voice": voice, "channel": channel, "statement": STATEMENT,
              "evidence": EVIDENCE[evidence]})
    for m in r.memories:
        assert m.channel == channel, "the validator never rewrites the channel"
        if voice not in JUNIPER_VOICES:
            assert m.voice not in JUNIPER_VOICES, f"{voice} was moved INTO Juniper's voice ({m.voice})"
        if m.voice in JUNIPER_VOICES:
            assert m.channel == "chat", "an internal channel can never carry Juniper's voice"
            assert any(e.verified and e.source_kind == "chat_prompt" for e in m.evidence)
        if m.voice == "worked_out_together":
            kinds = {e.source_kind for e in m.evidence if e.verified}
            assert kinds == {"chat_prompt", "chat_response"}
    if voice in JUNIPER_VOICES and channel != "chat":
        assert r.memories == []
    if voice in ("orion_read", "orion_self_knowledge") and evidence == "prompt_only":
        assert r.memories == [] and [x.reason for x in r.rejections] == ["voice_unsupported_by_evidence"]


@pytest.mark.parametrize("quote, ok", [
    ("I", False), ("Headed to", False), ("Headed to Austin", True),
    ("我们下周三在奥斯汀见面吧好吗呢", True),  # 15 CJK chars
    ("我们下周三", False),
])
def test_minimum_quote_length(quote, ok):
    text = "Headed to Austin and will fly back. 我们下周三在奥斯汀见面吧好吗呢"
    assert quote_in_text(quote, text) is ok


@pytest.mark.parametrize("quote, text", [
    ("I don't know yet", "Honestly I don’t know yet."),
    ("“headed to austin”", "Headed to Austin and back"),
    ("one - two - three", "one — two – three"),
    ("full width  spaces here", "full width spaces here"),
    ("ＡＢＣ is fine", "ABC is fine"),          # NFKC
])
def test_quotes_are_folded_on_both_sides(quote, text):
    assert quote_in_text(quote, text)


def test_stakes_are_the_distillers_own():
    low = _run({"purpose": "about_juniper", "voice": "juniper_said", "channel": "chat",
                "statement": "Juniper is anxious about flying to Austin next week.", "stakes": "low",
                "evidence": [{"turn": "t2", "field": "prompt", "quote": "Headed to Austin and"}]})
    assert (low.memories[0].stakes, low.memories[0].confirmation_state) == ("low", "auto")
    assert not any(e.op == "stakes_raised" for e in low.memories[0].events)
    high = _run({"purpose": "happened", "voice": "juniper_said", "channel": "chat", "stakes": "high",
                 "stakes_reason": "safety_location", "statement": "Juniper flew to Austin for a work offsite.",
                 "evidence": [{"turn": "t2", "field": "prompt", "quote": "Headed to Austin and"}]})
    assert (high.memories[0].stakes, high.memories[0].confirmation_state) == ("high", "pending_confirmation")


def test_no_word_list_is_imported():
    import orion.memory.episode.validate as v

    src = open(v.__file__).read()
    assert "intake_junk" not in src and "_STOPWORDS" not in src
