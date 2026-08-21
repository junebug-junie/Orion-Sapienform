import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.caption_sanitize import CAPTION_PROMPT, sanitize_answer, sanitize_caption


def test_sanitize_rejects_youtube_slop() -> None:
    text, ok, reason = sanitize_caption("youtube video watching webcam stream")
    assert ok is False
    assert text is None
    assert reason == "stoplist_ratio"


def test_sanitize_rejects_prompt_echo() -> None:
    text, ok, reason = sanitize_caption("describe this image. youtube")
    assert ok is False
    assert text is None
    assert reason == "prompt_echo"


def test_sanitize_rejects_caption_prompt_echo() -> None:
    text, ok, reason = sanitize_caption(CAPTION_PROMPT)
    assert ok is False
    assert text is None
    assert reason == "prompt_echo"


def test_sanitize_rejects_caption_prompt_echo_with_suffix() -> None:
    text, ok, reason = sanitize_caption(CAPTION_PROMPT + "s")
    assert ok is False
    assert text is None
    assert reason == "prompt_echo"


def test_sanitize_accepts_plain_scene() -> None:
    text, ok, reason = sanitize_caption("A desk with two monitors and an open door.")
    assert ok is True
    assert "monitors" in text
    assert reason is None


def test_caption_prompt_is_factual() -> None:
    assert "directly visible" in CAPTION_PROMPT.lower()


# --- sanitize_answer (VQA) ---------------------------------------------------


def test_sanitize_answer_rejects_youtube_slop() -> None:
    text, ok, reason = sanitize_answer("youtube video watching webcam stream", "is the door open?")
    assert ok is False
    assert text is None
    assert reason == "stoplist_ratio"


def test_sanitize_answer_rejects_question_echo() -> None:
    """The echo check must be against the real question, not CAPTION_PROMPT --
    this is the exact bug a naive reuse of sanitize_caption would have (it
    only ever checks against the fixed caption prompt, so a VQA echo of the
    question itself would sail through unnoticed)."""
    question = "Is the office chair occupied right now?"
    text, ok, reason = sanitize_answer(question, question)
    assert ok is False
    assert text is None
    assert reason == "prompt_echo"


def test_sanitize_answer_does_not_falsely_flag_against_caption_prompt() -> None:
    """A real answer that happens to overlap CAPTION_PROMPT's wording must not
    be rejected -- proves the echo check is scoped to the real question, not
    silently still checking the module-level CAPTION_PROMPT default."""
    text, ok, reason = sanitize_answer("Yes, visible objects include a laptop.", "what do you see?")
    assert ok is True
    assert text == "Yes, visible objects include a laptop."
    assert reason is None


def test_sanitize_answer_accepts_short_direct_answer() -> None:
    """Unlike sanitize_caption's 12-char floor, a short direct VQA answer is
    normal and correct, not degenerate -- must not be rejected for length."""
    text, ok, reason = sanitize_answer("Yes.", "is the door closed?")
    assert ok is True
    assert text == "Yes."
    assert reason is None


def test_sanitize_answer_rejects_empty() -> None:
    text, ok, reason = sanitize_answer("   ", "is the door closed?")
    assert ok is False
    assert text is None
    assert reason == "empty"


def test_sanitize_answer_rejects_punctuation_only_response() -> None:
    """Confirmed live 2026-08-20: BLIP-base answered a real question with a
    bare '?' -- schema-valid (a non-empty string) but not a real answer.
    The old whitespace-only token filter let this through as a 1-token,
    0%-stoplist 'answer'; must be rejected as empty instead."""
    text, ok, reason = sanitize_answer("?", "is there a person visible in this image?")
    assert ok is False
    assert text is None
    assert reason == "empty"


def test_sanitize_caption_rejects_punctuation_only_response_if_it_ever_gets_this_far() -> None:
    """Same fix as sanitize_answer's, applied for consistency even though
    the 12-char length floor above already rejects '?' first in practice
    -- this pins the token filter itself, not just the length floor's
    incidental coverage of it."""
    text, ok, reason = sanitize_caption("? ? ? ? ? ? ? ? ? ? ? ?")  # 12+ chars, all punctuation
    assert ok is False
    assert text is None
    assert reason == "empty"


def test_sanitize_answer_rejects_repetition_garbage() -> None:
    """Confirmed live 2026-08-20, same VQA smoke session: BLIP-base answered
    'what color is the door?' with '| by person | cci | cci | cci | cci |
    cci | cci |' -- non-empty, no echo, no stoplist hits (none of those
    tokens are on the topic-specific stoplist), but obvious repetition
    garbage, not a real answer."""
    text, ok, reason = sanitize_answer(
        "| by person | cci | cci | cci | cci | cci | cci |", "what color is the door?"
    )
    assert ok is False
    assert text is None
    assert reason == "repetition_degenerate"


def test_sanitize_answer_does_not_flag_a_short_answer_as_repetitive() -> None:
    """Repetition-dominance is meaningless below the minimum token count --
    a genuine short answer must not be rejected just because its rarer
    token is a numeric minority of a tiny token list."""
    text, ok, reason = sanitize_answer("Two monitors.", "how many monitors?")
    assert ok is True
    assert reason is None


def test_sanitize_caption_rejects_repetition_garbage_too() -> None:
    text, ok, reason = sanitize_caption("cci cci cci cci cci cci cci cci")
    assert ok is False
    assert text is None
    assert reason == "repetition_degenerate"
