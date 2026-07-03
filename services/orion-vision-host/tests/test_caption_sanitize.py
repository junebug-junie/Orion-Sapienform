import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.caption_sanitize import CAPTION_PROMPT, sanitize_caption


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
