"""Walkway captions (2026-10-15): the caller's instruction prompt reaches an
instruction-tuned VLM only, the agreed "nothing here" answer becomes no
caption, and the qwen lane's broadcast opens for named streams only."""
from __future__ import annotations

from app.caption_sanitize import sanitize_caption
from app.vlm_family import matches_null_answer, resolve_caption_prompt
from orion.vision.caption_echo import CAPTION_PROMPT

QWEN = "Qwen/Qwen2-VL-2B-Instruct"
BLIP = "Salesforce/blip-image-captioning-base"
WALKWAY_PROMPT = "Describe only people, animals and vehicles. If there are none, answer exactly: nothing notable."


def test_qwen_gets_the_callers_prompt():
    assert resolve_caption_prompt(QWEN, {"caption_prompt": WALKWAY_PROMPT}) == (WALKWAY_PROMPT, None)


def test_qwen_without_override_keeps_default_prompt():
    assert resolve_caption_prompt(QWEN, {}) == (CAPTION_PROMPT, None)
    assert resolve_caption_prompt(QWEN, None) == (CAPTION_PROMPT, None)


def test_blip_cannot_take_a_caller_prompt_and_says_so():
    # BLIP continues text instead of obeying it (#2542); the runner turns this
    # warning into "no caption" (test_run_caption_frame_walkway.py).
    prompt, warning = resolve_caption_prompt(BLIP, {"caption_prompt": WALKWAY_PROMPT})
    assert prompt is None
    assert warning == "caption_rejected:caller_prompt_unsupported"


def test_null_answer_matches_with_case_and_punctuation():
    for reply in ("nothing notable", "Nothing notable.", "  NOTHING NOTABLE!  ", '"Nothing notable."'):
        assert matches_null_answer(reply, ["nothing notable"]), reply
    assert matches_null_answer("nothing notable", "nothing notable")


def test_null_answer_is_exact_not_a_substring():
    assert not matches_null_answer("A man walks past; nothing notable otherwise.", ["nothing notable"])
    assert not matches_null_answer("A dog walks on the sidewalk.", ["nothing notable"])
    assert not matches_null_answer("nothing notable", None)
    assert not matches_null_answer("nothing notable", [])
    assert not matches_null_answer("", ["", "  "])


def test_echo_of_the_callers_prompt_is_rejected():
    text, ok, reason = sanitize_caption(WALKWAY_PROMPT, prompt=WALKWAY_PROMPT)
    assert (text, ok, reason) == (None, False, "prompt_echo")
    # default prompt still checked when no override
    assert sanitize_caption(CAPTION_PROMPT)[2] == "prompt_echo"


def test_real_walkway_sentence_passes():
    text, ok, _ = sanitize_caption("A person walks a dog along the sidewalk.", prompt=WALKWAY_PROMPT)
    assert ok and text.startswith("A person")
