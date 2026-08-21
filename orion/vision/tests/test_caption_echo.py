from orion.vision.caption_echo import CAPTION_PROMPT, is_caption_prompt_echo, strip_echoed_prompt_prefix


def test_is_caption_prompt_echo_matches_current_prompt() -> None:
    assert is_caption_prompt_echo(CAPTION_PROMPT) is True


def test_is_caption_prompt_echo_matches_legacy_phrase() -> None:
    assert is_caption_prompt_echo("Describe this image in detail.") is True


def test_is_caption_prompt_echo_rejects_blip_suffix_noise() -> None:
    assert is_caption_prompt_echo(CAPTION_PROMPT + "s") is True


def test_is_caption_prompt_echo_rejects_real_caption() -> None:
    assert is_caption_prompt_echo("A desk with two monitors and an open door.") is False


# --- generalized `prompt` param (added alongside VQA, 2026-08-20) -----------


def test_prompt_param_defaults_to_caption_prompt() -> None:
    """Every pre-existing caller (positional, no `prompt` kwarg) must keep
    its exact prior behavior -- this is the backward-compat contract the
    default argument exists to guarantee."""
    assert is_caption_prompt_echo(CAPTION_PROMPT) is True
    assert is_caption_prompt_echo("A desk with two monitors and an open door.") is False


def test_prompt_param_checks_against_the_given_prompt_not_caption_prompt() -> None:
    """A VQA-shaped call: the question is a real question, not CAPTION_PROMPT
    -- checking with the default would never flag an echo of it."""
    question = "Is the office chair occupied right now?"
    assert is_caption_prompt_echo(question, prompt=question) is True
    # And the same text must NOT be flagged as an echo of the (unrelated)
    # fixed caption prompt when no custom prompt is supplied.
    assert is_caption_prompt_echo(question) is False


# --- strip_echoed_prompt_prefix ----------------------------------------------


def test_strip_echoed_prompt_prefix_is_case_insensitive() -> None:
    """Reproduces the exact live bug (2026-08-20, VQA's first real smoke
    test): a mixed-case question echoed back lowercased by BLIP. A plain
    str.replace() (case-sensitive) silently fails to strip this."""
    question = "What color is the door?"
    generated = "what color is the door? a black and white photo of a room with a desk"
    result = strip_echoed_prompt_prefix(generated, prompt=question)
    assert result == "a black and white photo of a room with a desk"


def test_strip_echoed_prompt_prefix_no_echo_returns_text_unchanged() -> None:
    text = "A desk with two monitors and an open door."
    assert strip_echoed_prompt_prefix(text, prompt=CAPTION_PROMPT) == text


def test_strip_echoed_prompt_prefix_only_strips_a_genuine_leading_match() -> None:
    """Must not mangle a real answer that happens to reuse a prompt word
    mid-sentence, and must not strip anything when the text does not
    actually start with the prompt -- only a genuine prefix echo is
    removed, never a global find-and-replace."""
    question = "door"
    text = "The door is closed and there is a door mat in front of it."
    # text does NOT start with "door" (starts with "The") -- unchanged.
    assert strip_echoed_prompt_prefix(text, prompt=question) == text

    echoed = "door the door is closed and there is a door mat in front of it."
    assert strip_echoed_prompt_prefix(echoed, prompt=question) == (
        "the door is closed and there is a door mat in front of it."
    )


def test_strip_echoed_prompt_prefix_pure_echo_leaves_empty_string() -> None:
    question = "is the door open?"
    assert strip_echoed_prompt_prefix(question, prompt=question) == ""
