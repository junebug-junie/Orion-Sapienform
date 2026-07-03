from orion.vision.caption_echo import CAPTION_PROMPT, is_caption_prompt_echo


def test_is_caption_prompt_echo_matches_current_prompt() -> None:
    assert is_caption_prompt_echo(CAPTION_PROMPT) is True


def test_is_caption_prompt_echo_matches_legacy_phrase() -> None:
    assert is_caption_prompt_echo("Describe this image in detail.") is True


def test_is_caption_prompt_echo_rejects_blip_suffix_noise() -> None:
    assert is_caption_prompt_echo(CAPTION_PROMPT + "s") is True


def test_is_caption_prompt_echo_rejects_real_caption() -> None:
    assert is_caption_prompt_echo("A desk with two monitors and an open door.") is False
