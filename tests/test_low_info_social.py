from orion.memory.low_info_social import is_low_info_social


def test_greeting_is_low_info():
    assert is_low_info_social("hey Orion") is True


def test_substantive_is_not_low_info():
    assert is_low_info_social("still drowning in move logistics alone") is False


def test_thanks_is_low_info():
    assert is_low_info_social("thanks!") is True


def test_anytime_is_low_info():
    assert is_low_info_social("anytime") is True
