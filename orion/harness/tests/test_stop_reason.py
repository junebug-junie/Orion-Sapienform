from orion.harness.stop_reason import OUTSIDE_CANCEL_REASONS, is_outside_stop, turn_stop_reason


def test_a_cancel_wins_over_the_motor_code_and_nothing_is_none():
    assert turn_stop_reason(cancel_reason="user_stop", error_code="fcc_nonzero_exit") == "cancel:user_stop"
    assert turn_stop_reason(cancel_reason=None, error_code="fcc_timeout") == "fcc_timeout"
    assert turn_stop_reason(cancel_reason="  ", error_code="") is None


def test_only_known_cancels_are_outside_stops():
    assert all(is_outside_stop(f"cancel:{r}") for r in OUTSIDE_CANCEL_REASONS)
    for value in ("cancel:something_new", "cancel:", "fcc_timeout", "user_stop", None, 9, ""):
        assert is_outside_stop(value) is False
