from __future__ import annotations

from app.settings import Settings


def test_pre_turn_handler_defaults_enabled() -> None:
    s = Settings(_env_file=None)
    assert s.enable_pre_turn_appraisal_handler is True


def test_pre_turn_handler_lane_disabled_via_env() -> None:
    s = Settings(_env_file=None, ENABLE_PRE_TURN_APPRAISAL_HANDLER="false")
    assert s.enable_pre_turn_appraisal_handler is False
