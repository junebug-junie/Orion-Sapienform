from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_near_ratio_above_over_ratio_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ENERGY_STAKES_NEAR_RATIO"):
        Settings(ENERGY_STAKES_NEAR_RATIO=1.2, ENERGY_STAKES_OVER_RATIO=1.1)


def test_equal_and_default_ratios_are_accepted() -> None:
    assert Settings(ENERGY_STAKES_NEAR_RATIO=1.1, ENERGY_STAKES_OVER_RATIO=1.1).ENERGY_STAKES_NEAR_RATIO == 1.1
    s = Settings()
    assert s.ENERGY_STAKES_NEAR_RATIO <= s.ENERGY_STAKES_OVER_RATIO
