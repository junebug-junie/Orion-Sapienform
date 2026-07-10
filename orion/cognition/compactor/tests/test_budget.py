from __future__ import annotations

import pytest

from orion.cognition.compactor.budget import assert_fields_within_budget


def test_assert_fields_within_budget_raises_named_error() -> None:
    with pytest.raises(ValueError, match="compactor_output_over_budget:card_summary"):
        assert_fields_within_budget({"card_summary": ("x" * 10, 5)})
