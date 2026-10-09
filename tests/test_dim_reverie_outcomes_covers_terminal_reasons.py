"""Every visual chain terminal reason the schema can persist has a row in the analytics outcome
dimension. `resource_deferred` was missing (GPU pool parent spec, downstream readers item 8), so
deferred chains fell out of the dim join (assert_visual_reverie_joins_do_not_fan_out would fail on
the first one). Fixed in GPU pool stage 5.4; this keeps the next new reason from repeating it."""
from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from orion.schemas.reverie_visual import VisualTerminalReason

MART = Path(__file__).resolve().parents[1] / "services/orion-analytics/models/marts/dim_reverie_outcomes.sql"


def test_every_visual_terminal_reason_has_an_outcome_row():
    keys = set(re.findall(r"^\s*\('([a-z_]+)',", MART.read_text(), flags=re.M))
    missing = sorted(set(get_args(VisualTerminalReason)) - keys)
    assert not missing, f"dim_reverie_outcomes.sql lacks rows for {missing}"
    orders = [int(n) for n in re.findall(r",\s*(\d+)\)\s*,?\s*$", MART.read_text(), flags=re.M)]
    assert len(orders) == len(set(orders)), "sort_order must stay unique"
