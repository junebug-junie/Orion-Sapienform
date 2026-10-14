"""Acceptance check 4 (spec 2026-10-07): every self-reading number resolves to a persisted row
and matches it; missing or stale data renders nothing; a real high signal shows."""
from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.attention.self_readings import project_self_readings

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
from self_readings_truth import check_block  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "self_readings_history", _REPO / "tests" / "fixtures" / "self_readings_history.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)  # type: ignore[union-attr]


def _block(rows, now):
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=now)
    return project_self_readings(cache, now=now)


def test_busy_body_every_number_resolves_to_its_row() -> None:
    rows = fx.busy_body_rows()
    block = _block(rows, fx.NOW)
    assert block.text, block.skip_reason
    assert check_block(block.text, rows, fx.NOW) == []
    assert set(block.shown) == {"node:substrate.execution", "node:substrate.bus_synaptic",
                                "node:substrate.cabinet"}


@pytest.mark.parametrize("case", ["calm", "stale", "missing"])
def test_nothing_to_say_renders_nothing(case) -> None:
    rows, now = {
        "calm": (fx.calm_rows(), fx.NOW),
        "stale": (fx.busy_body_rows(), fx.NOW + timedelta(hours=2)),
        "missing": ([], fx.NOW),
    }[case]
    assert _block(rows, now).text == ""


def test_checker_has_teeth() -> None:
    rows = fx.busy_body_rows()
    text = _block(rows, fx.NOW).text
    assert check_block(text.replace("now 1.00", "now 0.40"), rows, fx.NOW)
    assert check_block(text.replace("fired 11 times", "fired 12 times"), rows, fx.NOW)
    assert check_block(text.replace("03:29:30Z", "03:29:31Z"), rows, fx.NOW)
    assert check_block(text + "\n- invented: 7/10 for me", rows, fx.NOW)
