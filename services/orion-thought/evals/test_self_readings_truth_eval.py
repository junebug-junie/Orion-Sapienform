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
    assert check_block(text.replace("went off 2 times", "went off 3 times"), rows, fx.NOW)
    assert check_block(text.replace("03:29:30Z", "03:29:31Z"), rows, fx.NOW)
    assert check_block(text + "\n- invented: 7/10 for me", rows, fx.NOW)


# --- "tends to" clauses (spec section E): every number resolves to its association row ----------

from datetime import datetime, timezone  # noqa: E402

from orion.attention.self_readings import build_self_readings, render_self_readings  # noqa: E402
from orion.schemas.self_signal_outcome import SelfSignalOutcomeAssociationV1  # noqa: E402

_EXE = "node:substrate.execution"
_ROW = SelfSignalOutcomeAssociationV1(
    association_id="ssa-0123456789", computed_at=datetime(2026, 10, 14, 3, tzinfo=timezone.utc),
    signal_id=_EXE, outcome="rpc_timeout", n_high=42, events_high=13, n_normal=120, events_normal=12,
    rate_high=13 / 42, rate_normal=0.1, ratio=3.0952, ci_low=1.6, ci_high=6.0, min_hours=24,
    verdict="tends_more",
    clause=("when this is high, my next hour had an RPC timeout 3.1x as often (31% of 42 high hours "
            "vs 10% of 120 normal) [assoc:ssa-0123456789]"),
)


def _clause_block():
    rows = fx.busy_body_rows()
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if r[1] >= since], now=fx.NOW)
    readings = build_self_readings(cache, now=fx.NOW, associations={_EXE: _ROW})
    return rows, render_self_readings(readings, now=fx.NOW, max_shown=2).text


def test_a_rendered_clause_resolves_to_its_association_row() -> None:
    rows, text = _clause_block()
    assert "[assoc:ssa-0123456789]" in text
    assert check_block(text, rows, fx.NOW, associations={_ROW.association_id: _ROW}) == []


def test_clause_checker_has_teeth() -> None:
    rows, text = _clause_block()
    ok = {_ROW.association_id: _ROW}
    assert check_block(text, rows, fx.NOW)  # a clause with no row behind it
    assert check_block(text.replace("3.1x", "4.1x"), rows, fx.NOW, associations=ok)
    assert check_block(text.replace("of 42 high", "of 41 high"), rows, fx.NOW, associations=ok)
    other = _ROW.model_copy(update={"signal_id": "node:substrate.chat"})
    assert check_block(text, rows, fx.NOW, associations={_ROW.association_id: other})
