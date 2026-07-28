from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from orion.substrate.bus_synaptic_surprise import latest_bus_synaptic_prediction_error


def _engine_with_row(row: dict | None) -> MagicMock:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = row
    return engine


def test_returns_real_value() -> None:
    engine = _engine_with_row({"value": "0.1675", "generated_at": datetime.now(timezone.utc)})
    assert latest_bus_synaptic_prediction_error(engine) == 0.1675


def test_none_when_no_row() -> None:
    engine = _engine_with_row(None)
    assert latest_bus_synaptic_prediction_error(engine) is None


def test_none_when_node_absent() -> None:
    engine = _engine_with_row({"value": None, "generated_at": datetime.now(timezone.utc)})
    assert latest_bus_synaptic_prediction_error(engine) is None


def test_none_when_stale() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    engine = _engine_with_row({"value": "0.1675", "generated_at": stale})
    assert latest_bus_synaptic_prediction_error(engine) is None


def test_none_when_unparseable() -> None:
    engine = _engine_with_row({"value": "not-a-float", "generated_at": datetime.now(timezone.utc)})
    assert latest_bus_synaptic_prediction_error(engine) is None
