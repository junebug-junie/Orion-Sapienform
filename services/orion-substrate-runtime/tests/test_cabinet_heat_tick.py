"""node:substrate.cabinet's prediction_error: the cabinet warming error, through the existing
prediction-error node path (attend-to-act loop item 2)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from orion.autonomy.cabinet_heat import CABINET_NODE_ID
from orion.hardware_watch.rules import TempPoint


def _worker(points):
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(cabinet_heat_rise_threshold_c=0.5)
    w._store = SimpleNamespace(load_cabinet_points=lambda since, until: points)
    w.writes = []
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw)
    return w


def _ramp(a, b, minutes=20):
    now = datetime.now(timezone.utc)
    n = minutes * 2
    return [TempPoint(now - timedelta(seconds=30 * (n - i)), a + (b - a) * i / n) for i in range(n + 1)]


def test_elevated_and_rising_writes_a_nonzero_error_on_the_cabinet_node():
    w = _worker(_ramp(29.6, 30.4))
    w._cabinet_heat_tick()
    (write,) = w.writes
    assert write["node_id"] == CABINET_NODE_ID and write["reducer_key"] == "cabinet_heat"
    assert 0.5 <= write["error"] <= 1.0


def test_elevated_but_flat_writes_a_calm_zero_every_tick():
    w = _worker(_ramp(30.5, 30.5))
    w._cabinet_heat_tick()
    assert w.writes[0]["error"] == 0.0     # written (not skipped), so the node can return to calm


def test_no_reading_writes_nothing():
    w = _worker([])
    w._cabinet_heat_tick()
    assert w.writes == []


def test_a_read_failure_never_raises_out_of_the_tick():
    w = _worker([])
    w._store = SimpleNamespace(load_cabinet_points=lambda since, until: (_ for _ in ()).throw(RuntimeError("db")))
    w._cabinet_heat_tick()
    assert w.writes == []
