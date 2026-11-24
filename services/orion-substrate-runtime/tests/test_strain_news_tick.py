"""node:substrate.strain's alarm value (signal_value since PE retire Patch 2): strain news through the node path
(strain loop Part 2, orion/regulation/strain_news.py)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.regulation.arousal_view import ArousalView
from orion.regulation.strain_news import STRAIN_NODE_ID
from orion.schemas.regulation import ArousalReadingV1


@pytest.fixture(autouse=True)
def _pe_nodes_on(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")


def _reading(now, level="strained", since=None):
    return ArousalReadingV1(arousal_level=level, since=since or now - timedelta(minutes=1),
                            observed_at=now, reasons=["gpu_queue_sustained"], cabinet_temp_c=32.0)


def _worker(reading_fn, *, enabled=True, history_rows=500, event_times=()):
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(enable_strain_attention=enabled, orion_situation_timezone="America/Denver",
                                  orion_bus_url="redis://unused")
    w.writes, w.recorded = [], []
    w._store = SimpleNamespace(
        # PE retire Patch 2: the seed reads both tables, the event row lands in the signal table.
        fetch_signal_event_times_read_both=lambda node_id, since: list(event_times),
        count_signal_history_read_both=lambda node_id, since: history_rows,
        save_signal_history_samples=lambda rows: w.recorded.extend(r[:3] for r in rows),
        save_prediction_error_history_samples=lambda samples: w.pe_recorded.extend(samples),
    )
    w.pe_recorded = []
    w._write_signal_node = lambda **kw: w.writes.append(kw)
    w._write_prediction_error_node = lambda **kw: w.pe_writes.append(kw)
    w.pe_writes = []

    def read(now):
        r = reading_fn(now)
        return (r, ArousalView(level=r.arousal_level, reason="ok")) if r is not None else (
            None, ArousalView(level="unknown", reason="stale"))

    w._read_regulation_reading = read
    return w


def test_flag_off_writes_nothing():
    w = _worker(lambda now: _reading(now), enabled=False)
    w._strain_attention_tick()
    assert w.writes == [] and w.recorded == []


def test_unknown_reading_writes_nothing():
    w = _worker(lambda now: None)
    w._strain_attention_tick()
    assert w.writes == [] and w.recorded == []


def test_onset_writes_one_and_records_the_event_row_then_holds_then_rests():
    w = _worker(lambda now: _reading(now))
    w._strain_attention_tick()
    (write,) = w.writes
    assert write["node_id"] == STRAIN_NODE_ID and write["value"] == 1.0
    assert write["reducer_key"] == "strain_news"
    assert [(n, v) for n, _, v in w.recorded] == [(STRAIN_NODE_ID, 1.0)]
    # Next tick inside the 300 s orienting window: no event and no rest write (1.0 carried).
    w._strain_attention_tick()
    assert len(w.writes) == 1
    # Past the window: a calm 0.0, no new event row.
    w._strain_news_state = w._strain_news_state.__class__(
        **{**w._strain_news_state.__dict__,
           "last_event_at": w._strain_news_state.last_event_at - timedelta(seconds=301)})
    w._strain_attention_tick()
    assert w.writes[-1]["value"] == 0.0 and len(w.recorded) == 1


def test_idle_writes_calm_zero_only():
    w = _worker(lambda now: _reading(now, level="idle"))
    w._strain_attention_tick()
    assert [x["value"] for x in w.writes] == [0.0] and w.recorded == []


def test_restart_with_a_recorded_event_does_not_refire_onset():
    # The event is > 30 min old, so the gap cannot be what stops a second onset: only the
    # "already recorded for this episode" seed can.
    since = datetime.now(timezone.utc) - timedelta(minutes=40)
    w = _worker(lambda now: _reading(now, since=since), event_times=[since + timedelta(seconds=30)])
    w._strain_attention_tick()
    assert w.recorded == []
    assert all(x["value"] == 0.0 for x in w.writes)
    # Control: same episode with no recorded event onsets.
    w2 = _worker(lambda now: _reading(now, since=since))
    w2._strain_attention_tick()
    assert [n for n, _, _ in w2.recorded] == [STRAIN_NODE_ID]


def test_pe_nodes_disabled_skips_the_tick(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "false")
    w = _worker(lambda now: _reading(now))
    w._strain_attention_tick()
    assert w.writes == [] and w.recorded == [] and w._strain_news_state is None


def test_a_failed_tick_backs_off():
    calls = []
    w = _worker(lambda now: _reading(now))
    w._store = SimpleNamespace(
        fetch_signal_event_times_read_both=lambda **kw: calls.append(1) or (_ for _ in ()).throw(RuntimeError("db")))
    w._strain_attention_tick()
    w._strain_attention_tick()
    assert calls == [1]


def test_warming_history_writes_rest_and_spends_no_event():
    w = _worker(lambda now: _reading(now), history_rows=10)
    w._strain_attention_tick()
    assert [x["value"] for x in w.writes] == [0.0] and w.recorded == []
    assert w._strain_news_state.events_today == 0


def test_a_failure_never_raises_out_of_the_tick():
    w = _worker(lambda now: _reading(now))
    w._store = SimpleNamespace(
        fetch_signal_event_times_read_both=lambda **kw: (_ for _ in ()).throw(RuntimeError("db")))
    w._strain_attention_tick()
    assert w.writes == []


def test_strain_never_touches_the_prediction_error_slot():
    """PE retire Patch 2: no PE node write and no PE-table row, the event row is a signal row."""
    now = datetime.now(timezone.utc)
    w = _worker(lambda now: _reading(now))
    w._strain_attention_tick()
    assert w.pe_writes == [] and w.pe_recorded == []
    assert [n for n, _, _ in w.recorded] == [STRAIN_NODE_ID]
