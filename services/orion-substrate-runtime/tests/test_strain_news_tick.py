"""node:substrate.strain's prediction_error: strain news through the existing PE node path
(strain loop Part 2, orion/regulation/strain_news.py)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from orion.regulation.arousal_view import ArousalView
from orion.regulation.strain_news import STRAIN_NODE_ID
from orion.schemas.regulation import ArousalReadingV1


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
        fetch_prediction_error_event_times=lambda node_id, since: list(event_times),
        count_prediction_error_history=lambda node_id, since: history_rows,
        save_prediction_error_history_samples=lambda samples: w.recorded.extend(samples),
    )
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw)

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
    assert write["node_id"] == STRAIN_NODE_ID and write["error"] == 1.0
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
    assert w.writes[-1]["error"] == 0.0 and len(w.recorded) == 1


def test_idle_writes_calm_zero_only():
    w = _worker(lambda now: _reading(now, level="idle"))
    w._strain_attention_tick()
    assert [x["error"] for x in w.writes] == [0.0] and w.recorded == []


def test_restart_with_a_recorded_event_does_not_refire_onset():
    since = datetime.now(timezone.utc) - timedelta(minutes=10)
    w = _worker(lambda now: _reading(now, since=since), event_times=[since + timedelta(seconds=30)])
    w._strain_attention_tick()
    assert w.recorded == []
    assert all(x["error"] == 0.0 for x in w.writes)


def test_warming_history_writes_rest_and_spends_no_event():
    w = _worker(lambda now: _reading(now), history_rows=10)
    w._strain_attention_tick()
    assert [x["error"] for x in w.writes] == [0.0] and w.recorded == []
    assert w._strain_news_state.events_today == 0


def test_a_failure_never_raises_out_of_the_tick():
    w = _worker(lambda now: _reading(now))
    w._store = SimpleNamespace(
        fetch_prediction_error_event_times=lambda **kw: (_ for _ in ()).throw(RuntimeError("db")))
    w._strain_attention_tick()
    assert w.writes == []
