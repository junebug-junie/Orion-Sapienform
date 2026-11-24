"""PR #3781 Patch 1 + Patch 2: the history sampler writes the five non-PE nodes into
substrate_node_signal_history. Patch 2: the three alarms carry signal_value and go to the signal
table ONLY (never the PE table, whatever the flag says); the two cameras are still dual-written
(PE row + signal row, flag-gated). A failing signal write never costs the PE write or the tick."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_worker_pe_history import OBSERVED, _make_worker, _pe_node, _run_tick  # noqa: E402

FIVE = (
    ("node:substrate.cabinet", "alarm"),
    ("node:substrate.strain", "alarm"),
    ("node:substrate.street", "alarm"),
    ("node:substrate.perception", "scene_change"),
    ("node:substrate.perception_walkway", "scene_change"),
)


# A live PE node beside the five (was node:substrate.execution until its 2026-11-24 retirement, #3576).
ALARMS = {n for n, k in FIVE if k == "alarm"}
CAMERAS = {n for n, k in FIVE if k == "scene_change"}


def _node(node_id, value, observed_at):
    """An alarm carries signal_value (PE retire Patch 2); every other node prediction_error."""
    node = _pe_node(node_id, value, observed_at)
    if node_id in ALARMS:
        node.metadata["signal_value"] = node.metadata.pop("prediction_error")
        node.metadata["signal_kind"] = "alarm"
    return node


def _nodes():
    nodes = [_node(node_id, 0.1 * i, OBSERVED) for i, (node_id, _) in enumerate(FIVE)]
    nodes.append(_pe_node("node:prediction.body", 0.3, OBSERVED))
    return nodes


def test_alarms_signal_only_cameras_dual_written(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, _nodes())
    pe_rows = worker._store.save_prediction_error_history_samples.call_args.args[0]
    # The PE table gets the cameras and the live PE node, never an alarm (Patch 2).
    assert {r[0] for r in pe_rows} == CAMERAS | {"node:prediction.body"}
    (sig_rows,) = worker._store.save_signal_history_samples.call_args.args
    assert sorted((n, k) for n, _, _, k in sig_rows) == sorted(FIVE)
    pe_by_id = {n: (t, v) for n, t, v in pe_rows}
    for n, t, v, _ in sig_rows:
        if n in CAMERAS:
            assert pe_by_id[n] == (t, v)
    # The alarms' readings still land in the in-memory window (they are ranked as before).
    assert set(worker._pe_history_cache) >= ALARMS
    worker._store.prune_signal_history.assert_called_once()


def test_stale_reading_is_not_rewritten_to_signal_table(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    node = _node("node:substrate.cabinet", 0.0, OBSERVED)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.save_signal_history_samples.call_count == 1
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, OBSERVED + timedelta(seconds=30))])
    assert worker._store.save_signal_history_samples.call_count == 2


def test_non_signal_nodes_never_reach_the_signal_table(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, [_pe_node("node:prediction.body", 0.3, OBSERVED)])
    worker._store.save_signal_history_samples.assert_not_called()


def test_flag_off_stops_only_the_camera_dual_write(monkeypatch):
    """Patch 2: the alarms live only in the signal table, so their rows and its prune run even
    with the flag off; the flag still stops the cameras' dual-write."""
    monkeypatch.setenv("SUBSTRATE_SIGNAL_HISTORY_ENABLED", "false")
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, _nodes())
    worker._store.save_prediction_error_history_samples.assert_called_once()
    (sig_rows,) = worker._store.save_signal_history_samples.call_args.args
    assert {r[0] for r in sig_rows} == ALARMS
    worker._store.prune_signal_history.assert_called_once()


def test_a_stale_prediction_error_key_on_an_alarm_is_never_sampled(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, [_pe_node("node:substrate.strain", 1.0, OBSERVED)])  # old key only
    worker._store.save_signal_history_samples.assert_not_called()
    assert "node:substrate.strain" not in (worker._pe_history_cache or {})


def test_a_failed_alarm_signal_write_is_not_added_to_the_window(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = RuntimeError("no table")
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, OBSERVED)])
    assert not worker._pe_history_cache.get("node:substrate.cabinet")
    assert worker._signal_history_pending  # kept for retry


def test_signal_write_failure_fails_open(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = RuntimeError("relation does not exist")
    worker._store.prune_signal_history.side_effect = RuntimeError("relation does not exist")
    projection = _run_tick(worker, _nodes())
    worker._store.save_prediction_error_history_samples.assert_called_once()
    worker._store.prune_prediction_error_history.assert_called_once()
    assert projection.frame.open_loops  # the broadcast still went out


def test_flag_ships_on_by_default(monkeypatch):
    monkeypatch.delenv("SUBSTRATE_SIGNAL_HISTORY_ENABLED", raising=False)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app.settings import Settings

    assert Settings().signal_history_enabled is True
    example = (SUBSTRATE_ROOT / ".env_example").read_text()
    assert "\nSUBSTRATE_SIGNAL_HISTORY_ENABLED=true\n" in example


def test_strain_event_row_is_a_signal_row_only(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))
    w._settings.signal_history_enabled = False  # the flag does not gate an alarm row
    w.signal = []
    w._store.save_signal_history_samples = lambda rows: w.signal.extend(rows)
    w._strain_attention_tick()
    assert w.pe_recorded == []
    assert [(n, v, k) for n, _, v, k in w.signal] == [("node:substrate.strain", 1.0, "alarm")]


def test_strain_event_signal_failure_keeps_the_event(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))
    w._settings.signal_history_enabled = True

    def boom(rows):
        raise RuntimeError("no table")

    w._store.save_signal_history_samples = boom
    w._strain_attention_tick()
    assert [x["value"] for x in w.writes] == [1.0]
    assert [r[0] for r in w._signal_history_pending] == ["node:substrate.strain"]


def test_failed_signal_write_is_retried_with_the_next_write(monkeypatch):
    """The PE sampler has moved past a reading once its PE row is saved, so a failed signal
    write is kept and sent again with the next one (review finding, PR #3781 Patch 1)."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = [RuntimeError("no table"), None]
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, OBSERVED)])
    later = OBSERVED + timedelta(seconds=30)
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, later)])
    retried = worker._store.save_signal_history_samples.call_args.args[0]
    assert [t for _, t, _, _ in retried] == [OBSERVED, later]
    assert worker._signal_history_pending == []


def test_pending_retry_buffer_is_capped(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = RuntimeError("no table")
    cap = worker._SIGNAL_HISTORY_PENDING_MAX
    worker._signal_history_pending = [("node:substrate.cabinet", OBSERVED, 0.0, "alarm")] * cap
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, OBSERVED + timedelta(seconds=30))])
    assert len(worker._signal_history_pending) == cap
    assert worker._signal_history_pending[-1][1] == OBSERVED + timedelta(seconds=30)
