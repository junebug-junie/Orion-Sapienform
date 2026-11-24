"""PR #3781 Patch 1: the PE-history sampler and the strain event row DUAL-WRITE the five non-PE
nodes into substrate_node_signal_history. The PE-table writes do not change; flag off writes
nothing new; a failing signal write never costs the PE write or the tick."""
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


def _nodes():
    nodes = [_pe_node(node_id, 0.1 * i, OBSERVED) for i, (node_id, _) in enumerate(FIVE)]
    nodes.append(_pe_node("node:substrate.execution", 0.3, OBSERVED))
    return nodes


def test_five_nodes_dual_written_with_kind_pe_rows_unchanged(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, _nodes())
    pe_rows = worker._store.save_prediction_error_history_samples.call_args.args[0]
    # The PE table still gets all six readings, unchanged.
    assert {r[0] for r in pe_rows} == {n for n, _ in FIVE} | {"node:substrate.execution"}
    (sig_rows,) = worker._store.save_signal_history_samples.call_args.args
    assert sorted((n, k) for n, _, _, k in sig_rows) == sorted(FIVE)
    pe_by_id = {n: (t, v) for n, t, v in pe_rows}
    for n, t, v, _ in sig_rows:
        assert pe_by_id[n] == (t, v)
    worker._store.prune_signal_history.assert_called_once()


def test_stale_reading_is_not_rewritten_to_signal_table(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    node = _pe_node("node:substrate.cabinet", 0.0, OBSERVED)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert worker._store.save_signal_history_samples.call_count == 1
    _run_tick(worker, [_pe_node("node:substrate.cabinet", 0.0, OBSERVED + timedelta(seconds=30))])
    assert worker._store.save_signal_history_samples.call_count == 2


def test_non_signal_nodes_never_reach_the_signal_table(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, [_pe_node("node:substrate.execution", 0.3, OBSERVED)])
    worker._store.save_signal_history_samples.assert_not_called()


def test_flag_off_no_signal_write_no_prune(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_SIGNAL_HISTORY_ENABLED", "false")
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, _nodes())
    worker._store.save_prediction_error_history_samples.assert_called_once()
    worker._store.save_signal_history_samples.assert_not_called()
    worker._store.prune_signal_history.assert_not_called()


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


def test_strain_event_row_is_dual_written(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))
    w._settings.signal_history_enabled = True
    w.signal = []
    w._store.save_signal_history_samples = lambda rows: w.signal.extend(rows)
    w._strain_attention_tick()
    assert [(n, v) for n, _, v in w.recorded] == [("node:substrate.strain", 1.0)]
    assert [(n, v, k) for n, _, v, k in w.signal] == [("node:substrate.strain", 1.0, "alarm")]
    assert w.signal[0][1] == w.recorded[0][1]


def test_strain_event_signal_failure_keeps_the_event(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))
    w._settings.signal_history_enabled = True

    def boom(rows):
        raise RuntimeError("no table")

    w._store.save_signal_history_samples = boom
    w._strain_attention_tick()
    assert [x["error"] for x in w.writes] == [1.0] and len(w.recorded) == 1


def test_failed_signal_write_is_retried_with_the_next_write(monkeypatch):
    """The PE sampler has moved past a reading once its PE row is saved, so a failed signal
    write is kept and sent again with the next one (review finding, PR #3781 Patch 1)."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = [RuntimeError("no table"), None]
    _run_tick(worker, [_pe_node("node:substrate.cabinet", 0.0, OBSERVED)])
    later = OBSERVED + timedelta(seconds=30)
    _run_tick(worker, [_pe_node("node:substrate.cabinet", 0.0, later)])
    retried = worker._store.save_signal_history_samples.call_args.args[0]
    assert [t for _, t, _, _ in retried] == [OBSERVED, later]
    assert worker._signal_history_pending == []


def test_pending_retry_buffer_is_capped(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = RuntimeError("no table")
    cap = worker._SIGNAL_HISTORY_PENDING_MAX
    worker._signal_history_pending = [("node:substrate.cabinet", OBSERVED, 0.0, "alarm")] * cap
    _run_tick(worker, [_pe_node("node:substrate.cabinet", 0.0, OBSERVED + timedelta(seconds=30))])
    assert len(worker._signal_history_pending) == cap
    assert worker._signal_history_pending[-1][1] == OBSERVED + timedelta(seconds=30)
