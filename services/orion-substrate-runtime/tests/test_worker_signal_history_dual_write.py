"""PR #3781 Patch 1-3: the history sampler writes the five non-PE nodes into
substrate_node_signal_history. Patch 2: the three alarms carry signal_value and go to the signal
table ONLY. Patch 3: the two cameras too (camera scene change, each row with its scale); the
Patch 1 dual-write flag is gone. A failing signal write never costs the PE write or the tick."""
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


def _node(node_id, value, observed_at, scale="v2"):
    """An alarm (Patch 2) and a camera (Patch 3) carry signal_value; every other node
    prediction_error. A camera also carries its signal_scale."""
    node = _pe_node(node_id, value, observed_at)
    if node_id in ALARMS or node_id in CAMERAS:
        node.metadata["signal_value"] = node.metadata.pop("prediction_error")
        node.metadata["signal_kind"] = "alarm" if node_id in ALARMS else "scene_change"
    if node_id in CAMERAS and scale is not None:
        node.metadata["signal_scale"] = scale
    return node


def _nodes():
    nodes = [_node(node_id, 0.1 * i, OBSERVED) for i, (node_id, _) in enumerate(FIVE)]
    nodes.append(_pe_node("node:prediction.body", 0.3, OBSERVED))
    return nodes


def test_all_five_signal_only_with_their_scale(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, _nodes())
    pe_rows = worker._store.save_prediction_error_history_samples.call_args.args[0]
    # The PE table gets only the live forecast node: never an alarm (Patch 2) or a camera (Patch 3).
    assert {r[0] for r in pe_rows} == {"node:prediction.body"}
    (sig_rows,) = worker._store.save_signal_history_samples.call_args.args
    assert sorted((n, k) for n, _, _, k, _ in sig_rows) == sorted(FIVE)
    for n, _, _, _, scale in sig_rows:
        assert scale == ("v2" if n in CAMERAS else "v1")
    # Alarm and current-scale camera readings land in the in-memory window.
    assert set(worker._pe_history_cache) >= ALARMS | CAMERAS
    worker._store.prune_signal_history.assert_called_once()


def test_an_old_scale_camera_row_is_stored_but_never_ranked(monkeypatch):
    """legacy_z rollback rows are scale v1: written, but kept out of the window and magnitudes
    (readers read current-scale camera rows only)."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, [_node("node:substrate.perception", 0.4, OBSERVED, scale="v1"),
                       _node("node:substrate.perception_walkway", 0.0, OBSERVED, scale=None)])
    (sig_rows,) = worker._store.save_signal_history_samples.call_args.args
    assert {(r[0], r[4]) for r in sig_rows} == {
        ("node:substrate.perception", "v1"), ("node:substrate.perception_walkway", "v1")
    }
    assert not (worker._pe_history_cache or {}).get("node:substrate.perception")
    assert not (worker._pe_history_cache or {}).get("node:substrate.perception_walkway")
    # Not re-sampled on the next tick either.
    _run_tick(worker, [_node("node:substrate.perception", 0.4, OBSERVED, scale="v1")])
    assert worker._store.save_signal_history_samples.call_count == 1


def test_a_stale_prediction_error_key_on_a_camera_is_never_sampled(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    _run_tick(worker, [_pe_node("node:substrate.perception", 0.8, OBSERVED)])  # old key only
    worker._store.save_signal_history_samples.assert_not_called()
    worker._store.save_prediction_error_history_samples.assert_not_called()


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


def test_dual_write_flag_is_gone(monkeypatch):
    """Patch 3: every signal node lives only in the signal table, so the Patch 1 flag is removed
    from settings, .env_example and compose (kill means kill)."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app.settings import Settings

    assert not hasattr(Settings(), "signal_history_enabled")
    for name in (".env_example", "docker-compose.yml"):
        assert "SUBSTRATE_SIGNAL_HISTORY_ENABLED=" not in (SUBSTRATE_ROOT / name).read_text()


def test_a_failed_alarm_row_is_pending_once_and_enters_the_window_when_it_lands(monkeypatch):
    """Review finding: the re-sampled row must not fill the retry cap with copies, and a retried
    alarm row (its only home is the signal table) must reach the window once it lands."""
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = [RuntimeError("down"), RuntimeError("down"), None]
    node = _node("node:substrate.cabinet", 0.0, OBSERVED)
    _run_tick(worker, [node])
    _run_tick(worker, [node])
    assert len(worker._signal_history_pending) == 1
    later = OBSERVED + timedelta(seconds=30)
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, later)])
    assert worker._signal_history_pending == []
    assert [t for t, _ in worker._pe_history_cache["node:substrate.cabinet"]] == [OBSERVED, later]


def test_strain_event_row_is_a_signal_row_only(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))
    w.signal = []
    w._store.save_signal_history_samples = lambda rows: w.signal.extend(rows)
    w._strain_attention_tick()
    assert w.pe_recorded == []
    assert [(n, v, k, sc) for n, _, v, k, sc in w.signal] == [("node:substrate.strain", 1.0, "alarm", "v1")]


def test_strain_event_signal_failure_keeps_the_event(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_WRITE_PREDICTION_ERROR_NODES", "true")
    from test_strain_news_tick import _reading, _worker

    w = _worker(lambda now: _reading(now))

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
    assert [r[1] for r in retried] == [OBSERVED, later]
    assert worker._signal_history_pending == []


def test_pending_retry_buffer_is_capped(monkeypatch):
    worker = _make_worker(monkeypatch, history_enabled=True)
    worker._store.save_signal_history_samples.side_effect = RuntimeError("no table")
    cap = worker._SIGNAL_HISTORY_PENDING_MAX
    worker._signal_history_pending = [
        ("node:substrate.cabinet", OBSERVED - timedelta(seconds=30 * (i + 1)), 0.0, "alarm") for i in range(cap)
    ]
    _run_tick(worker, [_node("node:substrate.cabinet", 0.0, OBSERVED + timedelta(seconds=30))])
    assert len(worker._signal_history_pending) == cap
    assert worker._signal_history_pending[-1][1] == OBSERVED + timedelta(seconds=30)
