"""Feedback runtime's settle-time world scoring pass (attend-to-act loop), with a fake store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.hardware_watch.rules import TempPoint

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(minutes=21)


class _Store:
    def __init__(self, episodes):
        self.episodes, self.saved = episodes, []

    def load_world_episodes_due(self, now):
        return self.episodes

    def load_effect_posteriors(self):
        return {}

    def load_control_posteriors(self):
        return {}

    def load_cabinet_points(self, since, until):
        return [TempPoint(T0 + timedelta(seconds=30 * i), 30.0 - 0.01 * i) for i in range(45)]

    def load_hardware_incidents_opened(self, since, until):
        return []

    def load_loop_salience(self, open_loop_id, since):
        return 0.31

    def save_world_score(self, episode_id, score, scored_at):
        self.saved.append((episode_id, score))
        return True


def _worker(store, enabled=True):
    from app.worker import FeedbackRuntimeWorker

    w = object.__new__(FeedbackRuntimeWorker)
    w._settings = SimpleNamespace(world_settlement_scoring_enabled=enabled, world_settlement_interval_sec=30.0)
    w._store = store
    return w


def _ep(arm="treated"):
    return {"episode_id": "d1", "arm": arm, "decided_at": T0, "dispatch_kind": "self_regulate",
            "target_id": "pool:background_gpu", "dispatch_frame_id": "f", "open_loop_id": "open-loop-cab",
            "expected_effect": {"direction": "decrease", "predicted_delta": 0.0}, "settlement_state": "expired",
            "settlement": {"ttl_sec": 900, "manipulation_check": {"started_at": T0.isoformat()}}}


def test_off_by_default_does_nothing():
    store = _Store([_ep()])
    assert _worker(store, enabled=False)._score_world_episodes(now=NOW) == 0 and store.saved == []


def test_scores_and_stamps_the_next_tick_salience_on_the_acted_verdict():
    store = _Store([_ep()])
    assert _worker(store)._score_world_episodes(now=NOW) == 1
    (_, score), = store.saved
    assert score.record.arm == "dispatched" and score.loop_outcome["salience_at_close"] == 0.31
    assert score.loop_outcome["features_at_close"]["next_tick_salience"] == 0.31


def test_save_refuses_anything_but_orions_acted_verdict():
    from app.store import FeedbackRuntimeStore

    store = object.__new__(FeedbackRuntimeStore)

    class _Conn:
        def execute(self, *a, **k):
            raise AssertionError("must refuse before writing")

    class _Engine:
        def begin(self):
            class _Ctx:
                def __enter__(self_inner):
                    return _Conn()

                def __exit__(self_inner, *a):
                    return False
            return _Ctx()

    store._engine = _Engine()
    bad = SimpleNamespace(outcome={}, record=None, control_cell=None,
                          loop_outcome={"loop_id": "l", "verdict": "resolved", "actor": "orion"})
    with pytest.raises(ValueError):
        store.save_world_score(episode_id="d1", score=bad, scored_at=NOW)
