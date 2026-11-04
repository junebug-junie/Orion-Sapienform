"""Settle-time world scoring pass (attend-to-act loop), hosted by sql-writer since the feedback rung
retired (2026-11-04), with a fake store."""
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


def _ep(arm="treated"):
    return {"episode_id": "d1", "arm": arm, "decided_at": T0, "dispatch_kind": "self_regulate",
            "target_id": "pool:background_gpu", "dispatch_frame_id": "f", "open_loop_id": "open-loop-cab",
            "expected_effect": {"direction": "decrease", "predicted_delta": 0.0}, "settlement_state": "expired",
            "settlement": {"ttl_sec": 900, "manipulation_check": {"started_at": T0.isoformat()}}}


def _score(store):
    from orion.autonomy.world_settlement_store import score_due_world_episodes

    return score_due_world_episodes(store, now=NOW)


def test_nothing_due_scores_nothing():
    store = _Store([])
    assert _score(store) == 0 and store.saved == []


def test_an_unreadable_input_skips_the_episode_never_scores_it_as_calm():
    store = _Store([_ep()])

    def boom(since, until):
        raise RuntimeError("incidents table unreadable")
    store.load_hardware_incidents_opened = boom
    assert _score(store) == 0 and store.saved == []


def test_scores_and_stamps_the_next_tick_salience_on_the_acted_verdict():
    store = _Store([_ep()])
    assert _score(store) == 1
    (_, score), = store.saved
    assert score.record.arm == "dispatched" and score.loop_outcome["salience_at_close"] == 0.31
    assert score.loop_outcome["features_at_close"]["next_tick_salience"] == 0.31


def test_save_refuses_anything_but_orions_acted_verdict():
    from orion.autonomy.world_settlement_store import WorldSettlementStore

    store = object.__new__(WorldSettlementStore)

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


def test_the_loop_is_on_by_default_and_wired_into_startup():
    from pathlib import Path

    from app.settings import Settings

    assert Settings.model_fields["sql_writer_world_settlement_interval_sec"].default > 0
    import ast

    main = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    tree = ast.parse(main)  # a text match alone passed once while main.py did not even compile
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "world_settlement_loop"]
    assert calls, "world_settlement_loop(...) is never started in app/main.py"


def test_the_loop_says_it_is_alive_even_when_nothing_is_due(monkeypatch, caplog):
    """Most passes score nothing; an hourly 'alive' line tells an idle loop from a dead one."""
    import asyncio
    import logging

    from app import world_settlement_loop as loop

    calls = {"n": 0}

    def one_pass():
        calls["n"] += 1
        return 0

    async def stop_after_two(_sec):
        if calls["n"] >= 2:
            raise asyncio.CancelledError
    monkeypatch.setattr(loop, "run_one_pass", one_pass)
    monkeypatch.setattr(loop, "ALIVE_EVERY_SEC", 0.0)
    monkeypatch.setattr(loop.asyncio, "sleep", stop_after_two)
    caplog.set_level(logging.INFO, logger="sql-writer.world_settlement_loop")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.world_settlement_loop(SimpleNamespace(sql_writer_world_settlement_interval_sec=30.0)))
    assert any("[WORLD_SETTLEMENT] alive passes=1 scored=0 failed=0" in r.message for r in caplog.records)
