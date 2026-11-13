"""Body economy Patch C3b: a training run's SHADOW bid at run start (app/services/physiology_shadow.py).

The contract: the run (its act events, its status writes) is identical whatever the economy answers
(funded, deferred, no answer, an error); the logged decision carries the run act's correlation id
and the real outcome (ran); the run never waits on the economy.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import pytest

from app.services import act_emit, physiology_shadow as ps, training
from orion.schemas.physiology_want import WantDecisionV1, WantV1
from test_training_run_act import _run, harness  # noqa: F401 -- harness is a fixture

ANSWERS = ["funded", "deferred", "none", "raises", "off"]


def _decision(want: WantV1, outcome: str) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome,
        reason="net_positive" if outcome == "funded" else "price", lane="fast", want=want,
    )


@pytest.fixture
def economy(monkeypatch):
    class _Economy:
        answer = "funded"
        gate = None

    eco = _Economy()
    eco.wants, eco.released, eco.shadows = [], [], []

    def fake_bid(want):
        eco.wants.append(want)
        if eco.gate is not None:
            eco.gate.wait(5.0)
        if eco.answer == "raises":
            raise RuntimeError("economy down")
        return None if eco.answer == "none" else _decision(want, eco.answer)

    real_cls = ps.ShadowBid

    class _Tracked(real_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            eco.shadows.append(self)

    monkeypatch.setattr(ps, "bid", fake_bid)
    monkeypatch.setattr(ps, "release", lambda want_id: eco.released.append(want_id))
    monkeypatch.setattr(ps, "ShadowBid", _Tracked)
    monkeypatch.setattr(ps, "_enabled", lambda: True)
    return eco


def _lines(caplog) -> list[dict]:
    out = []
    for r in caplog.records:
        msg = r.getMessage()
        if msg.startswith("physiology_shadow_bid "):
            out.append(dict(part.split("=", 1) for part in msg.split()[1:]))
    return out


def _wait(eco) -> None:
    for s in eco.shadows:
        s.join(5.0)


def _strip(event) -> tuple:
    outcome = event.act.outcome
    return (event.event_kind, event.act.act_id, event.correlation_id,
            None if outcome is None else (outcome.value, outcome.note))


def test_a_run_is_identical_for_every_answer_and_joins_its_act(harness, economy, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="topic-foundry.physiology_shadow")
    from uuid import uuid4

    run_id = uuid4()
    seen = []
    for answer in ANSWERS:
        economy.answer = answer
        economy.wants.clear()
        economy.shadows.clear()
        harness.events.clear()
        caplog.clear()
        monkeypatch.setattr(ps, "_enabled", lambda: answer != "off")
        statuses = []
        monkeypatch.setattr(training, "update_run", lambda run: statuses.append(run.status))
        _run(run_id)
        _wait(economy)
        assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
        if answer == "off":
            assert economy.wants == [] and _lines(caplog) == []
        else:
            (want,) = economy.wants
            # The join: want_id == the run act's correlation id.
            assert want.want_id == want.correlation_id == harness.events[0].correlation_id \
                == act_emit.run_correlation_id(run_id)
            assert (want.family, want.verb, want.tier) == ("topic_foundry.run", "maintain", 2)
            (line,) = _lines(caplog)
            assert (line["want_id"], line["real"]) == (want.want_id, "ran")
            assert economy.released == []
        seen.append(([_strip(e) for e in harness.events], statuses))
    assert all(s == seen[0] for s in seen)


def test_a_slow_economy_never_holds_the_run(harness, economy, caplog):
    caplog.set_level(logging.INFO, logger="topic-foundry.physiology_shadow")
    economy.gate = threading.Event()
    t0 = time.monotonic()
    _run()
    assert time.monotonic() - t0 < 2.0
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    assert _lines(caplog) == []                # the run finished; the economy has not answered
    economy.gate.set()
    _wait(economy)
    assert _lines(caplog)[0]["real"] == "ran"


def test_a_shadow_that_cannot_be_built_never_fails_the_run(harness, economy, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad deploy")

    monkeypatch.setattr(ps, "ShadowBid", boom)
    _run()
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    assert harness.events[1].act.outcome.value == 1.0


def test_helper_blocked_funded_releases_off_the_caller_thread(monkeypatch):
    """The helper's release rule, used by a producer whose real gate blocks. The release runs on
    its own thread, so settle() returns at once."""
    released = threading.Event()
    got = []

    def slow_release(want_id):
        time.sleep(0.3)
        got.append(want_id)
        released.set()

    monkeypatch.setattr(ps, "bid", lambda want: _decision(want, "funded"))
    monkeypatch.setattr(ps, "release", slow_release)
    s = ps.ShadowBid(family="topic_foundry.run", verb="maintain", tier=2, want_id="w-1", enabled=True)
    s.place()
    s.join(5.0)
    t0 = time.monotonic()
    s.settle("blocked:test")
    assert time.monotonic() - t0 < 0.2
    assert released.wait(5.0) and got == ["w-1"]
