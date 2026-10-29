"""Body economy Patch C3b: the dream cycle's SHADOW bid (app/physiology_shadow.py).

The contract: a check that would sleep bids; the cycle (its result, its act events, what it
persists) is identical whatever the economy answers (funded, deferred, no answer, an error); the
logged decision carries the sleep act's correlation id and the real outcome (ran, held, empty).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from orion.schemas.physiology_want import WantDecisionV1, WantV1
from test_dream_cycle_v2 import _Fakes, _rows
from test_dream_reads_arousal import _deps, _state

ANSWERS = ["funded", "deferred", "none", "raises"]


def _decision(want: WantV1, outcome: str) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome,
        reason="net_positive" if outcome == "funded" else "price", lane="metacog", want=want,
    )


@pytest.fixture
def economy(monkeypatch):
    from app import physiology_shadow as ps

    class _Economy:
        answer = "funded"
        gate = None

    eco = _Economy()
    eco.wants, eco.released, eco.ps = [], [], ps

    async def fake_abid(want):
        eco.wants.append(want)
        if eco.gate is not None:
            await eco.gate.wait()
        if eco.answer == "raises":
            raise RuntimeError("economy down")
        return None if eco.answer == "none" else _decision(want, eco.answer)

    monkeypatch.setattr(ps, "abid", fake_abid)
    monkeypatch.setattr(ps, "release", lambda want_id: eco.released.append(want_id))
    monkeypatch.setattr(ps, "_enabled", lambda: True)
    return eco


def _lines(caplog) -> list[dict]:
    out = []
    for r in caplog.records:
        msg = r.getMessage()
        if msg.startswith("physiology_shadow_bid "):
            out.append(dict(part.split("=", 1) for part in msg.split()[1:]))
    return out


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


def _run(deps, **kw):
    from app.cycle import run_cycle_once

    async def go():
        out = await run_cycle_once(deps, **kw)
        await _drain()
        return out

    return asyncio.run(go())


def _cycle_view(cycle) -> dict:
    """A cycle minus its random id, its wall clocks, and the hypotheses' content (control pairs
    are seeded by the random cycle id, so their text varies run to run with or without a bid)."""
    view = cycle.model_dump(exclude={"cycle_id", "started_at", "ended_at", "hypotheses"})
    view["pressure"] = {k: v for k, v in view["pressure"].items() if k not in ("since", "computed_at")}
    return view | {"n_hypotheses": len(cycle.hypotheses)}


def _act_view(event) -> tuple:
    outcome = event.act.outcome
    return (event.event_kind, event.act.actor, event.act.verb,
            None if outcome is None else (outcome.value, outcome.note))


def test_a_sleep_is_identical_for_every_answer_and_joins_its_act(economy, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="orion-dream.physiology_shadow")
    seen = []
    for answer in ANSWERS + ["off"]:
        if answer == "off":
            monkeypatch.setattr(economy.ps, "_enabled", lambda: False)
        economy.answer = answer
        economy.wants.clear()
        caplog.clear()
        f = _Fakes(_rows())
        deps, _ = _deps(f, _state("idle"))
        events = []

        async def publish(event):
            events.append(event)

        deps.publish_act = publish
        cycle = _run(deps)
        assert cycle is not None and cycle.status == "completed"
        assert [e.event_kind for e in events] == ["act_opened", "act_closed"]
        if answer == "off":
            assert economy.wants == [] and _lines(caplog) == []
        else:
            (want,) = economy.wants
            # The join: want_id == the emitted act_opened event's correlation id.
            assert want.want_id == want.correlation_id == events[0].correlation_id
            assert (want.family, want.verb, want.tier) == ("dream.cycle", "make", 1)
            (line,) = _lines(caplog)
            assert (line["want_id"], line["real"]) == (events[0].correlation_id, "ran")
        seen.append((_cycle_view(cycle), len(f.persisted), len(f.prompts), [_act_view(e) for e in events]))
    for s in seen[1:]:
        diff = {k: (seen[0][0].get(k), s[0].get(k)) for k in s[0] if s[0].get(k) != seen[0][0].get(k)}
        assert s == seen[0], (diff, s[1:], seen[0][1:])


@pytest.mark.parametrize("answer", ANSWERS)
def test_an_arousal_hold_is_unchanged_and_logged_blocked(economy, caplog, answer):
    caplog.set_level(logging.INFO, logger="orion-dream.physiology_shadow")
    economy.answer = answer
    f = _Fakes(_rows())
    deps, calls = _deps(f, _state("strained"))
    assert _run(deps) is None
    assert calls == [1] and f.persisted == [] and f.prompts == []
    assert deps.hold_reason == "arousal_strained"
    (line,) = _lines(caplog)
    assert (line["family"], line["real"]) == ("dream.cycle", "blocked:arousal_strained")
    assert economy.released == ([economy.wants[0].want_id] if answer == "funded" else [])


def test_a_check_that_is_not_due_never_bids(economy):
    f = _Fakes({})                       # nothing new: no pressure, no candidates
    deps, _ = _deps(f, _state("idle"))
    assert _run(deps) is None
    assert economy.wants == []


def test_an_empty_forced_sleep_is_logged_nothing_to_replay(economy, caplog):
    caplog.set_level(logging.INFO, logger="orion-dream.physiology_shadow")
    f = _Fakes({})
    deps, _ = _deps(f, _state("idle"))
    cycle = _run(deps, force=True)
    assert cycle is not None and cycle.status == "empty"
    (line,) = _lines(caplog)
    assert line["real"] == "blocked:nothing_to_replay"


def test_a_slow_economy_never_holds_the_sleep(economy, caplog):
    from app.cycle import run_cycle_once

    caplog.set_level(logging.INFO, logger="orion-dream.physiology_shadow")
    f = _Fakes(_rows())
    deps, _ = _deps(f, _state("idle"))

    async def go():
        economy.gate = asyncio.Event()
        cycle = await asyncio.wait_for(run_cycle_once(deps), timeout=5.0)
        assert cycle.status == "completed" and _lines(caplog) == []
        economy.gate.set()
        await _drain()
        return cycle

    asyncio.run(go())
    assert _lines(caplog)[0]["real"] == "ran"


def test_a_shadow_that_cannot_be_built_never_stops_a_sleep(economy, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad deploy")

    monkeypatch.setattr(economy.ps, "ShadowBid", boom)
    f = _Fakes(_rows())
    deps, _ = _deps(f, _state("idle"))
    cycle = _run(deps)
    assert cycle is not None and cycle.status == "completed" and economy.wants == []
