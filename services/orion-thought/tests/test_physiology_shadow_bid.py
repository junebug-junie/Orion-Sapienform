"""Body economy Patch C3b: orion-thought's SHADOW bid at the painting generate step.

The contract: the step's result, its GPU work and its act events are identical whatever the
economy answers (funded, deferred, no answer, an error), and the logged shadow decision carries
the generate step's correlation id (the act's correlation id when this step opens it) and the
step's real outcome.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from orion.schemas.physiology_want import WantDecisionV1, WantV1
from test_visual_steps import _dream_req, _GrammarBus, _req, _run_on, env  # noqa: F401 -- env is a fixture

ANSWERS = ["funded", "deferred", "none", "raises"]


def _decision(want: WantV1, outcome: str) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome,
        reason="net_positive" if outcome == "funded" else "price", lane="diffusion", want=want,
    )


@pytest.fixture
def economy(monkeypatch):
    from app import physiology_shadow as ps

    class _Economy:
        answer = "funded"

    eco = _Economy()
    eco.wants, eco.released, eco.ps = [], [], ps

    async def fake_abid(want):
        eco.wants.append(want)
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


def _stable(result) -> dict:
    """The step result minus wall time and the attempt id (a fresh claim per round)."""
    return result.model_dump(exclude={"elapsed_sec", "attempt_id"})


@pytest.mark.asyncio
async def test_generate_is_identical_for_every_answer_and_joins_the_act(env, economy, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="orion-thought.physiology_shadow")
    monkeypatch.setattr(env.steps.settings, "thought_act_ledger_enabled", True, raising=False)
    seen = []
    for answer in ANSWERS:
        economy.answer = answer
        economy.wants.clear()
        caplog.clear()
        env.store.rows.clear()
        env.generate_calls.clear()
        bus = _GrammarBus()
        prepared = await _run_on(env, bus, _req("prepare"))
        assert economy.wants == []          # prepare never bids: a plan is not GPU work
        out = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id, correlation_id="gen-corr"))
        await _drain()
        assert out.status == "done" and len(env.generate_calls) == 1
        opened = [e.payload for ch, e in bus.sent if ch == "orion:grammar:event"]
        assert [p["event_kind"] for p in opened] == ["act_opened"]
        (want,) = economy.wants
        # The join: want_id == the act_opened event's correlation id.
        assert want.want_id == want.correlation_id == opened[0]["correlation_id"] == "gen-corr"
        assert (want.family, want.verb, want.tier) == ("thought.painting", "make", 2)
        (line,) = _lines(caplog)
        assert (line["want_id"], line["real"]) == ("gen-corr", "ran")
        assert economy.released == []       # it ran: the reservation stands until the act closes
        events = [{k: v for k, v in p.items() if k not in ("emitted_at", "observed_at", "event_id")}
                  for p in opened]
        seen.append((_stable(out), events, env.generate_calls[:]))
    # Baseline: the same step with the shadow switched off answers and publishes the same.
    monkeypatch.setattr(economy.ps, "_enabled", lambda: False)
    economy.wants.clear()
    env.store.rows.clear()
    env.generate_calls.clear()
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    out = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id, correlation_id="gen-corr"))
    assert economy.wants == []
    events = [{k: v for k, v in e.payload.items() if k not in ("emitted_at", "observed_at", "event_id")}
              for ch, e in bus.sent if ch == "orion:grammar:event"]
    seen.append((_stable(out), events, env.generate_calls[:]))
    assert all(x == seen[0] for x in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ANSWERS)
async def test_thermal_refusal_is_unchanged_and_logged_blocked(env, economy, caplog, monkeypatch, answer):
    caplog.set_level(logging.INFO, logger="orion-thought.physiology_shadow")
    economy.answer = answer
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    monkeypatch.setattr(env.vc.settings, "thermal_gate_enabled", True)
    monkeypatch.setattr(env.vc, "_thermal_state", "normal")
    monkeypatch.setattr(env.vc, "read_cabinet_temp_c", lambda: (36.0, 1.0))
    out = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id))
    await _drain()
    assert (out.status, out.reason) == ("retry", "thermal_refused") and env.generate_calls == []
    assert bus.sent == []
    (line,) = _lines(caplog)
    assert (line["family"], line["want_id"], line["real"]) == ("thought.painting", "corr-1", "blocked:thermal_refused")
    assert line["shadow"] == {"funded": "funded", "deferred": "deferred"}.get(answer, "none")
    # A funded want the real gate refused frees its shadow reservation.
    assert economy.released == (["corr-1"] if answer == "funded" else [])


@pytest.mark.asyncio
async def test_replayed_generate_does_not_bid_again(env, economy):
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id))
    await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id, correlation_id="corr-2"))
    await _drain()
    assert [w.want_id for w in economy.wants] == ["corr-1"]


@pytest.mark.asyncio
async def test_dream_hop_bids_as_its_own_family(env, economy):
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _dream_req("prepare"))
    out = await _run_on(env, bus, _dream_req("generate", attempt_id=prepared.attempt_id))
    await _drain()
    assert out.status == "done"
    (want,) = economy.wants
    assert (want.family, want.want_id) == ("thought.dream_hop", "dream-corr-1")


@pytest.mark.asyncio
async def test_flag_off_never_bids(env, economy, monkeypatch):
    monkeypatch.setattr(economy.ps, "_enabled", lambda: False)
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    out = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id))
    await _drain()
    assert out.status == "done" and economy.wants == []


@pytest.mark.asyncio
async def test_a_later_generate_names_the_act_it_ran_under(env, economy, caplog, monkeypatch):
    """The act opens on the first generate. A re-generate after a failed render runs under that act,
    so its line names the act's correlation id for the join."""
    caplog.set_level(logging.INFO, logger="orion-thought.physiology_shadow")
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    real_generate = env.vc.call_diffusion_generate

    def broken(*a, **kw):
        raise env.vc.DiffusionGenerationError("HTTP 500")

    monkeypatch.setattr(env.vc, "call_diffusion_generate", broken)
    first = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id, correlation_id="gen-1"))
    assert first.status == "retry"
    monkeypatch.setattr(env.vc, "call_diffusion_generate", real_generate)
    second = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id, correlation_id="gen-2"))
    await _drain()
    assert second.status == "done"
    lines = _lines(caplog)
    assert [(x["want_id"], x["real"], x["act_correlation_id"]) for x in lines] == [
        ("gen-1", "ran", "-"), ("gen-2", "ran", "gen-1")]


@pytest.mark.asyncio
async def test_a_step_that_returns_before_its_work_starts_settles_when_the_work_does(env, economy, caplog,
                                                                                    monkeypatch):
    """Deadline before the generating mark lands: the line waits for the work, so a render that
    starts late is logged ran and its funded reservation is kept."""
    import time as _time

    caplog.set_level(logging.INFO, logger="orion-thought.physiology_shadow")
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    monkeypatch.setattr(env.steps, "visual_step_generate_deadline_sec", lambda: 0.05)
    real_update = env.store.update_visual_stage

    def slow_update(*a, **kw):
        _time.sleep(0.3)
        return real_update(*a, **kw)

    monkeypatch.setattr(env.steps.store, "update_visual_stage", slow_update)
    out = await _run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id))
    assert (out.status, out.reason) == ("retry", "generate_deadline_exceeded")
    await _drain()
    assert _lines(caplog) == []                 # the work has not decided yet
    await asyncio.gather(*list(env.steps._generate_tasks))
    await _drain()
    (line,) = _lines(caplog)
    assert line["real"] == "ran" and economy.released == []


@pytest.mark.asyncio
async def test_a_slow_economy_never_holds_the_step(env, economy, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="orion-thought.physiology_shadow")
    gate = asyncio.Event()

    async def slow_abid(want):
        economy.wants.append(want)
        await gate.wait()
        return _decision(want, "funded")

    monkeypatch.setattr(economy.ps, "abid", slow_abid)
    bus = _GrammarBus()
    prepared = await _run_on(env, bus, _req("prepare"))
    out = await asyncio.wait_for(_run_on(env, bus, _req("generate", attempt_id=prepared.attempt_id)), timeout=5.0)
    assert out.status == "done" and len(economy.wants) == 1
    assert _lines(caplog) == []
    gate.set()
    await _drain()
    assert _lines(caplog)[0]["shadow"] == "funded"
