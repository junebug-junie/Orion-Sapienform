"""Body economy Patch C3b: the in-process journal's SHADOW bid (app/physiology_shadow.py).

The contract: a journal that would compose bids (Tier 1 still bids); the compose (its act events,
its compose request) is identical whatever the economy answers (funded, deferred, no answer, an
error); the logged decision carries the journal act's correlation id and the real outcome.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from app import physiology_shadow as ps
from orion.schemas.physiology_want import WantDecisionV1, WantV1
from test_act_emit import _Harness, _journal_env

ANSWERS = ["funded", "deferred", "none", "raises", "off"]


def _decision(want: WantV1, outcome: str) -> WantDecisionV1:
    now = datetime.now(timezone.utc)
    return WantDecisionV1(
        decision_id=f"{want.want_id}@{now.isoformat()}", want_id=want.want_id, family=want.family,
        producer=want.producer, verb=want.verb, tier=want.tier, decided_at=now, outcome=outcome,
        reason="net_positive" if outcome == "funded" else "price", lane="metacog", want=want,
    )


@pytest.fixture
def economy(monkeypatch):
    class _Economy:
        answer = "funded"
        gate = None

    eco = _Economy()
    eco.wants, eco.released = [], []

    async def fake_abid(want):
        eco.wants.append(want)
        if eco.gate is not None:
            await eco.gate.wait()
        if eco.answer == "raises":
            raise RuntimeError("economy down")
        return None if eco.answer == "none" else _decision(want, eco.answer)

    monkeypatch.setattr(ps, "abid", fake_abid)
    monkeypatch.setattr(ps, "release", lambda want_id: eco.released.append(want_id))
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


def _strip(event: dict) -> dict:
    """An act event minus its clocks and random event id."""
    out = {k: v for k, v in event.items() if k not in ("emitted_at", "observed_at", "event_id")}
    act = dict(out["act"])
    for k in ("opened_at", "closed_at"):
        act.pop(k, None)
    if act.get("outcome"):
        act["outcome"] = {k: v for k, v in act["outcome"].items() if k != "observed_at"}
    if act.get("cost"):
        act["cost"] = {k: v for k, v in act["cost"].items() if k != "seconds"}
    out["act"] = act
    return out


def _journal_once(tmp_path, monkeypatch, env, *, orch_payload=None, rpc_error=None):
    async def body(app):
        await app.state.bus_handler(env)
        await _drain()

    h = _Harness(orch_payload=orch_payload, rpc_error=rpc_error)
    events = asyncio.run(h.run(tmp_path, body))
    return h, events


def test_compose_is_identical_for_every_answer_and_joins_the_act(tmp_path, economy, caplog, monkeypatch):
    from app import main as actions_main

    caplog.set_level(logging.INFO, logger="orion-actions.physiology_shadow")
    env = _journal_env()
    seen = []
    for answer in ANSWERS:
        economy.answer = answer
        economy.wants.clear()
        caplog.clear()
        monkeypatch.setattr(actions_main.settings, "actions_physiology_shadow_bid_enabled", answer != "off")
        h, events = _journal_once(tmp_path, monkeypatch, env, orch_payload={"ok": True})
        assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
        if answer == "off":
            assert economy.wants == [] and _lines(caplog) == []
        else:
            (want,) = economy.wants
            # The join: want_id == the journal act's correlation id == the compose request's id.
            assert want.want_id == want.correlation_id == events[0]["correlation_id"] == str(h.sent[0].correlation_id)
            assert (want.family, want.verb, want.tier) == ("actions.journal", "make", 1)
            (line,) = _lines(caplog)
            assert (line["want_id"], line["real"]) == (want.want_id, "ran")
        seen.append(([_strip(e) for e in events], [str(s.correlation_id) for s in h.sent]))
    assert all(s == seen[0] for s in seen)


@pytest.mark.parametrize("answer", ["funded", "none"])
def test_a_failed_compose_still_ran(tmp_path, economy, caplog, monkeypatch, answer):
    """The act opened and closed at 0: the real gates let it run, so the shadow line says ran and a
    funded reservation is kept until the act closes."""
    caplog.set_level(logging.INFO, logger="orion-actions.physiology_shadow")
    economy.answer = answer
    _, events = _journal_once(tmp_path, monkeypatch, _journal_env(), rpc_error=TimeoutError("RPC timeout"))
    assert events[1]["act"]["outcome"]["value"] == 0.0
    assert _lines(caplog)[0]["real"] == "ran" and economy.released == []


def test_a_cooldown_skip_never_bids(tmp_path, economy):
    env = _journal_env("same summary")

    async def body(app):
        await app.state.bus_handler(env)
        await app.state.bus_handler(_journal_env("same summary"))  # deduped by cooldown
        await _drain()

    asyncio.run(_Harness(orch_payload={"ok": True}).run(tmp_path, body))
    assert len(economy.wants) == 1


def test_a_slow_economy_never_holds_the_compose(tmp_path, economy, caplog):
    caplog.set_level(logging.INFO, logger="orion-actions.physiology_shadow")

    async def body(app):
        economy.gate = asyncio.Event()
        await asyncio.wait_for(app.state.bus_handler(_journal_env()), timeout=5.0)
        assert _lines(caplog) == []          # composed; the economy has not answered yet
        economy.gate.set()
        await _drain()

    events = asyncio.run(_Harness(orch_payload={"ok": True}).run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert _lines(caplog)[0]["shadow"] == "funded"


def test_a_shadow_that_cannot_be_built_never_fails_the_compose(tmp_path, economy, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad deploy")

    monkeypatch.setattr(ps, "ShadowBid", boom)
    _, events = _journal_once(tmp_path, monkeypatch, _journal_env(), orch_payload={"ok": True})
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["value"] == 1.0
