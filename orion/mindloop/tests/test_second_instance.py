"""A second loop instance runs on the library with its own identity, key, family and actor (design section 10)."""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

from orion.mindloop.tests.echo_instance import ECHO_KEY, Bus, EchoStateV1, advance, echo_runtime
from orion.mindloop.turns import JuniperTurns

NO_SEAT = (None, "empty_llm_response", {"gateway_error": {"error": "gpu_pool_unavailable", "reason": "deadline"}})
ECHO = ({"echo": "hello back"}, None, {})


def test_a_funded_request_bids_as_its_own_family_and_closes_its_own_act(caplog):
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    rt = echo_runtime([ECHO])
    asyncio.run(rt.tick())
    [want] = rt.bids
    assert (want.family, want.producer, want.tier, want.verb, want.lanes) == ("test.echo", "test.echo", 2, "infer",
                                                                             ["chat"])
    assert want.want_id.startswith("echo-think-") and want.cost.gpu_slot_s_by_lane == {"chat": 30.0}
    assert "ECHO_LLM_TIMEOUT_SEC" in want.cost.basis["gpu_slot_s_by_lane"]
    assert rt.released == [want.want_id]
    [call] = rt.llm.calls
    assert call["route"] == "echo" and call["user_prompt"] == "hello" and call["pool_wait_sec"] == 90.0
    acts = rt.bus.acts()
    assert [(k, a["actor"], a["verb"]) for k, a in acts] == [("act_opened", "echo", "infer"),
                                                             ("act_closed", "echo", "infer")]
    assert acts[1][1]["outcome"]["note"] == "woke: 1 request(s) | echoed 'hello back'"
    assert all(e.payload["provenance"]["source_component"] == "echo.thought" for _, e in rt.bus.published)
    # Its own key only; nothing under Prime's.
    assert set(rt.bus.redis.kv) == {ECHO_KEY}
    saved = EchoStateV1.model_validate_json(rt.bus.redis.kv[ECHO_KEY])
    assert saved.answers == ["hello back"] and saved.requests == [] and saved.last_thought_at is not None


def test_no_seat_is_not_a_thought_and_the_retry_goes_on_the_starved_route(caplog):
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    rt = echo_runtime([NO_SEAT, ECHO])
    asyncio.run(rt.tick())
    assert rt.bus.acts() == [] and rt.state.no_model_streak == 1 and rt.state.requests == ["hello"]
    assert any(r.getMessage().startswith("echo_try_no_model kind=no_seat route=echo ") for r in caplog.records)
    assert rt.state.told and "never reached the model" in rt.state.told[0]
    advance(rt, 15)
    asyncio.run(rt.tick())
    assert [c["route"] for c in rt.llm.calls] == ["echo", "echo_starved"]
    assert rt.state.answers == ["hello back"] and rt.state.no_model_streak == 0


def test_it_yields_to_juniper_and_never_fails_open(caplog):
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    turns = JuniperTurns()
    rt = echo_runtime([ECHO], turns=turns)
    with turns.turn():
        asyncio.run(rt.tick())
    assert rt.bids == [] and any(r.getMessage() == "echo_yield reason=juniper_turn pending=1" for r in caplog.records)
    deferred = echo_runtime([ECHO], outcome="deferred")
    asyncio.run(deferred.tick())
    assert deferred.llm.calls == [] and deferred.bus.acts() == []


def test_state_round_trips_through_its_own_key():
    bus = Bus()
    rt = echo_runtime([ECHO], bus=bus, requests=("a", "b"))
    asyncio.run(rt.save())
    again = echo_runtime([], bus=bus, requests=())
    asyncio.run(again.load())
    assert again.state.requests == ["a", "b"]


def test_the_library_imports_no_instance():
    """orion/mindloop/ must not depend on Prime (or any instance): instances depend on it."""
    root = Path(__file__).resolve().parents[1]
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:  # relative: resolve against orion.mindloop
                    pkg = "orion.mindloop".split(".")[: max(0, 2 - (node.level - 1))]
                    base = ".".join([*pkg, base] if base else pkg)
                names = [base]
            else:
                continue
            for name in names:
                for banned in ("orion.prime", "orion.curiosity", "app"):
                    assert not (name == banned or name.startswith(banned + ".")), f"{path.name} imports {name}"


def test_prime_state_re_exports_the_library_backoff():
    from orion.mindloop import backoff
    from orion.prime import state

    for name in backoff.__all__:
        assert getattr(state, name) is getattr(backoff, name)


def test_instance_hooks_see_why_no_thought_happened_and_shape_the_want():
    from orion.mindloop.tests.echo_instance import EchoRuntime

    seen = []

    class Asking(EchoRuntime):
        def _want_extras(self):
            return {"object_ref": "echo_request:1"}

        async def _on_not_funded(self, decision):
            seen.append(("not_funded", decision.outcome))

        async def _on_no_model(self, reply):
            seen.append(("no_model", reply.kind))

    rt = echo_runtime([NO_SEAT])
    rt.__class__ = Asking
    asyncio.run(rt.tick())
    assert rt.bids[0].object_ref == "echo_request:1" and seen == [("no_model", "no_seat")]
    deferred = echo_runtime([], outcome="deferred")
    deferred.__class__ = Asking
    asyncio.run(deferred.tick())
    assert seen[-1] == ("not_funded", "deferred")


def test_wake_ends_the_tick_wait_early():
    rt = echo_runtime([])
    rt.s.ECHO_TICK_SEC = 60

    async def go():
        waiter = asyncio.create_task(rt._wait_tick())
        await asyncio.sleep(0)
        rt.wake()
        await asyncio.wait_for(waiter, timeout=2)
        assert not rt._wake.is_set()

    asyncio.run(go())


def test_a_state_model_without_the_loop_fields_is_refused_at_build():
    import pytest
    from pydantic import BaseModel

    from orion.mindloop.tests.echo_instance import EchoRuntime

    class Bare(BaseModel):
        requests: list[str] = []

    class Broken(EchoRuntime):
        state_model = Bare

    with pytest.raises(TypeError, match="no_model_streak"):
        Broken(settings=echo_runtime([]).s, bus=Bus(), llm=None, bid=lambda w: None, release=lambda i: None)


def _full(net):
    from types import SimpleNamespace

    return SimpleNamespace(outcome="deferred", reason="capacity_reserved", binding_constraint="lane:chat", net=net,
                           retry_after_sec=60.0)


def test_a_capacity_deferral_worth_its_price_thinks_and_queues_at_the_pool(caplog):
    # The lane is full, but the thought is worth its price: the GPU pool's queue holds it (bind_verdict).
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    rt = echo_runtime([ECHO], decision=_full(0.67))
    asyncio.run(rt.tick())
    [want] = rt.bids
    assert len(rt.llm.calls) == 1 and rt.state.answers == ["hello back"]
    assert [k for k, _ in rt.bus.acts()] == ["act_opened", "act_closed"]
    assert rt.released == [want.want_id]
    msgs = [r.getMessage() for r in caplog.records]
    assert ("echo_bid_queued_at_pool outcome=deferred reason=capacity_reserved binding=lane:chat net=0.67 pending=1"
            in msgs)
    assert not any("bid_not_funded" in m for m in msgs)


def test_a_capacity_deferral_not_worth_its_price_waits(caplog):
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    for net in (0.0, -0.1, None, float("nan")):
        rt = echo_runtime([ECHO], decision=_full(net))
        asyncio.run(rt.tick())
        assert rt.llm.calls == [] and rt.bus.acts() == [] and rt.released == [] and rt.state.requests == ["hello"]
    assert any(r.getMessage().startswith("echo_bid_not_funded outcome=deferred reason=capacity_reserved")
               for r in caplog.records)


def test_another_deferral_with_a_positive_net_still_waits():
    from types import SimpleNamespace

    rt = echo_runtime([ECHO], decision=SimpleNamespace(outcome="deferred", reason="price", binding_constraint=None,
                                                       net=0.5))
    asyncio.run(rt.tick())
    assert rt.llm.calls == [] and rt.bus.acts() == []


def test_no_answer_from_the_market_waits(caplog):
    # Never fail open: no answer is "not funded".
    caplog.set_level(logging.INFO, logger="test.mindloop.echo")
    rt = echo_runtime([ECHO], decision=None)
    asyncio.run(rt.tick())
    assert len(rt.bids) == 1 and rt.llm.calls == [] and rt.bus.acts() == [] and rt.released == []
    assert any(r.getMessage().startswith("echo_bid_not_funded outcome=none reason=-") for r in caplog.records)


def test_a_consent_block_never_goes_ahead_even_when_worth_its_price():
    from types import SimpleNamespace

    blocked = SimpleNamespace(outcome="deferred", reason="capacity_reserved", binding_constraint="lane:chat", net=0.9,
                              consent=SimpleNamespace(verdict="refuse"))
    rt = echo_runtime([ECHO], decision=blocked)
    asyncio.run(rt.tick())
    assert rt.llm.calls == [] and rt.bus.acts() == []
