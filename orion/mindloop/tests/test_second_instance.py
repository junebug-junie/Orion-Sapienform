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
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith("orion.prime"), f"{path.name} imports {name}"
                assert not name.startswith("orion.curiosity"), f"{path.name} imports {name}"


def test_prime_state_re_exports_the_library_backoff():
    from orion.mindloop import backoff
    from orion.prime import state

    for name in backoff.__all__:
        assert getattr(state, name) is getattr(backoff, name)
