"""gpu-pool stage 6.2: two clocks per call, per granted role, plus decode speed.

wait = pool acquire -> grant (the line), model = grant -> reply (the worker). The old single
clock started before the lease and mixed the two, per machine. These tests pin that the two
clocks are disjoint, grouped by the granted role, carried for the HTTP passthroughs too, and
that tokens/sec is llama.cpp's own reading, never derived from wall time.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app import grammar_emit
from app import main as gateway
from app.grammar_emit import (
    CallClock,
    InferenceWindowRecorder,
    build_window_events,
    classify_http_outcome,
    decode_tps_from,
)
from app.settings import settings


class _Tick:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _kv(summary: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in summary.split() if "=" in part)


def _roles(summary: str) -> dict[str, dict[str, str]]:
    import re

    out = {}
    for role, body in re.findall(r"([a-z0-9_.-]+)\[([^\]]*)\]", _kv(summary)["roles"]):
        out[role] = dict(p.split(":", 1) for p in body.split("|"))
    return out


@pytest.fixture(autouse=True)
def _fresh_recorder(monkeypatch):
    grammar_emit.reset_recorder_for_tests()
    monkeypatch.setattr(settings, "llm_gateway_grammar_enabled", True)
    monkeypatch.setattr(settings, "llm_lane_routing_enabled", False)
    yield
    grammar_emit.reset_recorder_for_tests()


# ── CallClock ──────────────────────────────────────────────────────────────────────────────


def test_wait_and_model_are_separate_disjoint_intervals():
    tick = _Tick()
    clock = CallClock(clock=tick)
    clock.waiting()
    tick.t = 5.0
    clock.granted("chat")
    tick.t = 7.5
    clock.replied()
    assert (clock.wait_ms, clock.model_ms, clock.role) == (5000, 2500, "chat")


def test_a_re_lease_sums_each_interval_and_keeps_the_last_role():
    tick = _Tick()
    clock = CallClock(clock=tick)
    clock.waiting()
    tick.t = 1.0
    clock.granted("fast")
    tick.t = 1.2  # overflowed quickly
    clock.replied()
    clock.waiting()
    tick.t = 4.0
    clock.granted("agent")
    tick.t = 10.0
    clock.replied()
    assert clock.wait_ms == 1000 + 2800
    assert clock.model_ms == 200 + 6000
    assert clock.role == "agent"


def test_a_call_that_never_got_a_grant_has_wait_but_no_model_time():
    tick = _Tick()
    clock = CallClock(clock=tick)
    clock.waiting()
    tick.t = 60.0
    clock.not_granted()
    assert (clock.wait_ms, clock.model_ms, clock.role) == (60000, None, None)


def test_close_ends_an_interval_left_open_by_a_crash():
    tick = _Tick()
    clock = CallClock(clock=tick)
    clock.waiting()
    tick.t = 1.0
    clock.granted("chat")
    tick.t = 3.0
    clock.close()
    assert (clock.wait_ms, clock.model_ms) == (1000, 2000)


# ── window aggregation ─────────────────────────────────────────────────────────────────────


def _clock(wait_ms: int | None, model_ms: int | None, role: str | None) -> CallClock:
    c = CallClock()
    c.wait_ms, c.model_ms, c.role = wait_ms, model_ms, role
    return c


def test_window_groups_the_clocks_by_granted_role():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    ok = lambda tps: {"text": "ok", "raw": {"timings": {"predicted_per_second": tps}}}  # noqa: E731
    rec.record(ok(40.0), served_by="circe-worker-chat", timing=_clock(100, 2000, "chat"))
    rec.record(ok(20.0), served_by="circe-worker-chat", timing=_clock(300, 4000, "chat"))
    rec.record(ok(30.0), served_by="circe-worker-chat", timing=_clock(200, 3000, "chat"))
    rec.record(ok(90.0), served_by="circe-worker-metacog", timing=_clock(5, 400, "metacog"))
    # a timeout waited in line like everyone else, but its model time is its budget, not a speed
    rec.record({"text": "[Error: llamacpp timed out]", "raw": {}}, served_by="circe-worker-chat",
               timing=_clock(900, 120000, "chat"))
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets)
    roles = _roles(events[0].atom.summary)

    assert set(roles) == {"chat", "metacog"}
    chat = roles["chat"]
    assert chat["calls"] == "4" and chat["served"] == "3" and chat["upstream_failed"] == "1"
    assert chat["wait_p50_ms"] == "300"  # 100, 200, 300, 900 -> index round(0.5*3)=2
    assert chat["wait_p95_ms"] == "900"
    assert chat["model_p50_ms"] == "3000"  # served only: 2000, 3000, 4000
    assert chat["model_p95_ms"] == "4000"
    assert chat["decode_tps_p50"] == "30.0" and chat["decode_tps_n"] == "3"
    assert roles["metacog"]["model_p50_ms"] == "400" and roles["metacog"]["decode_tps_p50"] == "90.0"


def test_ungranted_calls_are_their_own_role_and_have_no_model_clock():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    rec.record({"text": "", "raw": {"error": "gpu_pool_unavailable"}}, served_by=None,
               timing=_clock(60000, None, None))
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets)
    role = _roles(events[0].atom.summary)["ungranted"]
    assert role["refused"] == "1" and role["wait_p50_ms"] == "60000"
    assert "model_p50_ms" not in role and "decode_tps_p50" not in role


def test_decode_tps_is_absent_not_zero_when_the_backend_did_not_report_it():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    rec.record({"text": "ok", "raw": {}}, served_by="circe-worker-fast", timing=_clock(1, 50, "fast"))
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets)
    fast = _roles(events[0].atom.summary)["fast"]
    assert "decode_tps_p50" not in fast and fast["decode_tps_n"] == "0"
    assert fast["model_p50_ms"] == "50"


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"timings": {"predicted_per_second": 31.25}}, 31.25),
        ({"timings": {"predicted_per_second": "12.5"}}, 12.5),
        ({"timings": {"predicted_per_second": 0}}, None),
        ({"timings": {"predicted_per_second": -3}}, None),
        ({"timings": {"predicted_per_second": "nan"}}, None),
        ({"timings": {}}, None),
        ({"usage": {"completion_tokens": 500}}, None),  # never derived from tokens / wall time
        ({}, None),
        # a stream's tail: the last reported value wins
        (b'data: {"timings":{"predicted_per_second":10.0}}\n\ndata: {"timings":{"predicted_per_second": 22.5}}\n\n', 22.5),
        (b"data: [DONE]\n\n", None),
        (None, None),
    ],
)
def test_decode_tps_reads_llamacpp_timings_only(payload, expected):
    assert decode_tps_from(payload) == expected


@pytest.mark.parametrize(
    "status, overflow, expected",
    [(200, False, "served"), (400, False, "upstream_http_4xx"), (400, True, "context_overflow"),
     (503, False, "upstream_http_5xx"), (500, False, "upstream_http_5xx")],
)
def test_classify_http_outcome(status, overflow, expected):
    assert classify_http_outcome(status, context_overflow=overflow) == expected


# ── bus path: the clock no longer starts before the lease ─────────────────────────────────


def _hold_role_busy(pool, role: str, seconds: float) -> None:
    """Fill ``role``'s slots so an acquire must wait, then free them after ``seconds``."""
    pool.max_wait_sec = 5.0  # the fake pool gives up after 50 ms by default
    pool.busy[role] = pool.slots[role]
    threading.Timer(seconds, lambda: pool.busy.__setitem__(role, 0)).start()


@pytest.mark.asyncio
async def test_bus_call_records_wait_and_model_time_separately_under_the_granted_role(fake_pool, monkeypatch):
    def run(body, plan):
        time.sleep(0.15)  # the model
        return {"text": "hello", "raw": {"timings": {"predicted_per_second": 25.0}}, "route": plan.route,
                "served_by": plan.route_target.served_by}

    monkeypatch.setattr(gateway, "run_llm_chat", run)
    fake_pool.choose = lambda kw: "chat"
    _hold_role_busy(fake_pool, "chat", 0.3)  # the line
    for role in fake_pool.slots:
        if role != "chat":
            fake_pool.busy[role] = fake_pool.slots[role]
    env = BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="chat").model_dump(mode="json"),
    )
    await gateway.handle_chat(env)

    _, _, buckets = grammar_emit.get_recorder().drain()
    role = buckets["circe"].roles["chat"]
    assert role.calls == 1
    (wait,), (model,) = role.wait_ms, role.model_ms
    assert 250 <= wait < 1500, wait  # waited for the slot, not for the model
    assert 120 <= model < 1000, model  # the model, not the line
    assert role.decode_tps == [25.0]


@pytest.mark.asyncio
async def test_pool_refusal_records_the_wait_under_ungranted(fake_pool, monkeypatch):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    fake_pool.unavailable = "no_serviceable_role"
    env = BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="quick").model_dump(mode="json"),
    )
    await gateway.handle_chat(env)
    _, _, buckets = grammar_emit.get_recorder().drain()
    # node counts unchanged (no worker: unattributed), the wait filed under the pool's host node
    assert buckets["unrouted"].classes == {"gpu_pool_unavailable": 1} and buckets["unrouted"].roles == {}
    role = buckets["circe"].roles["ungranted"]
    assert role.classes == {"gpu_pool_unavailable": 1}
    assert len(role.wait_ms) == 1 and role.model_ms == []
    assert buckets["circe"].calls == 0


# ── HTTP passthroughs are counted too ──────────────────────────────────────────────────────


@pytest.fixture
def passthrough_client(monkeypatch, fake_pool):
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    from app.main import app

    return TestClient(app)


def test_openai_passthrough_is_counted_with_both_clocks(passthrough_client, fake_pool, monkeypatch):
    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        import asyncio

        await asyncio.sleep(0.1)
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            "timings": {"predicted_per_second": 48.0},
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    _hold_role_busy(fake_pool, "fast", 0.2)
    for role in fake_pool.slots:
        if role != "fast":
            fake_pool.busy[role] = fake_pool.slots[role]
    response = passthrough_client.post(
        "/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200

    _, _, buckets = grammar_emit.get_recorder().drain()
    node = buckets["circe"]
    # record-only: the node counts behind inference_failure_pressure stay bus-RPC calls
    assert node.calls == 0 and node.classes == {} and node.worker_attempted == {}
    role = node.roles["fast"]
    assert (role.calls, role.http_calls, role.classes) == (1, 1, {"served": 1})
    (wait,), (model,) = role.wait_ms, role.model_ms
    assert 150 <= wait < 1500, wait
    assert 80 <= model < 1000, model
    assert role.decode_tps == [48.0]


def test_openai_passthrough_upstream_5xx_is_a_backend_failure_in_its_role(passthrough_client, fake_pool, monkeypatch):
    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    passthrough_client.post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    _, _, buckets = grammar_emit.get_recorder().drain()
    node = buckets["circe"]
    assert node.roles["fast"].classes == {"upstream_http_5xx": 1}
    assert node.roles["fast"].count(grammar_emit.UPSTREAM_FAILURE_CLASSES) == 1
    # not (yet) in inference_failure_pressure's population: that is a metric-definition change
    assert node.worker_failed == {} and node.calls == 0


def test_passthrough_pool_refusal_is_counted_as_ungranted(passthrough_client, fake_pool, monkeypatch):
    fake_pool.unavailable = "no_serviceable_role"
    response = passthrough_client.post(
        "/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    _, _, buckets = grammar_emit.get_recorder().drain()
    assert "unrouted" not in buckets  # HTTP calls never touch node counts
    assert buckets["circe"].roles["ungranted"].classes == {"gpu_pool_unavailable": 1}
    assert buckets["circe"].roles["ungranted"].http_calls == 1


def test_passthrough_records_nothing_when_grammar_is_off(passthrough_client, fake_pool, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_grammar_enabled", False)

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    passthrough_client.post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert grammar_emit.get_recorder().drain()[2] == {}


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_streamed_passthrough_is_counted_once_at_stream_end_with_its_tail_timings(
    mock_client_cls: MagicMock, passthrough_client, fake_pool
) -> None:
    class _Upstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield (b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2},'
                   b'"timings":{"predicted_per_second":33.3}}\n\n')
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            return None

    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=_Upstream())
    mock_client.aclose = AsyncMock()
    mock_client_cls.return_value = mock_client

    with passthrough_client.stream("POST", "/v1/chat/completions", json={
            "model": "quick", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as response:
        b"".join(response.iter_bytes())

    _, _, buckets = grammar_emit.get_recorder().drain()
    node = buckets["circe"]
    role = node.roles[fake_pool.leases[-1].grant.role]
    assert (role.calls, role.http_calls, role.classes) == (1, 1, {"served": 1})
    assert role.decode_tps == [33.3] and len(role.model_ms) == 1


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_stream_that_breaks_is_counted_as_an_upstream_failure(
    mock_client_cls: MagicMock, passthrough_client, fake_pool
) -> None:
    class _Upstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b"data: {}\n\n"
            raise httpx.ReadError("connection reset")

        async def aclose(self) -> None:
            return None

    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=_Upstream())
    mock_client.aclose = AsyncMock()
    mock_client_cls.return_value = mock_client

    with pytest.raises(Exception):
        with passthrough_client.stream("POST", "/v1/chat/completions", json={
                "model": "quick", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as response:
            b"".join(response.iter_bytes())
    _, _, buckets = grammar_emit.get_recorder().drain()
    (role,) = buckets["circe"].roles.values()
    assert role.classes == {"upstream_error": 1}
    assert role.calls == 1  # not twice: close() runs from the body and from cleanup


def test_anthropic_messages_passthrough_is_counted_under_its_granted_role(fake_pool, monkeypatch):
    """The FCC / Claude Code path (/v1/messages) was not counted at all before stage 6.2."""
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    from app.main import app

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "msg_1", "content": [{"type": "text", "text": "OK"}],
            "usage": {"input_tokens": 40, "output_tokens": 2},
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    response = TestClient(app).post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01", "x-api-key": "freecc"},
        json={"model": "llamacpp/agent", "max_tokens": 64, "stream": False,
              "messages": [{"role": "user", "content": "Say OK."}]},
    )
    assert response.status_code == 200
    _, _, buckets = grammar_emit.get_recorder().drain()
    node = buckets["circe"]
    granted = fake_pool.leases[-1].grant.role
    role = node.roles[granted]
    assert (role.calls, role.http_calls, role.classes) == (1, 1, {"served": 1})
    assert len(role.wait_ms) == 1 and len(role.model_ms) == 1
    assert role.decode_tps == []  # no llama.cpp timings in this body: absent, not guessed


def test_a_passthrough_only_window_leaves_the_failure_population_untouched():
    """The node atom still goes out (so the role clocks reach the reducer), with calls=0 and no
    worker counts, so inference_failure_pressure's rolling window gains nothing from it."""
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    rec.record_outcome("upstream_http_5xx", served_by="circe-worker-agent",
                       timing=_clock(10, 200, "agent"), http=True)
    rec.record_outcome("served", served_by="circe-worker-agent", timing=_clock(5, 900, "agent"),
                       tokens=(100, 10), timings=grammar_emit.ReplyTimings(decode_tps=20.0), http=True)
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets)
    kv = _kv(events[0].atom.summary)
    assert (kv["calls"], kv["served"], kv["upstream_failed"]) == ("0", "0", "0")
    assert kv["worker_attempted"] == "none" and kv["worker_failed"] == "none"
    agent = _roles(events[0].atom.summary)["agent"]
    assert (agent["calls"], agent["http_calls"], agent["upstream_failed"]) == ("2", "2", "1")
    assert agent["model_p50_ms"] == "900" and agent["decode_tps_p50"] == "20.0"


# ── stage 7 covariates: occupancy at grant, slots, llama.cpp prompt-cache counts ───────────

# The exact timings block a leased /v1/chat/completions call to the fast role returned
# through this gateway on 2026-09-30 (llama.cpp b10398): prompt_n + cache_n == usage.prompt_tokens.
LIVE_TIMINGS = {
    "cache_n": 3, "prompt_n": 11, "prompt_ms": 33.016, "prompt_per_token_ms": 3.0014545454545454,
    "prompt_per_second": 333.1717954930943, "predicted_n": 2, "predicted_ms": 11.283,
    "predicted_per_token_ms": 5.6415, "predicted_per_second": 177.25782150137377,
}


def test_timings_from_reads_the_live_llamacpp_shape():
    t = grammar_emit.timings_from({"timings": LIVE_TIMINGS, "usage": {"prompt_tokens": 14}})
    assert (t.prompt_n, t.cache_n) == (11, 3)
    assert t.decode_tps == pytest.approx(177.2578, rel=1e-4)
    tail = b'data: {"timings":' + __import__("json").dumps(LIVE_TIMINGS).encode() + b"}\n\ndata: [DONE]\n\n"
    assert grammar_emit.timings_from(tail) == t
    assert grammar_emit.timings_from({}) == grammar_emit.ReplyTimings()


def _busy_clock(busy: int, role: str = "metacog") -> CallClock:
    c = _clock(5, 500, role)
    c.busy_at_grant = busy
    return c


def test_decode_speed_is_banded_by_occupancy_at_grant_and_carries_slots_and_cache():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    for busy, tps in [(1, 100.0), (1, 90.0), (3, 40.0), (4, 30.0), (2, 50.0)]:
        rec.record({"text": "ok", "raw": {"timings": {"predicted_per_second": tps, "prompt_n": 10, "cache_n": 90}}},
                   served_by="circe-worker-metacog", timing=_busy_clock(busy))
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets,
                                 role_slots={"metacog": 4})
    m = _roles(events[0].atom.summary)["metacog"]
    assert (m["decode_tps_solo_p50"], m["decode_tps_solo_n"]) == ("90.0", "2")  # [90, 100] nearest-rank
    assert (m["decode_tps_shared_p50"], m["decode_tps_shared_n"]) == ("40.0", "3")
    assert (m["busy_p50"], m["busy_max"], m["slots"]) == ("2", "4", "4")
    assert (m["prompt_n"], m["cache_n"], m["cache_reports"]) == ("50", "450", "5")


def test_unknown_occupancy_and_no_cache_report_are_omitted_not_zero():
    rec = InferenceWindowRecorder(clock=lambda: 0.0)
    rec.record({"text": "ok", "raw": {"timings": {"predicted_per_second": 10.0}}},
               served_by="circe-worker-fast", timing=_clock(1, 50, "fast"))
    _, _, buckets = rec.drain()
    events = build_window_events(gateway_node="athena", window_start=0, window_end=60, buckets=buckets)
    f = _roles(events[0].atom.summary)["fast"]
    for key in ("busy_p50", "busy_max", "slots", "prompt_n", "cache_n", "decode_tps_solo_p50"):
        assert key not in f, key
    assert f["cache_reports"] == "0" and f["decode_tps_p50"] == "10.0"


@pytest.mark.asyncio
async def test_concurrent_bus_calls_see_each_other_at_grant_and_occupancy_never_leaks(fake_pool, monkeypatch):
    import asyncio

    from app import pool_placement

    pool_placement.reset_occupancy_for_tests()
    gate = threading.Event()

    def run(body, plan):
        gate.wait(2.0)
        return {"text": "hi", "raw": {"timings": {"predicted_per_second": 20.0}}, "route": plan.route,
                "served_by": plan.route_target.served_by}

    monkeypatch.setattr(gateway, "run_llm_chat", run)
    fake_pool.choose = lambda kw: "metacog"

    def _env():
        return BaseEnvelope(
            kind="llm.chat.request", source=ServiceRef(name="cortex-exec", node="n", version="0"),
            correlation_id=str(uuid.uuid4()),
            payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")],
                                       route="metacog").model_dump(mode="json"),
        )

    first = asyncio.ensure_future(gateway.handle_chat(_env()))
    await asyncio.sleep(0.1)
    second = asyncio.ensure_future(gateway.handle_chat(_env()))
    await asyncio.sleep(0.1)
    assert pool_placement.role_in_flight() == {"metacog": 2}
    gate.set()
    await asyncio.gather(first, second)
    assert pool_placement.role_in_flight() == {}  # released on every path, never leaks

    _, _, buckets = grammar_emit.get_recorder().drain()
    role = buckets["circe"].roles["metacog"]
    assert sorted(role.busy) == [1, 2]
    assert role.decode_tps_solo == [20.0] and role.decode_tps_shared == [20.0]


@pytest.mark.asyncio
async def test_occupancy_is_released_when_the_upstream_call_raises(fake_pool, monkeypatch):
    from app import pool_placement

    pool_placement.reset_occupancy_for_tests()

    def boom(body, plan):
        raise RuntimeError("worker thread crashed")

    monkeypatch.setattr(gateway, "run_llm_chat", boom)
    with pytest.raises(RuntimeError):
        await gateway._dispatch_chat(
            __import__("app.models", fromlist=["ChatBody"]).ChatBody(
                route="quick", messages=[{"role": "user", "content": "x"}]),
            correlation_id="c")
    assert pool_placement.role_in_flight() == {}


def test_passthrough_carries_occupancy_and_cache_counts(passthrough_client, fake_pool, monkeypatch):
    from app import pool_placement

    pool_placement.reset_occupancy_for_tests()

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"choices": [], "timings": LIVE_TIMINGS})

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    passthrough_client.post("/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    _, _, buckets = grammar_emit.get_recorder().drain()
    (role,) = buckets["circe"].roles.values()
    assert role.busy == [1]
    assert (role.prompt_n, role.cache_n, role.cache_reports) == (11, 3, 1)
    assert pool_placement.role_in_flight() == {}


@pytest.mark.asyncio
async def test_role_slots_reads_llm_roles_from_pool_state(monkeypatch):
    from app import pool_placement

    async def _state():
        return {"roles": [
            {"role": "metacog", "kind": "llm", "slots": 4},
            {"role": "agent", "kind": "llm", "slots": 1},
            {"role": "diffusion", "kind": "service", "slots": 1},
            {"role": "chat", "kind": "llm", "slots": 0},  # down: unknown, not zero
        ]}

    monkeypatch.setattr(pool_placement, "fetch_pool_state", _state)
    assert await pool_placement.role_slots() == {"metacog": 4, "agent": 1}

    async def _none():
        return None

    monkeypatch.setattr(pool_placement, "fetch_pool_state", _none)
    assert await pool_placement.role_slots() == {}


@pytest.mark.asyncio
async def test_publisher_stamps_slots_and_survives_a_slots_failure():
    import asyncio

    for provider, expect in ((lambda: _async({"metacog": 4}), "slots:4"), (_raise, None)):
        grammar_emit.reset_recorder_for_tests()
        grammar_emit.get_recorder().record({"text": "ok", "raw": {}}, served_by="circe-worker-metacog",
                                           timing=_busy_clock(1))
        published = []

        class _Bus:
            async def publish(self, channel, envelope):
                published.append(envelope)

        stop = asyncio.Event()

        async def _stop_soon():
            await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(
            grammar_emit.run_window_publisher(_Bus(), gateway_node="athena", window_sec=60, stop=stop,
                                              slots_provider=provider),
            _stop_soon(),
        )
        summary = published[0].payload["atom"]["summary"]
        if expect:
            assert expect in summary
        else:
            assert "slots:" not in summary and "metacog[" in summary


async def _async(value):
    return value


async def _raise():
    raise RuntimeError("pool state rpc down")


# ── passthrough exit paths: each call recorded exactly once, with its real outcome ─────────

_OVERFLOW_BODY = {"error": {"message": "the request exceeds the available context size"}}


def _roles_total(buckets) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in buckets.values():
        for role in node.roles.values():
            for cls, n in role.classes.items():
                out[cls] = out.get(cls, 0) + n
    return out


def test_overflow_then_re_lease_is_one_served_call(passthrough_client, fake_pool, monkeypatch):
    replies = [httpx.Response(400, json=_OVERFLOW_BODY), httpx.Response(200, json={"choices": []})]

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return replies.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    response = passthrough_client.post(
        "/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200 and len(fake_pool.calls) == 2
    _, _, buckets = grammar_emit.get_recorder().drain()
    assert _roles_total(buckets) == {"served": 1}


def test_overflow_with_nothing_bigger_is_one_context_overflow(passthrough_client, fake_pool, monkeypatch):
    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        fake_pool.unavailable = "min_ctx_exceeds_class:4096"  # the re-lease finds nothing bigger
        return httpx.Response(400, json=_OVERFLOW_BODY)

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    response = passthrough_client.post(
        "/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 400
    _, _, buckets = grammar_emit.get_recorder().drain()
    assert _roles_total(buckets) == {"context_overflow": 1}


def test_lease_revoked_mid_call_is_one_gpu_pool_recalled(passthrough_client, fake_pool, monkeypatch):
    import asyncio

    fake_pool.on_grant = lambda lease: lease.lost.set()  # the pool drops the lease at once

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    response = passthrough_client.post(
        "/v1/chat/completions", json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    _, _, buckets = grammar_emit.get_recorder().drain()
    assert _roles_total(buckets) == {"gpu_pool_recalled": 1}


@pytest.mark.asyncio
async def test_a_grant_landing_as_the_client_leaves_is_released(fake_pool):
    """Review finding: the acquire finished during the disconnect check; the lease (and the
    role occupancy count) must be released, not held for the life of the process."""
    import asyncio

    from app import passthrough_proxy, pool_placement

    pool_placement.reset_occupancy_for_tests()
    handle = pool_placement.PoolLease(route="quick", spec=pool_placement.route_spec("quick"), holder="t",
                                      turn_correlation_id=None, min_ctx_tokens=10, deadline_sec=5.0)
    task = asyncio.ensure_future(handle.acquire())
    await asyncio.sleep(0.05)
    assert task.done() and pool_placement.role_in_flight() == {"fast": 1}
    await passthrough_proxy._withdraw(task, handle)
    assert pool_placement.role_in_flight() == {} and fake_pool.active == 0


def test_client_gone_while_queued_is_recorded_once(fake_pool, monkeypatch):
    """_ClientGone path: the queued acquire is withdrawn and the call is client_gone, once."""
    import asyncio

    from app import passthrough_proxy, pool_placement

    fake_pool.max_wait_sec = 5.0
    for role in fake_pool.slots:
        fake_pool.busy[role] = fake_pool.slots[role]  # nothing free: the acquire queues

    class _GoneRequest:
        async def is_disconnected(self) -> bool:
            return True

    monkeypatch.setattr(passthrough_proxy, "_DISCONNECT_POLL_SEC", 0.01)
    monkeypatch.setattr(passthrough_proxy, "_http_helpers",
                        lambda: (lambda request: {}, lambda headers: {}, lambda: None))

    async def _run():
        return await passthrough_proxy.proxy_on_pool(
            request=_GoneRequest(), route_key="quick", forward_body={}, path="/v1/chat/completions",
            holder="http:openai", correlation_id=None, min_ctx_tokens=10, anthropic=False)

    response = asyncio.run(_run())
    assert response.status_code == passthrough_proxy.CLIENT_CLOSED_STATUS
    _, _, buckets = grammar_emit.get_recorder().drain()
    assert _roles_total(buckets) == {grammar_emit.CLIENT_GONE: 1}
    assert pool_placement.role_in_flight() == {}
