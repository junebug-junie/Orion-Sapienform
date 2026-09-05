"""One saturated upstream must not stall any other upstream, and the gateway must not
generate for callers that already gave up.

Incident 2026-09-05 (docs/superpowers/pr-reports/2026-09-05-stance-react-attention-
salience-gateway-starvation-incident.md): every bus request took a thread from the one
32-thread default executor for its whole life. `quick`'s 4-slot worker fell behind,
topic-foundry / memory-consolidation traffic to it filled all 32 threads, and an
interactive stance_react turn for the idle `chat` worker waited 18 minutes for a thread.
Measured over 05:00-08:00Z: median 21 min receipt-to-dispatch on `quick`, 19 min on the
equally idle `metacog` lane.

The reproduction below is that shape in miniature: a flood on one lane, one request on
another, and the assertion that the other lane does not notice.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app import main as gw_main
from app import upstream_admission as ua
from app.llm_backend import _load_route_targets
from app.models import ChatBody
from app.settings import settings

QUICK = "http://quick:8013"
CHAT = "http://chat:8011"


# ------------------------------------------------------------------ the gate itself


@pytest.mark.asyncio
async def test_lanes_are_independent() -> None:
    gate = ua.UpstreamAdmission(max_inflight=1)
    async with gate.admit(QUICK, max_wait_s=1.0) as first:
        assert first is True
        assert gate.snapshot()[QUICK]["longest_wait_s"] == 0.0, "an uncontended acquire is not a wait"
        # QUICK is full. CHAT must still admit instantly.
        started = time.monotonic()
        async with gate.admit(CHAT, max_wait_s=1.0) as other:
            assert other is True
        assert time.monotonic() - started < 0.1
    snap = gate.snapshot()
    assert snap[QUICK]["admitted"] == 1 and snap[CHAT]["admitted"] == 1
    assert snap[QUICK]["inflight"] == 0 and snap[CHAT]["inflight"] == 0


@pytest.mark.asyncio
async def test_a_full_lane_sheds_at_the_budget_instead_of_waiting_forever() -> None:
    gate = ua.UpstreamAdmission(max_inflight=1)
    async with gate.admit(QUICK, max_wait_s=1.0):
        adm = gate.admit(QUICK, max_wait_s=0.05)
        async with adm as admitted:
            assert admitted is False
        assert 0.04 <= adm.waited_s < 0.5
    lane = gate.snapshot()[QUICK]
    assert lane["shed"] == 1
    assert lane["admitted"] == 1
    assert lane["waiting"] == 0, "a shed request must not be counted as still waiting"
    assert lane["inflight"] == 0
    assert lane["last_shed_age_s"] is not None


@pytest.mark.asyncio
async def test_a_permit_is_released_on_exception_and_a_shed_request_releases_nothing() -> None:
    gate = ua.UpstreamAdmission(max_inflight=1)
    with pytest.raises(RuntimeError):
        async with gate.admit(QUICK, max_wait_s=1.0):
            raise RuntimeError("upstream blew up")
    assert gate.lane(QUICK).sem._value == 1
    async with gate.admit(QUICK, max_wait_s=1.0):
        async with gate.admit(QUICK, max_wait_s=0.01) as admitted:
            assert admitted is False
        # the shed request must not have released a permit it never held
        assert gate.lane(QUICK).sem._value == 0
    assert gate.lane(QUICK).sem._value == 1


@pytest.mark.asyncio
async def test_a_queued_request_is_admitted_when_the_permit_frees_and_its_wait_is_recorded() -> None:
    gate = ua.UpstreamAdmission(max_inflight=1)

    async def _hold(seconds: float) -> None:
        async with gate.admit(QUICK, max_wait_s=1.0):
            await asyncio.sleep(seconds)

    holder = asyncio.create_task(_hold(0.1))
    await asyncio.sleep(0.01)
    adm = gate.admit(QUICK, max_wait_s=2.0)
    async with adm as admitted:
        assert admitted is True
    await holder
    assert adm.queued is True
    assert adm.waited_s >= 0.05
    assert gate.snapshot()[QUICK]["longest_wait_s"] >= 0.05


def test_executor_is_sized_from_the_route_table_not_the_cpu_count() -> None:
    gate = ua.UpstreamAdmission(max_inflight=8)
    assert gate.executor_workers(4) == 4 * 8 + 4
    assert gate.executor_workers(1, headroom=0) == 8
    assert gate.executor_workers(0) >= 8, "no route table still gets one full lane"


def test_max_inflight_floor_is_one() -> None:
    assert ua.UpstreamAdmission(max_inflight=0).max_inflight == 1
    assert ua.UpstreamAdmission(max_inflight=-3).max_inflight == 1


# ------------------------------------------------------------------ wired into main.py


@pytest.fixture
def _two_lanes(monkeypatch: pytest.MonkeyPatch):
    """Route table with `quick` and `chat` on different upstreams, cap 2 per lane."""
    original = settings.llm_route_table_json
    settings.llm_route_table_json = (
        '{"quick":{"url":"%s","served_by":"fast","backend":"llamacpp"},'
        '"chat":{"url":"%s","served_by":"big","backend":"llamacpp"},'
        '"quick_background":{"url":"%s","served_by":"fast","backend":"llamacpp",'
        '"priority":"background","reserved_free_slots":2}}' % (QUICK, CHAT, QUICK)
    )
    _load_route_targets.cache_clear()
    monkeypatch.setattr(settings, "llm_gateway_upstream_max_inflight", 2)
    ua.reset_upstream_admission_for_tests()
    gw_main.reset_executor_for_tests()
    try:
        yield
    finally:
        settings.llm_route_table_json = original
        _load_route_targets.cache_clear()
        ua.reset_upstream_admission_for_tests()
        gw_main.reset_executor_for_tests()


def _body(route: str) -> ChatBody:
    return ChatBody(route=route, messages=[{"role": "user", "content": "ping"}], trace_id=str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_a_flood_on_quick_does_not_delay_chat(_two_lanes) -> None:
    """The incident shape. 12 slow `quick` requests against a cap of 2 (so 10 queue), one
    `chat` request arriving last. Before this fix all 13 shared one executor FIFO and chat
    waited behind every quick. Now chat must complete in well under one quick's duration."""
    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        if plan.route == "quick":
            time.sleep(0.3)
        return {"text": f"ok:{plan.route}", "raw": {}, "route": plan.route}

    with patch.object(gw_main, "run_llm_chat", fake_run):
        quick_tasks = [
            asyncio.create_task(gw_main._dispatch_chat(_body("quick"), correlation_id=f"q{i}"))
            for i in range(12)
        ]
        await asyncio.sleep(0.02)  # let the flood take its permits and start queueing
        started = time.monotonic()
        chat = await gw_main._dispatch_chat(_body("chat"), correlation_id="chat-1")
        chat_elapsed = time.monotonic() - started
        results = await asyncio.gather(*quick_tasks)

    assert chat["text"] == "ok:chat"
    assert chat_elapsed < 0.25, f"chat waited {chat_elapsed:.3f}s behind the quick flood"
    assert all(r["text"] == "ok:quick" for r in results), "budget is 30s+, nothing should shed"
    snap = ua.get_upstream_admission().snapshot()
    assert snap[QUICK]["admitted"] == 12 and snap[QUICK]["longest_wait_s"] > 0.2
    assert snap[CHAT]["admitted"] == 1 and snap[CHAT]["longest_wait_s"] == 0.0


@pytest.mark.asyncio
async def test_a_request_past_its_own_budget_is_shed_not_generated(_two_lanes, caplog) -> None:
    calls = {"n": 0}

    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        calls["n"] += 1
        time.sleep(0.2)
        return {"text": "ok", "raw": {}, "route": plan.route}

    with patch.object(gw_main, "run_llm_chat", fake_run), \
         patch.object(gw_main, "resolve_caller_budget_sec", lambda body: 0.05):
        tasks = [
            asyncio.create_task(gw_main._dispatch_chat(_body("quick"), correlation_id=f"q{i}"))
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)

    shed = [r for r in results if r.get("raw", {}).get("error") == "gateway_overloaded"]
    served = [r for r in results if r.get("text") == "ok"]
    assert len(served) == 2 and len(shed) == 2, results
    assert calls["n"] == 2, "a shed request must never reach run_llm_chat"
    details = shed[0]["raw"]["details"]
    assert details["upstream"] == QUICK and details["route"] == "quick"
    assert details["stage"] == "upstream_queue"
    assert details["served_by"] == "fast"
    assert details["waited_s"] >= 0.04 and details["budget_s"] == 0.05
    assert details["lane"]["max_inflight"] == 2
    assert shed[0]["served_by"] == "fast" and shed[0]["route"] == "quick"
    assert any("gateway_overloaded" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_background_waits_for_slack_before_taking_a_lane_permit(_two_lanes, monkeypatch) -> None:
    """Order of gates: a background request must poll /slots while holding NOTHING. If it
    held its upstream's permit during the (up to 30s) slack wait, a foreground request on
    the same upstream would queue behind a request that is itself waiting for room."""
    from app import priority_admission as pa

    gate = ua.get_upstream_admission()
    seen: list[int] = []

    async def fake_free(url: str, **kw: Any):
        seen.append(gate.lane(QUICK).inflight)
        return 4

    monkeypatch.setattr(pa, "_free_slot_count", fake_free)
    monkeypatch.setattr(settings, "llm_gateway_background_poll_interval_sec", 0.01)
    monkeypatch.setattr(settings, "llm_gateway_background_max_wait_sec", 1.0)

    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        return {"text": "ok", "raw": {}, "route": plan.route}

    with patch.object(gw_main, "run_llm_chat", fake_run):
        out = await gw_main._dispatch_chat(_body("quick_background"), correlation_id="bg")
    assert out["text"] == "ok"
    assert seen == [0], "slack was polled while a lane permit was held"


@pytest.mark.asyncio
async def test_lane_rejection_short_circuits_before_any_gate(_two_lanes, monkeypatch) -> None:
    from app.llm_backend import ChatDispatchPlan

    err = {"text": "", "raw": {"error": "llm_route_unavailable"}, "route": "spark"}
    monkeypatch.setattr(gw_main, "plan_llm_chat", lambda body: ChatDispatchPlan(body=body, route="spark", error=err))
    with patch.object(gw_main, "run_llm_chat") as run:
        out = await gw_main._dispatch_chat(_body("spark"), correlation_id="x")
    assert out["raw"]["error"] == "llm_route_unavailable"
    run.assert_not_called()
    assert ua.get_upstream_admission().snapshot() == {}


@pytest.mark.asyncio
async def test_admission_endpoint_exposes_lane_depth(_two_lanes) -> None:
    gate = ua.get_upstream_admission()
    async with gate.admit(QUICK, max_wait_s=1.0):
        snap = await gw_main.admission_snapshot()
    assert snap["upstreams"][QUICK]["inflight"] == 1
    assert snap["upstreams"][QUICK]["max_inflight"] == 2
    assert "checked" in snap, "the existing ledger fields must survive"


@pytest.mark.asyncio
async def test_configure_executor_builds_a_dedicated_pool_sized_from_the_route_table(_two_lanes) -> None:
    loop = asyncio.get_running_loop()
    workers = gw_main.configure_executor(loop, ua.get_upstream_admission())
    # two distinct upstreams (quick and quick_background share one) x cap 2 + headroom 4
    assert workers == 2 * 2 + 4
    pool = gw_main._chat_executor
    assert isinstance(pool, ThreadPoolExecutor)
    assert pool._max_workers == workers  # noqa: SLF001
    assert loop._default_executor is not pool, "the chassis heartbeat keeps the stock default executor"  # noqa: SLF001


@pytest.mark.asyncio
async def test_plain_routes_never_enter_the_background_gate(_two_lanes) -> None:
    """The negative half of the gate-order test: `quick`/`chat` must not touch
    background_admission at all (its /slots poll, its per-route semaphore)."""
    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        return {"text": "ok", "raw": {}, "route": plan.route}

    with patch.object(gw_main, "run_llm_chat", fake_run), \
         patch.object(gw_main, "background_admission") as bg:
        for route in ("quick", "chat"):
            out = await gw_main._dispatch_chat(_body(route), correlation_id=route)
            assert out["text"] == "ok"
    bg.assert_not_called()


@pytest.mark.asyncio
async def test_a_background_request_stuck_on_its_route_permit_is_shed_at_budget(_two_lanes, monkeypatch) -> None:
    """Review finding: the background route's own concurrency semaphore is a queue too,
    and it must obey the caller's deadline like every other wait. Hold the
    `quick_background` permit, send a request with a tiny budget, and it must come back
    `gateway_overloaded` stage=background_queue without ever polling /slots or running."""
    from app import priority_admission as pa

    monkeypatch.setattr(pa, "_free_slot_count", pytest.fail)  # must never be reached
    sem = pa._semaphore_for("quick_background", 1)
    await sem.acquire()
    try:
        with patch.object(gw_main, "run_llm_chat") as run, \
             patch.object(gw_main, "resolve_caller_budget_sec", lambda body: 0.05):
            out = await gw_main._dispatch_chat(_body("quick_background"), correlation_id="bg")
    finally:
        sem.release()
    run.assert_not_called()
    assert out["raw"]["error"] == "gateway_overloaded"
    assert out["raw"]["details"]["stage"] == "background_queue"
    assert out["raw"]["details"]["waited_s"] >= 0.04


@pytest.mark.asyncio
async def test_an_8s_caller_gets_an_8s_queue_budget_not_the_30s_read_floor() -> None:
    from app.llm_backend import _resolve_http_read_timeout_sec, resolve_caller_budget_sec

    eight = ChatBody(route="quick", messages=[{"role": "user", "content": "x"}],
                     options={"gateway_read_timeout_sec": 8.0})
    assert resolve_caller_budget_sec(eight) == 8.0
    assert _resolve_http_read_timeout_sec(eight) == 30.0, "the HTTP client floor is unchanged"
    none = ChatBody(route="quick", messages=[{"role": "user", "content": "x"}])
    assert resolve_caller_budget_sec(none) == _resolve_http_read_timeout_sec(none)
    junk = ChatBody(route="quick", messages=[{"role": "user", "content": "x"}],
                    options={"gateway_read_timeout_sec": "nope"})
    assert resolve_caller_budget_sec(junk) == _resolve_http_read_timeout_sec(junk)
    assert resolve_caller_budget_sec(ChatBody(route="quick", messages=[], options={"gateway_read_timeout_sec": -4})) > 0


@pytest.mark.asyncio
async def test_time_spent_queueing_comes_out_of_the_read_timeout(_two_lanes) -> None:
    """Review finding: a request that waited 390s of a 393s budget used to be dispatched
    with a fresh 393s read. The read timeout handed to run_llm_chat must be what is left."""
    seen: list[float] = []

    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        seen.append(float(body.options["gateway_read_timeout_sec"]))
        assert plan.body is body
        time.sleep(0.15)
        return {"text": "ok", "raw": {}, "route": plan.route}

    with patch.object(gw_main, "run_llm_chat", fake_run), \
         patch.object(gw_main, "resolve_caller_budget_sec", lambda body: 1.0):
        # cap is 2: the third request queues behind two 0.15s generations
        tasks = [asyncio.create_task(gw_main._dispatch_chat(_body("quick"), correlation_id=f"q{i}")) for i in range(3)]
        await asyncio.gather(*tasks)
    assert len(seen) == 3
    assert max(seen) <= 1.0
    assert min(seen) <= 0.9, f"the queued request must have had its wait deducted: {seen}"


@pytest.mark.asyncio
async def test_a_permit_admitted_with_nothing_left_is_shed_before_dispatch(_two_lanes) -> None:
    """Edge of the deadline: admitted (lane free) but the budget already expired while
    waiting elsewhere. `asyncio.timeout(0)` does not fire on an uncontended acquire, so
    this needs its own check after admission."""
    with patch.object(gw_main, "run_llm_chat") as run, \
         patch.object(gw_main, "resolve_caller_budget_sec", lambda body: 0.0):
        out = await gw_main._dispatch_chat(_body("quick"), correlation_id="late")
    run.assert_not_called()
    assert out["raw"]["details"]["stage"] == "budget_exhausted"
    assert ua.get_upstream_admission().snapshot()[QUICK]["inflight"] == 0


@pytest.mark.asyncio
async def test_a_cancelled_handler_keeps_its_permit_until_the_thread_finishes(_two_lanes) -> None:
    """Review finding: Rabbit cancels every in-flight handler on reconnect. Cancelling the
    await cannot stop the thread, so releasing the permit on cancel let a lane run 2x its
    cap in real threads while the gauge said otherwise."""
    import threading

    started = threading.Event()
    finished = threading.Event()

    def fake_run(body: ChatBody, plan=None) -> Dict[str, Any]:
        started.set()
        time.sleep(0.2)
        finished.set()
        return {"text": "ok", "raw": {}, "route": plan.route}

    gate = ua.get_upstream_admission()
    with patch.object(gw_main, "run_llm_chat", fake_run):
        task = asyncio.create_task(gw_main._dispatch_chat(_body("quick"), correlation_id="c"))
        await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.snapshot()[QUICK]["inflight"] == 1, "thread still running: permit must still be held"
        await asyncio.to_thread(finished.wait, 1.0)
        await asyncio.sleep(0.05)  # let the done-callback land on the loop
    assert gate.snapshot()[QUICK]["inflight"] == 0
    assert gate.lane(QUICK).sem._value == 2


@pytest.mark.asyncio
async def test_an_unconfigured_route_never_takes_a_permit_or_a_thread(_two_lanes) -> None:
    out = await gw_main._dispatch_chat(_body("nope"), correlation_id="x")
    assert out["raw"]["error"] == "route_not_configured"
    assert ua.get_upstream_admission().snapshot() == {}
