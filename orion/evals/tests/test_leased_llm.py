"""orion/evals/leased_llm.py: an eval's calls go only to the gateway under an operator hold, a recall
stops the eval, and the hold is released on every exit path (leased-evals design, Phase 1)."""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx
import pytest

import orion.gpu_pool.client as pool_client
from orion.evals import leased_llm
from orion.evals.leased_llm import (
    LaneCallError, LaneRecalled, LaneUnavailable, operator_class, operator_lane, operator_lane_sync,
)
from orion.gpu_pool.client import Lease, LeaseUnavailable
from orion.llm.resource_lease import GPU_LEASE_HEADER, decode_gpu_lease_header
from orion.schemas.gpu_pool import GpuLeaseGrantV1

GATEWAY = "http://gateway.test:8210"


class FakePool:
    """Stands in for ``orion.gpu_pool.client.gpu_lease``: records what was asked and how it ended."""

    def __init__(self, refuse: str | None = None):
        self.asked: dict[str, Any] = {}
        self.exits: list[str | None] = []
        self.lease: Lease | None = None
        self.refuse = refuse

    def __call__(self, bus: Any, **kw: Any):
        self.asked = kw
        pool = self

        @contextlib.asynccontextmanager
        async def cm():
            if pool.refuse:
                raise LeaseUnavailable(pool.refuse)
            pool.lease = Lease("hold-1", GpuLeaseGrantV1(lease_id="hold-1", generation=2, role="chat", cards=["gpu0"],
                                                          url="http://worker.invalid:8011", served_by="w"))
            try:
                yield pool.lease
            except BaseException as exc:
                pool.exits.append(pool.lease.release_outcome or "upstream_error")
                raise
            else:
                pool.exits.append(pool.lease.release_outcome or "ok")

        return cm()


class Gateway:
    def __init__(self, status: int = 200, body: Any = None):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = body if body is not None else {"choices": [{"message": {"content": "{}"}}]}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(pool_client, "gpu_lease", fake)

    async def no_bus(url):
        return _Bus()
    monkeypatch.setattr(leased_llm, "_connect_bus", no_bus)
    return fake


class _Bus:
    async def close(self):
        pass


def _sync(gw: Gateway, **kw):
    return operator_lane_sync("chat", holder="operator:test-eval", gateway=GATEWAY,
                              http_transport=httpx.MockTransport(gw), **kw)


def test_lane_names_map_to_operator_classes():
    assert operator_class("chat") == "operator_chat" and operator_class("agent-gpu2") == "operator_agent_gpu2"
    with pytest.raises(ValueError):
        operator_class("gpu0")


def test_hold_is_background_operator_class_and_every_call_goes_to_the_gateway_with_the_hold(pool):
    gw = Gateway()
    with _sync(gw) as lane:
        lane.chat({"messages": [{"role": "user", "content": "hi"}], "model": "something-else"})
        lane.messages({"messages": []})
        assert lane.summary()["work_class"] == "operator_chat" and lane.summary()["calls"] == 2
    assert pool.asked["work_class"] == "operator_chat"
    assert (pool.asked["priority"], pool.asked["kind"], pool.asked["holder"]) == ("background", "hold",
                                                                                  "operator:test-eval")
    assert [str(r.url) for r in gw.requests] == [f"{GATEWAY}/v1/chat/completions", f"{GATEWAY}/v1/messages"]
    for r in gw.requests:
        assert json.loads(r.content)["model"] == "operator_chat"   # the route, never the worker's model
        ref = decode_gpu_lease_header(r.headers[GPU_LEASE_HEADER])
        assert (ref.lease_id, ref.generation, ref.role, ref.holder) == ("hold-1", 2, "chat", "operator:test-eval")
    assert pool.exits == ["ok"]


def test_a_recall_stops_the_eval_and_releases_the_hold_as_cancelled(pool):
    gw = Gateway()
    with pytest.raises(LaneRecalled):
        with _sync(gw) as lane:
            lane.chat({"messages": []})
            pool.lease.recalled.set()   # the pool's heartbeat answered "recall"
            lane.chat({"messages": []})
    assert len(gw.requests) == 1        # no call after the recall
    assert pool.exits == ["cancelled"]


def test_a_gateway_refusal_for_a_gone_hold_is_a_recall_not_a_call_error(pool):
    gw = Gateway(503, {"error": {"type": "gpu_pool_unavailable", "reason": "hold_not_granted:released"}})
    with pytest.raises(LaneRecalled):
        with _sync(gw) as lane:
            lane.chat({"messages": []})
    assert pool.exits == ["cancelled"]


def test_an_ordinary_failure_is_a_call_error_and_the_eval_may_continue(pool):
    gw = Gateway(500, {"error": {"type": "upstream_error"}})
    with _sync(gw) as lane:
        with pytest.raises(LaneCallError) as err:
            lane.chat({"messages": []})
        assert err.value.status == 500
    assert pool.exits == ["ok"]


def test_the_hold_is_released_when_the_eval_raises(pool):
    with pytest.raises(KeyError):
        with _sync(Gateway()):
            raise KeyError("boom")
    assert pool.exits == ["upstream_error"]


def test_a_refused_hold_is_lane_unavailable_and_nothing_is_called(monkeypatch):
    fake = FakePool(refuse="deadline")
    monkeypatch.setattr(pool_client, "gpu_lease", fake)

    async def no_bus(url):
        return _Bus()
    monkeypatch.setattr(leased_llm, "_connect_bus", no_bus)
    gw = Gateway()
    with pytest.raises(LaneUnavailable, match="deadline"):
        with _sync(gw):
            pass
    assert gw.requests == []


def test_holder_must_name_operator_work():
    async def go():
        async with operator_lane("chat", holder="orion-mind", bus=_Bus()):
            pass
    with pytest.raises(ValueError):
        asyncio.run(go())


def test_async_lane_releases_on_recall(pool):
    async def go():
        async with operator_lane("chat", holder="operator:t", bus=_Bus(), gateway=GATEWAY,
                                 http_transport=httpx.MockTransport(Gateway())) as lane:
            pool.lease.recalled.set()
            pool.lease.lost.set()
            lane.check()
    with pytest.raises(LaneRecalled, match="lost"):
        asyncio.run(go())
    assert pool.exits == ["cancelled"]


def test_no_bus_url_is_a_clear_refusal(monkeypatch):
    monkeypatch.delenv("ORION_BUS_URL", raising=False)
    with pytest.raises(LaneUnavailable, match="ORION_BUS_URL"):
        asyncio.run(leased_llm._connect_bus(None))


def test_the_helper_names_no_worker_address():
    """The only URL in the helper is the gateway default: no circe address, no worker port."""
    import inspect
    import re

    src = inspect.getsource(leased_llm)
    urls = set(re.findall(r"https?://[\w.\-]+:\d+", src))
    assert urls == {"http://127.0.0.1:8210"}, urls


def test_operator_lanes_take_a_hold_only_for_a_lane_the_run_uses(pool):
    from orion.evals.leased_llm import OperatorLanes

    gw = Gateway()
    with OperatorLanes(holder="operator:t", gateway=GATEWAY, http_transport=httpx.MockTransport(gw)) as lanes:
        lanes.get("agent").chat({"messages": []})
        lanes.get("agent").chat({"messages": []})
        assert [s["work_class"] for s in lanes.summary()] == ["operator_agent"]
    assert pool.asked["work_class"] == "operator_agent"
    assert pool.exits == ["ok"]


def test_ctrl_c_while_queued_withdraws_the_queued_hold(monkeypatch):
    """Review finding: an interrupt during the queue wait must cancel the acquire (gpu_lease then
    withdraws the queued lease), not leave it queued for hours."""
    import threading

    withdrawn = []

    def queued_forever(bus, **kw):
        @contextlib.asynccontextmanager
        async def cm():
            try:
                await asyncio.sleep(3600)   # waiting in the pool's queue
            except asyncio.CancelledError:
                withdrawn.append(True)      # gpu_lease's _acquire withdraws here
                raise
            yield None
        return cm()

    monkeypatch.setattr(pool_client, "gpu_lease", queued_forever)

    async def no_bus(url):
        return _Bus()
    monkeypatch.setattr(leased_llm, "_connect_bus", no_bus)

    class InterruptingEvent(threading.Event):
        def wait(self, timeout=None):
            super().wait(0.2)
            raise KeyboardInterrupt

    monkeypatch.setattr(leased_llm, "_ready_event", InterruptingEvent)
    with pytest.raises(KeyboardInterrupt):
        with operator_lane_sync("chat", holder="operator:t", gateway=GATEWAY):
            pass
    assert withdrawn == [True]


def test_ctrl_c_in_the_eval_body_releases_as_cancelled(pool):
    with pytest.raises(KeyboardInterrupt):
        with _sync(Gateway()):
            raise KeyboardInterrupt
    assert pool.exits == ["cancelled"]


def test_operator_class_match_is_exact():
    from orion.gpu_pool.value_order import is_operator_class

    assert not is_operator_class("operators_x") and not is_operator_class("operator-ish")
