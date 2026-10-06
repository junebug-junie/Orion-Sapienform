"""Unified-turn latency L4: stance_context_prepare (producer side, orion-thought).

orion-thought sends the prepare at the same time as the orion-mind call, on the
prepare channel of the same exec lane stance_react uses, and marks the
stance_react request so cortex-exec uses the prepared context.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.stance_context_prepare import (
    STANCE_CONTEXT_PREPARE_REQUEST_KIND,
    STANCE_CONTEXT_PREPARE_RESULT_KIND,
    STANCE_PREPARE_REQUESTED_CTX_KEY,
    StanceContextPrepareRequestV1,
    StanceContextPrepareResultV1,
)
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

CORR = "7f0c2a52-6a8f-4b7e-9d1f-2b0e0e6c1a11"


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id=CORR,
        session_id="sess-1",
        user_message="how are you today?",
        association=HubAssociationBundleV1(
            correlation_id=CORR,
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how are you today?"},
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        f'"strain_refs":["hub:turn:{CORR}"],"evidence_refs":["hub:turn:{CORR}"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


class _Client:
    def __init__(self, request_channel: str, events: list[str]) -> None:
        self.request_channel = request_channel
        self.events = events
        self.context: dict[str, Any] | None = None

    async def execute_plan(self, *, req, **_kwargs) -> dict:
        self.events.append("stance_sent")
        self.context = req.context
        return {
            "final_text": _stance_json(),
            "metadata": {"stance_prepare_overlap": {"outcome": "used", "wait_ms": 0.0, "build_ms": 9000.0}},
        }


def _reload(monkeypatch, *, flag: str, mind: str = "true"):
    monkeypatch.setenv("ORION_THOUGHT_STANCE_PREPARE_PARALLEL", flag)
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", mind)
    import app.settings as s

    importlib.reload(s)
    import app.mind_enrichment as me

    importlib.reload(me)
    import app.bus_listener as bl

    importlib.reload(bl)
    return bl


def _wire(monkeypatch, bl, events: list[str], sent: list[dict]):
    async def _mind(*_a, **_k):
        events.append("mind_start")
        await asyncio.sleep(0.05)
        events.append("mind_done")
        return None

    async def _send(request, *, request_channel, bus=None):
        events.append("prepare_sent")
        sent.append({"request": request, "request_channel": request_channel})
        return StanceContextPrepareResultV1(correlation_id=request.correlation_id, status="ready", build_ms=42.0)

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(bl, "send_stance_context_prepare", _send)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane_channel", ["orion:cortex:exec:request", "orion:cortex:exec:request:chat"])
async def test_prepare_fires_with_mind_on_the_stance_lane(monkeypatch, caplog, lane_channel) -> None:
    bl = _reload(monkeypatch, flag="true")
    events: list[str] = []
    sent: list[dict] = []
    _wire(monkeypatch, bl, events, sent)
    client = _Client(lane_channel, events)

    with caplog.at_level(logging.INFO, logger="orion-thought.bus"):
        thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)

    assert thought.imperative == "Stay present with Juniper."
    # Sent before mind finished, i.e. overlapping it; stance only after mind.
    assert events.index("prepare_sent") < events.index("mind_done") < events.index("stance_sent")
    assert sent[0]["request_channel"] == lane_channel
    assert client.context[STANCE_PREPARE_REQUESTED_CTX_KEY] is True
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("stance_prepare_overlap"))
    assert f"corr={CORR}" in line and "mind_ms=" in line
    assert "build_ms=42.0" in line and "wait_ms=0.0" in line and "outcome=used" in line


@pytest.mark.asyncio
async def test_flag_off_sends_no_prepare_and_no_marker(monkeypatch) -> None:
    bl = _reload(monkeypatch, flag="false")
    events: list[str] = []
    sent: list[dict] = []
    _wire(monkeypatch, bl, events, sent)
    client = _Client("orion:cortex:exec:request", events)
    await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert sent == []
    assert STANCE_PREPARE_REQUESTED_CTX_KEY not in client.context


@pytest.mark.asyncio
async def test_unknown_exec_channel_sends_no_prepare(monkeypatch) -> None:
    bl = _reload(monkeypatch, flag="true")
    events: list[str] = []
    sent: list[dict] = []
    _wire(monkeypatch, bl, events, sent)
    client = _Client("orion:custom:exec", events)
    await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert sent == []
    assert STANCE_PREPARE_REQUESTED_CTX_KEY not in client.context


class _FakeBus:
    def __init__(self, *, fail: bool = False) -> None:
        self.codec = OrionCodec()
        self.fail = fail
        self.calls: list[dict] = []

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.calls.append({"channel": channel, "env": env, "reply_channel": reply_channel})
        if self.fail:
            raise TimeoutError("no reply")
        reply = BaseEnvelope(
            kind=STANCE_CONTEXT_PREPARE_RESULT_KIND,
            source=ServiceRef(name="cortex-exec", node="n", version="1"),
            correlation_id=env.correlation_id,
            payload=StanceContextPrepareResultV1(
                correlation_id=CORR, status="ready", build_ms=8800.0, lane="chat"
            ).model_dump(mode="json"),
        )
        return {"data": self.codec.encode(reply)}


@pytest.mark.asyncio
async def test_send_builds_the_registered_request_on_the_lane_prepare_channel(monkeypatch) -> None:
    bl = _reload(monkeypatch, flag="true")
    bus = _FakeBus()
    result = await bl.send_stance_context_prepare(
        _request(), request_channel="orion:cortex:exec:request:chat", bus=bus
    )
    assert result is not None and result.status == "ready" and result.build_ms == 8800.0
    call = bus.calls[0]
    assert call["channel"] == "orion:cortex:exec:stance_prepare:chat"
    assert call["reply_channel"].startswith("orion:cortex:exec:stance_prepare_result:")
    env = call["env"]
    assert env.kind == STANCE_CONTEXT_PREPARE_REQUEST_KIND
    payload = StanceContextPrepareRequestV1.model_validate(env.payload)
    assert payload.correlation_id == CORR
    assert payload.plan_request.plan.verb_name == "stance_react"
    ctx = payload.plan_request.context
    # The same ctx stance_react sends, minus mind coloring and the marker.
    assert ctx["user_message"] == "how are you today?"
    assert "mind_coloring" not in ctx
    assert STANCE_PREPARE_REQUESTED_CTX_KEY not in ctx


@pytest.mark.asyncio
async def test_send_failure_is_fail_open(monkeypatch) -> None:
    bl = _reload(monkeypatch, flag="true")
    result = await bl.send_stance_context_prepare(
        _request(), request_channel="orion:cortex:exec:request", bus=_FakeBus(fail=True)
    )
    assert result is None


@pytest.mark.asyncio
async def test_prepare_task_is_cancelled_when_the_turn_fails_before_stance(monkeypatch) -> None:
    bl = _reload(monkeypatch, flag="true")
    events: list[str] = []
    started = asyncio.Event()
    holder: dict[str, asyncio.Task] = {}

    async def _slow_send(request, *, request_channel, bus=None):
        holder["task"] = asyncio.current_task()
        started.set()
        await asyncio.sleep(30)

    async def _mind_boom(*_a, **_k):
        await started.wait()
        raise asyncio.CancelledError()

    monkeypatch.setattr(bl, "send_stance_context_prepare", _slow_send)
    monkeypatch.setattr(bl, "_maybe_build_mind_coloring", _mind_boom)
    client = _Client("orion:cortex:exec:request", events)
    with pytest.raises(asyncio.CancelledError):
        await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    await asyncio.sleep(0)
    assert holder["task"].cancelled() or holder["task"].cancelling()
    assert "stance_sent" not in events
