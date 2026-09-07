"""Cockpit hop WS frames built from the unified-turn emit helpers."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_THOUGHT = {
    "disposition": "proceed",
    "disposition_reasons": ["x"],
    "imperative": "go",
    "tone": "calm",
}

_MIN_ASSOC: dict = {}
_MIN_STANCE: dict = {}


def test_orchestrator_emits_thick_pre_motor_hops():
    from orion.hub.cockpit_emit import emit_pre_motor_hops

    frames = emit_pre_motor_hops(
        "corr-1",
        _THOUGHT,
        association={
            "schema_version": "hub.association.bundle.v1",
            "correlation_id": "corr-1",
            "broadcast_stale": True,
            "broadcast": None,
            "execution_trajectory_slice": None,
            "repair_bundle": None,
            "read_source": "felt_state_reader",
        },
        stance_inputs={
            "user_message": "hi",
            "session_id": None,
            "llm_profile": "brain",
            "stance_inputs": {"user_message": "hi"},
        },
    )
    hops = [f["hop"] for f in frames if f["kind"] == "cockpit_hop"]
    stages = [h["stage"] for h in hops]
    assert stages == ["ingress", "association", "stance_inputs", "stance_decision"]
    assert hops[0]["status"] == "gap"
    assert hops[0]["summary"]["deferred_to"] == "slice_c"
    assert hops[1]["status"] == "ok"
    assert hops[1]["raw"]["broadcast_stale"] is True
    assert hops[2]["status"] == "ok"
    assert hops[2]["raw"]["user_message"] == "hi"
    assert hops[3]["stage"] == "stance_decision"
    assert hops[3]["seq"] == 3


def test_motor_hop_from_drained_claude_step_increments_seq():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_pre_motor_hops,
    )

    emit_pre_motor_hops(
        "corr-1", _THOUGHT, association=_MIN_ASSOC, stance_inputs=_MIN_STANCE
    )
    frame = emit_motor_hop_from_claude_step(
        "corr-1",
        {
            "kind": "claude_step",
            "correlation_id": "corr-1",
            "step_index": 2,
            "step": {"type": "tool_use", "name": "Read"},
        },
    )
    assert frame is not None
    assert frame["kind"] == "cockpit_hop"
    assert frame["correlation_id"] == "corr-1"
    assert frame["hop"]["stage"] == "motor_hop"
    assert frame["hop"]["seq"] == 4
    assert frame["hop"]["summary"]["step_index"] == 2


def test_motor_hop_helper_ignores_non_claude_step():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_pre_motor_hops,
    )

    emit_pre_motor_hops(
        "corr-1", _THOUGHT, association=_MIN_ASSOC, stance_inputs=_MIN_STANCE
    )
    assert emit_motor_hop_from_claude_step("corr-1", {"kind": "other"}) is None


def test_finalize_hops_from_run_artifact_and_timeline_complete():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_slice_a_finalize_hops,
        emit_pre_motor_hops,
    )

    emit_pre_motor_hops(
        "corr-1", _THOUGHT, association=_MIN_ASSOC, stance_inputs=_MIN_STANCE
    )
    emit_motor_hop_from_claude_step(
        "corr-1",
        {
            "kind": "claude_step",
            "step_index": 0,
            "step": {"type": "assistant"},
        },
    )
    frames = emit_slice_a_finalize_hops(
        "corr-1",
        {
            "draft_text": "substrate read",
            "reflection": "wrapped up",
            "final_text": "hello",
            "finalize_ran": True,
            "compliance_verdict": "completed",
        },
    )
    hops = [f for f in frames if f.get("kind") == "cockpit_hop"]
    stages = [f["hop"]["stage"] for f in hops]
    assert stages == ["draft_appraisal", "finalize"]
    assert hops[0]["hop"]["seq"] == 5
    assert hops[1]["hop"]["seq"] == 6
    assert frames[-1] == {
        "kind": "cockpit_timeline_complete",
        "correlation_id": "corr-1",
    }


def test_finalize_hops_emit_outcome_when_present():
    from orion.cockpit.sequencer import reset_seq
    from orion.hub.cockpit_emit import emit_slice_a_finalize_hops

    reset_seq("corr-out")
    frames = emit_slice_a_finalize_hops(
        "corr-out",
        {
            "final_text": "done",
            "finalize_ran": True,
            "outcome": {"status": "ok"},
        },
    )
    stages = [f["hop"]["stage"] for f in frames if f.get("kind") == "cockpit_hop"]
    assert "finalize" in stages
    assert "closure" in stages


@pytest.mark.asyncio
async def test_publish_cockpit_frames_fail_open():
    from orion.hub.cockpit_emit import emit_pre_motor_hops, publish_cockpit_frames

    frames = emit_pre_motor_hops(
        "corr-pub", _THOUGHT, association=_MIN_ASSOC, stance_inputs=_MIN_STANCE
    )
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    await publish_cockpit_frames(bus, frames)


@pytest.mark.asyncio
async def test_publish_cockpit_frames_skips_when_bus_missing():
    from orion.hub.cockpit_emit import emit_pre_motor_hops, publish_cockpit_frames

    frames = emit_pre_motor_hops(
        "corr-nobus", _THOUGHT, association=_MIN_ASSOC, stance_inputs=_MIN_STANCE
    )
    await publish_cockpit_frames(None, frames)


@pytest.mark.asyncio
async def test_deliver_cockpit_frames_fail_open_on_sink_error():
    from orion.hub.turn_orchestrator import _deliver_cockpit_frames

    async def boom(_frames: list[dict]) -> None:
        raise RuntimeError("websocket closed")

    await _deliver_cockpit_frames(
        [{"kind": "cockpit_hop", "correlation_id": "corr-ws"}],
        bus=MagicMock(),
        cockpit_sink=boom,
    )


@pytest.mark.asyncio
async def test_pre_motor_ws_send_failure_does_not_abort_turn():
    from orion.hub.cockpit_emit import emit_pre_motor_hops
    from orion.hub.turn_orchestrator import run_unified_turn

    class _BoomOnPreMotor:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, frame: dict) -> None:
            hop = frame.get("hop") if isinstance(frame.get("hop"), dict) else {}
            if frame.get("kind") == "cockpit_hop" and hop.get("stage") == "ingress":
                raise RuntimeError("websocket closed")
            self.sent.append(frame)

    ws = _BoomOnPreMotor()

    async def _fake_execute(**kwargs):
        holder = kwargs.get("cockpit_run_holder")
        if isinstance(holder, dict):
            holder["run"] = _run_dump()
        sink = kwargs.get("cockpit_sink")
        if sink is not None:
            await sink(
                emit_pre_motor_hops(
                    "corr-ws",
                    _THOUGHT,
                    association=_MIN_ASSOC,
                    stance_inputs=_MIN_STANCE,
                )
            )
        return _SUCCESS_FINAL

    with patch("orion.hub.turn_orchestrator.execute_unified_turn", _fake_execute):
        await run_unified_turn(
            ws,
            bus=MagicMock(),
            correlation_id="corr-ws",
            session_id="sess-1",
            user_message="hello",
        )

    assert any(f.get("type") == "final" for f in ws.sent)
    assert ws.sent[-1] == {"state": "idle"}


@pytest.mark.asyncio
async def test_run_unified_turn_converts_drained_claude_step_to_cockpit_hop():
    from orion.hub.turn_orchestrator import run_unified_turn

    sent: list[dict] = []

    class _FakeWS:
        async def send_json(self, frame: dict) -> None:
            sent.append(frame)

    class _FakeRelay:
        def register_queue(self, correlation_id: str, queue) -> None:
            return None

        def unregister_queue(self, correlation_id: str, queue) -> None:
            return None

        def forget(self, correlation_id: str) -> None:
            return None

    success = [
        {
            "type": "final",
            "correlation_id": "corr-ws",
            "mode": "orion",
            "llm_response": "hi",
        }
    ]

    from orion.hub.cockpit_emit import emit_pre_motor_hops

    async def _fake_execute(**kwargs):
        holder = kwargs.get("cockpit_run_holder")
        if isinstance(holder, dict):
            holder["run"] = {
                "draft_text": "draft",
                "final_text": "hi",
                "finalize_ran": True,
                "compliance_verdict": "completed",
            }
        sink = kwargs.get("cockpit_sink")
        if sink is not None:
            await sink(
                emit_pre_motor_hops(
                    "corr-ws",
                    _THOUGHT,
                    association=_MIN_ASSOC,
                    stance_inputs=_MIN_STANCE,
                )
            )
        queue = kwargs.get("harness_step_queue")
        if queue is not None:
            queue.put_nowait(
                {
                    "kind": "claude_step",
                    "mode": "orion",
                    "correlation_id": "corr-ws",
                    "step_index": 0,
                    "step": {"type": "assistant", "name": "think"},
                }
            )
        return success

    with patch(
        "orion.hub.turn_orchestrator.execute_unified_turn",
        _fake_execute,
    ):
        await run_unified_turn(
            _FakeWS(),
            bus=MagicMock(),
            correlation_id="corr-ws",
            session_id="sess-1",
            user_message="hello",
            harness_step_relay=_FakeRelay(),
        )

    kinds = [f.get("kind") for f in sent]
    assert "claude_step" in kinds
    hop_stages = [f["hop"]["stage"] for f in sent if f.get("kind") == "cockpit_hop"]
    assert hop_stages[:4] == ["ingress", "association", "stance_inputs", "stance_decision"]
    assert "stance_decision" in hop_stages
    assert "motor_hop" in hop_stages
    assert "draft_appraisal" in hop_stages or "finalize" in hop_stages
    assert {"kind": "cockpit_timeline_complete", "correlation_id": "corr-ws"} in sent
    assert sent[-1] == {"state": "idle"}


class _CollectingWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, frame: dict) -> None:
        self.sent.append(frame)


class _NoopRelay:
    def register_queue(self, correlation_id: str, queue) -> None:
        return None

    def unregister_queue(self, correlation_id: str, queue) -> None:
        return None

    def forget(self, correlation_id: str) -> None:
        return None


_SUCCESS_FINAL = [
    {
        "type": "final",
        "correlation_id": "corr-ws",
        "mode": "orion",
        "llm_response": "hi",
    }
]


def _run_dump() -> dict:
    return {
        "draft_text": "draft",
        "final_text": "hi",
        "finalize_ran": True,
        "compliance_verdict": "completed",
    }


@pytest.mark.asyncio
async def test_finalize_hop_failure_does_not_block_idle_or_success():
    from orion.hub.turn_orchestrator import run_unified_turn

    ws = _CollectingWS()

    async def _fake_execute(**kwargs):
        holder = kwargs.get("cockpit_run_holder")
        if isinstance(holder, dict):
            holder["run"] = _run_dump()
        return _SUCCESS_FINAL

    with (
        patch("orion.hub.turn_orchestrator.execute_unified_turn", _fake_execute),
        patch(
            "orion.hub.turn_orchestrator.emit_slice_a_finalize_hops",
            side_effect=RuntimeError("finalize hop boom"),
        ),
    ):
        await run_unified_turn(
            ws,
            bus=MagicMock(),
            correlation_id="corr-ws",
            session_id="sess-1",
            user_message="hello",
        )

    assert any(f.get("type") == "final" for f in ws.sent)
    assert ws.sent[-1] == {"state": "idle"}


@pytest.mark.asyncio
async def test_motor_hop_emit_failure_does_not_block_idle_or_success():
    from orion.hub.turn_orchestrator import run_unified_turn

    ws = _CollectingWS()

    async def _fake_execute(**kwargs):
        holder = kwargs.get("cockpit_run_holder")
        if isinstance(holder, dict):
            holder["run"] = _run_dump()
        queue = kwargs.get("harness_step_queue")
        if queue is not None:
            queue.put_nowait(
                {
                    "kind": "claude_step",
                    "mode": "orion",
                    "correlation_id": "corr-ws",
                    "step_index": 0,
                    "step": {"type": "assistant", "name": "think"},
                }
            )
        return _SUCCESS_FINAL

    with (
        patch("orion.hub.turn_orchestrator.execute_unified_turn", _fake_execute),
        patch(
            "orion.hub.turn_orchestrator.emit_motor_hop_from_claude_step",
            side_effect=RuntimeError("motor hop boom"),
        ),
    ):
        await run_unified_turn(
            ws,
            bus=MagicMock(),
            correlation_id="corr-ws",
            session_id="sess-1",
            user_message="hello",
            harness_step_relay=_NoopRelay(),
        )

    assert any(f.get("type") == "final" for f in ws.sent)
    assert any(f.get("kind") == "claude_step" for f in ws.sent)
    assert ws.sent[-1] == {"state": "idle"}


@pytest.mark.asyncio
async def test_drain_shutdown_does_not_drop_queued_motor_hop():
    from orion.hub.turn_orchestrator import run_unified_turn

    sent: list[dict] = []
    emit_started = asyncio.Event()

    class _SlowMotorHopWS:
        async def send_json(self, frame: dict) -> None:
            hop = frame.get("hop") if isinstance(frame.get("hop"), dict) else {}
            if frame.get("kind") == "cockpit_hop" and hop.get("stage") == "motor_hop":
                emit_started.set()
                await asyncio.sleep(0.2)
            sent.append(frame)

    async def _fake_execute(**kwargs):
        holder = kwargs.get("cockpit_run_holder")
        if isinstance(holder, dict):
            holder["run"] = _run_dump()
        queue = kwargs.get("harness_step_queue")
        if queue is not None:
            queue.put_nowait(
                {
                    "kind": "claude_step",
                    "mode": "orion",
                    "correlation_id": "corr-ws",
                    "step_index": 0,
                    "step": {"type": "assistant", "name": "think"},
                }
            )
            await emit_started.wait()
        return _SUCCESS_FINAL

    with patch("orion.hub.turn_orchestrator.execute_unified_turn", _fake_execute):
        await run_unified_turn(
            _SlowMotorHopWS(),
            bus=MagicMock(),
            correlation_id="corr-ws",
            session_id="sess-1",
            user_message="hello",
            harness_step_relay=_NoopRelay(),
        )

    hop_stages = [f["hop"]["stage"] for f in sent if f.get("kind") == "cockpit_hop"]
    assert "motor_hop" in hop_stages
    assert any(f.get("kind") == "claude_step" for f in sent)
    assert sent[-1] == {"state": "idle"}
