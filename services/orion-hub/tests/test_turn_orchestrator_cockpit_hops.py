"""Cockpit hop WS frames built from the unified-turn emit helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_THOUGHT = {
    "disposition": "proceed",
    "disposition_reasons": ["x"],
    "imperative": "go",
    "tone": "calm",
}


def test_orchestrator_emits_gap_then_stance_cockpit_hops():
    from orion.hub.cockpit_emit import emit_slice_a_pre_motor_hops

    frames = emit_slice_a_pre_motor_hops("corr-1", _THOUGHT)
    kinds = [f.get("kind") for f in frames]
    assert kinds.count("cockpit_hop") >= 5  # 4 gaps + stance
    stages = [f["hop"]["stage"] for f in frames if f["kind"] == "cockpit_hop"]
    assert stages[:4] == ["ingress", "association", "stance_inputs", "motor_boot"]
    assert stages[4] == "stance_decision"
    assert frames[0]["correlation_id"] == "corr-1"
    assert frames[0]["hop"]["seq"] == 0
    assert frames[4]["hop"]["seq"] == 4
    assert frames[4]["hop"]["status"] == "ok"
    assert frames[0]["hop"]["status"] == "gap"


def test_motor_hop_from_drained_claude_step_increments_seq():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_slice_a_pre_motor_hops,
    )

    emit_slice_a_pre_motor_hops("corr-1", _THOUGHT)
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
    assert frame["hop"]["seq"] == 5
    assert frame["hop"]["summary"]["step_index"] == 2


def test_motor_hop_helper_ignores_non_claude_step():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_slice_a_pre_motor_hops,
    )

    emit_slice_a_pre_motor_hops("corr-1", _THOUGHT)
    assert emit_motor_hop_from_claude_step("corr-1", {"kind": "other"}) is None


def test_finalize_hops_from_run_artifact_and_timeline_complete():
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_slice_a_finalize_hops,
        emit_slice_a_pre_motor_hops,
    )

    emit_slice_a_pre_motor_hops("corr-1", _THOUGHT)
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
    assert hops[0]["hop"]["seq"] == 6
    assert hops[1]["hop"]["seq"] == 7
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
    from orion.hub.cockpit_emit import emit_slice_a_pre_motor_hops, publish_cockpit_frames

    frames = emit_slice_a_pre_motor_hops("corr-pub", _THOUGHT)
    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    await publish_cockpit_frames(bus, frames)


@pytest.mark.asyncio
async def test_publish_cockpit_frames_skips_when_bus_missing():
    from orion.hub.cockpit_emit import emit_slice_a_pre_motor_hops, publish_cockpit_frames

    frames = emit_slice_a_pre_motor_hops("corr-nobus", _THOUGHT)
    await publish_cockpit_frames(None, frames)


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

    from orion.hub.cockpit_emit import emit_slice_a_pre_motor_hops

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
            await sink(emit_slice_a_pre_motor_hops("corr-ws", _THOUGHT))
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
    assert hop_stages[:4] == ["ingress", "association", "stance_inputs", "motor_boot"]
    assert "stance_decision" in hop_stages
    assert "motor_hop" in hop_stages
    assert "draft_appraisal" in hop_stages or "finalize" in hop_stages
    assert {"kind": "cockpit_timeline_complete", "correlation_id": "corr-ws"} in sent
    assert sent[-1] == {"state": "idle"}
