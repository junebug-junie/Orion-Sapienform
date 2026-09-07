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
    assert hops[0]["status"] == "ok"
    assert hops[0]["raw"]["user_message"] == "hi"
    assert hops[0]["summary"]["user_message_len"] == 2
    assert "2" in hops[0]["visor_line"]
    assert hops[0]["raw"].get("observation_published") is False
    assert "deferred_to" not in hops[0].get("summary", {})
    assert hops[1]["status"] == "ok"
    assert hops[1]["raw"]["broadcast_stale"] is True
    assert hops[2]["status"] == "ok"
    assert hops[2]["raw"]["user_message"] == "hi"
    assert hops[3]["stage"] == "stance_decision"
    assert hops[3]["seq"] == 3


def test_emit_progress_hop_failed_status():
    from orion.hub.cockpit_emit import begin_cockpit_timeline, emit_progress_hop

    begin_cockpit_timeline(
        "corr-prog",
        ingress={"user_message": "x", "attachment_count": 0, "observation_published": False},
    )
    frame = emit_progress_hop(
        "corr-prog",
        stage="pre_turn_appraisal",
        status="failed",
        visor_line="appraisal · FAILED TimeoutError",
        summary={"error": "TimeoutError"},
        raw={"error": "TimeoutError"},
    )
    assert frame["hop"]["stage"] == "pre_turn_appraisal"
    assert frame["hop"]["status"] == "failed"
    assert frame["hop"]["seq"] == 1
    assert "FAILED" in frame["hop"]["visor_line"]


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


def test_motor_boot_from_drained_claude_step_before_motor_hop():
    from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
    from orion.hub.cockpit_emit import (
        emit_motor_hop_from_claude_step,
        emit_pre_motor_hops,
    )

    emit_pre_motor_hops(
        "corr-boot",
        _THOUGHT,
        association={"broadcast_stale": True, "read_source": "felt_state_reader"},
        stance_inputs={"user_message": "hi", "stance_inputs": {"user_message": "hi"}},
    )
    boot = emit_motor_hop_from_claude_step(
        "corr-boot",
        {
            "kind": "claude_step",
            "step_index": -1,
            "step": {
                "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
                "prompt": "EXACT PREFIX\nUSER: hi",
                "prompt_char_len": len("EXACT PREFIX\nUSER: hi"),
            },
        },
    )
    assert boot is not None
    assert boot["hop"]["stage"] == "motor_boot"
    assert boot["hop"]["raw"]["prompt"] == "EXACT PREFIX\nUSER: hi"
    assert boot["hop"]["seq"] == 4
    assert boot["hop"]["producer"] == "orion-harness-governor"

    hop = emit_motor_hop_from_claude_step(
        "corr-boot",
        {
            "kind": "claude_step",
            "step_index": 0,
            "step": {"type": "tool_use", "name": "Read"},
        },
    )
    assert hop["hop"]["stage"] == "motor_hop"
    assert hop["hop"]["seq"] == 5


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


@pytest.mark.asyncio
async def test_motor_boot_claude_step_not_relayed_to_live_ws():
    """Synthetic motor_boot steps must become cockpit hops, not live FCC steps."""
    from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
    from orion.hub.turn_orchestrator import run_unified_turn

    sent: list[dict] = []

    class _FakeWS:
        async def send_json(self, frame: dict) -> None:
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
                    "correlation_id": "corr-boot-ws",
                    "step_index": -1,
                    "step": {
                        "_cockpit": COCKPIT_MOTOR_BOOT_MARKER,
                        "prompt": "SYSTEM\n\nUSER\nhi",
                    },
                }
            )
        return _SUCCESS_FINAL

    with patch("orion.hub.turn_orchestrator.execute_unified_turn", _fake_execute):
        await run_unified_turn(
            _FakeWS(),
            bus=MagicMock(),
            correlation_id="corr-boot-ws",
            session_id="sess-1",
            user_message="hello",
            harness_step_relay=_NoopRelay(),
        )

    hop_stages = [f["hop"]["stage"] for f in sent if f.get("kind") == "cockpit_hop"]
    assert "motor_boot" in hop_stages
    boot_claude = [
        f
        for f in sent
        if f.get("kind") == "claude_step"
        and isinstance(f.get("step"), dict)
        and f["step"].get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER
    ]
    assert boot_claude == []
    assert sent[-1] == {"state": "idle"}


def _ensure_hub_paths() -> None:
    import os
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    hub = Path(__file__).resolve().parents[1]
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for candidate in (repo, hub):
        try:
            sys.path.remove(str(candidate))
        except ValueError:
            pass
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(hub))
    os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
    os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
    os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
    os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
    os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def _live_thought():
    from datetime import datetime, timezone

    from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1

    return ThoughtEventV1(
        event_id="t-cockpit-1",
        correlation_id="00000000-0000-4000-8000-000000000301",
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer.",
        tone="calm",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        disposition="proceed",
        disposition_reasons=[],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )


def _live_association():
    from orion.schemas.thought import HubAssociationBundleV1

    return HubAssociationBundleV1(
        correlation_id="00000000-0000-4000-8000-000000000301",
        broadcast=None,
        broadcast_stale=True,
        read_source="felt_state_reader",
    )


@pytest.mark.asyncio
async def test_execute_unified_turn_streams_progress_before_thought_returns():
    """Association + thought_rpc started land before ThoughtClient.react returns."""
    _ensure_hub_paths()
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from orion.hub.turn_orchestrator import execute_unified_turn
    from orion.schemas.harness_finalize import HarnessRunV1
    import scripts.harness_governor_client as harness_governor_client
    import scripts.thought_client as thought_client
    import scripts.pre_turn_appraisal_client as pta_client_mod

    corr = "00000000-0000-4000-8000-000000000301"
    collected: list[dict] = []

    async def sink(frames: list[dict]) -> None:
        collected.extend(frames)

    async def slow_react(*_a, **_k):
        stages_so_far = [
            f["hop"]["stage"]
            for f in collected
            if f.get("kind") == "cockpit_hop" and isinstance(f.get("hop"), dict)
        ]
        assert "association" in stages_so_far
        assert "thought_rpc" in stages_so_far
        thought_hops = [
            f["hop"]
            for f in collected
            if f.get("kind") == "cockpit_hop" and f["hop"].get("stage") == "thought_rpc"
        ]
        assert thought_hops[-1]["status"] == "started"
        return thought_client.ThoughtReactResult(thought=_live_thought())

    harness_entered = asyncio.Event()
    stages_before_harness: list[str] = []

    async def harness_run(*_a, **_k):
        stages_before_harness.extend(
            f["hop"]["stage"]
            for f in collected
            if f.get("kind") == "cockpit_hop" and isinstance(f.get("hop"), dict)
        )
        harness_entered.set()
        return HarnessRunV1(
            correlation_id=corr,
            final_text="hi",
            finalize_ran=True,
            step_count=1,
            compliance_verdict="completed",
            grounding_status="grounded",
        )

    settings = SimpleNamespace(
        ENABLE_PRE_TURN_APPRAISAL=True,
        PRE_TURN_APPRAISAL_PARADIGMS="repair_pressure",
        PRE_TURN_APPRAISAL_TIMEOUT_MS=800,
        ENABLE_UNIFIED_TURN_CHAT_GRAMMAR=False,
    )

    with (
        patch(
            "orion.hub.turn_orchestrator.build_hub_association_bundle",
            return_value=_live_association(),
        ),
        patch.object(
            pta_client_mod.PreTurnAppraisalClient,
            "appraise",
            AsyncMock(return_value=None),
        ),
        patch(
            "scripts.pre_turn_appraisal_wiring._publish_repair_pressure_appraisal",
            AsyncMock(),
        ),
        patch.object(thought_client.ThoughtClient, "react", slow_react),
        patch.object(harness_governor_client.HarnessGovernorClient, "run", harness_run),
        patch(
            "orion.hub.turn_orchestrator._publish_unified_turn_chat_grammar",
            AsyncMock(),
        ),
        patch(
            "orion.hub.turn_orchestrator._build_situation_prompt_fragment",
            AsyncMock(return_value=None),
        ),
    ):
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=corr,
            session_id="sess-1",
            user_message="hello",
            settings=settings,
            cockpit_sink=sink,
        )

    assert harness_entered.is_set()
    assert "harness_dispatch" in stages_before_harness
    assert stages_before_harness[-1] == "harness_dispatch"
    harness_hops = [
        f["hop"]
        for f in collected
        if f.get("kind") == "cockpit_hop" and f["hop"].get("stage") == "harness_dispatch"
    ]
    assert harness_hops
    assert harness_hops[0]["status"] == "ok"
    assert "contacting governor" in harness_hops[0]["visor_line"]
    stages = [
        f["hop"]["stage"]
        for f in collected
        if f.get("kind") == "cockpit_hop" and isinstance(f.get("hop"), dict)
    ]
    assert stages[0] == "ingress"
    ingress_hops = [
        f["hop"]
        for f in collected
        if f.get("kind") == "cockpit_hop" and f["hop"].get("stage") == "ingress"
    ]
    assert len(ingress_hops) == 1
    assert ingress_hops[0]["status"] == "ok"
    assert ingress_hops[0]["raw"]["user_message"] == "hello"
    assert ingress_hops[0]["raw"]["observation_published"] is False
    assert "deferred_to" not in ingress_hops[0].get("summary", {})
    assert "pre_turn_appraisal" in stages
    assert "association" in stages
    assert "thought_rpc" in stages
    assert "mind_enrichment" in stages
    assert "stance_inputs" in stages
    assert "stance_decision" in stages
    assert "harness_dispatch" in stages
    assert stages.index("association") < stages.index("thought_rpc")
    failed_appraisal = [
        f["hop"]
        for f in collected
        if f.get("kind") == "cockpit_hop"
        and f["hop"].get("stage") == "pre_turn_appraisal"
        and f["hop"].get("status") == "failed"
    ]
    assert failed_appraisal
    assert "FAILED" in failed_appraisal[0]["visor_line"]
    assert "appraisal_unavailable" in failed_appraisal[0]["visor_line"]
