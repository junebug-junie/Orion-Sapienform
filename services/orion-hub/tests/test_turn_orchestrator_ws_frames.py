from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts."):
        del sys.modules[key]
    if key == "app" or key.startswith("app."):
        del sys.modules[key]
for candidate in (REPO_ROOT, HUB_ROOT):
    try:
        sys.path.remove(str(candidate))
    except ValueError:
        pass
for candidate in (REPO_ROOT, HUB_ROOT):
    sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.hub.turn_orchestrator import _success_frames, execute_unified_turn
from orion.schemas.harness_finalize import HarnessRunV1
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    ThoughtEventV1,
)

_CORR_ID = "00000000-0000-4000-8000-000000000201"


def _thought(*, disposition: str = "proceed") -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-orch-1",
        correlation_id=_CORR_ID,
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        disposition=disposition,
        disposition_reasons=["stale_broadcast_no_evidence"] if disposition != "proceed" else [],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )


def _association() -> HubAssociationBundleV1:
    return HubAssociationBundleV1(
        correlation_id=_CORR_ID,
        broadcast=None,
        broadcast_stale=True,
        read_source="felt_state_reader",
    )


def _ensure_hub_import_paths() -> None:
    other_services = tuple(
        p
        for p in REPO_ROOT.glob("services/orion-*")
        if p.is_dir() and p.resolve() != HUB_ROOT.resolve()
    )
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for candidate in (REPO_ROOT, HUB_ROOT, *other_services):
        try:
            sys.path.remove(str(candidate))
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


def _hub_client_patches(*, thought: ThoughtEventV1, harness_run: HarnessRunV1 | AsyncMock):
    _ensure_hub_import_paths()
    import scripts.harness_governor_client as harness_governor_client
    import scripts.thought_client as thought_client

    return (
        patch(
            "orion.hub.turn_orchestrator.build_hub_association_bundle",
            return_value=_association(),
        ),
        patch.object(
            thought_client.ThoughtClient,
            "react",
            AsyncMock(return_value=thought),
        ),
        patch.object(
            harness_governor_client.HarnessGovernorClient,
            "run",
            harness_run if isinstance(harness_run, AsyncMock) else AsyncMock(return_value=harness_run),
        ),
    )


@pytest.mark.asyncio
async def test_turn_orchestrator_defaults_fcc_model_label() -> None:
    harness_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="hello",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    bus = MagicMock()
    harness_client_run = AsyncMock(return_value=harness_run)
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1], patches[2]:
        await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
        )

    harness_client_run.assert_awaited_once()
    req = harness_client_run.await_args.args[0]
    assert req.fcc_model_label == "MODEL_SONNET"


@pytest.mark.asyncio
async def test_turn_orchestrator_publishes_chat_history_on_success() -> None:
    harness_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="final answer",
        finalize_ran=True,
        step_count=2,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    bus = MagicMock()
    bus.enabled = True
    publish_history = AsyncMock()
    publish_turn = AsyncMock()
    publish_spark = AsyncMock()
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_run)
    with patches[0], patches[1], patches[2], patch(
        "scripts.chat_history.publish_chat_history", publish_history
    ), patch(
        "scripts.chat_history.publish_chat_turn", publish_turn
    ), patch(
        "scripts.spark_candidate.publish_spark_introspect_candidate", publish_spark
    ):
        frames = await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="user asks",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
        )

    assert frames[-1]["type"] == "final"
    publish_history.assert_awaited_once()
    publish_turn.assert_awaited_once()
    publish_spark.assert_awaited_once()


@pytest.mark.asyncio
async def test_turn_orchestrator_skips_chat_history_when_no_write() -> None:
    harness_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="final answer",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    bus = MagicMock()
    bus.enabled = True
    publish_turn = AsyncMock()
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_run)
    with patches[0], patches[1], patches[2], patch(
        "scripts.chat_history.publish_chat_turn", publish_turn
    ):
        await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="user asks",
            payload={"no_write": True},
            emit_observation_fn=lambda **_kwargs: None,
        )

    publish_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_turn_orchestrator_never_publishes_draft_text() -> None:
    failed_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text=None,
        draft_text="secret internal draft must not leak",
        finalize_ran=False,
        step_count=2,
        compliance_verdict="failed",
        grounding_status="motor_failed",
    )
    bus = MagicMock()
    patches = _hub_client_patches(thought=_thought(), harness_run=failed_run)
    with patches[0], patches[1], patches[2]:
        frames = await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            emit_observation_fn=lambda **_kwargs: None,
        )

    assert all("draft_text" not in frame for frame in frames)
    assert any(frame.get("type") == "turn_error" for frame in frames)


@pytest.mark.asyncio
async def test_turn_orchestrator_turn_error_on_harness_fail() -> None:
    failed_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text=None,
        draft_text="internal",
        substrate_appraisal=None,
        finalize_ran=False,
        step_count=3,
        compliance_verdict="failed",
        grounding_status="substrate_timeout",
    )
    bus = MagicMock()
    patches = _hub_client_patches(thought=_thought(), harness_run=failed_run)
    with patches[0], patches[1], patches[2]:
        frames = await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            emit_observation_fn=lambda **_kwargs: None,
        )

    assert frames[-1]["type"] == "turn_error"
    assert frames[-1]["phase"] == "substrate_appraisal"
    assert frames[-1]["finalize_ran"] is False


@pytest.mark.asyncio
async def test_turn_orchestrator_turn_deferred_on_stance_defer() -> None:
    bus = MagicMock()
    harness_run = AsyncMock()
    patches = _hub_client_patches(thought=_thought(disposition="defer"), harness_run=harness_run)
    with patches[0], patches[1], patches[2]:
        frames = await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            emit_observation_fn=lambda **_kwargs: None,
        )

    assert frames == [
        {
            "type": "turn_deferred",
            "correlation_id": _CORR_ID,
            "reason": "stale_broadcast_no_evidence",
        }
    ]
    harness_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_turn_orchestrator_passes_empty_answer_contract_not_heuristic() -> None:
    harness_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="hello",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    bus = MagicMock()
    harness_client_run = AsyncMock(return_value=harness_run)
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1], patches[2]:
        await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="docker compose logs show traceback",
            emit_observation_fn=lambda **_kwargs: None,
        )

    req = harness_client_run.await_args.args[0]
    assert req.answer_contract.request_kind == "conceptual"
    assert req.answer_contract.requires_repo_grounding is False
    assert req.answer_contract.requires_runtime_grounding is False


def test_turn_orchestrator_source_has_no_heuristic_answer_contract() -> None:
    source = (REPO_ROOT / "orion/hub/turn_orchestrator.py").read_text(encoding="utf-8")
    assert "heuristic_answer_contract" not in source


def test_success_frames_final_includes_recall_when_present() -> None:
    recall_debug = {
        "source": "pcr_phase3",
        "pcr_ran": True,
        "memory_digest": "recalled: prior turn about X",
    }
    run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="answer",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
        recall_debug=recall_debug,
        memory_digest="recalled: prior turn about X",
    )
    frames = _success_frames(run, correlation_id="corr-1")
    final_frame = next(frame for frame in frames if frame["type"] == "final")
    assert final_frame["recall_debug"] == run.recall_debug
    assert final_frame["memory_digest"] == "recalled: prior turn about X"


def test_success_frames_final_omits_recall_when_absent() -> None:
    run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="answer",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
        recall_debug=None,
        memory_digest=None,
    )
    frames = _success_frames(run, correlation_id="corr-1")
    final_frame = next(frame for frame in frames if frame["type"] == "final")
    assert "recall_debug" not in final_frame
    assert "memory_digest" not in final_frame


@pytest.mark.asyncio
async def test_run_unified_turn_emits_trailing_idle_state_frame() -> None:
    """Regression: the Hub status line is set to 'Sent...' on send and only resets when a
    frame carries state 'idle' (classic lane sends one). The unified terminal frames omit
    state, so run_unified_turn must emit a trailing {'state': 'idle'} to unstick the status."""
    from orion.hub.turn_orchestrator import run_unified_turn

    sent: list[dict] = []

    class _FakeWS:
        async def send_json(self, frame: dict) -> None:
            sent.append(frame)

    frames = [
        {"type": "final", "correlation_id": _CORR_ID, "mode": "orion", "llm_response": "hi"},
    ]
    with patch(
        "orion.hub.turn_orchestrator.execute_unified_turn",
        AsyncMock(return_value=frames),
    ):
        await run_unified_turn(
            _FakeWS(),
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
        )

    assert any(f.get("type") == "final" for f in sent), "final frame must still be sent"
    assert sent[-1] == {"state": "idle"}, "must end with an idle-state frame so status resets to Ready"
