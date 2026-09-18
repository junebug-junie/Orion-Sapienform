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

from orion.hub.turn_orchestrator import execute_unified_turn, run_unified_turn
from orion.schemas.harness_finalize import HarnessRunV1
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    ThoughtEventV1,
)

_CORR_ID = "00000000-0000-4000-8000-000000000301"


def _thought() -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-origin-1",
        correlation_id=_CORR_ID,
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
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
            AsyncMock(return_value=thought_client.ThoughtReactResult(thought=thought)),
        ),
        patch.object(
            harness_governor_client.HarnessGovernorClient,
            "run",
            harness_run if isinstance(harness_run, AsyncMock) else AsyncMock(return_value=harness_run),
        ),
    )


def _harness_run() -> HarnessRunV1:
    return HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="hello",
        finalize_ran=True,
        step_count=1,
        compliance_verdict="completed",
        grounding_status="grounded",
    )


@pytest.mark.asyncio
async def test_execute_unified_turn_threads_juniper_origin_into_stance_inputs() -> None:
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello from juniper",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin="juniper",
        )

    react_mock.assert_awaited_once()
    stance_req = react_mock.await_args.args[0]
    assert stance_req.stance_inputs.get("utterance_origin") == "juniper"


@pytest.mark.asyncio
async def test_execute_unified_turn_threads_orion_origin_into_stance_inputs() -> None:
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="investigate claim X",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin="orion",
        )

    react_mock.assert_awaited_once()
    stance_req = react_mock.await_args.args[0]
    assert stance_req.stance_inputs.get("utterance_origin") == "orion"


@pytest.mark.asyncio
async def test_execute_unified_turn_omits_origin_when_unset() -> None:
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="legacy caller",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
        )

    react_mock.assert_awaited_once()
    stance_req = react_mock.await_args.args[0]
    assert "utterance_origin" not in stance_req.stance_inputs


@pytest.mark.asyncio
async def test_run_unified_turn_passes_juniper_utterance_origin() -> None:
    captured: dict = {}

    async def _fake_execute_unified_turn(**kwargs):
        captured.update(kwargs)
        return [{"type": "final", "correlation_id": _CORR_ID, "llm_response": "hi"}]

    class _FakeWS:
        async def send_json(self, _frame: dict) -> None:
            return None

    with patch(
        "orion.hub.turn_orchestrator.execute_unified_turn",
        _fake_execute_unified_turn,
    ):
        await run_unified_turn(
            _FakeWS(),
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
        )

    assert captured.get("utterance_origin") == "juniper"
    assert captured.get("reading_context") == "unified_chat"
