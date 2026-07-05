from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.hub.turn_orchestrator import execute_unified_turn
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
    with patch(
        "orion.hub.turn_orchestrator.build_hub_association_bundle",
        return_value=_association(),
    ), patch(
        "scripts.thought_client.ThoughtClient.react",
        AsyncMock(return_value=_thought()),
    ), patch(
        "scripts.harness_governor_client.HarnessGovernorClient.run",
        AsyncMock(return_value=failed_run),
    ):
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
    with patch(
        "orion.hub.turn_orchestrator.build_hub_association_bundle",
        return_value=_association(),
    ), patch(
        "scripts.thought_client.ThoughtClient.react",
        AsyncMock(return_value=_thought()),
    ), patch(
        "scripts.harness_governor_client.HarnessGovernorClient.run",
        AsyncMock(return_value=failed_run),
    ):
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
    with patch(
        "orion.hub.turn_orchestrator.build_hub_association_bundle",
        return_value=_association(),
    ), patch(
        "scripts.thought_client.ThoughtClient.react",
        AsyncMock(return_value=_thought(disposition="defer")),
    ), patch(
        "scripts.harness_governor_client.HarnessGovernorClient.run",
        harness_run,
    ):
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
