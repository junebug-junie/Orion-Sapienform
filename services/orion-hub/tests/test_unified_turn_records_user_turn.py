"""Only Juniper's own unified turns record "the user just spoke".

execute_unified_turn also runs turns Orion authors itself -- endogenous
outreach posts into Juniper's live session with no utterance_origin,
curiosity passes "orion". Once Hub bound the conversation-phase store, each
of those would have stamped last_user_turn_at and made her next real reply
read as a short gap. The situation builder now records only when
``utterance_origin == "juniper"``; the phase store side is covered in
orion/situational/tests/test_conversation_phase_user_turn.py.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_CORR = "00000000-0000-4000-8000-000000000477"


def _ensure_hub_paths() -> None:
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
    for key, value in {
        "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
        "CHANNEL_VOICE_LLM": "orion:voice:llm",
        "CHANNEL_VOICE_TTS": "orion:voice:tts",
        "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
        "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
    }.items():
        os.environ.setdefault(key, value)


def _captured_situation_ctx(**kwargs) -> dict:
    """Run the real _build_situation_prompt_fragment with the builder stubbed."""
    import orion.hub.turn_orchestrator as turn_orchestrator

    captured: dict = {}

    async def _fake_build(ctx, runtime_ns):
        captured["ctx"] = ctx
        return None, {"compact_text": "Situation: stub"}

    with patch.object(turn_orchestrator, "build_situation_for_ctx", _fake_build):
        asyncio.run(
            turn_orchestrator._build_situation_prompt_fragment(
                session_id="s1",
                user_message="hello",
                payload={},
                settings=SimpleNamespace(),
                correlation_id="corr-1",
                **kwargs,
            )
        )
    return captured["ctx"]


def test_fragment_does_not_record_unless_told():
    assert _captured_situation_ctx()["record_user_turn"] is False


def test_fragment_records_when_told():
    assert _captured_situation_ctx(record_user_turn=True)["record_user_turn"] is True


async def _record_flag_for(utterance_origin):
    """Drive the real execute_unified_turn (same stubbing as
    test_turn_orchestrator_cockpit_hops.py) and return the record_user_turn
    it hands the situation builder."""
    _ensure_hub_paths()
    from orion.hub.turn_orchestrator import execute_unified_turn
    from orion.schemas.harness_finalize import HarnessRunV1
    from orion.schemas.thought import HubAssociationBundleV1, StanceHarnessSliceV1, ThoughtEventV1
    import scripts.harness_governor_client as harness_governor_client
    import scripts.pre_turn_appraisal_client as pta_client_mod
    import scripts.thought_client as thought_client

    from datetime import datetime, timezone

    thought = ThoughtEventV1(
        event_id="t-record-1",
        correlation_id=_CORR,
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
    situation = AsyncMock(
        return_value={
            "compact_text": None,
            "status": "empty",
            "provider_status": {},
            "source_summary": {},
            "perception_enabled": False,
            "diagnostics": {},
        }
    )
    settings = SimpleNamespace(
        ENABLE_PRE_TURN_APPRAISAL=False,
        ENABLE_UNIFIED_TURN_CHAT_GRAMMAR=False,
    )
    with (
        patch(
            "orion.hub.turn_orchestrator.build_hub_association_bundle",
            return_value=HubAssociationBundleV1(
                correlation_id=_CORR,
                broadcast=None,
                broadcast_stale=True,
                read_source="felt_state_reader",
            ),
        ),
        patch.object(pta_client_mod.PreTurnAppraisalClient, "appraise", AsyncMock(return_value=None)),
        patch.object(
            thought_client.ThoughtClient,
            "react",
            AsyncMock(return_value=thought_client.ThoughtReactResult(thought=thought)),
        ),
        patch.object(
            harness_governor_client.HarnessGovernorClient,
            "run",
            AsyncMock(
                return_value=HarnessRunV1(
                    correlation_id=_CORR,
                    final_text="hi",
                    finalize_ran=True,
                    step_count=1,
                    compliance_verdict="completed",
                    grounding_status="grounded",
                )
            ),
        ),
        patch("orion.hub.turn_orchestrator._publish_unified_turn_chat_grammar", AsyncMock()),
        patch("orion.hub.turn_orchestrator._build_situation_prompt_fragment", situation),
    ):
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR,
            session_id="sess-1",
            user_message="hello",
            settings=settings,
            utterance_origin=utterance_origin,
        )
    situation.assert_awaited_once()
    return situation.await_args.kwargs["record_user_turn"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance_origin", "expected"),
    [
        ("juniper", True),  # Hub chat, WS and HTTP
        (None, False),  # endogenous outreach, world-pulse reads, collapse-mirror reply
        ("orion", False),  # curiosity investigations
    ],
)
async def test_only_juniper_turns_record_the_user_turn(utterance_origin, expected):
    assert await _record_flag_for(utterance_origin) is expected
