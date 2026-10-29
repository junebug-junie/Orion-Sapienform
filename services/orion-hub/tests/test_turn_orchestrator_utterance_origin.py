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


@pytest.mark.asyncio
async def test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness() -> None:
    """Stance/Mind see the short subject; the motor still gets the full prompt."""
    kickoff = (
        "ASKING FOR CONTRACTOR HELP (orion_worldview).\n"
        '  MERGE (h:HelpRequest {help_id: "<unique help id>"})\n'
        "Pick something. Nobody asked you."
    )
    subject = (
        "Orion investigation subject (self-authored).\n"
        "Investigation claim: not yet chosen.\n"
        "Continue note: still do not know why substrate.route has no edges."
    )
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message=kickoff,
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin="orion",
            mind_appraisal_text=subject,
        )

    react_mock.assert_awaited_once()
    stance_req = react_mock.await_args.args[0]
    assert stance_req.user_message == subject
    assert stance_req.stance_inputs.get("user_message") == subject
    assert "MERGE (h:HelpRequest" not in stance_req.user_message
    harness_client_run.assert_awaited_once()
    harness_req = harness_client_run.await_args.args[0]
    # The full kickoff reaches the motor intact. A HelpRequest kickoff may also
    # carry the role-teach "Mind work-shape" advisory in front of it
    # (orion/curiosity/role_teach_disclosure.py, 2026-09-19), so not `==`.
    assert harness_req.user_message.endswith(kickoff)
    assert "MERGE (h:HelpRequest" in harness_req.user_message


@pytest.mark.asyncio
async def test_execute_unified_turn_gives_stance_the_view_and_harness_the_full_prompt() -> None:
    """A reading turn: the stance reads the task without the reader's JSON contract,
    the motor gets the full prompt with it."""
    full = 'Read https://example.org/a.\nRequired JSON shape: {"summary": "..."}'
    view = "Read https://example.org/a.\n[contract withheld]"
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="orion_world_pulse_read",
            user_message=full,
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            reading_only=True,
            stance_view_message=view,
        )

    stance_req = react_mock.await_args.args[0]
    assert stance_req.user_message == view
    assert '"summary"' not in str(stance_req.stance_inputs)
    harness_req = harness_client_run.await_args.args[0]
    assert harness_req.user_message == full


@pytest.mark.asyncio
async def test_stance_view_also_replaces_harness_user_message_on_the_appraisal_path() -> None:
    full = 'Read it.\nRequired JSON shape: {"summary": "..."}'
    patches = _hub_client_patches(thought=_thought(), harness_run=_harness_run())
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message=full,
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            mind_appraisal_text="subject",
            stance_view_message="Read it.",
        )

    stance_req = react_mock.await_args.args[0]
    assert stance_req.user_message == "subject"
    assert stance_req.stance_inputs["harness_user_message"] == "Read it."


@pytest.mark.asyncio
async def test_execute_unified_turn_sends_caller_retrieval_query_to_stance() -> None:
    """Recall retrieval design phase 3: the caller's search text rides the stance request
    (orion-thought copies it to ctx["retrieval_query"] for cortex-exec's recall calls)."""
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="A 30k-character instruction prompt. " * 10,
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin="orion",
            retrieval_query="  What do I keep noticing about my own recall?  " + "x" * 2000,
        )

    stance_req = react_mock.await_args.args[0]
    assert stance_req.retrieval_query.startswith("What do I keep noticing about my own recall?")
    assert len(stance_req.retrieval_query) == 1000
    # Not in stance_inputs: stance_react.j2 renders those into the stance prompt.
    assert "retrieval_query" not in stance_req.stance_inputs
    assert "What do I keep noticing" not in str(stance_req.stance_inputs)


@pytest.mark.asyncio
async def test_execute_unified_turn_without_retrieval_query_sends_none() -> None:
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
        )

    stance_req = react_mock.await_args.args[0]
    assert stance_req.retrieval_query is None
    assert "retrieval_query" not in stance_req.stance_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["juniper", "orion", None])
async def test_execute_unified_turn_threads_origin_into_harness_request(origin) -> None:
    """L7: the governor keeps auto-memory out of chat replies only, so the
    harness request must say who started the turn."""
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1], patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="hello",
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin=origin,
        )

    harness_req = harness_client_run.await_args.args[0]
    assert harness_req.utterance_origin == origin


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("with_focus", "origin", "expected"),
    [(True, "orion", True), (False, "orion", False),
     # A Hub chat client's payload is the raw request body: it never reaches the system prompt.
     (True, "juniper", False), (True, None, False)],
)
async def test_execute_unified_turn_puts_a_curiosity_focus_on_the_harness_request(with_focus, origin, expected) -> None:
    """The focus a curiosity run took rides payload -> HarnessRunRequestV1.curiosity_focus, where the
    governor anchors it in the system prompt (orion/harness/task_anchor.py::build_focus_anchor)."""
    from orion.schemas.curiosity_focus import CuriosityFocusAnchorV1

    focus = CuriosityFocusAnchorV1(
        focus_id="a21f96c9-f1aa-5c19-9562-91ed82a64b80", run_id="480ee593ead6",
        question="Do walkway regulars keep a weekly rhythm?", origin="self", graph="orion_worldview",
    )
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    payload = {"curiosity_focus": focus.model_dump(mode="json")} if with_focus else {}
    with patches[0], patches[1], patches[2]:
        await execute_unified_turn(
            bus=MagicMock(), correlation_id=_CORR_ID, session_id="sess-1", user_message="investigate",
            payload=payload, emit_observation_fn=lambda **_kwargs: None, utterance_origin=origin,
        )
    harness_req = harness_client_run.await_args.args[0]
    assert harness_req.curiosity_focus == (focus if expected else None)
