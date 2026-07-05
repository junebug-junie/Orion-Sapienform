from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
_OTHER_SERVICES = tuple(
    p
    for p in REPO_ROOT.glob("services/orion-*")
    if p.is_dir() and p.resolve() != HUB_ROOT.resolve()
)
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts."):
        del sys.modules[key]
    if key == "app" or key.startswith("app."):
        del sys.modules[key]
for candidate in (REPO_ROOT, HUB_ROOT, *_OTHER_SERVICES):
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

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)
from scripts.settings import settings
from scripts.thought_client import ThoughtClient


_CORR_ID = "00000000-0000-4000-8000-000000000101"


def _stance_request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id=_CORR_ID,
        session_id="sess-1",
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id=_CORR_ID,
            broadcast=None,
            broadcast_stale=True,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hello"},
    )


def _thought_event() -> ThoughtEventV1:
    return ThoughtEventV1(
        event_id="t-1",
        correlation_id=_CORR_ID,
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        imperative="Answer directly.",
        tone="neutral",
        strain_refs=["n-1"],
        evidence_refs=["n-1"],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    )


@pytest.mark.asyncio
async def test_thought_client_react_returns_thought_event() -> None:
    bus = MagicMock()
    thought = _thought_event()
    bus.codec.decode.return_value = MagicMock(
        ok=True,
        envelope=BaseEnvelope(
            kind="thought.event.v1",
            source=ServiceRef(name="orion-thought", version="0.1.0"),
            correlation_id=_CORR_ID,
            payload=thought.model_dump(mode="json"),
        ),
    )
    bus.rpc_request = AsyncMock(return_value={"data": b"x"})
    client = ThoughtClient(bus)

    result = await client.react(_stance_request())

    assert result is not None
    assert result.event_id == "t-1"
    bus.rpc_request.assert_awaited_once()
    call_kwargs = bus.rpc_request.await_args.kwargs
    assert call_kwargs["reply_channel"] == f"{settings.CHANNEL_THOUGHT_RESULT_PREFIX}{_CORR_ID}"
    sent_envelope = bus.rpc_request.await_args.args[1]
    assert sent_envelope.kind == "stance.react.request.v1"
    assert bus.rpc_request.await_args.args[0] == settings.CHANNEL_THOUGHT_REQUEST


@pytest.mark.asyncio
async def test_thought_client_react_timeout_returns_none() -> None:
    bus = MagicMock()
    bus.rpc_request = AsyncMock(side_effect=TimeoutError())
    client = ThoughtClient(bus)

    result = await client.react(_stance_request(), timeout_sec=0.2)

    assert result is None
