from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)


@pytest.mark.asyncio
async def test_thought_artifact_published() -> None:
    from app import bus_listener

    req = StanceReactRequestV1(
        correlation_id="c-1",
        session_id=None,
        user_message="hello",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "hello"},
    )
    fake_thought = ThoughtEventV1(
        event_id="t-1",
        correlation_id="c-1",
        session_id=None,
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
    bus = AsyncMock()
    with patch.object(bus_listener, "run_stance_react", AsyncMock(return_value=fake_thought)):
        reply = await bus_listener.handle_stance_react_request(
            bus,
            req,
            reply_to="orion:thought:result:c-1",
        )
    assert reply.correlation_id == "c-1"
    bus.publish.assert_called()
