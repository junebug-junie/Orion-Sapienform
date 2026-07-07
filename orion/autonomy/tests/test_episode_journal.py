from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from orion.autonomy.episode_journal import dispatch_autonomy_episode_journal
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


@pytest.mark.asyncio
async def test_dispatch_autonomy_episode_journal_publishes_write() -> None:
    bus = AsyncMock()
    bus.connect = AsyncMock()
    decode_result = MagicMock(
        ok=True,
        envelope=MagicMock(
            payload={
                "ok": True,
                "final_text": '{"mode":"digest","title":"Episode","body":"## Gap\\nGPU gap"}',
            }
        ),
    )
    bus.codec = MagicMock()
    bus.codec.decode = MagicMock(return_value=decode_result)
    bus.rpc_request = AsyncMock(return_value={"data": b"payload"})
    corr = uuid4()
    parent = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        correlation_id=corr,
        source=ServiceRef(service="test", version="0", node="test"),
        payload={},
    )
    source = ServiceRef(service="concept-induction", version="0.1.0", node="test")

    with patch(
        "orion.autonomy.episode_journal.draft_from_cortex_result",
        return_value=MagicMock(mode="digest", title="Episode", body="body", model_dump=lambda mode="json": {}),
    ):
        result = await dispatch_autonomy_episode_journal(
            bus=bus,
            parent=parent,
            source=source,
            goal_artifact_id="goal-gap-gpu",
            spawned_correlation_id=str(corr),
            narrative_seed="fetch outcome: fetched 2 article(s)",
            cortex_request_channel="orion:cortex:request",
            cortex_result_prefix="orion:cortex:result",
            journal_write_channel="orion:journal:write",
            timeout_sec=12.0,
        )

    bus.rpc_request.assert_awaited_once()
    bus.publish.assert_awaited_once()
    publish_channel, publish_env = bus.publish.await_args.args
    assert publish_channel == "orion:journal:write"
    assert publish_env.kind == "journal.entry.write.v1"
    assert result["write"] is not None
