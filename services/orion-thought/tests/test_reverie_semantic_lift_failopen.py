import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(
            open_loops=[
                OpenLoopV1(
                    id="ol-1",
                    description="deploy",
                    source_refs=["harness_closure:corr-1"],
                )
            ]
        ),
        attended_node_ids=["harness_closure:corr-1"],
        selected_open_loop_id="ol-1",
        coalition_stability_score=0.6,
    )


@pytest.mark.asyncio
async def test_semantic_lift_resolver_error_fail_open(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    bus = AsyncMock()
    cortex = AsyncMock()
    with patch.object(
        reverie,
        "resolve_concern_cards",
        side_effect=RuntimeError("resolver boom"),
    ):
        result = await reverie.run_reverie_once(
            bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
        )
    assert result is None
    cortex.execute_plan.assert_not_called()
