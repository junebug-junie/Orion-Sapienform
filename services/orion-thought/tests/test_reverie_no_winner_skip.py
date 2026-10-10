"""World-first: a no-winner broadcast tick has no coalition to narrate.

Live 2026-10-07..10: 438 of 2,251 stored reveries were narrated over an empty
coalition, and they read as fixation on stale prediction-error loops ("The
coalition is fixated on biometrics prediction error...").
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)


@pytest.mark.asyncio
async def test_no_winner_tick_is_skipped_without_an_llm_call(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    cortex = AsyncMock()

    def empty():
        return AttentionBroadcastProjectionV1(
            frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-stale", description="old PE loop")]),
            attended_node_ids=[], selected_open_loop_id=None, selected_action_type="none",
        )

    thought = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=empty, cortex_client=cortex)
    assert thought is None
    cortex.execute_plan.assert_not_called()
