import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

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
async def test_semantic_lift_tick_skips_without_cards(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    bus = AsyncMock()
    cortex = AsyncMock()
    with patch.object(reverie, "resolve_concern_cards", return_value=[]):
        result = await reverie.run_reverie_once(
            bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
        )
    assert result is None
    cortex.execute_plan.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_lift_uses_background_cortex_channel(monkeypatch):
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    monkeypatch.setattr(
        reverie.settings,
        "channel_reverie_cortex_exec_request",
        "orion:cortex:exec:request:background",
    )
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="Will the deploy slip if we cut testing?",
        stance_imperative="Name the testing tradeoff before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    bus = AsyncMock()
    mock_client = AsyncMock()
    mock_client.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({
            "interpretation": (
                "I keep circling whether cutting testing makes the deploy slip — "
                "that tradeoff is still open."
            ),
            "evidence_refs": ["ol-1"],
        }),
    })

    with patch.object(reverie, "resolve_concern_cards", return_value=[card]), patch.object(
        reverie, "CortexExecClient", return_value=mock_client
    ) as mock_client_cls:
        result = await reverie.run_reverie_once(
            bus,
            broadcast_reader=lambda: _broadcast(),
        )
    mock_client_cls.assert_called_once()
    assert mock_client_cls.call_args.kwargs["request_channel"] == (
        "orion:cortex:exec:request:background"
    )
    assert result is not None
    assert result.llm_profile == "metacog"


@pytest.mark.asyncio
async def test_semantic_lift_plan_uses_metacog_background_channel(monkeypatch):
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    monkeypatch.setattr(reverie.settings, "reverie_semantic_lift_enabled", True)
    card = ConcernCardV1.from_harness_turn(
        coalition_ref="harness_closure:corr-1",
        user_message_excerpt="Will the deploy slip if we cut testing?",
        stance_imperative="Name the testing tradeoff before reassuring.",
        created_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
    )
    assert card is not None
    bus = AsyncMock()
    captured = {}

    class _Cortex:
        def __init__(self, *_a, **_k):
            pass

        async def execute_plan(self, **kwargs):
            captured.update(kwargs)
            return {
                "final_text": json.dumps({
                    "interpretation": (
                        "I keep circling whether cutting testing makes the deploy slip — "
                        "that tradeoff is still open."
                    ),
                    "evidence_refs": ["ol-1"],
                })
            }

    with patch.object(reverie, "resolve_concern_cards", return_value=[card]):
        result = await reverie.run_reverie_once(
            bus,
            broadcast_reader=lambda: _broadcast(),
            cortex_client=_Cortex(),
        )
    assert result is not None
    req = captured["req"]
    assert req.context.get("mode") == "metacog"
    assert req.context.get("llm_route") == "metacog"
    assert req.args.extra.get("execution_lane") == "background"
