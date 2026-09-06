"""Reverie chain -> attention schema surface: the chain runner emits one
AttentionSchemaV1 row per chain on orion:attention:schema, alongside (never
instead of) its existing reverie.chain.v1 publish and table write.

docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
    )


def _thought(idx: int) -> SpontaneousThoughtV1:
    return SpontaneousThoughtV1(
        thought_id=f"th-{idx}", correlation_id="corr-r",
        coalition=CoalitionSnapshotV1(attended_node_ids=["n-1"], selected_open_loop_id="ol-1", open_loop_ids=["ol-1"], generated_at=NOW),
        interpretation=f"thought {idx}: the loop ol-1 keeps recurring", salience=0.7,
        next_focus="the deploy" if idx == 1 else None,
    )


async def _step(chain_id, index):
    return _thought(index)


def _published(bus: AsyncMock) -> dict[str, list]:
    out: dict[str, list] = {}
    for call in bus.publish.await_args_list:
        channel, envelope = call.args
        out.setdefault(channel, []).append(envelope)
    return out


@pytest.mark.asyncio
async def test_chain_publishes_one_attention_schema_row_alongside_the_chain():
    from app import chain

    bus = AsyncMock()
    with patch.object(chain, "persist_reverie_chain", return_value=True), \
         patch.object(chain, "_maybe_emit_resonance_alert", new=AsyncMock()):
        c = await chain.run_reverie_chain(
            bus, step_fn=_step, refractory_store=chain.InMemoryRefractoryStore(),
            broadcast_reader=_broadcast, max_steps=2, publish=True, now_fn=lambda: NOW,
        )
    assert c is not None
    by_channel = _published(bus)
    assert len(by_channel[chain.settings.channel_reverie_chain]) == 1  # unchanged
    rows = by_channel[ATTENTION_SCHEMA_CHANNEL]
    assert len(rows) == 1
    env = rows[0]
    assert env.kind == ATTENTION_SCHEMA_KIND
    payload = env.payload
    assert payload["process"] == "reverie"
    assert payload["entry_id"] == f"reverie-{c.chain_id}"
    assert payload["attended_id"] == "ol-1"
    assert payload["attention_reason"] == "coalition_broadcast"
    assert payload["reason_narrative"] == "thought 0: the loop ol-1 keeps recurring"
    assert payload["narrative_kind"] == "self_report"
    assert payload["predicted_next"] == "the deploy"  # from the LAST thought
    assert payload["correlation_id"] == "corr-r"


@pytest.mark.asyncio
async def test_publish_false_emits_nothing_on_the_surface():
    from app import chain

    bus = AsyncMock()
    await chain.run_reverie_chain(
        bus, step_fn=_step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=2, publish=False, now_fn=lambda: NOW,
    )
    assert ATTENTION_SCHEMA_CHANNEL not in _published(bus)


@pytest.mark.asyncio
async def test_a_failing_surface_publish_does_not_break_the_chain():
    """The adapter is write-only and best-effort: the chain readout, its own
    publish, and its table write all complete even if the surface publish
    raises."""
    from app import chain

    bus = AsyncMock()

    async def _publish(channel, envelope):
        if channel == ATTENTION_SCHEMA_CHANNEL:
            raise RuntimeError("bus down")

    bus.publish.side_effect = _publish
    with patch.object(chain, "persist_reverie_chain", return_value=True) as persist, \
         patch.object(chain, "_maybe_emit_resonance_alert", new=AsyncMock()):
        c = await chain.run_reverie_chain(
            bus, step_fn=_step, refractory_store=chain.InMemoryRefractoryStore(),
            broadcast_reader=_broadcast, max_steps=2, publish=True, now_fn=lambda: NOW,
        )
    assert c is not None and c.terminal_reason == "max_steps"
    persist.assert_called_once()
