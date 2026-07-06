"""Phase D — episode + motif grounding (read-only). Refs are capped, degrade to
empty, attach to the thought, and the adapter never writes."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)

GROUNDED = "the loop ol-1 keeps recurring and has not discharged"


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
    )


def test_collect_grounding_caps_and_shapes():
    from app import grounding
    m, e = grounding.collect_grounding(
        motif_loader=lambda n: [f"frame-{i}" for i in range(10)],
        episode_loader=lambda n: [f"ep-{i}" for i in range(10)],
        cap=3,
    )
    assert m == ["frame-0", "frame-1", "frame-2"]
    assert e == ["ep-0", "ep-1", "ep-2"]


def test_collect_grounding_degrades_to_empty_on_loader_error():
    from app import grounding

    def boom(n):
        raise RuntimeError("db down")

    m, e = grounding.collect_grounding(motif_loader=boom, episode_loader=boom)
    assert m == [] and e == []


def test_grounding_adapter_is_read_only():
    # The adapter must never contain a write statement.
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "grounding.py").read_text().upper()
    for write_kw in ("INSERT", "UPDATE ", "DELETE", "DROP", "CREATE TABLE"):
        assert write_kw not in src, f"grounding must be read-only (found {write_kw})"


@pytest.mark.asyncio
async def test_tick_attaches_grounding_when_flag_on(monkeypatch):
    from app import reverie, grounding
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_ground_consolidation", True)
    monkeypatch.setattr(grounding, "default_motif_loader", lambda n: ["frame-9"])
    monkeypatch.setattr(grounding, "default_episode_loader", lambda n: ["ep-9"])
    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)

    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED, "evidence_refs": ["ol-1"]}),
    })
    thought = await reverie.run_reverie_once(
        bus, broadcast_reader=_broadcast, cortex_client=cortex,
    )
    assert thought is not None
    assert thought.motif_refs == ["frame-9"]
    assert thought.episode_summary_refs == ["ep-9"]


@pytest.mark.asyncio
async def test_tick_no_grounding_when_flag_off(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_ground_consolidation", False)
    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: True)
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED, "evidence_refs": ["ol-1"]}),
    })
    thought = await reverie.run_reverie_once(
        bus, broadcast_reader=_broadcast, cortex_client=cortex,
    )
    assert thought is not None
    assert thought.motif_refs == [] and thought.episode_summary_refs == []
