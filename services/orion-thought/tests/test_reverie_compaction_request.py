"""Phase E — compaction request (queue only). A settled chain queues a typed
ask; nothing consumes it. Requests degrade, cap, and are gate-controlled."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.reverie import CompactionRequestV1, ReverieChainV1

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
    )


def _chain(terminal="max_steps", theme="loop:ol-1", ids=("th-0", "th-1")):
    return ReverieChainV1(
        chain_id="ch-1", theme_key=theme, terminal_reason=terminal, thought_ids=list(ids),
    )


# --- pure builder -------------------------------------------------------------

def test_settled_chain_builds_consolidate_request():
    from app import chain
    req = chain.build_compaction_request(_chain("max_steps"))
    assert req is not None
    assert req.op_hint == "consolidate"  # awake never asks to downscale/prune
    assert req.theme == "loop:ol-1"
    assert req.origin_chain_id == "ch-1"
    assert req.evidence_refs == ["th-0", "th-1"]


def test_unsettled_chain_yields_no_request():
    from app import chain
    assert chain.build_compaction_request(_chain("no_coalition")) is None
    assert chain.build_compaction_request(_chain("refractory")) is None


def test_unknown_theme_yields_no_request():
    from app import chain
    assert chain.build_compaction_request(_chain("max_steps", theme="unknown")) is None


def test_request_evidence_refs_capped():
    with pytest.raises(Exception):
        CompactionRequestV1(request_id="r", theme="t",
                            evidence_refs=[str(i) for i in range(51)] * 5)  # > 200


# --- queue-only emission ------------------------------------------------------

@pytest.mark.asyncio
async def test_chain_queues_request_when_flag_on(monkeypatch):
    from app import chain
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_compaction_request_enabled", True)
    monkeypatch.setattr(chain, "persist_reverie_chain", lambda c: True)
    queued = []
    monkeypatch.setattr(chain, "persist_compaction_request", lambda r: queued.append(r) or True)

    from orion.schemas.reverie import SpontaneousThoughtV1
    from orion.schemas.thought import CoalitionSnapshotV1

    async def step2(chain_id, index):
        return SpontaneousThoughtV1(
            thought_id=f"th-{index}", correlation_id="c",
            coalition=CoalitionSnapshotV1(attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
                                         open_loop_ids=["ol-1"], generated_at=NOW),
            interpretation="loop ol-1 keeps recurring and has not discharged",
            salience=0.8, evidence_refs=["ol-1"],
        )

    bus = AsyncMock()
    c = await chain.run_reverie_chain(
        bus, step_fn=step2, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=2, publish=True, now_fn=lambda: NOW,
    )
    assert c is not None
    assert len(queued) == 1  # a request was enqueued
    published_channels = [call.args[0] for call in bus.publish.call_args_list]
    assert "orion:dream:compaction-request" in published_channels


@pytest.mark.asyncio
async def test_chain_no_request_when_flag_off(monkeypatch):
    from app import chain
    from app.settings import settings
    from orion.schemas.reverie import SpontaneousThoughtV1
    from orion.schemas.thought import CoalitionSnapshotV1

    monkeypatch.setattr(settings, "reverie_compaction_request_enabled", False)
    monkeypatch.setattr(chain, "persist_reverie_chain", lambda c: True)
    queued = []
    monkeypatch.setattr(chain, "persist_compaction_request", lambda r: queued.append(r) or True)

    async def step(chain_id, index):
        return SpontaneousThoughtV1(
            thought_id=f"th-{index}", correlation_id="c",
            coalition=CoalitionSnapshotV1(attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
                                         open_loop_ids=["ol-1"], generated_at=NOW),
            interpretation="loop ol-1 keeps recurring and has not discharged",
            salience=0.8, evidence_refs=["ol-1"],
        )

    await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=2, publish=True, now_fn=lambda: NOW,
    )
    assert queued == []


# --- contract: queue has no consumer -----------------------------------------

def test_compaction_request_channel_has_no_consumer():
    channels = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())["channels"]
    entry = next(c for c in channels if c["name"] == "orion:dream:compaction-request")
    assert entry.get("consumer_services") == [], "Phase E must be a dead-end queue"
