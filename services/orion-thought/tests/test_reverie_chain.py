"""Phase C — reverie chain. Deterministic control is fully testable with
injected fakes: termination, refractory, lossy EMA, never-raises, no dream read.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.reverie import MAX_CHAIN_THOUGHTS, ReverieChainV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


def _broadcast(selected="ol-1", attended=("n-1",)):
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id=selected or "ol-1", description="d")]),
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
    )


def _thought(idx=0, salience=0.8):
    return SpontaneousThoughtV1(
        thought_id=f"th-{idx}", correlation_id="c",
        coalition=CoalitionSnapshotV1(
            attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
            open_loop_ids=["ol-1"], generated_at=NOW,
        ),
        interpretation="the loop ol-1 keeps recurring and has not discharged",
        salience=salience, evidence_refs=["ol-1"],
    )


def _step_always(salience=0.8):
    async def step(chain_id, index):
        return _thought(index, salience)
    return step


# --- pure helpers -------------------------------------------------------------

def test_theme_key_prefers_selected_loop():
    from app import chain
    assert chain.theme_key_for(_thought().coalition) == "loop:ol-1"


def test_theme_key_none_is_unknown():
    from app import chain
    assert chain.theme_key_for(None) == "unknown"


def test_update_ema_is_lossy_low_pass():
    from app import chain
    # a fresh spike does not fully overwrite prior state
    ema = chain.update_ema(1.0, 0.0, alpha=0.5)
    assert ema == pytest.approx(0.5)
    ema2 = chain.update_ema(ema, 0.0, alpha=0.5)
    assert 0.0 < ema2 < ema  # decays, never a verbatim recall of the spike


# --- orchestration ------------------------------------------------------------

@pytest.mark.asyncio
async def test_chain_terminates_at_max_steps():
    from app import chain
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=3, publish=False, now_fn=lambda: NOW,
    )
    assert c is not None
    assert c.terminal_reason == "max_steps"
    assert len(c.thought_ids) == 3


@pytest.mark.asyncio
async def test_chain_terminates_on_pressure_discharge():
    from app import chain
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, pressure_reader=lambda: 0.10, max_steps=10,
        publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "pressure_discharged"
    assert len(c.thought_ids) == 1  # discharged after the first step


@pytest.mark.asyncio
async def test_chain_none_when_no_coalition():
    from app import chain
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=lambda: None, publish=False, now_fn=lambda: NOW,
    )
    assert c is None


@pytest.mark.asyncio
async def test_chain_suppressed_by_refractory():
    from app import chain
    store = chain.InMemoryRefractoryStore()
    store.suppress("loop:ol-1", NOW + timedelta(seconds=600))
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=store,
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
    )
    assert c is None  # theme in refractory → not re-triggered


@pytest.mark.asyncio
async def test_chain_suppresses_theme_after_completion():
    from app import chain
    store = chain.InMemoryRefractoryStore()
    await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=store,
        broadcast_reader=_broadcast, max_steps=2, refractory_sec=900, publish=False, now_fn=lambda: NOW,
    )
    assert store.is_suppressed("loop:ol-1", NOW) is True


@pytest.mark.asyncio
async def test_chain_step_none_terminates_without_raise():
    from app import chain

    async def step_none(chain_id, index):
        return None

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_none, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "no_coalition"
    assert c.thought_ids == []


@pytest.mark.asyncio
async def test_chain_never_raises_on_step_error():
    from app import chain

    async def step_boom(chain_id, index):
        raise RuntimeError("narration exploded")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_boom, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
    )
    assert c is not None  # degraded to a terminated chain, not a raise


@pytest.mark.asyncio
async def test_ema_summary_is_not_a_verbatim_window():
    from app import chain
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=_step_always(), refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=3, publish=False, now_fn=lambda: NOW,
    )
    # wide-n memory is the lossy scalar, not the verbatim interpretation text
    assert "keeps recurring" not in c.ema_summary
    assert "ema_salience" in c.ema_summary


def test_chain_thought_ids_capped_by_schema():
    with pytest.raises(Exception):
        ReverieChainV1(
            chain_id="x", terminal_reason="max_steps",
            thought_ids=[str(i) for i in range(MAX_CHAIN_THOUGHTS + 1)],
        )


@pytest.mark.asyncio
async def test_standalone_reverie_superseded_when_chain_enabled(monkeypatch):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_enabled", True)
    monkeypatch.setattr(settings, "reverie_chain_enabled", True)
    # Returns before touching the bus — no double-emission with chain mode.
    await reverie.run_reverie_worker(stop_event=None)


def test_chain_module_never_reads_a_dream():
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "chain.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    for mod in mods:
        assert "dream" not in mod.lower(), f"chain must not read a dream ({mod!r})"
