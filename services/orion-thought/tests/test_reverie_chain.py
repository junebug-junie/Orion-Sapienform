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


_DISTINCT = [
    "the loop ol-1 keeps recurring and has not discharged",
    "maybe the recurrence tracks the nightly backup window rather than load",
    "if it is the backup, then the fix belongs in scheduling, not in alerting",
    "what is still unclear is why the spike outlasts the backup by an hour",
    "checking that would need the disk latency history beside the bus noise",
]


def _thought(idx=0, salience=0.8, next_focus="what drives the recurrence", interpretation=None):
    return SpontaneousThoughtV1(
        thought_id=f"th-{idx}", correlation_id="c",
        coalition=CoalitionSnapshotV1(
            attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
            open_loop_ids=["ol-1"], generated_at=NOW,
        ),
        interpretation=interpretation or _DISTINCT[idx % len(_DISTINCT)],
        salience=salience, evidence_refs=["ol-1"],
        next_focus=next_focus,
    )


def _step_always(salience=0.8):
    async def step(chain_id, index, continuity=None):
        return _thought(index, salience)
    return step


# --- pure helpers -------------------------------------------------------------

def test_theme_key_prefers_selected_loop():
    from app import chain
    # Bare loop id — must match attention_loops_store.suppress_loop's key format
    # (orion-hub) so a human's Resolve/Dismiss action actually suppresses this chain.
    assert chain.theme_key_for(_thought().coalition) == "ol-1"


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
async def test_chain_settles_when_step_has_no_next_focus():
    from app import chain

    async def step(chain_id, index, continuity=None):
        return _thought(index, next_focus=None if index == 1 else "the backup window")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=4, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "settled"
    assert len(c.thought_ids) == 2  # step 1 said there was nothing further


@pytest.mark.asyncio
async def test_chain_settles_when_step_repeats_previous_thought():
    from app import chain

    async def step(chain_id, index, continuity=None):
        return _thought(index, interpretation="the loop ol-1 keeps recurring and has not discharged")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=4, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "settled"
    assert len(c.thought_ids) == 2  # second step restated the first


@pytest.mark.asyncio
async def test_each_step_sees_the_previous_thought_and_hint():
    """Spec 2026-07-14 #2/#3: the old chain ran the same prompt N times."""
    from app import chain

    seen = []

    async def step(chain_id, index, continuity=None):
        seen.append(continuity)
        return _thought(index, next_focus=f"focus-{index}")

    await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=3, publish=False, now_fn=lambda: NOW,
    )
    assert [c.index for c in seen] == [0, 1, 2]
    assert seen[0].prior_thoughts == [] and seen[0].next_focus_hint is None
    assert seen[1].prior_thoughts == [_DISTINCT[0]] and seen[1].next_focus_hint == "focus-0"
    assert seen[2].prior_thoughts == [_DISTINCT[0], _DISTINCT[1]] and seen[2].next_focus_hint == "focus-1"
    assert [c.is_final for c in seen] == [False, False, True]


@pytest.mark.asyncio
async def test_chain_settles_on_a_b_a_restatement():
    from app import chain

    texts = [_DISTINCT[0], _DISTINCT[1], _DISTINCT[0] + " still"]

    async def step(chain_id, index, continuity=None):
        return _thought(index, interpretation=texts[index])

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=4, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "settled"
    assert len(c.thought_ids) == 3


@pytest.mark.asyncio
async def test_lift_mode_chain_is_not_cut_by_missing_next_focus():
    """The concern-card (metacog) prompt never asks for next_focus, so its
    absence must not settle the chain after step 0."""
    from app import chain

    async def step(chain_id, index, continuity=None):
        return _thought(index, next_focus=None).model_copy(update={"llm_profile": "metacog"})

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=3, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "max_steps"
    assert len(c.thought_ids) == 3


def test_is_repeat_thought():
    from app import chain

    assert chain.is_repeat_thought("the bus is noisy again", "The bus is noisy again.")
    assert not chain.is_repeat_thought(_DISTINCT[1], _DISTINCT[0])
    assert not chain.is_repeat_thought("", "anything")


@pytest.mark.asyncio
async def test_calm_tick_starts_no_chain_and_no_unknown_refractory(caplog):
    """Regression: a no-winner broadcast used to persist an "unknown" chain row
    (terminal no_coalition) and arm an "unknown" refractory entry, ~190/day."""
    import logging

    from app import chain

    steps = []

    async def step(chain_id, index, continuity=None):
        steps.append(index)
        return _thought(index)

    store = chain.InMemoryRefractoryStore()
    persisted = []
    calm = _broadcast(selected=None, attended=())
    before = chain.CHAIN_SKIP_COUNTS["calm_no_winner"]
    with caplog.at_level(logging.INFO, logger="orion-thought.chain"):
        c = await chain.run_reverie_chain(
            AsyncMock(), step_fn=step, refractory_store=store,
            broadcast_reader=lambda: calm, publish=False, now_fn=lambda: NOW,
            no_coalition_streak_loader=lambda k: persisted.append(k) or 0,
        )
    assert c is None
    assert steps == []  # no LLM step ran
    assert store._until == {}  # no "unknown" refractory entry
    assert chain.CHAIN_SKIP_COUNTS["calm_no_winner"] == before + 1
    assert "reverie_chain_skipped reason=calm_no_winner" in caplog.text


def test_has_attention_winner():
    from app import chain

    assert chain.has_attention_winner(_broadcast())
    assert chain.has_attention_winner(_broadcast(selected=None, attended=("n-1",)))
    assert not chain.has_attention_winner(_broadcast(selected=None, attended=()))
    assert not chain.has_attention_winner(None)


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
    store.suppress("ol-1", NOW + timedelta(seconds=600))
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
    assert store.is_suppressed("ol-1", NOW) is True


@pytest.mark.asyncio
async def test_chain_step_none_terminates_without_raise():
    from app import chain

    async def step_none(chain_id, index, continuity=None):
        return None

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_none, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "no_coalition"
    assert c.thought_ids == []


# --- stuck-loop breaker (no_coalition doesn't arm refractory on its own,
# but a repeated streak of it forces the cooldown anyway) ---------------------

def test_stuck_loop_pure_below_threshold_does_not_force_suppress():
    from app import chain
    force, next_streak = chain.resolve_reverie_chain_stuck_loop("no_coalition", 1, 3)
    assert force is False
    assert next_streak == 2  # climbing, but not there yet


def test_stuck_loop_pure_at_threshold_forces_suppress_and_resets():
    from app import chain
    force, next_streak = chain.resolve_reverie_chain_stuck_loop("no_coalition", 2, 3)
    assert force is True
    assert next_streak == 0  # the forced cooldown is itself the reset


def test_stuck_loop_pure_non_no_coalition_terminal_resets_streak():
    from app import chain
    # A real step happened -- whatever streak preceded it is broken.
    force, next_streak = chain.resolve_reverie_chain_stuck_loop("pressure_discharged", 7, 3)
    assert force is False
    assert next_streak == 0


@pytest.mark.asyncio
async def test_chain_below_streak_threshold_does_not_force_refractory():
    from app import chain

    async def step_none(chain_id, index, continuity=None):
        return None

    store = chain.InMemoryRefractoryStore()
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_none, refractory_store=store,
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
        no_coalition_max_repeats=3, no_coalition_streak_loader=lambda theme_key: 1,
    )
    assert c.terminal_reason == "no_coalition"
    assert c.no_coalition_streak == 2  # 1 (prior) + 1 (this chain)
    assert store.is_suppressed("ol-1", NOW) is False  # not there yet -- can retry next tick


@pytest.mark.asyncio
async def test_chain_at_streak_threshold_forces_refractory():
    from app import chain

    async def step_none(chain_id, index, continuity=None):
        return None

    store = chain.InMemoryRefractoryStore()
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_none, refractory_store=store,
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
        refractory_sec=900, no_coalition_max_repeats=3, no_coalition_streak_loader=lambda theme_key: 2,
    )
    assert c.terminal_reason == "no_coalition"
    assert c.no_coalition_streak == 0  # forced cooldown is the reset
    # The loop that was resonating every tick is now actually suppressed.
    assert store.is_suppressed("ol-1", NOW) is True
    assert store.is_suppressed("ol-1", NOW + timedelta(seconds=899)) is True
    assert store.is_suppressed("ol-1", NOW + timedelta(seconds=901)) is False


@pytest.mark.asyncio
async def test_chain_streak_loader_failure_degrades_to_zero_not_raise():
    from app import chain

    async def step_none(chain_id, index, continuity=None):
        return None

    def boom(theme_key):
        raise RuntimeError("db down")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step_none, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, publish=False, now_fn=lambda: NOW,
        no_coalition_streak_loader=boom,
    )
    assert c is not None  # never raises
    assert c.terminal_reason == "no_coalition"
    assert c.no_coalition_streak == 1  # degraded prior streak of 0, plus this chain


@pytest.mark.asyncio
async def test_chain_never_raises_on_step_error():
    from app import chain

    async def step_boom(chain_id, index, continuity=None):
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


@pytest.mark.asyncio
async def test_resonance_tripwire_disabled_is_noop(monkeypatch):
    from app import chain
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_resonance_alert_enabled", False)
    bus = AsyncMock()
    await chain._maybe_emit_resonance_alert(bus)
    bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_resonance_tripwire_fires_on_runaway(monkeypatch):
    from app import chain
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_resonance_alert_enabled", True)
    monkeypatch.setattr(settings, "reverie_refractory_sec", 900.0)
    monkeypatch.setattr(settings, "reverie_resonance_window", 200)

    # Recent chains: one theme re-igniting every 60s (inside the 900s bound).
    events = [("loop:ol-1", NOW + timedelta(seconds=60 * i)) for i in range(5)]
    monkeypatch.setattr(chain, "load_recent_chain_theme_events", lambda n: events)

    persisted = []
    monkeypatch.setattr(chain, "persist_resonance_alert", lambda a: persisted.append(a) or True)

    bus = AsyncMock()
    await chain._maybe_emit_resonance_alert(bus)
    bus.publish.assert_awaited_once()
    assert bus.publish.await_args.args[0] == settings.channel_reverie_resonance_alert
    assert persisted and persisted[0].theme_key == "loop:ol-1"


@pytest.mark.asyncio
async def test_resonance_tripwire_silent_when_damped(monkeypatch):
    from app import chain
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_resonance_alert_enabled", True)
    monkeypatch.setattr(settings, "reverie_refractory_sec", 900.0)
    # Every recurrence waited out the refractory bound.
    events = [("loop:ol-1", NOW + timedelta(seconds=1000 * i)) for i in range(5)]
    monkeypatch.setattr(chain, "load_recent_chain_theme_events", lambda n: events)

    bus = AsyncMock()
    await chain._maybe_emit_resonance_alert(bus)
    bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_resonance_tripwire_never_raises(monkeypatch):
    from app import chain
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_resonance_alert_enabled", True)

    def boom(n):
        raise RuntimeError("db down")

    monkeypatch.setattr(chain, "load_recent_chain_theme_events", boom)
    await chain._maybe_emit_resonance_alert(AsyncMock())  # degrades, no raise


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
