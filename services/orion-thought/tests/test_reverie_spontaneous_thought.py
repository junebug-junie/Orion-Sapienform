"""Phase A — spontaneous thought (reverie inside orion-thought).

Covers the load-bearing anti-hollow guard and the producer's graceful
degradation: a reverie tick must never raise and must never publish empty-shell
cognition.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)


def _coalition(attended=("n-1",), open_loops=("ol-1",), selected="ol-1"):
    return CoalitionSnapshotV1(
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
        open_loop_ids=list(open_loops),
        generated_at="2026-07-06T00:00:00Z",
    )


def _broadcast(attended=("n-1",), loops=(("ol-1", {"predictive_value": 0.8}),),
               selected="ol-1", stability=0.4):
    frame = AttentionFrameV1(
        open_loops=[OpenLoopV1(id=oid, description="d", **scores) for oid, scores in loops],
    )
    return AttentionBroadcastProjectionV1(
        frame=frame,
        attended_node_ids=list(attended),
        selected_open_loop_id=selected,
        coalition_stability_score=stability,
    )


# --- schema: the anti-hollow guard --------------------------------------------

def test_absent_coalition_is_hollow():
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=None,
                             interpretation=GROUNDED_TEXT, evidence_refs=["n-1"]).marked_hollow()
    assert t.is_hollow() and t.hollow_reason == "absent_coalition"


def test_short_interpretation_is_hollow():
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation="hmm", evidence_refs=["n-1"]).marked_hollow()
    assert t.is_hollow() and t.hollow_reason == "interpretation_too_short"


def test_no_evidence_is_unanchored_hollow():
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation=GROUNDED_TEXT, evidence_refs=[]).marked_hollow()
    assert t.is_hollow() and t.hollow_reason == "unanchored_no_evidence"


def test_evidence_outside_coalition_is_hollow():
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation=GROUNDED_TEXT, evidence_refs=["not-in-coalition"]).marked_hollow()
    assert t.is_hollow() and t.hollow_reason == "unanchored_evidence_outside_coalition"


def test_grounded_thought_is_not_hollow():
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation=GROUNDED_TEXT, evidence_refs=["ol-1"]).marked_hollow()
    assert not t.is_hollow() and t.hollow_reason is None


def test_evidence_refs_capped_at_50():
    with pytest.raises(Exception):
        SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation=GROUNDED_TEXT, evidence_refs=[str(i) for i in range(51)])


# --- producer: grounding + salience -------------------------------------------

def test_build_coalition_snapshot_maps_broadcast():
    from app import reverie

    snap = reverie.build_coalition_snapshot(_broadcast())
    assert snap is not None
    assert snap.attended_node_ids == ["n-1"]
    assert snap.open_loop_ids == ["ol-1"]
    assert snap.selected_open_loop_id == "ol-1"


def test_build_coalition_snapshot_none_on_absent():
    from app import reverie

    assert reverie.build_coalition_snapshot(None) is None


def test_derive_salience_max_of_open_loop_scores():
    from app import reverie

    # selected loop predictive_value 0.8 dominates → salience 0.8 (no invented weights)
    assert reverie.derive_salience(_broadcast()) == pytest.approx(0.8)


def test_derive_salience_fallback_to_stability():
    from app import reverie

    # selected id not present among loops → fall back to coalition stability
    bcast = _broadcast(selected="missing", stability=0.33)
    assert reverie.derive_salience(bcast) == pytest.approx(0.33)


def test_parse_reverie_payload_marks_unanchored_hollow():
    from app import reverie

    raw = json.dumps({"interpretation": GROUNDED_TEXT, "salience": 0.7, "evidence_refs": ["outside"]})
    t = reverie.parse_reverie_payload(raw, coalition=_coalition(), correlation_id="c", broadcast=_broadcast())
    assert t.hollow and t.hollow_reason == "unanchored_evidence_outside_coalition"


def test_parse_reverie_payload_grounded_is_clean():
    from app import reverie

    # LLM proposes salience 0.7 but code owns it deterministically: predictive_value 0.8.
    raw = json.dumps({"interpretation": GROUNDED_TEXT, "salience": 0.7, "evidence_refs": ["ol-1"]})
    t = reverie.parse_reverie_payload(raw, coalition=_coalition(), correlation_id="c", broadcast=_broadcast())
    assert not t.hollow
    assert t.salience == pytest.approx(0.8)


# --- producer: the tick never raises, never ships empty shells -----------------

@pytest.mark.asyncio
async def test_tick_returns_none_when_no_coalition():
    from app import reverie

    bus = AsyncMock()
    result = await reverie.run_reverie_once(bus, broadcast_reader=lambda: None)
    assert result is None
    bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_tick_publishes_grounded_thought():
    from app import reverie

    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "salience": 0.7, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
    )
    assert result is not None and not result.hollow
    bus.publish.assert_called_once()
    channel, envelope = bus.publish.call_args.args
    assert channel == "orion:reverie:thought"
    assert envelope.kind == "reverie.thought.v1"


@pytest.mark.asyncio
async def test_tick_persists_published_thought(monkeypatch):
    from app import reverie

    calls = []
    monkeypatch.setattr(reverie, "persist_reverie_thought", lambda t: calls.append(t) or True)
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
    )
    assert result is not None
    assert len(calls) == 1 and calls[0].thought_id == result.thought_id


def test_persist_never_raises_on_bad_db(monkeypatch):
    from app import store

    # Force a fresh engine against an unroutable DB; persistence must swallow it.
    monkeypatch.setattr(store, "_engine", None)
    monkeypatch.setattr(store, "_database_url", lambda: "postgresql://x:x@127.0.0.1:1/nope")
    t = SpontaneousThoughtV1(thought_id="t", correlation_id="c", coalition=_coalition(),
                             interpretation=GROUNDED_TEXT, evidence_refs=["ol-1"]).marked_hollow()
    assert store.persist_reverie_thought(t) is False  # returned, not raised


@pytest.mark.asyncio
async def test_tick_drops_hollow_thought():
    from app import reverie

    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": "x", "salience": 0.9, "evidence_refs": []}),
    })
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
    )
    assert result is None
    bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_tick_swallows_narration_failure():
    from app import reverie

    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(side_effect=RuntimeError("gateway down"))
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
    )
    assert result is None  # degraded, not raised
    bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_tick_swallows_publish_failure():
    from app import reverie

    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    # Must degrade to None, not raise, even though the thought was valid.
    result = await reverie.run_reverie_once(
        bus, broadcast_reader=lambda: _broadcast(), cortex_client=cortex,
    )
    assert result is None


# --- default broadcast reader (real signature coverage, no live DB) ------------

def test_default_broadcast_reader_returns_projection(monkeypatch):
    """Covers the direct-DB reader wiring the mocked ticks skip."""
    from app import broadcast_reader, reverie

    monkeypatch.setattr(broadcast_reader, "read_latest_broadcast", lambda *a, **k: _broadcast())
    result = reverie._default_broadcast_reader()
    assert result is not None
    assert result.attended_node_ids == ["n-1"]


def test_default_broadcast_reader_none_on_empty(monkeypatch):
    from app import broadcast_reader, reverie

    monkeypatch.setattr(broadcast_reader, "read_latest_broadcast", lambda *a, **k: None)
    assert reverie._default_broadcast_reader() is None


def test_default_broadcast_reader_never_raises(monkeypatch):
    from app import broadcast_reader, reverie

    def boom(*a, **k):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(broadcast_reader, "read_latest_broadcast", boom)
    assert reverie._default_broadcast_reader() is None  # swallowed


def test_broadcast_reader_none_when_no_row(monkeypatch):
    """The direct reader itself is fail-open on an empty/absent table."""
    from app import broadcast_reader

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k):
            class _R:
                def mappings(self_): return self_
                def first(self_): return None
            return _R()

    class _Engine:
        def connect(self): return _Conn()

    monkeypatch.setattr(broadcast_reader, "_get_engine", lambda: _Engine())
    assert broadcast_reader.read_latest_broadcast() is None
