from datetime import datetime, timezone

import pytest

import orion.substrate.attention.salience as salience_mod
from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.salience import SalienceHistory
from orion.substrate.attention.scoring import build_open_loops, score_loop


def _ctx_signals():
    return [
        AttentionSignalV1(
            signal_id="s1", source="current_turn", target_text="the reactor plan",
            target_type_hint="plan", signal_kind="test", salience=0.9, confidence=0.9,
            evidence_refs=["r1"],
        )
    ]


def test_build_open_loops_populates_salience_features():
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    assert loops, "expected at least one loop"
    loop = loops[0]
    assert loop.salience_features, "salience_features must be populated"
    assert loop.salience > 0.0


def test_score_loop_uses_combiner_when_v2_on(monkeypatch):
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    assert score_loop(loop) == loop.salience


def test_score_loop_legacy_when_v2_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    assert score_loop(loop) >= 0.0
    assert salience_mod.salience_v2_enabled() is False


def test_build_open_loops_now_param_reaches_recency_calculation():
    """Direct, wall-clock-independent proof that build_open_loops()'s `now`
    parameter actually reaches compute_salience()/compute_features() instead
    of being silently dropped (found missing entirely until this patch --
    see orion/substrate/attention_broadcast.py's _first_selected_at comment
    for the live incident this was part of). Two widely-separated fixed
    `now` values against the SAME first_seen_at must produce different
    recency -- if `now` were ignored and compute_features fell back to real
    datetime.now(), both calls would (almost certainly) still differ from
    each other by real elapsed test-runtime, but neither would show the
    exact, deterministic half-life relationship asserted here."""
    from orion.substrate.attention.common import compact, stable_id

    first_seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    phrase = compact("the reactor plan", 120)
    loop_id = stable_id("open-loop", phrase.lower())
    history = SalienceHistory(first_seen_at={loop_id: first_seen})

    fresh = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
        history=history, now=first_seen,
    )[0]
    six_hours_later = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
        history=history, now=first_seen.replace(hour=6),
    )[0]

    assert fresh.id == loop_id == six_hours_later.id
    assert fresh.salience_features["recency"] == 1.0
    # One half-life (see salience._recency's docstring: half-life ~6h).
    assert six_hours_later.salience_features["recency"] == pytest.approx(0.5, abs=0.01)


def test_dwell_scoped_to_dwelling_loop_only():
    """dwell must only penalize the loop actually recorded as dwelling.

    Previously `history.dwell_ticks` (a single module-level tick counter,
    see attention_broadcast.py's `_dwell_ticks`) was applied uniformly to
    every competing loop scored in a tick -- a uniform per-tick offset
    shared by every candidate cannot demote the specific stuck loop
    relative to its competitors, since it changes nothing about who wins.
    Found live 2026-07-14 alongside the recency bug."""
    from orion.substrate.attention.common import compact, stable_id

    signals = [
        AttentionSignalV1(
            signal_id="s1", source="current_turn", target_text="the reactor plan",
            target_type_hint="plan", signal_kind="test", salience=0.9, confidence=0.9,
            evidence_refs=["r1"],
        ),
        AttentionSignalV1(
            signal_id="s2", source="current_turn", target_text="a fresh competitor thread",
            target_type_hint="concept", signal_kind="test", salience=0.9, confidence=0.9,
            evidence_refs=["r2"],
        ),
    ]
    dwelling_id = stable_id("open-loop", compact("the reactor plan", 120).lower())
    history = SalienceHistory(dwell_ticks=5, dwelling_loop_id=dwelling_id)

    loops = build_open_loops(
        signals=signals,
        ctx={"user_message": "the reactor plan a fresh competitor thread"},
        inputs={}, belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5, history=history,
    )
    by_id = {loop.id: loop for loop in loops}
    assert len(by_id) == 2
    other_id = next(lid for lid in by_id if lid != dwelling_id)

    assert by_id[dwelling_id].salience_features["dwell"] > 0.0
    assert by_id[other_id].salience_features["dwell"] == 0.0
    assert by_id[dwelling_id].salience_features["habituation"] > by_id[other_id].salience_features["habituation"]
