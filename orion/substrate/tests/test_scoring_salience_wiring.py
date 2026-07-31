"""Wiring tests for build_open_loops()/score_loop() over the GWT-coalition
Borda formula (replaces the killed SEED_WEIGHTS linear combiner + legacy
score_loop() fallback -- see orion.substrate.attention.salience's module
docstring for the kill/replace rationale).
"""

import pytest

from orion.schemas.attention_frame import AttentionSignalV1
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
    assert set(loop.salience_features) == {"schema_version", "evidence_strength", "evidence_breadth"}
    # Single-candidate tick -> raw mean(evidence_strength, evidence_breadth)
    # fallback (see borda_coalition_salience's n==1 case).
    assert loop.salience > 0.0


def test_score_loop_returns_precomputed_coalition_salience():
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    assert score_loop(loop) == pytest.approx(loop.salience)


def test_score_loop_ignores_stale_salience_v2_env(monkeypatch):
    """score_loop() has exactly one formula now -- ORION_ATTENTION_
    SALIENCE_V2_ENABLED no longer has any effect on it (the legacy
    constant-ladder fallback that flag used to gate was deleted 2026-07-31)."""
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    off = score_loop(loop)
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    on = score_loop(loop)
    assert off == on == pytest.approx(loop.salience)


def test_build_open_loops_rank_aggregates_across_the_real_competing_set():
    """The design brief's core requirement: salience is computed over the
    real competing set of loops scored together in ONE build_open_loops()
    call, not per-loop in isolation. A strong-evidence loop and a
    weak-evidence loop built in the SAME call must get distinguishable
    Borda salience."""
    signals = [
        AttentionSignalV1(
            signal_id="s-strong", source="current_turn", target_text="a strong reactor concern",
            target_type_hint="plan", signal_kind="test", salience=0.95, confidence=0.9,
            evidence_refs=["r1", "r2"],
        ),
        AttentionSignalV1(
            signal_id="s-weak", source="current_turn", target_text="a faint side note",
            target_type_hint="concept", signal_kind="test", salience=0.2, confidence=0.4,
            evidence_refs=["r3"],
        ),
    ]
    loops = build_open_loops(
        signals=signals,
        ctx={"user_message": "a strong reactor concern a faint side note"},
        inputs={}, belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    assert len(loops) == 2
    by_desc = {loop.description: loop for loop in loops}
    strong = by_desc["a strong reactor concern"]
    weak = by_desc["a faint side note"]
    assert strong.salience > weak.salience
