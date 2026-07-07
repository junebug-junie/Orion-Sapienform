import orion.substrate.attention.salience as salience_mod
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
