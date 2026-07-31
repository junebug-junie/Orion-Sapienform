from datetime import datetime, timezone

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from app.reverie import derive_salience


def _broadcast(selected_id: str, loops: list[OpenLoopV1]) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=loops)
    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at, frame=frame,
        selected_open_loop_id=selected_id, coalition_stability_score=0.3,
    )


def test_derive_salience_reads_loop_salience_directly():
    """2026-07-31: there is exactly one salience formula now
    (orion.substrate.attention.salience's GWT-coalition Borda
    rank-aggregation) -- derive_salience() always reads loop.salience,
    unconditionally, with no ORION_ATTENTION_SALIENCE_V2_ENABLED branch
    left to test. See derive_salience()'s own docstring for what used to
    live here (the deleted legacy branch: max of OpenLoopV1's seven raw
    constant-ladder fields)."""
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.81)
    b = _broadcast("loop-a", [loop])
    assert derive_salience(b) == 0.81


def test_derive_salience_falls_back_to_stability_when_salience_is_zero():
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.0)
    b = _broadcast("loop-a", [loop])
    assert derive_salience(b) == pytest.approx(0.3)


def test_derive_salience_zero_can_be_a_real_outranked_loop_not_only_no_evidence():
    """2026-07-31 review finding: under Borda rank-aggregation, `loop.
    salience == 0.0` is NOT only a "no evidence at all" case -- a loop with
    real, nonzero evidence_strength/evidence_breadth legitimately lands at
    exactly 0.0 whenever it is the strict last-place candidate on BOTH
    voters in a multi-loop tick (see
    orion.substrate.attention.salience.borda_coalition_salience). This
    test constructs exactly that scenario directly against a real
    build_open_loops() competing set (not a hand-set salience=0.0 stub) to
    make the disclosed gap inspectable: derive_salience() still falls back
    to coalition_stability_score for this loop even though it has real
    evidence, simply because it lost this tick's ranking. Not fixed here
    -- see derive_salience()'s own docstring for why."""
    from orion.substrate.attention.scoring import build_open_loops
    from orion.schemas.attention_frame import AttentionSignalV1

    signals = [
        AttentionSignalV1(
            signal_id="s-weak", source="current_turn", target_text="a faint concern",
            signal_kind="test", salience=0.1, confidence=0.5, evidence_refs=[],
        ),
        AttentionSignalV1(
            signal_id="s-strong-a", source="current_turn", target_text="a strong concern",
            signal_kind="test", salience=0.9, confidence=0.9, evidence_refs=["r2"],
        ),
        AttentionSignalV1(
            signal_id="s-strong-b", source="autonomy", target_text="another strong concern",
            signal_kind="test", salience=0.85, confidence=0.85, evidence_refs=["r3"],
        ),
    ]
    loops = build_open_loops(
        signals=signals,
        ctx={"user_message": "a faint concern a strong concern another strong concern"},
        inputs={}, belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    weak_loop = next(l for l in loops if l.description == "a faint concern")
    assert weak_loop.salience == 0.0, "expected the weak loop to be strict last-place"

    b = _broadcast(weak_loop.id, loops)
    # Real evidence existed (evidence_strength=0.1*0.5=0.05 > 0), but
    # derive_salience() cannot tell "real but outranked" from "no evidence"
    # -- it falls back to the coalition's stability score either way.
    assert derive_salience(b) == pytest.approx(0.3)
