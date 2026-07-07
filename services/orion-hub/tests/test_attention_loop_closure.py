from scripts.attention_loops_store import build_loop_outcome


def test_build_loop_outcome_resolve():
    outcome = build_loop_outcome(
        loop_id="open-loop-x", theme_key="open-loop-x", verdict="resolved",
        actor="juniper", note="handled", salience_at_close=0.7, features_at_close={"x": 1},
    )
    assert outcome.verdict == "resolved"
    assert outcome.loop_id == "open-loop-x"
    assert outcome.outcome_id


def test_build_loop_outcome_rejects_bad_verdict():
    import pytest
    with pytest.raises(ValueError):
        build_loop_outcome(
            loop_id="x", theme_key="x", verdict="banana", actor="juniper",
            note="", salience_at_close=0.0, features_at_close={},
        )
