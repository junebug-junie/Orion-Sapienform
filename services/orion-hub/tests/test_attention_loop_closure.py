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


def test_persist_loop_outcome_never_raises(monkeypatch):
    import scripts.attention_loops_store as store

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "_engine", boom)
    outcome = build_loop_outcome(
        loop_id="x", theme_key="x", verdict="resolved", actor="j",
        note="", salience_at_close=0.5, features_at_close={},
    )
    assert store.persist_loop_outcome(outcome) is False
    assert store.suppress_loop("x") is False


def test_build_loop_outcome_accepts_orions_non_final_acted_verdict():
    """2026-10-01 attend-to-act loop: Orion's world-action settle path writes `acted`; the Hub
    validator must accept it, and it must never be terminal (verdicts.TERMINAL_VERDICTS)."""
    from orion.substrate.attention.verdicts import TERMINAL_VERDICTS

    outcome = build_loop_outcome(
        loop_id="open-loop-cab", theme_key="open-loop-cab", verdict="acted",
        actor="orion", note="shed_background_gpu expired", salience_at_close=0.4, features_at_close={},
    )
    assert outcome.verdict == "acted" and outcome.actor == "orion"
    assert "acted" not in TERMINAL_VERDICTS
