import scripts.attention_loops_store as store


def test_resolve_persists_outcome_and_suppresses(monkeypatch):
    calls = {"outcome": None, "suppressed": None}
    monkeypatch.setattr(store, "persist_loop_outcome", lambda o: calls.__setitem__("outcome", o) or True)
    monkeypatch.setattr(store, "suppress_loop", lambda k, **kw: calls.__setitem__("suppressed", k) or True)

    outcome = store.build_loop_outcome(
        loop_id="open-loop-x", theme_key="open-loop-x", verdict="resolved",
        actor="juniper", note="", salience_at_close=0.6, features_at_close={"evidence_strength": 0.8},
    )
    assert store.persist_loop_outcome(outcome) is True
    assert store.suppress_loop("open-loop-x") is True
    assert calls["outcome"].verdict == "resolved"
    assert calls["suppressed"] == "open-loop-x"
