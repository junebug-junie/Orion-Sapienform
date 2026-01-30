from orion.schemas.telemetry.turn_effect_explanations import explain_alerts, summarize_explanations


def test_explain_alerts_coherence_drop():
    expl = explain_alerts([{"rule": "coherence_drop", "severity": "error"}])
    assert "context_fragmentation" in expl["likely_causes"]
    assert "What outcome are we optimizing for right now?" in expl["suggested_questions"]


def test_explain_alerts_novelty_spike():
    expl = explain_alerts([{"rule": "novelty_spike", "severity": "warn"}])
    assert "topic_shift" in expl["likely_causes"]
    assert "Is this a new direction or a side-thread?" in expl["suggested_questions"]


def test_summarize_explanations():
    expl = explain_alerts([{"rule": "valence_drop", "severity": "warn"}])
    summary = summarize_explanations(expl)
    assert summary.startswith("Alert Explanation:")
