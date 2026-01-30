from orion.schemas.telemetry.turn_effect_policy import (
    recommend_actions_from_alerts,
    summarize_recommended_actions,
)


def test_policy_coherence_drop_error_actions():
    policy = recommend_actions_from_alerts([{"rule": "coherence_drop", "severity": "error"}])
    assert "stabilize_mode" in policy["actions"]
    assert "shorten_responses" in policy["actions"]
    assert policy["severity_max"] == "error"


def test_policy_novelty_spike_warn_actions():
    policy = recommend_actions_from_alerts([{"rule": "novelty_spike", "severity": "warn"}])
    assert "capture_insight_candidate" in policy["actions"]
    assert "summarize_key_shift" in policy["actions"]


def test_policy_dedup_and_ordering():
    policy = recommend_actions_from_alerts(
        [
            {"rule": "coherence_drop", "severity": "warn"},
            {"rule": "coherence_drop", "severity": "warn"},
        ]
    )
    assert policy["actions"].count("shorten_responses") == 1
    assert policy["actions"][0] in {"shorten_responses", "stabilize_mode", "ask_one_clarifying_question"}


def test_policy_summary():
    policy = recommend_actions_from_alerts([{"rule": "valence_drop", "severity": "warn"}])
    summary = summarize_recommended_actions(policy)
    assert summary.startswith("Actions:")
