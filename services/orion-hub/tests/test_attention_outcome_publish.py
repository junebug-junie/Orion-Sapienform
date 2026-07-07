from orion.schemas.attention_salience import AttentionLoopOutcomeV1
from scripts.bus_publish import build_loop_outcome_envelope


def test_envelope_kind_and_payload():
    outcome = AttentionLoopOutcomeV1(
        outcome_id="o1", loop_id="l1", theme_key="l1", verdict="resolved",
        actor="juniper", salience_at_close=0.5,
    )
    env = build_loop_outcome_envelope(outcome)
    assert env.kind == "attention.loop.outcome.v1"
    assert env.payload["verdict"] == "resolved"
