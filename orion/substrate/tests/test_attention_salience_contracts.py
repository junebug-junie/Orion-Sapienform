from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    AttentionSalienceTraceV1,
    PendingAttentionCardV1,
)


def test_schemas_registered():
    for name in ("AttentionSalienceTraceV1", "AttentionLoopOutcomeV1", "PendingAttentionCardV1"):
        assert resolve(name) is not None
    assert SCHEMA_REGISTRY["AttentionSalienceTraceV1"].kind == "attention.salience.trace.v1"
    assert SCHEMA_REGISTRY["AttentionLoopOutcomeV1"].kind == "attention.loop.outcome.v1"


def test_loop_outcome_verdicts():
    o = AttentionLoopOutcomeV1(
        outcome_id="o1", loop_id="open-loop-x", theme_key="open-loop-x",
        verdict="resolved", actor="juniper", salience_at_close=0.7,
    )
    assert o.verdict == "resolved"


def test_pending_card_requires_plain_text():
    c = PendingAttentionCardV1(
        loop_id="open-loop-x", theme_key="open-loop-x",
        title="The reactor rollout plan", why_it_matters="You flagged it as urgent and it is unresolved.",
        what_triggered="3 detectors (current turn, autonomy, concept)", salience=0.7,
        weights_version="seed-v1",
    )
    assert c.status == "pending"
    assert c.title and not c.title.startswith("open-loop-")
