"""The label Hub puts on each unified turn for harness_turn_trace.workflow (#3255)."""
from __future__ import annotations

from orion.hub import turn_orchestrator as to


def test_label_is_the_callers_source_tag_else_reading_else_chat():
    assert to.turn_workflow_label({"source": "endogenous_outreach"}) == "endogenous_outreach"
    assert to.turn_workflow_label({"source": " curiosity_investigation "}, utterance_origin="orion") \
        == "curiosity_investigation"
    assert to.turn_workflow_label({}, reading_only=True) == "reading"
    assert to.turn_workflow_label({"source": ""}, utterance_origin="juniper") == "chat"
    assert to.turn_workflow_label(None) is None
    assert to.turn_workflow_label({"source": None}, utterance_origin="orion") is None
