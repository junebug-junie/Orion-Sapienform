from orion.cockpit.builders import (
    gap_hop,
    hop_from_motor_step,
    hop_from_thought,
)


def test_hop_from_thought_proceed():
    hop = hop_from_thought(
        correlation_id="c1",
        seq=0,
        thought={
            "disposition": "proceed",
            "disposition_reasons": ["ok"],
            "imperative": "stay with it",
            "tone": "steady",
        },
    )
    assert hop.stage == "stance_decision"
    assert hop.status == "ok"
    assert "proceed" in hop.visor_line
    assert hop.raw["disposition"] == "proceed"
    assert hop.producer == "orion-hub"


def test_hop_from_motor_step():
    hop = hop_from_motor_step(
        correlation_id="c1",
        seq=3,
        step_index=2,
        step={"type": "tool_use", "name": "Read", "input": {"path": "x"}},
    )
    assert hop.stage == "motor_hop"
    assert hop.summary["step_index"] == 2
    assert hop.raw["step"]["name"] == "Read"


def test_gap_hop_motor_boot():
    hop = gap_hop(correlation_id="c1", seq=1, stage="motor_boot")
    assert hop.status == "gap"
    assert hop.stage == "motor_boot"
