from orion.cockpit.builders import (
    gap_hop,
    hop_from_motor_step,
    hop_from_run_artifact,
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


def test_hop_from_run_artifact_draft_and_finalize_consecutive_seq():
    hops = hop_from_run_artifact(
        correlation_id="c1",
        seq=0,
        run={
            "draft_text": "substrate read",
            "reflection": "wrapped up",
            "compliance_verdict": "pass",
        },
    )
    assert len(hops) == 2
    assert hops[0].seq == 0
    assert hops[0].stage == "draft_appraisal"
    assert hops[1].seq == 1
    assert hops[1].stage == "finalize"
    assert hops[0].producer == "orion-hub"
    assert hops[1].producer == "orion-hub"
