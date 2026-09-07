from orion.cockpit.builders import (
    gap_hop,
    hop_from_association,
    hop_from_motor_boot,
    hop_from_motor_step,
    hop_from_run_artifact,
    hop_from_stance_inputs,
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


def test_hop_from_association_ok():
    association = {
        "schema_version": "hub.association.bundle.v1",
        "correlation_id": "c1",
        "broadcast_stale": True,
        "broadcast": None,
        "execution_trajectory_slice": {"tick": 1},
        "repair_bundle": {"status": "ok"},
        "read_source": "felt_state_reader",
    }
    hop = hop_from_association(
        correlation_id="c1",
        seq=1,
        association=association,
    )
    assert hop.stage == "association"
    assert hop.status == "ok"
    assert hop.raw["broadcast_stale"] is True
    assert hop.raw["execution_trajectory_slice"]["tick"] == 1
    assert "association" in hop.visor_line
    assert hop.producer == "orion-hub"


def test_hop_from_stance_inputs_ok():
    payload = {
        "user_message": "hello there",
        "session_id": "s1",
        "llm_profile": "brain",
        "stance_inputs": {"user_message": "hello there", "surface_context": {"k": 1}},
    }
    hop = hop_from_stance_inputs(
        correlation_id="c1",
        seq=2,
        stance_inputs=payload,
    )
    assert hop.stage == "stance_inputs"
    assert hop.status == "ok"
    assert hop.raw["user_message"] == "hello there"
    assert hop.raw["stance_inputs"]["surface_context"]["k"] == 1
    assert hop.summary["user_message_len"] == len("hello there")
    assert hop.producer == "orion-hub"


def test_hop_from_motor_boot_carries_exact_prompt():
    prompt = "WHO YOU ARE\n...\nUSER: hi"
    hop = hop_from_motor_boot(
        correlation_id="c1",
        seq=4,
        prompt=prompt,
    )
    assert hop.stage == "motor_boot"
    assert hop.status == "ok"
    assert hop.raw["prompt"] == prompt
    assert hop.raw["prompt_char_len"] == len(prompt)
    assert hop.summary["prompt_char_len"] == len(prompt)
    assert str(len(prompt)) in hop.visor_line
    assert hop.producer == "orion-harness-governor"


def test_gap_hop_ingress_deferred_to_slice_c():
    hop = gap_hop(
        correlation_id="c1",
        seq=0,
        stage="ingress",
        deferred_to="slice_c",
    )
    assert hop.status == "gap"
    assert hop.summary["deferred_to"] == "slice_c"
    assert "slice_c" in hop.visor_line
