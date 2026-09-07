from orion.cockpit.builders import (
    extract_mind_quality_fields,
    gap_hop,
    hop_from_association,
    hop_from_ingress,
    hop_from_motor_boot,
    hop_from_motor_step,
    hop_from_progress,
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


def test_hop_from_progress_failed_appraisal():
    hop = hop_from_progress(
        correlation_id="c1",
        seq=1,
        stage="pre_turn_appraisal",
        status="failed",
        visor_line="appraisal · FAILED TimeoutError",
        summary={"error": "TimeoutError", "failed_paradigms": ["repair_pressure"]},
        raw={"error": "TimeoutError", "failed_paradigms": ["repair_pressure"]},
    )
    assert hop.stage == "pre_turn_appraisal"
    assert hop.status == "failed"
    assert "FAILED" in hop.visor_line
    assert hop.raw["error"] == "TimeoutError"


def test_hop_from_progress_harness_dispatch_started():
    hop = hop_from_progress(
        correlation_id="c1",
        seq=7,
        stage="harness_dispatch",
        status="started",
        visor_line="harness_dispatch · contacting governor",
        summary={"phase": "started"},
        raw={"phase": "started"},
    )
    assert hop.stage == "harness_dispatch"
    assert hop.status == "started"


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


def test_hop_from_association_fresh_hollow():
    association = {
        "broadcast_stale": False,
        "read_source": "felt_state_reader",
        "broadcast": {
            "frame": {
                "open_loops": [],
                "debug": {"signal_count": 0},
            }
        },
    }
    hop = hop_from_association(
        correlation_id="c1",
        seq=2,
        association=association,
    )
    assert hop.summary["hollow"] is True
    assert hop.summary["signal_count"] == 0
    assert "empty" in hop.visor_line


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


def test_hop_from_situation_carries_exact_compact_text():
    from orion.cockpit.builders import hop_from_situation

    text = "Situation:\n- Your cabinet sensors (read just now): temp=30.1C"
    hop = hop_from_situation(
        correlation_id="c1",
        seq=5,
        compact_text=text,
        provider_status={"cabinet": "ok", "weather": "ok", "perception": "disabled"},
        source_summary={"cabinet": "file", "weather": "openmeteo"},
        perception_enabled=False,
    )
    assert hop.stage == "situation"
    assert hop.status == "ok"
    assert hop.raw["compact_text"] == text
    assert hop.summary["has_fragment"] is True
    assert hop.summary["cabinet_mentioned"] is True
    assert hop.summary["perception_enabled"] is False
    assert "cabinet" in hop.visor_line


def test_hop_from_situation_failed_and_empty():
    from orion.cockpit.builders import hop_from_situation

    failed = hop_from_situation(
        correlation_id="c1", seq=1, compact_text=None, status="failed"
    )
    assert failed.status == "failed"
    assert "failed" in failed.visor_line

    empty = hop_from_situation(correlation_id="c1", seq=2, compact_text="", status="ok")
    assert empty.status == "ok"
    assert "empty" in empty.visor_line
    assert empty.summary["has_fragment"] is False


def test_hop_from_ingress_carries_exact_user_message():
    hop = hop_from_ingress(
        correlation_id="c1",
        seq=0,
        ingress={
            "user_message": "hello there",
            "session_id": "s1",
            "mode": "orion",
            "attachment_count": 0,
            "observation_published": False,
        },
    )
    assert hop.stage == "ingress"
    assert hop.status == "ok"
    assert hop.raw["user_message"] == "hello there"
    assert hop.summary["user_message_len"] == len("hello there")
    assert hop.summary["session_id"] == "s1"
    assert hop.summary["mode"] == "orion"
    assert str(len("hello there")) in hop.visor_line
    assert hop.raw["observation_published"] is False
    assert hop.producer == "orion-hub"


def test_hop_from_ingress_empty_message():
    hop = hop_from_ingress(
        correlation_id="c1",
        seq=0,
        ingress={"user_message": "", "attachment_count": 0, "observation_published": False},
    )
    assert hop.status == "ok"
    assert hop.visor_line == "ingress · empty"


def test_gap_hop_still_supports_deferred_stages():
    hop = gap_hop(
        correlation_id="c1",
        seq=0,
        stage="closure",
        deferred_to="slice_c",
    )
    assert hop.status == "gap"
    assert hop.summary["deferred_to"] == "slice_c"
    assert "slice_c" in hop.visor_line


def test_extract_mind_quality_fields_nested():
    assert extract_mind_quality_fields({"disposition": "proceed"}) is None
    found = extract_mind_quality_fields(
        {
            "mind": {
                "mind_quality": "fallback_contract_only",
                "authorized_for_stance_use": False,
                "coloring_skipped": True,
            }
        }
    )
    assert found is not None
    assert found["mind_quality"] == "fallback_contract_only"
    assert found["authorized_for_stance_use"] is False
    assert found["coloring_skipped"] is True
