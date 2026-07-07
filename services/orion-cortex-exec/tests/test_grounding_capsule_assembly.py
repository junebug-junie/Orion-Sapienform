from __future__ import annotations

from app.grounding_capsule import (
    build_grounding_capsule,
    stance_slice_brief_from_step_text,
)


def test_build_grounding_capsule_from_ctx() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": ["Juniper is my collaborator."],
        "response_policy_summary": ["Speak plainly."],
        "continuity_digest": "We were mid-refactor.",
        "belief_digest": "Orion values continuity.",
        "memory_digest": "We were mid-refactor.\n\nOrion values continuity.",
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=True)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.relationship_summary == ["Juniper is my collaborator."]
    assert capsule.response_policy_summary == ["Speak plainly."]
    assert capsule.memory_digest == "We were mid-refactor.\n\nOrion values continuity."
    assert capsule.provenance["identity_source"] == "configured_yaml"
    assert capsule.provenance["pcr_ran"] is True


def test_build_grounding_capsule_identity_only_when_pcr_missing() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": [],
        "response_policy_summary": [],
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=False)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.continuity_digest is None
    assert capsule.belief_digest is None
    assert capsule.memory_digest is None
    assert capsule.provenance["pcr_ran"] is False


def test_stance_slice_brief_from_step_text_extracts_mode_and_frame() -> None:
    text = (
        '{"imperative":"Stay present.","tone":"warm",'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )
    brief = stance_slice_brief_from_step_text(text)
    assert brief["task_mode"] == "reflective_dialogue"
    assert brief["conversation_frame"] == "reflective"


def test_stance_slice_brief_from_step_text_tolerates_garbage() -> None:
    assert stance_slice_brief_from_step_text("not json") == {}
    assert stance_slice_brief_from_step_text("") == {}
