from __future__ import annotations

from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY, resolve
from orion.schemas.thought import GroundingCapsuleV1, ThoughtEventV1


def test_grounding_capsule_round_trip() -> None:
    capsule = GroundingCapsuleV1(
        identity_summary=["I am Oríon."],
        relationship_summary=["Juniper is my collaborator."],
        response_policy_summary=["Speak plainly."],
        continuity_digest="We were mid-refactor.",
        belief_digest="Orion values continuity.",
        memory_digest="We were mid-refactor.\n\nOrion values continuity.",
        provenance={"identity_source": "configured_yaml", "pcr_ran": True},
    )
    dumped = capsule.model_dump(mode="json")
    restored = GroundingCapsuleV1.model_validate(dumped)
    assert restored == capsule
    assert restored.schema_version == "grounding.capsule.v1"


def test_grounding_capsule_registered() -> None:
    assert "GroundingCapsuleV1" in _REGISTRY
    assert resolve("GroundingCapsuleV1") is GroundingCapsuleV1
    assert SCHEMA_REGISTRY["GroundingCapsuleV1"].kind == "grounding.capsule.v1"


def test_thought_event_capsule_optional_and_defaults_none() -> None:
    thought = ThoughtEventV1.model_validate(
        {
            "event_id": "t-1",
            "correlation_id": "c-1",
            "session_id": None,
            "created_at": "2026-07-07T00:00:00+00:00",
            "imperative": "Answer directly.",
            "tone": "neutral",
            "strain_refs": [],
            "stance_harness_slice": {
                "task_mode": "direct_response",
                "conversation_frame": "mixed",
                "answer_strategy": "direct",
            },
        }
    )
    assert thought.grounding_capsule is None
    thought2 = thought.model_copy(
        update={"grounding_capsule": GroundingCapsuleV1(identity_summary=["I am Oríon."])}
    )
    dumped = thought2.model_dump(mode="json")
    restored = ThoughtEventV1.model_validate(dumped)
    assert restored.grounding_capsule is not None
    assert restored.grounding_capsule.identity_summary == ["I am Oríon."]
