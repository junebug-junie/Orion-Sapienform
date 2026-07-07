from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.harness_finalize import (
    FinalizeReflectionV1,
    HarnessRepairOverlayV1,
    SubstrateFinalizeAppraisalV1,
)
from orion.schemas.thought import GroundingCapsuleV1, StanceHarnessSliceV1, ThoughtEventV1


def make_thought(**overrides: object) -> ThoughtEventV1:
    base = {
        "event_id": "t-1",
        "correlation_id": "c-1",
        "session_id": "sess-1",
        "created_at": datetime.now(timezone.utc),
        "imperative": "Answer directly.",
        "tone": "neutral",
        "strain_refs": ["node-a"],
        "evidence_refs": ["node-a"],
        "stance_harness_slice": StanceHarnessSliceV1(
            task_mode="direct_response",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    }
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def make_appraisal(**overrides: object) -> SubstrateFinalizeAppraisalV1:
    base = {
        "correlation_id": "c-1",
        "molecule_id": "appraisal-1",
        "draft_hash": "draft-hash-1",
        "surprise_level": 0.02,
        "strain_shift_refs": [],
        "open_loop_pressure": 0.0,
        "prediction_error_refs": [],
        "learning_refs": ["learn-1"],
        "alignment_hints": [],
    }
    base.update(overrides)
    return SubstrateFinalizeAppraisalV1.model_validate(base)


def make_repair_overlay(**overrides: object) -> HarnessRepairOverlayV1:
    base: dict[str, object] = {"mode": "default"}
    base.update(overrides)
    return HarnessRepairOverlayV1.model_validate(base)


def make_reflection(
    *,
    alignment_verdict: str = "aligned",
    **overrides: object,
) -> FinalizeReflectionV1:
    base = {
        "correlation_id": "c-1",
        "thought_event_id": "t-1",
        "substrate_appraisal_id": "appraisal-1",
        "draft_hash": "draft-hash-1",
        "imperative": "Answer directly.",
        "tone": "neutral",
        "strain_refs": ["node-a"],
        "alignment_verdict": alignment_verdict,
        "alignment_notes": [],
        "strain_unresolved": False,
    }
    base.update(overrides)
    return FinalizeReflectionV1.model_validate(base)


def make_grounding_capsule(**overrides: object) -> GroundingCapsuleV1:
    base = {
        "identity_summary": ["I am Oríon, a digital mind in development."],
        "relationship_summary": ["Juniper is my collaborator and steward."],
        "response_policy_summary": ["Speak plainly; no generic-assistant framing."],
        "continuity_digest": "We were mid-way through the grounding refactor.",
        "belief_digest": "Orion values continuity and self-coherence.",
        "memory_digest": "We were mid-way through the grounding refactor.\n\nOrion values continuity and self-coherence.",
        "provenance": {"identity_source": "configured_yaml", "pcr_ran": True},
    }
    base.update(overrides)
    return GroundingCapsuleV1.model_validate(base)
