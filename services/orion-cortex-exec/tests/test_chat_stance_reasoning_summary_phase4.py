from __future__ import annotations

from datetime import datetime, timezone

from app.chat_stance import build_chat_stance_inputs, fallback_chat_stance_brief
from app.endogenous_runtime import EndogenousRuntimeAdoptionService, EndogenousRuntimeConfig
from orion.core.schemas.reasoning import ClaimV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning import InMemoryReasoningRepository


def _claim() -> ClaimV1:
    return ClaimV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        status="canonical",
        authority="local_inferred",
        confidence=0.9,
        salience=0.8,
        novelty=0.4,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["ev:1"],
            "source_channel": "orion:test",
            "source_kind": "unit",
            "producer": "pytest",
        },
        claim_text="Reasoning continuity is strong.",
        claim_kind="identity_signal",
    )


def test_chat_stance_uses_reasoning_summary_when_available() -> None:
    repo = InMemoryReasoningRepository()
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="unit",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=[_claim()],
        )
    )

    ctx = {
        "user_message": "who are you?",
        "reasoning_repository": repo,
    }
    built = build_chat_stance_inputs(ctx)
    assert built["reasoning_summary"]["active_claims"]
    assert ctx["chat_reasoning_summary_used"] is True
    assert ctx["chat_reasoning_debug"]["compiler_succeeded"] is True


def test_chat_stance_reasoning_fallback_safe_when_unavailable() -> None:
    ctx = {"user_message": "help me debug"}
    built = build_chat_stance_inputs(ctx)
    assert built["reasoning_summary"]["fallback_recommended"] is True
    assert ctx["chat_reasoning_summary_used"] is False


def test_chat_stance_records_endogenous_runtime_audit() -> None:
    from app import chat_stance as chat_stance_module

    original_runtime_service = chat_stance_module.runtime_service
    chat_stance_module.runtime_service = lambda: EndogenousRuntimeAdoptionService(
        config=EndogenousRuntimeConfig(
            enabled=True,
            surface_chat_reflective_enabled=True,
            surface_operator_enabled=False,
            allowed_workflow_types=frozenset({"reflective_journal"}),
            allow_mentor_branch=False,
            sample_rate=1.0,
            max_actions=4,
        )
    )
    try:
        ctx = {"user_message": "please reflect on contradictions"}
        _ = build_chat_stance_inputs(ctx)
        assert isinstance(ctx.get("chat_endogenous_runtime"), dict)
        assert ctx["chat_endogenous_runtime"]["audit"]["status"] in {"ok", "surface_disabled", "not_selected", "disabled", "sampled_out"}
    finally:
        chat_stance_module.runtime_service = original_runtime_service


def test_fallback_brief_consumes_reasoning_summary_fields() -> None:
    ctx = {
        "user_message": "who are you?",
        "chat_reasoning_summary": {
            "fallback_recommended": False,
            "active_claims": [{"claim_text": "continuity_signal"}],
            "relationship_signals": ["trusted_collaboration"],
            "tensions": ["open contradiction"],
            "hazards": ["unresolved_contradictions_present"],
        },
    }
    brief = fallback_chat_stance_brief(ctx)
    assert "continuity_signal" in brief.active_identity_facets
    assert "trusted_collaboration" in brief.active_relationship_facets
    assert "open contradiction" in brief.active_tensions
    assert "unresolved_contradictions_present" in brief.response_hazards
