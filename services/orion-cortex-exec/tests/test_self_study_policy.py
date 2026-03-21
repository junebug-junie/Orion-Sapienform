from __future__ import annotations

from app.self_study_policy import (
    build_self_study_consumer_context,
    build_self_study_consumer_request,
    render_self_study_consumer_context,
    resolve_self_study_consumer_policy,
)
from orion.schemas.self_study import SelfStudyRetrieveResultV1


def test_legacy_plan_project_planning_caps_to_conceptual() -> None:
    decision = resolve_self_study_consumer_policy(
        consumer_name="legacy.plan",
        output_mode="project_planning",
        config={"enabled": True, "retrieval_mode": "reflective"},
    )

    assert decision.consumer_kind == "planning_architecture"
    assert decision.enabled is True
    assert decision.retrieval_mode == "conceptual"
    assert decision.max_mode == "conceptual"
    assert decision.downgraded is True
    assert decision.allowed_trust_tiers == ["authoritative", "induced"]
    assert decision.policy_reason == "policy_downgraded_to_conceptual"


def test_legacy_plan_delivery_mode_stays_factual() -> None:
    decision = resolve_self_study_consumer_policy(
        consumer_name="legacy.plan",
        output_mode="implementation_guide",
        config={"enabled": True},
    )

    assert decision.consumer_kind == "delivery_debug"
    assert decision.retrieval_mode == "factual"
    assert decision.allowed_trust_tiers == ["authoritative"]


def test_collapse_mirror_allows_reflective_mode() -> None:
    decision = resolve_self_study_consumer_policy(
        consumer_name="actions.respond_to_juniper_collapse_mirror.v1",
        output_mode="implementation_guide",
        config={"enabled": True, "retrieval_mode": "reflective"},
    )

    assert decision.consumer_kind == "metacog_self"
    assert decision.retrieval_mode == "reflective"
    assert decision.max_mode == "reflective"
    assert decision.allowed_trust_tiers == ["authoritative", "induced", "reflective"]


def test_unknown_consumer_is_disabled() -> None:
    decision = resolve_self_study_consumer_policy(
        consumer_name="unknown.consumer",
        output_mode="reflective_depth",
        config={"enabled": True},
    )

    assert decision.enabled is False
    assert decision.policy_reason == "unknown_consumer"
    assert decision.retrieval_mode is None


def test_context_and_request_helpers_round_trip() -> None:
    decision = resolve_self_study_consumer_policy(
        consumer_name="legacy.plan",
        output_mode="project_planning",
        config={"enabled": True, "filters": {"limit": 3, "text_query": "planner"}},
    )
    request = build_self_study_consumer_request(decision, {"filters": {"limit": 3, "text_query": "planner"}})
    result = SelfStudyRetrieveResultV1.model_validate(
        {
            "run_id": "run-1",
            "retrieval_mode": request.retrieval_mode,
            "applied_filters": request.filters.model_dump(),
            "groups": [
                {
                    "trust_tier": "induced",
                    "items": [
                        {
                            "stable_id": "concept-1",
                            "trust_tier": "induced",
                            "record_type": "concept",
                            "title": "Planner pipeline",
                            "content_preview": "Planner delegates through runtime packs.",
                            "source_kind": "self_study",
                            "source_snapshot_id": "snapshot-1",
                            "metadata": {},
                        }
                    ],
                }
            ],
            "counts": {"total": 1, "authoritative": 0, "induced": 1, "reflective": 0, "facts": 0, "concepts": 1, "reflections": 0},
            "backend_status": [],
            "notes": [],
        }
    )

    context = build_self_study_consumer_context(decision, result=result, notes=["used in prompt"])
    rendered = render_self_study_consumer_context(context)

    assert request.retrieval_mode == "conceptual"
    assert request.filters.limit == 3
    assert request.filters.text_query == "planner"
    assert context.used is True
    assert context.notes == ["used in prompt"]
    assert "SELF-STUDY CONTEXT mode=conceptual consumer=legacy.plan" in rendered
    assert "Planner pipeline" in rendered
