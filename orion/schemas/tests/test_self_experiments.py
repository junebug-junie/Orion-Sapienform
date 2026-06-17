"""Schema tests for typed self-experiments."""

from orion.schemas.self_experiments import (
    SelfExperimentCreateRequestV1,
    SelfExperimentSpecV1,
)


def test_create_request_legacy_skill_id_field() -> None:
    req = SelfExperimentCreateRequestV1(skill_id="skills.system.time_now.v1")
    assert req.skill_id == "skills.system.time_now.v1"
    assert req.experiment_type is None


def test_create_request_typed_fields() -> None:
    req = SelfExperimentCreateRequestV1(
        experiment_type="runtime_drift_check",
        question="Check lag.",
        source="daily_metacog_v1",
    )
    assert req.experiment_type == "runtime_drift_check"
    assert req.question == "Check lag."
