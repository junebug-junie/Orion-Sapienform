from __future__ import annotations

from orion.social import SocialRoomShakedownWorkflow, load_shakedown_pack
from orion.social.scenario_replay import load_scenarios


def test_shakedown_pack_models_link_issues_to_regressions() -> None:
    issues, fixes = load_shakedown_pack()
    fix_issue_ids = {item.issue_id for item in fixes}

    assert len(issues) >= 14
    assert len(fixes) >= 14
    assert all(item.linked_regression_scenario for item in issues)
    assert all(item.linked_regression_test for item in issues)
    assert all(item.issue_id in fix_issue_ids for item in issues)


def test_shakedown_workflow_verifies_linked_replay_fixes() -> None:
    workflow = SocialRoomShakedownWorkflow()
    issues, fixes = load_shakedown_pack()
    scenarios = load_scenarios()

    report = workflow.run(issues=issues, fixes=fixes, scenarios=scenarios)

    assert report["summary"]["issue_count"] >= 14
    assert report["summary"]["verified_count"] >= 14
    assert report["summary"]["missing_regression_links"] == []
    assert all(item["verified"] is True for item in report["results"])


def test_shakedown_failure_output_remains_inspectable() -> None:
    workflow = SocialRoomShakedownWorkflow()
    issues, fixes = load_shakedown_pack(only_issue_ids=["issue-local-landing-over-bridged"])
    bad_fix = fixes[0].model_copy(update={"status": "open"})

    report = workflow.run(issues=issues, fixes=[bad_fix], scenarios=load_scenarios())

    assert report["summary"]["verified_count"] == 0
    result = report["results"][0]
    assert result["verified"] is False
    assert result["evaluation"]["passed"] is True
    assert result["issue"]["linked_regression_scenario"] == "explicit-landing-stays-local"


def test_shakedown_keeps_safety_visible_for_private_boundary_issue() -> None:
    workflow = SocialRoomShakedownWorkflow()
    issues, fixes = load_shakedown_pack()
    scenarios = load_scenarios(only=["blocked-private-material-stays-blocked"])

    report = workflow.run(issues=[], fixes=fixes, scenarios=scenarios)
    replay_result = scenarios[0]
    harness_report = workflow.harness.run([replay_result])
    result = harness_report["results"][0]

    assert report["summary"]["issue_count"] == 0
    assert result["observed_outcomes"]["private_material_blocked"] is True
    assert any("blocked string remained absent" in item for item in result["safety_observations"])
