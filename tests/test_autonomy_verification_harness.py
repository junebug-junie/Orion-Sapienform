from __future__ import annotations

from pathlib import Path

from orion.autonomy.verification import AutonomyVerificationHarness, GraphDBClient, load_scenarios, write_report


def test_harness_verifies_all_fixture_scenarios_locally(tmp_path: Path):
    harness = AutonomyVerificationHarness()
    report = harness.run(load_scenarios(), publish_bus=False, verify_graphdb=False)

    assert report["summary"]["scenario_count"] == 6
    assert not report["summary"]["local_failures"]
    assert report["summary"]["passed_count"] == 6


def test_harness_marks_graphdb_skipped_when_unconfigured(monkeypatch):
    monkeypatch.delenv("GRAPHDB_URL", raising=False)
    harness = AutonomyVerificationHarness(graphdb_client=GraphDBClient(base_url="", repo=""))
    report = harness.run(load_scenarios(only=["self-model-snapshot"]), verify_graphdb=True)

    result = report["results"][0]
    assert result["graphdb_verify"] == "skipped"
    assert "GraphDB not configured" in result["graphdb_detail"]


def test_write_report_emits_json_and_markdown(tmp_path: Path):
    harness = AutonomyVerificationHarness()
    report = harness.run(load_scenarios(only=["proposal-only-goal"]), publish_bus=False, verify_graphdb=False)

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    write_report(report, json_out=json_out, md_out=md_out)

    assert json_out.exists()
    assert md_out.exists()
    assert "proposal-only-goal" in md_out.read_text()
