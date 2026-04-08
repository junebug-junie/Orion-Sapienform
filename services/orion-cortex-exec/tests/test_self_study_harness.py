from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_DIR.parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.self_study_harness import render_self_study_harness, run_self_study_harness
from orion.core.bus.bus_schemas import ServiceRef


def _scenario_map(result):
    return {scenario.name: scenario for scenario in result.scenarios_run}


def test_harness_runs_golden_path_scenarios() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenarios = _scenario_map(result)

    assert result.summary.failed == 0
    assert set(scenarios) == {
        "factual_self_study",
        "concept_induction",
        "reflection",
        "factual_retrieval",
        "conceptual_retrieval",
        "reflective_retrieval",
        "factual_consumer",
        "conceptual_consumer",
        "reflective_consumer",
        "degraded_consumer_backend",
    }
    assert all(scenario.success for scenario in scenarios.values())
    assert "summary: passed=" in render_self_study_harness(result)


def test_factual_retrieval_stays_authoritative_only() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenario = _scenario_map(result)["factual_retrieval"]

    assert scenario.retrieval_mode == "factual"
    assert scenario.boundary_violations == []
    assert scenario.summary_counts["authoritative"] > 0
    assert scenario.summary_counts["induced"] == 0
    assert scenario.summary_counts["reflective"] == 0
    assert scenario.summary_counts["concepts"] == 0
    assert scenario.summary_counts["reflections"] == 0


def test_conceptual_retrieval_exposes_authoritative_and_induced_only() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenario = _scenario_map(result)["conceptual_retrieval"]

    assert scenario.retrieval_mode == "conceptual"
    assert scenario.boundary_violations == []
    assert scenario.summary_counts["authoritative"] > 0
    assert scenario.summary_counts["induced"] > 0
    assert scenario.summary_counts["reflective"] == 0
    assert scenario.summary_counts["concepts"] > 0
    assert scenario.summary_counts["reflections"] == 0


def test_reflective_retrieval_exposes_all_tiers_without_upcasting() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenario = _scenario_map(result)["reflective_retrieval"]

    assert scenario.retrieval_mode == "reflective"
    assert scenario.boundary_violations == []
    assert scenario.summary_counts["authoritative"] > 0
    assert scenario.summary_counts["induced"] > 0
    assert scenario.summary_counts["reflective"] > 0
    assert scenario.summary_counts["concepts"] > 0
    assert scenario.summary_counts["reflections"] > 0


def test_consumer_modes_preserve_trust_boundaries() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenarios = _scenario_map(result)

    factual = scenarios["factual_consumer"]
    conceptual = scenarios["conceptual_consumer"]
    reflective = scenarios["reflective_consumer"]

    assert factual.consumer_name == "legacy.plan"
    assert factual.retrieval_mode == "factual"
    assert factual.summary_counts["authoritative"] > 0
    assert factual.summary_counts["induced"] == 0
    assert factual.summary_counts["reflective"] == 0

    assert conceptual.consumer_name == "legacy.plan"
    assert conceptual.retrieval_mode == "conceptual"
    assert conceptual.summary_counts["authoritative"] > 0
    assert conceptual.summary_counts["induced"] > 0
    assert conceptual.summary_counts["reflective"] == 0

    assert reflective.consumer_name == "actions.respond_to_juniper_collapse_mirror.v1"
    assert reflective.retrieval_mode == "reflective"
    assert reflective.summary_counts["authoritative"] > 0
    assert reflective.summary_counts["induced"] > 0
    assert reflective.summary_counts["reflective"] > 0


def test_degraded_consumer_backend_is_reported_cleanly() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=1))
    scenario = _scenario_map(result)["degraded_consumer_backend"]

    assert scenario.success is True
    assert scenario.summary_counts == {}
    assert scenario.boundary_violations == []
    assert scenario.degradation_notes == ["self_study_unavailable:backend_down"]


def test_soak_mode_reports_stable_repeated_runs() -> None:
    result = asyncio.run(run_self_study_harness(source=ServiceRef(name="test-harness"), soak_iterations=3))

    assert result.soak is not None
    assert result.soak.enabled is True
    assert result.soak.iterations == 3
    assert result.soak.warnings == []
    assert result.summary.duplicate_notes == []
    assert result.summary.drift_notes == []


def test_cli_json_output_shape_is_stable(tmp_path: Path) -> None:
    json_out = tmp_path / "self-study-harness.json"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/self_study_harness.py",
            "--format",
            "json",
            "--skip-degraded",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["run_id"].startswith("self-study-harness-")
    assert payload["summary"]["failed"] == 0
    assert payload["scenarios_run"][0]["name"] == "factual_self_study"
    assert json.loads(json_out.read_text(encoding="utf-8"))["summary"]["failed"] == 0
