"""Tests for operator proposal CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "orion_proposal_cli.py"
PYTHON = ROOT / "orion_dev" / "bin" / "python"


def _run_cli(*args: str, store: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        str(CLI),
        *args,
        "--store",
        str(store),
    ]
    env = {"PYTHONPATH": str(ROOT)}
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "proposals.json"


def test_proposal_cli_seed_demo_creates_fixture_records(store_path: Path) -> None:
    result = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["seeded"] is True
    assert store_path.exists()


def test_proposal_cli_list_filters_pending_review(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    seed_payload = json.loads(seed.stdout)
    pending_id = seed_payload["records"]["pending_review_memory"]

    result = _run_cli("list", "--status", "pending_review", "--json", store=store_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["proposal_id"] == pending_id
    assert rows[0]["status"] == "pending_review"

    stored_result = _run_cli("list", "--status", "stored", "--json", store=store_path)
    stored_rows = [
        json.loads(line) for line in stored_result.stdout.strip().splitlines() if line.strip()
    ]
    assert all(row["status"] == "stored" for row in stored_rows)
    assert pending_id not in {row["proposal_id"] for row in stored_rows}


def test_proposal_cli_show_displays_envelope_and_inner_artifact(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["pending_review_memory"]

    result = _run_cli("show", proposal_id, store=store_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proposal_id"] == proposal_id
    assert payload["envelope"]["proposal_type"] == "memory_correction_proposal"
    assert payload["inner_artifact_summary"]["artifact_type"] == "MemoryCorrectionProposalV1"
    assert payload["inner_artifact_summary"]["current_belief"]
    assert "execution_eligibility" in payload


def test_proposal_cli_triage_store_only_does_not_create_human_chore(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["stored_patch"]

    result = _run_cli(
        "triage",
        proposal_id,
        "--action",
        "store_only",
        "--reason",
        "not worth human attention",
        store=store_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "stored"
    assert payload["attention_required"] is False


def test_proposal_cli_triage_promote_sets_pending_review_attention(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["stored_patch"]

    result = _run_cli(
        "triage",
        proposal_id,
        "--action",
        "promote_to_review",
        "--reason",
        "identity memory correction",
        store=store_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "pending_review"
    assert payload["attention_required"] is True


def test_proposal_cli_review_reject_does_not_execute(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["pending_review_memory"]

    result = _run_cli(
        "review",
        proposal_id,
        "--decision",
        "reject",
        "--reason",
        "unsupported evidence",
        "--reviewer",
        "human:june",
        store=store_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "rejected"
    assert payload["execution_eligibility"]["eligible"] is False
    assert payload["execution_eligibility"]["execution_requested"] is False


def test_proposal_cli_review_approve_creates_eligibility_not_execution(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["pending_review_memory"]

    result = _run_cli(
        "review",
        proposal_id,
        "--decision",
        "approve",
        "--reason",
        "bounded and reversible",
        "--reviewer",
        "human:june",
        store=store_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "approved"
    assert payload["execution_eligibility"]["eligible"] is True
    assert payload["execution_eligibility"]["execution_requested"] is False


def test_proposal_cli_rejects_context_exec_as_reviewer_for_approval(store_path: Path) -> None:
    seed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "seed-demo",
            "--store",
            str(store_path),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    proposal_id = json.loads(seed.stdout)["records"]["pending_review_memory"]

    result = _run_cli(
        "review",
        proposal_id,
        "--decision",
        "approve",
        "--reason",
        "bad actor",
        "--reviewer",
        "context-exec",
        store=store_path,
    )
    assert result.returncode != 0
    assert "context-exec" in (result.stderr + result.stdout).lower()
