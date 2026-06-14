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


def test_proposal_cli_requires_explicit_store_path() -> None:
    result = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "list",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--store" in (result.stderr + result.stdout).lower()


def test_proposal_cli_show_missing_id_fails_cleanly(store_path: Path) -> None:
    result = _run_cli("show", "prop_does_not_exist", store=store_path)
    assert result.returncode != 0
    assert "proposal not found" in (result.stderr + result.stdout).lower()


def test_proposal_cli_review_missing_id_fails_cleanly(store_path: Path) -> None:
    result = _run_cli(
        "review",
        "prop_does_not_exist",
        "--decision",
        "approve",
        "--reason",
        "missing",
        store=store_path,
    )
    assert result.returncode != 0
    assert "proposal not found" in (result.stderr + result.stdout).lower()


def test_json_ledger_malformed_store_fails_cleanly_without_overwrite(tmp_path: Path) -> None:
    store = tmp_path / "proposals.json"
    corrupt = '{"records": ['
    store.write_text(corrupt, encoding="utf-8")

    result = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else sys.executable),
            str(CLI),
            "list",
            "--store",
            str(store),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "malformed json" in (result.stderr + result.stdout).lower()
    assert store.read_text(encoding="utf-8") == corrupt


def test_proposal_cli_list_shows_triage_and_attention_fields(store_path: Path) -> None:
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

    result = _run_cli("list", "--json", store=store_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.strip().splitlines() if line.strip()]
    pending_row = next(row for row in rows if row["proposal_id"] == pending_id)
    assert pending_row["triage_action"] == "promote_to_review"
    assert pending_row["attention_required"] is True
    assert pending_row["attention_reason"]
    assert pending_row["proposal_type"] == "memory_correction_proposal"
    assert pending_row["risk"]
    assert pending_row["title"]


def test_proposal_cli_show_shows_triage_reason(store_path: Path) -> None:
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
    assert payload["triage_action"] == "promote_to_review"
    assert payload["attention_required"] is True
    assert payload["attention_reason"]
    assert payload["review_status"]
    assert payload["execution_eligibility"]["execution_requested"] is False
    assert payload["execution_eligibility"]["eligible"] is False


def _seed_and_approve(store_path: Path) -> str:
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
    approve = _run_cli(
        "review",
        proposal_id,
        "--decision",
        "approve",
        "--reason",
        "dry run test",
        "--reviewer",
        "human:june",
        store=store_path,
    )
    assert approve.returncode == 0, approve.stderr
    return proposal_id


def test_dry_run_execute_requires_approved_proposal(store_path: Path) -> None:
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
    stored_id = json.loads(seed.stdout)["records"]["stored_patch"]

    result = _run_cli("dry-run-execute", stored_id, "--executor", "dry-run", store=store_path)
    assert result.returncode != 0
    assert "approved" in (result.stderr + result.stdout).lower()


def test_dry_run_execute_rejects_pending_review(store_path: Path) -> None:
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
    pending_id = json.loads(seed.stdout)["records"]["pending_review_memory"]

    result = _run_cli("dry-run-execute", pending_id, "--executor", "dry-run", store=store_path)
    assert result.returncode != 0
    assert "approved" in (result.stderr + result.stdout).lower()


def test_dry_run_execute_creates_receipt_without_mutation(store_path: Path) -> None:
    proposal_id = _seed_and_approve(store_path)
    before = store_path.read_text(encoding="utf-8")

    result = _run_cli("dry-run-execute", proposal_id, "--executor", "dry-run", store=store_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)

    assert receipt["status"] == "simulated"
    assert receipt["dry_run"] is True
    assert receipt["mutation_performed"] is False
    assert receipt["proposal_id"] == proposal_id
    assert receipt["executor_name"] == "dry-run"
    assert receipt["receipt_id"].startswith("rec_")
    assert receipt["planned_actions"]
    assert "changed_targets" not in receipt

    after = store_path.read_text(encoding="utf-8")
    assert after == before


def test_dry_run_execute_does_not_change_memory(store_path: Path, tmp_path: Path) -> None:
    proposal_id = _seed_and_approve(store_path)
    memory_probe = tmp_path / "memory_probe.txt"
    memory_probe.write_text("unchanged", encoding="utf-8")
    before_probe = memory_probe.read_text(encoding="utf-8")

    result = _run_cli("dry-run-execute", proposal_id, "--executor", "dry-run", store=store_path)
    assert result.returncode == 0, result.stderr

    assert memory_probe.read_text(encoding="utf-8") == before_probe


def test_dry_run_execute_does_not_change_repo(store_path: Path) -> None:
    proposal_id = _seed_and_approve(store_path)
    readme = ROOT / "README.md"
    before_mtime = readme.stat().st_mtime_ns

    result = _run_cli("dry-run-execute", proposal_id, "--executor", "dry-run", store=store_path)
    assert result.returncode == 0, result.stderr

    assert readme.stat().st_mtime_ns == before_mtime


def test_dry_run_execute_does_not_mark_executed_success(store_path: Path) -> None:
    proposal_id = _seed_and_approve(store_path)

    dry_run = _run_cli("dry-run-execute", proposal_id, "--executor", "dry-run", store=store_path)
    assert dry_run.returncode == 0, dry_run.stderr

    show = _run_cli("show", proposal_id, store=store_path)
    assert show.returncode == 0, show.stderr
    payload = json.loads(show.stdout)
    assert payload["status"] == "approved"
    assert payload["execution_eligibility"]["eligible"] is True
    assert payload["execution_eligibility"]["execution_requested"] is False
