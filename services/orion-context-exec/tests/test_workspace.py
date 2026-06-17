from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import app.workspace as workspace_mod
from app.workspace import allocate_workspace, workspace_dir, workspace_health_block
from app.runner import ContextExecRunner
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.fixture()
def workspace_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws_root = tmp_path / "workspaces"
    monkeypatch.setattr(workspace_mod.settings, "context_exec_storage_root", str(tmp_path))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_root", str(ws_root))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_repo_root", str(tmp_path / "canonical"))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_enabled", True)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_materialize_repo", False)
    return ws_root


def test_workspace_dir(workspace_roots: Path) -> None:
    assert workspace_dir("ctxrun_abc") == workspace_roots / "ctxrun_abc"


def test_allocate_workspace_creates_layout(workspace_roots: Path) -> None:
    ws = allocate_workspace("ctxrun_test1")
    assert ws.root.is_dir()
    assert ws.scratch_dir.is_dir()
    assert ws.outputs_dir.is_dir()
    assert ws.patches_dir.is_dir()
    assert ws.repo_dir.is_dir()
    assert ws.manifest_path.is_file()


def test_manifest_safety_posture(workspace_roots: Path) -> None:
    ws = allocate_workspace("ctxrun_manifest")
    manifest = json.loads(ws.manifest_path.read_text())
    assert manifest["schema"] == "orion.context_exec.workspace.v1"
    assert manifest["run_id"] == "ctxrun_manifest"
    assert manifest["canonical_repo_write_allowed"] is False
    assert manifest["workspace_write_allowed"] is True
    assert manifest["repo_materialized"] is False
    assert manifest["repo_materialize_mode"] == "none"
    assert manifest["canonical_repo_root"]


def test_allocate_workspace_idempotent(workspace_roots: Path) -> None:
    first = allocate_workspace("ctxrun_idem")
    created_at = json.loads(first.manifest_path.read_text())["created_at"]
    second = allocate_workspace("ctxrun_idem")
    assert second.root == first.root
    assert json.loads(second.manifest_path.read_text())["created_at"] == created_at


def test_workspace_health_block(workspace_roots: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_roots.mkdir(parents=True, exist_ok=True)
    block = workspace_health_block()
    assert block["enabled"] is True
    assert block["root"] == str(workspace_roots)
    assert block["present"] is True
    assert block["writable"] is True
    assert block["materialize_repo"] is False


def test_materialize_repo_respects_byte_cap(
    workspace_roots: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "big.bin").write_bytes(b"x" * 100)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_materialize_repo", True)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_copy_max_bytes", 50)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_repo_root", str(canonical))

    ws = allocate_workspace("ctxrun_cap")
    manifest = json.loads(ws.manifest_path.read_text())
    assert manifest["repo_materialized"] is False
    assert manifest["repo_materialize_mode"] == "aborted_over_limit"
    assert manifest.get("warnings")


def test_materialize_repo_skips_git(
    workspace_roots: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    git_dir = canonical / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (canonical / "README.md").write_text("hello")
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_materialize_repo", True)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_copy_max_bytes", 1_000_000)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_repo_root", str(canonical))

    ws = allocate_workspace("ctxrun_skipgit")
    assert not (ws.repo_dir / ".git").exists()
    assert (ws.repo_dir / "README.md").is_file()
    manifest = json.loads(ws.manifest_path.read_text())
    assert manifest["repo_materialized"] is True


@pytest.mark.asyncio
async def test_runner_workspace_allocation_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.runner.settings.context_exec_workspace_enabled", True)
    monkeypatch.setattr("app.runner.settings.context_exec_run_ledger_enabled", False)
    monkeypatch.setattr("app.runner.settings.context_exec_agent_synthesis_enabled", False)
    monkeypatch.setattr("app.runner.settings.context_exec_proposal_ledger_enabled", False)
    monkeypatch.setattr("app.runner.settings.context_exec_fake_organs_enabled", True)
    monkeypatch.setattr("app.runner.settings.orion_bus_enabled", False)
    monkeypatch.setattr("app.runner.settings.rlm_engine", "fake")
    with patch("app.workspace.allocate_workspace", side_effect=RuntimeError("boom")):
        runner = ContextExecRunner()
        req = ContextExecRequestV1(text="hello", mode="belief_provenance")
        run = await runner.run(req)
    assert run.status in {"ok", "schema_invalid", "error"}


@pytest.mark.asyncio
async def test_runner_allocates_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ws_root = tmp_path / "workspaces"
    monkeypatch.setattr("app.settings.settings.context_exec_workspace_root", str(ws_root))
    monkeypatch.setattr("app.settings.settings.context_exec_storage_root", str(tmp_path))
    monkeypatch.setattr("app.settings.settings.context_exec_repo_root", str(tmp_path / "repo"))
    monkeypatch.setattr("app.settings.settings.context_exec_workspace_enabled", True)
    monkeypatch.setattr("app.settings.settings.context_exec_run_ledger_enabled", False)
    monkeypatch.setattr("app.settings.settings.context_exec_agent_synthesis_enabled", False)
    monkeypatch.setattr("app.settings.settings.context_exec_proposal_ledger_enabled", False)
    monkeypatch.setattr("app.settings.settings.context_exec_fake_organs_enabled", True)
    monkeypatch.setattr("app.settings.settings.orion_bus_enabled", False)
    monkeypatch.setattr("app.settings.settings.rlm_engine", "fake")

    from app.runner import FAKE_ORGANS

    FAKE_ORGANS.memory_hits = [
        {"claim": "User is from Denver", "source_ref": "m:1", "verified": True, "confidence": 0.9}
    ]
    runner = ContextExecRunner()
    req = ContextExecRequestV1(text="hello", mode="belief_provenance")
    run = await runner.run(req)
    assert run.runtime_debug.get("workspace", {}).get("allocated") is True
    assert (ws_root / run.run_id / "manifest.json").is_file()
