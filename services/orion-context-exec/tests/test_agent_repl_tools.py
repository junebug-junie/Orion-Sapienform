"""Tests for agent_repl workbench tools: repo navigation, workspace artifacts, patch validation."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(CTX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture()
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import settings as settings_mod

    (tmp_path / "services" / "demo").mkdir(parents=True)
    py_file = tmp_path / "services" / "demo" / "sample.py"
    py_file.write_text(
        "import os\n\n\nclass Demo:\n    def run(self):\n        return 1\n\n\ndef helper():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "services" / "demo" / "notes.txt").write_text("plain\n", encoding="utf-8")
    (tmp_path / "services" / "demo" / ".env").write_text("SECRET=1\n", encoding="utf-8")
    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))
    return tmp_path


def test_context_exec_agent_repl_max_steps_default_is_32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_EXEC_AGENT_REPL_MAX_STEPS", raising=False)
    from app.settings import ContextExecSettings

    cfg = ContextExecSettings()
    assert cfg.context_exec_agent_repl_max_steps == 32


def test_repo_read_range_happy_path(repo_root: Path) -> None:
    from app import repo_tools

    out = repo_tools.repo_read_range("services/demo/sample.py", 1, 3)
    assert "1|import os" in out
    assert "3|" in out


def test_repo_read_range_rejects_traversal_and_denied(repo_root: Path) -> None:
    from app import repo_tools

    assert "traversal" in repo_tools.repo_read_range("../outside", 1, 1).lower()
    assert "blocked" in repo_tools.repo_read_range("services/demo/.env", 1, 1).lower()


def test_repo_find_files_respects_limit(repo_root: Path) -> None:
    from app import repo_tools

    out = repo_tools.repo_find_files("*.py", path="services/demo", limit=1)
    assert "services/demo/sample.py" in out
    assert "truncated" in out


def test_repo_outline_python(repo_root: Path) -> None:
    from app import repo_tools

    out = repo_tools.repo_outline("services/demo/sample.py")
    assert "class Demo" in out
    assert "def helper" in out
    assert "L" in out


def test_repo_outline_unsupported_for_non_python(repo_root: Path) -> None:
    from app import repo_tools

    out = repo_tools.repo_outline("services/demo/notes.txt")
    assert "outline unsupported" in out


def test_repo_grep_literal_mode(repo_root: Path) -> None:
    from app import repo_tools

    hits = repo_tools.repo_grep("class Demo", literal=True, limit=10)
    assert any(h.path == "services/demo/sample.py" for h in hits)


def test_patch_validate_accepts_valid_diff(repo_root: Path) -> None:
    from app import repo_tools

    diff = """--- a/services/demo/sample.py
+++ b/services/demo/sample.py
@@ -1,3 +1,3 @@
 import os
 
-
+#
"""
    out = repo_tools.patch_validate(diff)
    assert out.startswith("valid:")
    assert "services/demo/sample.py" in out


def test_patch_validate_rejects_env_and_traversal(repo_root: Path) -> None:
    from app import repo_tools

    env_diff = """--- a/services/demo/.env
+++ b/services/demo/.env
@@ -1 +1 @@
-SECRET=1
+SECRET=2
"""
    assert "invalid" in repo_tools.patch_validate(env_diff).lower()
    assert "denied" in repo_tools.patch_validate(env_diff).lower()

    traversal_diff = """--- a/../outside.py
+++ b/../outside.py
@@ -1 +1 @@
-x
+y
"""
    out = repo_tools.patch_validate(traversal_diff)
    assert "invalid" in out.lower()
    assert "traversal" in out.lower()


@pytest.fixture()
def workspace_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.workspace as workspace_mod
    from app.workspace import allocate_workspace

    ws_root = tmp_path / "workspaces"
    monkeypatch.setattr(workspace_mod.settings, "context_exec_storage_root", str(tmp_path))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_root", str(ws_root))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_repo_root", str(tmp_path / "canonical"))
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_enabled", True)
    monkeypatch.setattr(workspace_mod.settings, "context_exec_workspace_materialize_repo", False)

    ws = allocate_workspace("ctxrun_tools")
    info = {
        "enabled": True,
        "allocated": True,
        "root": str(ws.root),
        "scratch_dir": str(ws.scratch_dir),
        "outputs_dir": str(ws.outputs_dir),
        "patches_dir": str(ws.patches_dir),
        "repo_dir": str(ws.repo_dir),
    }
    return ws, info


def test_workspace_write_read_list(workspace_bundle) -> None:
    from app import workspace_tools

    ws, info = workspace_bundle
    wrote = workspace_tools.workspace_write(info, ws, "scratch/note.txt", "hello")
    assert wrote.startswith("wrote scratch/note.txt")
    assert workspace_tools.workspace_read(info, ws, "scratch/note.txt") == "hello"
    listing = workspace_tools.workspace_list(info, ws, "scratch")
    assert "scratch/note.txt" in listing


def test_workspace_tools_reject_absolute_and_traversal(workspace_bundle) -> None:
    from app import workspace_tools

    ws, info = workspace_bundle
    assert "absolute" in workspace_tools.workspace_write(info, ws, "/etc/passwd", "x").lower()
    assert "traversal" in workspace_tools.workspace_read(info, ws, "../outside").lower()


def test_workspace_write_patch_only_under_patches(workspace_bundle) -> None:
    from app import workspace_tools

    ws, info = workspace_bundle
    out = workspace_tools.workspace_write_patch(info, ws, "fix-demo", "diff text")
    assert out.startswith("wrote patches/fix-demo.patch")
    assert (ws.patches_dir / "fix-demo.patch").is_file()
    assert not (ws.root / "fix-demo.patch").is_file()


def test_workspace_tools_missing_handle_when_allocated(workspace_bundle) -> None:
    from app import workspace_tools

    _, info = workspace_bundle
    out = workspace_tools.workspace_write(info, None, "scratch/x.txt", "x")
    assert "unavailable" in out.lower()
    assert "handle missing" in out.lower()


def test_workspace_tools_unavailable_when_not_allocated() -> None:
    from app import workspace_tools

    info = {"enabled": True, "allocated": False, "error": "boom"}
    out = workspace_tools.workspace_write(info, None, "scratch/x.txt", "x")
    assert "unavailable" in out.lower()


def test_make_tools_includes_new_tools() -> None:
    from app.smolcode_engine import _make_tools

    runtime = MagicMock()
    runtime.request = MagicMock()
    runtime.request.permissions.read_repo = True
    runtime.repo_grep.return_value = []
    runtime.repo_read.return_value = None

    import asyncio

    loop = asyncio.new_event_loop()
    try:
        tools = _make_tools(runtime, loop, workspace_info=None, workspace=None)
        names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools}
    finally:
        loop.close()

    expected = {
        "repo_grep",
        "repo_read",
        "repo_read_range",
        "repo_find_files",
        "repo_tree",
        "repo_outline",
        "repo_list",
        "patch_validate",
        "workspace_write",
        "workspace_read",
        "workspace_list",
        "workspace_write_patch",
        "workspace_write_report",
        "recall_query",
    }
    assert expected.issubset(names)


@pytest.mark.asyncio
async def test_health_includes_agent_budget_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from httpx import ASGITransport, AsyncClient

    service_dir = Path(__file__).resolve().parents[1]
    root = service_dir.parents[1]
    if str(service_dir) not in sys.path:
        sys.path.insert(0, str(service_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for mod in ("app.main", "app.api", "app.proposal_review_api"):
        sys.modules.pop(mod, None)
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    data = resp.json()
    assert "agent_repl_max_steps" in data
    assert "max_seconds" in data
    assert "llm_timeout_sec" in data
    assert isinstance(data["agent_repl_max_steps"], int)


@pytest.mark.asyncio
async def test_agent_repl_runtime_debug_includes_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import runner as runner_mod
    from app.runner import ContextExecRunner
    from app.rlm_engine import RLMEngine
    from orion.schemas.context_exec import (
        ContextExecRequestV1,
        context_exec_permissions_for_llm_profile,
    )

    class StubEngine(RLMEngine):
        engine_name = "smolcode"

        async def run(self, request, namespace, *, organ_runtime=None,
                      step_callbacks=None, max_steps=None, per_step_timeout=None,
                      workspace_info=None, workspace=None):
            return {"summary": "done", "engine": "smolcode", "mode": request.mode}

    r = ContextExecRunner(engine=StubEngine())

    async def fake_resolve(profile):
        from app.llm_profile_resolver import LLMProfileSelection
        return LLMProfileSelection(requested=profile, selected="agent", route_used="agent")

    monkeypatch.setattr(runner_mod, "resolve_llm_profile", fake_resolve)
    monkeypatch.setattr(runner_mod.settings, "context_exec_run_ledger_enabled", False)
    monkeypatch.setattr(runner_mod.settings, "context_exec_agent_repl_max_steps", 32)
    monkeypatch.setattr(runner_mod.settings, "context_exec_workspace_enabled", False)

    req = ContextExecRequestV1(
        text="inspect repo",
        mode="agent_repl",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        llm_profile="agent",
    )
    run = await r.run(req)
    assert run.runtime_debug.get("agent_repl_max_steps") == 32
