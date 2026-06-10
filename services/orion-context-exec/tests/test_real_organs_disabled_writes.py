from __future__ import annotations

import pytest

from app.callable_namespace import ContextNamespace
from app.runner import ContextExecRunner, FAKE_ORGANS
from orion.schemas.context_exec import ContextExecPermissionV1, ContextExecRequestV1


@pytest.mark.asyncio
async def test_real_organs_repo_grep_when_enabled(tmp_path, monkeypatch):
    from app import settings as settings_mod

    root = tmp_path / "repo"
    (root / "services" / "orion-context-exec").mkdir(parents=True)
    (root / "services" / "orion-context-exec" / "README.md").write_text(
        "context-exec replaces AgentChainService intake\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(root))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(root))
    monkeypatch.setattr(settings_mod.settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "context_exec_fake_organs_enabled", False)

    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None

    perms = ContextExecPermissionV1(read_repo=True)
    run = await ContextExecRunner().run(
        ContextExecRequestV1(
            text="What breaks if I replace agent-chain-service with context-exec?",
            mode="repo_impact_analysis",
            permissions=perms,
        )
    )
    assert run.status == "ok"
    assert run.artifact.get("affected_paths")
    assert run.runtime_debug.get("real_repo_enabled") is True


def test_namespace_write_paths_blocked():
    perms = ContextExecPermissionV1(read_repo=True, write_repo=True)
    ns = ContextNamespace(permissions=perms)
    with pytest.raises(Exception):
        ns.repo.write("services/x.py", "data")
    with pytest.raises(Exception):
        ns.memory.write("x", "y")
