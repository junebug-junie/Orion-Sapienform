from __future__ import annotations

from pathlib import Path

from app import repo_tools


def test_repo_grep_and_read_respect_allow_deny(tmp_path, monkeypatch):
    from app import settings as settings_mod

    (tmp_path / "services" / "demo").mkdir(parents=True)
    allowed = tmp_path / "services" / "demo" / "chain.py"
    allowed.write_text("class AgentChainService:\n    pass\n", encoding="utf-8")
    denied = tmp_path / "services" / "demo" / ".env"
    denied.write_text("SECRET=1\n", encoding="utf-8")

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))

    hits = repo_tools.repo_grep("AgentChainService", limit=20)
    paths = {h.path for h in hits}
    assert "services/demo/chain.py" in paths
    assert not any(".env" in p for p in paths)

    rf = repo_tools.repo_read("services/demo/chain.py")
    assert rf is not None
    assert "AgentChainService" in rf.content
    assert repo_tools.repo_read(".env") is None
    assert repo_tools.repo_read("../outside") is None


def test_repo_grep_allows_app_prefix(tmp_path, monkeypatch):
    from app import settings as settings_mod

    (tmp_path / "app").mkdir(parents=True)
    target = tmp_path / "app" / "rlm_engine.py"
    target.write_text("def build_engine():\n    pass\n", encoding="utf-8")

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))

    hits = repo_tools.repo_grep("build_engine", path="app", limit=20)
    assert any(h.path == "app/rlm_engine.py" for h in hits)


def test_repo_write_blocked():
    try:
        repo_tools.repo_write()
        assert False, "expected PermissionError"
    except PermissionError:
        pass


def test_repo_list_top_level(tmp_path, monkeypatch):
    from app import settings as settings_mod

    (tmp_path / "services" / "demo").mkdir(parents=True)
    (tmp_path / "services" / "demo" / "main.py").write_text("pass\n")
    (tmp_path / "services" / "demo" / ".env").write_text("SECRET=1\n")
    (tmp_path / "docker").mkdir()

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))

    from app import repo_tools
    entries = repo_tools.repo_list("")
    assert "services/" in entries
    assert "docker/" not in entries          # docker/ not in ALLOW_PREFIXES
    assert not any(".env" in e for e in entries)


def test_repo_list_subpath(tmp_path, monkeypatch):
    from app import settings as settings_mod

    svc = tmp_path / "services" / "orion-context-exec" / "app"
    svc.mkdir(parents=True)
    (svc / "runner.py").write_text("pass\n")
    (svc / "__pycache__").mkdir()

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))

    from app import repo_tools
    entries = repo_tools.repo_list("services/orion-context-exec/app")
    assert "services/orion-context-exec/app/runner.py" in entries
    assert not any("__pycache__" in e for e in entries)


def test_repo_list_traversal_blocked(tmp_path, monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_repo_root", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "orion_repo_root", str(tmp_path))

    from app import repo_tools
    assert repo_tools.repo_list("../outside") == []
    assert repo_tools.repo_list("/etc") == []
