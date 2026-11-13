"""export_consult_source: tracked files only, no .git, no .env; current swaps; old exports pruned."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import export_consult_source as ecs  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", "https://user:secret-token@example.invalid/x.git")
    (repo / ".gitignore").write_text(".env\n")
    (repo / "code.py").write_text("x = 1\n")
    (repo / ".env").write_text("TOKEN=secret\n")
    _git(repo, "add", ".gitignore", "code.py")
    _git(repo, "commit", "-qm", "one")
    return repo


def test_export_has_tracked_files_only_and_swaps_current(tmp_path):
    repo, dest = _repo(tmp_path), tmp_path / "out"
    sha1, created = ecs.export(repo, "HEAD", dest)
    current = dest / "current"
    assert created and current.is_symlink() and (current / "code.py").read_text() == "x = 1\n"
    assert not (current / ".git").exists() and not (current / ".env").exists()
    assert ecs.find_forbidden(current.resolve()) == []
    assert "secret-token" not in "".join(p.read_text() for p in current.resolve().rglob("*") if p.is_file())
    assert ecs.export(repo, "HEAD", dest) == (sha1, False), "same commit: no new export"
    for n in range(3):
        (repo / "code.py").write_text(f"x = {n + 2}\n")
        _git(repo, "commit", "-qam", f"c{n}")
        sha, _ = ecs.export(repo, "HEAD", dest)
    assert (dest / "current").resolve().name == sha
    kept = [p for p in dest.iterdir() if p.is_dir() and not p.is_symlink()]
    assert len(kept) == ecs.KEEP


def test_never_fail_prints_one_failed_line(tmp_path, capsys):
    assert ecs.main(["--repo", str(tmp_path), "--dest", str(tmp_path / "o"), "--never-fail"]) == 0
    assert capsys.readouterr().out.startswith("failed reason=")
