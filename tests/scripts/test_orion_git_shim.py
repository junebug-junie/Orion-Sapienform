from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "scripts" / "git_hooks" / "orion-git-shim"
INSTALLER = ROOT / "scripts" / "install_orion_git_shim.sh"


def _real_git() -> str:
    found = shutil.which("git")
    assert found is not None, "real git must be on PATH to run these tests"
    return found


@pytest.fixture
def shim_env(tmp_path: Path) -> dict:
    """A PATH with the shim shadowing the real git, isolated to this
    fixture's own env dict -- never mutates the test process's real PATH
    or touches the real ~/.local/bin."""
    shim_dir = tmp_path / "shim_bin"
    shim_dir.mkdir()
    shim_path = shim_dir / "git"
    shutil.copy(SHIM, shim_path)
    shim_path.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    env.pop("ORION_ALLOW_SHARED_CHECKOUT_WRITE", None)
    return env


def _git(*args: str, cwd: Path, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=check
    )


@pytest.fixture
def primary_repo(tmp_path: Path) -> Path:
    primary = tmp_path / "primary"
    primary.mkdir()
    real = _real_git()
    subprocess.run([real, "init", "-q", "-b", "main"], cwd=primary, check=True)
    subprocess.run([real, "config", "user.email", "test@example.com"], cwd=primary, check=True)
    subprocess.run([real, "config", "user.name", "Test"], cwd=primary, check=True)
    (primary / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run([real, "add", "f.txt"], cwd=primary, check=True)
    subprocess.run([real, "commit", "-q", "-m", "init"], cwd=primary, check=True)
    return primary


def _opt_in(repo: Path) -> None:
    (repo / ".git" / "orion-safety-guard-enabled").touch()


@pytest.fixture
def opted_in_repo(primary_repo: Path) -> Path:
    _opt_in(primary_repo)
    return primary_repo


# --- self-resolution: no recursion, correctly skips other shim copies -----

def test_version_passes_through_no_recursion(primary_repo: Path, shim_env: dict) -> None:
    proc = _git("--version", cwd=primary_repo, env=shim_env, check=False)
    assert proc.returncode == 0
    assert "git version" in proc.stdout


def test_two_shim_copies_on_path_do_not_infinite_loop(
    primary_repo: Path, tmp_path: Path
) -> None:
    """Regression: two independently-installed shim copies used to only
    recognize themselves by exact path identity, so each treated the OTHER
    as "the real git" and exec'd into it forever. Content-marker matching
    fixes this -- verified here it terminates promptly instead of hanging."""
    dir_a = tmp_path / "shim_a"
    dir_b = tmp_path / "shim_b"
    dir_a.mkdir()
    dir_b.mkdir()
    for d in (dir_a, dir_b):
        shim_copy = d / "git"
        shutil.copy(SHIM, shim_copy)
        shim_copy.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{dir_a}:{dir_b}:{env['PATH']}"
    proc = subprocess.run(
        ["git", "--version"], cwd=primary_repo, capture_output=True, text=True,
        env=env, timeout=10,
    )
    assert proc.returncode == 0
    assert "git version" in proc.stdout


# --- passthrough behavior: must never break normal git usage ---------------

def test_non_destructive_command_passes_through(primary_repo: Path, shim_env: dict) -> None:
    proc = _git("status", "--short", cwd=primary_repo, env=shim_env)
    assert proc.returncode == 0


def test_destructive_command_allowed_when_repo_not_opted_in(
    primary_repo: Path, shim_env: dict
) -> None:
    proc = _git("reset", "--hard", cwd=primary_repo, env=shim_env, check=False)
    assert proc.returncode == 0
    assert "BLOCKED" not in proc.stderr


# --- blocking behavior, once opted in ---------------------------------------

def test_reset_hard_blocked_in_opted_in_shared_checkout(
    opted_in_repo: Path, shim_env: dict
) -> None:
    proc = _git("reset", "--hard", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr
    assert "orion-git-shim" in proc.stderr


def test_clean_force_blocked(opted_in_repo: Path, shim_env: dict) -> None:
    proc = _git("clean", "-fd", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


def test_clean_dry_run_allowed_even_with_force_flag(opted_in_repo: Path, shim_env: dict) -> None:
    proc = _git("clean", "-nfd", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 0
    assert "BLOCKED" not in proc.stderr


def test_checkout_force_blocked(opted_in_repo: Path, shim_env: dict) -> None:
    proc = _git("checkout", "-f", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


def test_checkout_without_force_allowed(opted_in_repo: Path, shim_env: dict) -> None:
    proc = _git("checkout", "main", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 0


def test_switch_force_blocked(opted_in_repo: Path, shim_env: dict) -> None:
    subprocess.run([_real_git(), "branch", "other"], cwd=opted_in_repo, check=True)
    proc = _git("switch", "--force", "other", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


# --- escape hatch ------------------------------------------------------------

def test_escape_hatch_env_var_allows(opted_in_repo: Path, shim_env: dict) -> None:
    env = dict(shim_env)
    env["ORION_ALLOW_SHARED_CHECKOUT_WRITE"] = "1"
    proc = _git("reset", "--hard", cwd=opted_in_repo, env=env, check=False)
    assert proc.returncode == 0


# --- worktree (not the shared checkout) allowed ------------------------------

def test_allowed_in_linked_worktree(opted_in_repo: Path, tmp_path: Path, shim_env: dict) -> None:
    linked = tmp_path / "linked"
    subprocess.run(
        [_real_git(), "worktree", "add", "-q", str(linked), "-b", "feat/x"],
        cwd=opted_in_repo, check=True,
    )
    proc = _git("reset", "--hard", cwd=linked, env=shim_env, check=False)
    assert proc.returncode == 0
    assert "BLOCKED" not in proc.stderr


# --- -C flag targeting the shared checkout from elsewhere --------------------

def test_dash_c_targeting_opted_in_shared_checkout_blocked(
    opted_in_repo: Path, tmp_path: Path, shim_env: dict
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = subprocess.run(
        ["git", "-C", str(opted_in_repo), "reset", "--hard"],
        cwd=elsewhere, capture_output=True, text=True, env=shim_env,
    )
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


def test_stacked_dash_c_chains_correctly(
    opted_in_repo: Path, tmp_path: Path, shim_env: dict
) -> None:
    """Regression: multiple -C flags used to overwrite _TARGET_DIR instead
    of chaining relative to the previous one, the way real git does."""
    parent = opted_in_repo.parent
    repo_name = opted_in_repo.name
    proc = subprocess.run(
        ["git", "-C", str(parent), "-C", repo_name, "reset", "--hard"],
        cwd=tmp_path, capture_output=True, text=True, env=shim_env,
    )
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


# --- --git-dir / --work-tree (space-separated global flags) ----------------

def test_work_tree_flag_targeting_shared_checkout_blocked(
    opted_in_repo: Path, tmp_path: Path, shim_env: dict
) -> None:
    """Regression: `--git-dir <path> --work-tree <path> reset --hard` used
    to have the PATH ARGUMENT after --git-dir misidentified as the
    subcommand, silently passing the real destructive command straight
    through with zero inspection -- a total bypass using ordinary syntax."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = subprocess.run(
        [
            "git", "--git-dir", str(opted_in_repo / ".git"),
            "--work-tree", str(opted_in_repo),
            "reset", "--hard",
        ],
        cwd=elsewhere, capture_output=True, text=True, env=shim_env,
    )
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


def test_git_dir_alone_does_not_misattribute_subcommand(
    opted_in_repo: Path, shim_env: dict
) -> None:
    """Even without --work-tree, --git-dir's value must not be mistaken
    for the subcommand -- the real subcommand (status, harmless) must
    still be recognized and executed normally."""
    proc = subprocess.run(
        ["git", "--git-dir", str(opted_in_repo / ".git"), "status"],
        cwd=opted_in_repo, capture_output=True, text=True, env=shim_env,
    )
    assert proc.returncode == 0
    assert "BLOCKED" not in proc.stderr


# --- alias resolution --------------------------------------------------------

def test_simple_alias_to_reset_hard_blocked(opted_in_repo: Path, shim_env: dict) -> None:
    """Regression: `git config alias.hr 'reset --hard'; git hr` used to
    completely bypass the guard -- the literal subcommand token 'hr'
    matched none of clean/reset/checkout/switch, so the whole invocation
    took the immediate passthrough path with zero destructive-shape
    inspection."""
    subprocess.run(
        [_real_git(), "config", "alias.hr", "reset --hard"], cwd=opted_in_repo, check=True
    )
    proc = _git("hr", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr


def test_alias_to_non_destructive_command_allowed(opted_in_repo: Path, shim_env: dict) -> None:
    subprocess.run(
        [_real_git(), "config", "alias.st", "status --short"], cwd=opted_in_repo, check=True
    )
    proc = _git("st", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 0
    assert "BLOCKED" not in proc.stderr


def test_shell_alias_passes_through_disclosed_gap(opted_in_repo: Path, shim_env: dict) -> None:
    """A `!`-prefixed alias is an arbitrary shell command, not argv-
    analyzable -- documented, disclosed gap, not a bug. Confirms it at
    least doesn't crash the shim."""
    subprocess.run(
        [_real_git(), "config", "alias.nuke", "!git reset --hard"], cwd=opted_in_repo, check=True
    )
    proc = _git("nuke", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 0  # passes through; the inner `git reset --hard`
    # the inner git IS re-resolved through the shim too (real git re-execs
    # `git` for `!git ...` aliases, which itself goes through PATH again) --
    # this assertion intentionally does not check for BLOCKED either way,
    # since covering that interaction is out of scope for this gap.


# --- clean.requireForce config -----------------------------------------------

def test_clean_requireforce_false_blocked_without_explicit_force_flag(
    opted_in_repo: Path, shim_env: dict
) -> None:
    """Regression: clean.requireForce=false makes `git clean -d` (no -f
    anywhere on the command line) actually destructive -- the shim used to
    only scan argv for -f/--force and missed this entirely, verified live
    to cause real data loss."""
    subprocess.run(
        [_real_git(), "config", "clean.requireForce", "false"], cwd=opted_in_repo, check=True
    )
    (opted_in_repo / "untracked.txt").write_text("junk", encoding="utf-8")
    proc = _git("clean", "-d", cwd=opted_in_repo, env=shim_env, check=False)
    assert proc.returncode == 1
    assert "BLOCKED" in proc.stderr
    assert (opted_in_repo / "untracked.txt").exists()  # not deleted


def test_clean_requireforce_true_default_still_needs_explicit_force(
    opted_in_repo: Path, shim_env: dict
) -> None:
    (opted_in_repo / "untracked.txt").write_text("junk", encoding="utf-8")
    proc = _git("clean", "-d", cwd=opted_in_repo, env=shim_env, check=False)
    # real git itself refuses without -f when requireForce is (default) true
    assert proc.returncode != 0
    assert "BLOCKED" not in proc.stderr
    assert (opted_in_repo / "untracked.txt").exists()


# --- installer ---------------------------------------------------------------

def test_installer_installs_and_creates_marker(primary_repo: Path, tmp_path: Path) -> None:
    shim_dir = tmp_path / "install_target"
    env = dict(os.environ)
    env["ORION_GIT_SHIM_DIR"] = str(shim_dir)
    proc = subprocess.run(
        ["sh", str(INSTALLER), str(primary_repo)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert (shim_dir / "git").exists()
    assert os.access(shim_dir / "git", os.X_OK)
    assert (primary_repo / ".git" / "orion-safety-guard-enabled").exists()


def test_installer_creates_marker_in_shared_dir_when_run_from_worktree(
    primary_repo: Path, tmp_path: Path
) -> None:
    linked = tmp_path / "linked"
    subprocess.run(
        [_real_git(), "worktree", "add", "-q", str(linked), "-b", "feat/x"],
        cwd=primary_repo, check=True,
    )
    shim_dir = tmp_path / "install_target"
    env = dict(os.environ)
    env["ORION_GIT_SHIM_DIR"] = str(shim_dir)
    proc = subprocess.run(
        ["sh", str(INSTALLER), str(linked)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    # marker lands in the PRIMARY checkout's .git, not the worktree's
    assert (primary_repo / ".git" / "orion-safety-guard-enabled").exists()


def test_installer_refuses_unrelated_existing_file(primary_repo: Path, tmp_path: Path) -> None:
    shim_dir = tmp_path / "install_target"
    shim_dir.mkdir()
    (shim_dir / "git").write_text("#!/bin/sh\necho unrelated\n", encoding="utf-8")
    (shim_dir / "git").chmod(0o755)
    env = dict(os.environ)
    env["ORION_GIT_SHIM_DIR"] = str(shim_dir)
    proc = subprocess.run(
        ["sh", str(INSTALLER), str(primary_repo)], capture_output=True, text=True, env=env
    )
    assert proc.returncode != 0
    assert "not an orion-git-shim install" in proc.stderr
    assert "unrelated" in (shim_dir / "git").read_text(encoding="utf-8")


def test_installer_idempotent(primary_repo: Path, tmp_path: Path) -> None:
    shim_dir = tmp_path / "install_target"
    env = dict(os.environ)
    env["ORION_GIT_SHIM_DIR"] = str(shim_dir)
    subprocess.run(["sh", str(INSTALLER), str(primary_repo)], env=env, check=True)
    proc = subprocess.run(
        ["sh", str(INSTALLER), str(primary_repo)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0
    assert "installed shim" in proc.stdout
