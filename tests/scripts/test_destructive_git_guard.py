from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hooks"))

from destructive_git_guard import (  # noqa: E402
    _escape_hatch_set,
    _evaluate,
    _is_shared_checkout,
    _resolve,
    _statement_is_destructive,
    _strip_quoted,
    main,
)


# --- _statement_is_destructive ------------------------------------------------

@pytest.mark.parametrize(
    "statement,expect_blocked",
    [
        ("git clean -fd", True),
        ("git clean -f", True),
        ("git clean --force -d", True),
        ("git clean -nfd", False),  # -n (dry-run) is authoritative over -f
        ("git clean --dry-run --force", False),
        ("git clean -n", False),
        ("git status", False),
        ("git reset --hard", True),
        ("git reset --hard HEAD~1", True),
        ("git reset --hard origin/main", True),
        ("git reset --soft HEAD~1", False),
        ("git reset --mixed", False),
        ("echo hello", False),
        ("git -C /some/dir clean -fd", True),
        ("git -C /some/dir reset --hard", True),
        ("git checkout -f", True),
        ("git checkout --force main", True),
        ("git checkout main", False),
        ("git checkout -- some/file", False),
        ("git switch -f other", True),
        ("git switch --force other", True),
        ("git switch other", False),
        ("git push --force", False),  # intentionally out of scope, see PR report
        ("git branch -D foo", False),  # intentionally out of scope, see PR report
        # line-continuation: `git reset \` + newline + `--hard` is a single
        # real command in bash. .* without DOTALL used to miss this.
        ("git reset \\\n  --hard", True),
    ],
)
def test_statement_is_destructive(statement: str, expect_blocked: bool) -> None:
    result = _statement_is_destructive(statement)
    assert (result is not None) == expect_blocked, f"{statement!r} -> {result!r}"


# --- _strip_quoted -------------------------------------------------------------

def test_strip_quoted_blanks_double_quotes_same_length() -> None:
    s = 'git commit -m "git reset --hard is dangerous"'
    stripped = _strip_quoted(s)
    assert "reset" not in stripped
    assert len(stripped) == len(s)


def test_strip_quoted_blanks_single_quotes() -> None:
    s = "echo 'git clean -fd'"
    stripped = _strip_quoted(s)
    assert "clean" not in stripped


def test_strip_quoted_leaves_unquoted_text() -> None:
    s = "git reset --hard"
    assert _strip_quoted(s) == s


# --- _resolve --------------------------------------------------------------

def test_resolve_relative_against_base() -> None:
    assert _resolve("/repo", "../other") == "/other"


def test_resolve_absolute_ignores_base() -> None:
    assert _resolve("/repo", "/elsewhere") == "/elsewhere"


def test_resolve_dot_is_base() -> None:
    assert _resolve("/repo", ".") == "/repo"


# --- _escape_hatch_set -------------------------------------------------------

def test_escape_hatch_leading_prefix() -> None:
    assert _escape_hatch_set("ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git reset --hard")


def test_escape_hatch_with_other_leading_assignments() -> None:
    assert _escape_hatch_set("FOO=bar ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git reset --hard")


def test_escape_hatch_not_set() -> None:
    assert not _escape_hatch_set("git reset --hard")


def test_escape_hatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_ALLOW_SHARED_CHECKOUT_WRITE", "1")
    assert _escape_hatch_set("git reset --hard")


def test_escape_hatch_trailing_is_not_a_bypass() -> None:
    """Regression: the token must be a genuine leading prefix of the
    statement, not appear anywhere in it (e.g. after the real command, or
    in a shell comment) -- a real shell never scopes an inline assignment
    that way."""
    assert not _escape_hatch_set("git reset --hard ORION_ALLOW_SHARED_CHECKOUT_WRITE=1")
    assert not _escape_hatch_set("git reset --hard # ORION_ALLOW_SHARED_CHECKOUT_WRITE=1")


# --- _is_shared_checkout (real git repos, no mocks) --------------------------

def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "primary"
    primary.mkdir()
    _git("init", "-q", cwd=primary)
    _git("config", "user.email", "test@example.com", cwd=primary)
    _git("config", "user.name", "Test", cwd=primary)
    (primary / "f.txt").write_text("x", encoding="utf-8")
    _git("add", "f.txt", cwd=primary)
    _git("commit", "-q", "-m", "init", cwd=primary)
    linked = tmp_path / "linked"
    _git("worktree", "add", "-q", str(linked), "-b", "feat/x", cwd=primary)
    return primary, linked


def test_is_shared_checkout_true_for_primary(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    assert _is_shared_checkout(str(primary)) is True


def test_is_shared_checkout_false_for_linked_worktree(
    repo_with_worktree: tuple[Path, Path]
) -> None:
    _, linked = repo_with_worktree
    assert _is_shared_checkout(str(linked)) is False


def test_is_shared_checkout_false_outside_any_repo(tmp_path: Path) -> None:
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    assert _is_shared_checkout(str(outside)) is False


# --- _evaluate: the regression tests for every review-confirmed bug ----------

def test_evaluate_blocks_simple_case(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    assert _evaluate(str(primary), "git reset --hard") is not None


def test_evaluate_allows_in_worktree(repo_with_worktree: tuple[Path, Path]) -> None:
    _, linked = repo_with_worktree
    assert _evaluate(str(linked), "git reset --hard") is None


def test_evaluate_multi_hop_cd_bug(repo_with_worktree: tuple[Path, Path]) -> None:
    """Regression: a cd that isn't the first token used to be invisible to
    directory resolution entirely."""
    primary, linked = repo_with_worktree
    command = f"echo starting && cd {primary} && git reset --hard"
    result = _evaluate(str(linked), command)
    assert result is not None
    assert result[1] == str(primary)


def test_evaluate_sequential_cd_statements(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, linked = repo_with_worktree
    command = f"cd {linked} && cd {primary} && git reset --hard"
    result = _evaluate("/nonexistent-start", command)
    assert result is not None
    assert result[1] == str(primary)


def test_evaluate_decoy_dash_c_in_earlier_statement_no_longer_hijacks(
    repo_with_worktree: tuple[Path, Path]
) -> None:
    """Regression: a `-C <dir>` substring inside an earlier, unrelated
    statement used to be picked up by a whole-string search and hijack
    directory resolution for a later, unrelated destructive statement."""
    primary, _ = repo_with_worktree
    command = 'echo "see git -C /tmp/decoy note" && git reset --hard'
    result = _evaluate(str(primary), command)
    assert result is not None
    assert result[1] == str(primary)


def test_evaluate_quoted_mention_is_not_a_false_positive(
    repo_with_worktree: tuple[Path, Path]
) -> None:
    """Regression: a commit message that merely mentions the destructive
    command as text used to be misidentified as the command itself."""
    primary, _ = repo_with_worktree
    command = 'git commit -m "docs: explain why git reset --hard is dangerous"'
    assert _evaluate(str(primary), command) is None


def test_evaluate_escape_hatch_scoped_to_its_own_statement(
    repo_with_worktree: tuple[Path, Path]
) -> None:
    """Regression: the escape hatch used to be searched across the whole
    command string, so it could authorize an earlier, unrelated destructive
    statement in the same chain."""
    primary, _ = repo_with_worktree
    command = "git clean -fd; ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 echo hi"
    result = _evaluate(str(primary), command)
    assert result is not None
    assert result[0].startswith("git clean")


def test_evaluate_escape_hatch_on_correct_statement_allows(
    repo_with_worktree: tuple[Path, Path]
) -> None:
    primary, _ = repo_with_worktree
    command = "ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git reset --hard"
    assert _evaluate(str(primary), command) is None


def test_evaluate_checkout_force_blocked(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    assert _evaluate(str(primary), "git checkout -f main") is not None


def test_evaluate_switch_force_blocked(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    assert _evaluate(str(primary), "git switch --force other") is not None


def test_evaluate_dash_c_with_prior_cd(repo_with_worktree: tuple[Path, Path]) -> None:
    """`git -C .` after a `cd` should resolve relative to the updated
    current directory, not the hook's original cwd."""
    primary, linked = repo_with_worktree
    command = f"cd {linked} && git -C . reset --hard"
    result = _evaluate("/nonexistent-start", command)
    assert result is None  # linked worktree, not shared


def test_evaluate_line_continuation_reset_hard(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    assert _evaluate(str(primary), "git reset \\\n  --hard") is not None


# --- main() end-to-end, real subprocess over stdin/stdout --------------------

def _run_main(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / "destructive_git_guard.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_main_blocks_reset_hard_in_primary(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    proc = _run_main({"cwd": str(primary), "tool_input": {"command": "git reset --hard"}})
    decision = json.loads(proc.stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "git-safety" in decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_allows_reset_hard_in_worktree(repo_with_worktree: tuple[Path, Path]) -> None:
    _, linked = repo_with_worktree
    proc = _run_main({"cwd": str(linked), "tool_input": {"command": "git reset --hard"}})
    assert proc.stdout == ""


def test_main_allows_with_escape_hatch(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    proc = _run_main(
        {
            "cwd": str(primary),
            "tool_input": {
                "command": "ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git reset --hard"
            },
        }
    )
    assert proc.stdout == ""


def test_main_allows_non_destructive_command(repo_with_worktree: tuple[Path, Path]) -> None:
    primary, _ = repo_with_worktree
    proc = _run_main({"cwd": str(primary), "tool_input": {"command": "git status"}})
    assert proc.stdout == ""


def test_main_ignores_malformed_input() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / "destructive_git_guard.py")],
        input="not json",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.stdout == ""
    assert proc.returncode == 0


def test_main_ignores_missing_tool_input() -> None:
    proc = _run_main({"cwd": "/repo"})
    assert proc.stdout == ""
    assert proc.returncode == 0
