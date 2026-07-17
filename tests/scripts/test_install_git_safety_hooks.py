from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install_git_safety_hooks.sh"


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _run_installer(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(SCRIPT), str(target)], capture_output=True, text=True, timeout=30
    )


ALL_HOOKS = ("pre-commit", "post-merge", "post-commit")


def test_installs_all_hooks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    proc = _run_installer(repo)
    assert proc.returncode == 0, proc.stderr
    for name in ALL_HOOKS:
        hook = repo / ".git" / "hooks" / name
        assert hook.exists()
        assert "orion-git-safety-guard" in hook.read_text(encoding="utf-8")


def test_all_hooks_executable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run_installer(repo)

    for name in ALL_HOOKS:
        hook = repo / ".git" / "hooks" / name
        assert os.access(hook, os.X_OK)


def test_rerun_refreshes_all_without_duplicating_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _run_installer(repo)
    proc = _run_installer(repo)
    assert proc.returncode == 0
    assert "refreshed" in proc.stdout
    # pre-commit/post-merge each carry one marker line; post-commit carries
    # two (an explicit BEGIN and END, so a refresh can find both ends of its
    # own block and preserve whatever another tool put on either side of it).
    expected_marker_count = {"pre-commit": 1, "post-merge": 1, "post-commit": 2}
    for name in ALL_HOOKS:
        hook = repo / ".git" / "hooks" / name
        assert hook.read_text(encoding="utf-8").count("# orion-git-safety-guard") == expected_marker_count[name]


def test_preserves_existing_foreign_hook_as_backup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "post-merge").write_text("#!/bin/sh\necho unrelated-hook\n", encoding="utf-8")

    proc = _run_installer(repo)
    assert proc.returncode == 0
    backup = hooks_dir / "post-merge.pre-orion-safety.bak"
    assert backup.exists()
    assert "unrelated-hook" in backup.read_text(encoding="utf-8")
    assert "orion-git-safety-guard" in (hooks_dir / "post-merge").read_text(encoding="utf-8")


def test_post_commit_appends_to_existing_foreign_hook_without_backup(tmp_path: Path) -> None:
    """post-commit is a hook name real tooling in this repo already claims --
    `graphify hook install` ships a post-commit hook that rebuilds the
    knowledge graph after every commit, confirmed live against this repo's
    own primary checkout during development of this feature. Installing our
    hook there the way pre-commit/post-merge install (overwrite + backup)
    would silently disable that feature on first install. This must append
    instead: the foreign hook's own content stays in place, unbackuped
    (nothing was destroyed), and runs before ours since git executes the
    hook script top-to-bottom."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "post-commit").write_text(
        "#!/bin/sh\necho some-other-tools-post-commit-hook-ran\n", encoding="utf-8"
    )
    (hooks_dir / "post-commit").chmod(0o755)

    proc = _run_installer(repo)
    assert proc.returncode == 0, proc.stderr

    backup = hooks_dir / "post-commit.pre-orion-safety.bak"
    assert not backup.exists()
    text = (hooks_dir / "post-commit").read_text(encoding="utf-8")
    assert "some-other-tools-post-commit-hook-ran" in text
    assert "orion-git-safety-guard" in text
    assert text.index("some-other-tools-post-commit-hook-ran") < text.index("orion-git-safety-guard")

    # Re-running must refresh only our own block -- the foreign hook's
    # content must not be duplicated or disturbed.
    proc2 = _run_installer(repo)
    assert proc2.returncode == 0, proc2.stderr
    text2 = (hooks_dir / "post-commit").read_text(encoding="utf-8")
    assert text2.count("some-other-tools-post-commit-hook-ran") == 1
    assert text2.count("# orion-git-safety-guard") == 2  # explicit BEGIN + END
    assert not backup.exists()


def test_post_commit_refresh_preserves_content_appended_after_our_block(tmp_path: Path) -> None:
    """Regression test for a real bug caught live: re-running
    `graphify hook install` after our hook was already appended causes
    graphify to append ITS content after ours (graphify's own installer
    uses the same append-to-whatever-exists strategy). An earlier version
    of this refresh logic assumed our block was always the file's tail and
    silently discarded that trailing content on the next
    `install_git_safety_hooks.sh` run. Our block must track its own BEGIN
    *and* END so a refresh preserves content on both sides of it."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    _run_installer(repo)

    hooks_dir = repo / ".git" / "hooks"
    post_commit = hooks_dir / "post-commit"
    with post_commit.open("a", encoding="utf-8") as f:
        f.write("\necho appended-by-another-tool-after-ours\n")

    proc = _run_installer(repo)
    assert proc.returncode == 0, proc.stderr

    text = post_commit.read_text(encoding="utf-8")
    assert "echo appended-by-another-tool-after-ours" in text
    assert text.count("echo appended-by-another-tool-after-ours") == 1
    assert text.count("# orion-git-safety-guard") == 2
    assert text.index("(post-commit) END") < text.index("appended-by-another-tool-after-ours")


def test_post_commit_heartbeat_survives_foreign_hook_early_exit(tmp_path: Path) -> None:
    """Regression test for a real bug caught live: graphify's own post-commit
    hook calls `exit 0` for any commit made from a linked worktree (its
    knowledge-graph rebuild only runs from the primary checkout). A bare
    `exit` in a concatenated shell script terminates the WHOLE file, not just
    the fragment that called it -- so simply appending our block after the
    foreign one silently killed our heartbeat on every worktree commit, which
    is this repo's common case (AGENTS.md section 2: worktrees only). Each
    fragment must be isolated in its own `( ... )` subshell so one's exit
    can't block the other, regardless of which fragment runs first."""
    primary = tmp_path / "primary"
    _init_repo(primary)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=primary, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=primary, check=True)

    board_scripts_dir = primary / "scripts"
    board_scripts_dir.mkdir()
    for name in ("agent_board.py", "agent_board_lib.py", "worktree_lib.py"):
        (board_scripts_dir / name).write_text(
            (ROOT / "scripts" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    subprocess.run(["git", "add", "-A"], cwd=primary, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=primary, check=True)

    # Write the foreign hook AFTER the init commit, not before -- otherwise
    # that fixture-setup commit would trigger it too, polluting the sentinel
    # this test uses to prove the foreign hook only ran once.
    hooks_dir = primary / ".git" / "hooks"
    sentinel = tmp_path / "foreign-hook-ran.txt"
    (hooks_dir / "post-commit").write_text(
        f"#!/bin/sh\necho ran >> {sentinel}\nexit 0\necho unreachable >> {sentinel}\n",
        encoding="utf-8",
    )
    (hooks_dir / "post-commit").chmod(0o755)

    proc = _run_installer(primary)
    assert proc.returncode == 0, proc.stderr

    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(worktree), "-b", "chore/test-early-exit"],
        cwd=primary,
        check=True,
    )

    board = tmp_path / "agent-board.jsonl"
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board)}

    (worktree / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=worktree, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test: verify heartbeat survives foreign early exit"],
        cwd=worktree,
        check=True,
        env=env,
    )

    # Foreign hook ran and stopped at its own `exit 0` -- confirms the test
    # fixture actually exercises the early-exit condition it claims to.
    sentinel_text = sentinel.read_text(encoding="utf-8")
    assert sentinel_text.count("ran") == 1
    assert "unreachable" not in sentinel_text

    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), "list", "--all"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert listed.returncode == 0, listed.stderr
    assert "test: verify heartbeat survives foreign early exit" in listed.stdout


def test_installed_post_commit_hook_heartbeats_the_agent_board(tmp_path: Path) -> None:
    """End-to-end: a real `git commit` in a repo with the installed hook must
    write a presence heartbeat to the agent board carrying the commit's own
    subject line -- this is the actual coupling this hook exists for, not
    just "the file got copied somewhere".

    Two things this repo's own conventions require of the fixture:
    - the hook looks for a sibling `scripts/agent_board.py` in the repo it's
      installed into (same pattern as the existing post-merge hook's
      `scripts/worktree_status.py` lookup), so the fake repo needs a real
      copy of it, not just an empty tree;
    - the installed pre-commit hook itself refuses commits made directly in
      a repo with no linked worktree (that's what "shared/primary checkout"
      collapses to for a solo repo) -- so the actual commit under test has
      to happen from a linked worktree, matching how this repo is meant to
      be used, not by working around the guard.
    """
    primary = tmp_path / "primary"
    _init_repo(primary)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=primary, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=primary, check=True)

    board_scripts_dir = primary / "scripts"
    board_scripts_dir.mkdir()
    for name in ("agent_board.py", "agent_board_lib.py", "worktree_lib.py"):
        (board_scripts_dir / name).write_text(
            (ROOT / "scripts" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    subprocess.run(["git", "add", "-A"], cwd=primary, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=primary, check=True)

    proc = _run_installer(primary)
    assert proc.returncode == 0, proc.stderr

    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(worktree), "-b", "chore/test-commit"],
        cwd=primary,
        check=True,
    )

    board = tmp_path / "agent-board.jsonl"
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board)}

    (worktree / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=worktree, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test: verify post-commit board heartbeat"],
        cwd=worktree,
        check=True,
        env=env,
    )

    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), "list", "--all"],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert listed.returncode == 0, listed.stderr
    assert "test: verify post-commit board heartbeat" in listed.stdout


def test_post_commit_hook_tags_heartbeat_with_claude_session_id_when_set(tmp_path: Path) -> None:
    """The Stop/SessionStart hooks need a session_id-tagged presence row to
    resolve the real worktree despite their own fixed process cwd (see
    resolve_current_identity() in agent_board_lib.py) -- this only works if
    the post-commit hook actually passes $CLAUDE_CODE_SESSION_ID through
    when Claude Code set it for the commit's own subprocess."""
    primary = tmp_path / "primary"
    _init_repo(primary)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=primary, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=primary, check=True)

    board_scripts_dir = primary / "scripts"
    board_scripts_dir.mkdir()
    for name in ("agent_board.py", "agent_board_lib.py", "worktree_lib.py"):
        (board_scripts_dir / name).write_text(
            (ROOT / "scripts" / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    subprocess.run(["git", "add", "-A"], cwd=primary, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=primary, check=True)

    proc = _run_installer(primary)
    assert proc.returncode == 0, proc.stderr

    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(worktree), "-b", "chore/test-session-id"],
        cwd=primary,
        check=True,
    )

    board = tmp_path / "agent-board.jsonl"
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board), "CLAUDE_CODE_SESSION_ID": "sess-xyz-789"}

    (worktree / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=worktree, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test: verify session id passthrough"],
        cwd=worktree,
        check=True,
        env=env,
    )

    raw_events = board.read_text(encoding="utf-8")
    assert '"session_id": "sess-xyz-789"' in raw_events
