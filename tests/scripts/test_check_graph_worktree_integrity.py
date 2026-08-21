"""Tests for the graphify-out/ working-tree auto-restore detector.

Uses a real throwaway git repo per test (not a mock) so the detector is
exercised against real `git diff`/`git show`/`git checkout` behavior, the
same commands the live 2026-08-21 incident actually needed. Fixtures are
hand-computed, not derived from the code under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_graph_worktree_integrity.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_graph_worktree_integrity as integrity  # noqa: E402


def _graph_doc(*, nodes: int, links: int = 0) -> dict:
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": f"n{i}"} for i in range(nodes)],
        "edges": [],
        "links": [{"source": "a", "target": "b"} for _ in range(links)],
        "hyperedges": [],
    }


def _write_bundle(root: Path, *, nodes: int, links: int = 0, manifest_n: int = 5, report_lines: int = 10) -> None:
    (root / "graphify-out").mkdir(parents=True, exist_ok=True)
    (root / "graphify-out" / "graph.json").write_text(
        json.dumps(_graph_doc(nodes=nodes, links=links)), encoding="utf-8"
    )
    (root / "graphify-out" / "manifest.json").write_text(
        json.dumps({f"file{i}.py": {"hash": str(i)} for i in range(manifest_n)}), encoding="utf-8"
    )
    (root / "graphify-out" / "GRAPH_REPORT.md").write_text(
        "\n".join(f"line {i}" for i in range(report_lines)) + "\n", encoding="utf-8"
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _init_repo(tmp_path: Path, *, nodes: int, links: int = 0, manifest_n: int = 5, report_lines: int = 10) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _write_bundle(root, nodes=nodes, links=links, manifest_n=manifest_n, report_lines=report_lines)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial graph")
    return root


@pytest.fixture(autouse=True)
def _sandbox_backup_root(tmp_path, monkeypatch):
    """Keep restore backups inside the test's own tmp dir, never the real
    /tmp/graphify_worktree_guard_backups shared across the whole machine."""
    monkeypatch.setattr(integrity, "BACKUP_ROOT", str(tmp_path / "backups"))


def _node_count(root: Path) -> int:
    doc = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return len(doc["nodes"])


def test_clean_worktree_is_a_no_op(tmp_path):
    root = _init_repo(tmp_path, nodes=1000)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["dirty"] == []
    assert result["restored"] is False


def test_small_shrink_within_threshold_left_alone(tmp_path):
    root = _init_repo(tmp_path, nodes=1000)
    _write_bundle(root, nodes=960)  # 4% loss, real incremental update
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is False
    assert result["restored"] is False
    assert _node_count(root) == 960  # left as-is


def test_destructive_shrink_is_auto_restored(tmp_path):
    """The real 2026-08-21 signature: 28306 -> 2475, 91.26% loss."""
    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is True
    assert result["restored"] is True
    assert result["node_loss_pct"] == pytest.approx(91.2563, abs=0.001)
    assert _node_count(root) == 28306  # back to HEAD

    # The discarded (destroyed) content must be recoverable, not just gone.
    backup_dir = Path(result["backup_dir"])
    backed_up = json.loads((backup_dir / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(backed_up["nodes"]) == 2475


def test_sibling_desync_without_graph_change_is_restored(tmp_path):
    """The second live incident: graph.json already matches HEAD, but a
    sibling (manifest.json here) is still stale from a partial restore."""
    root = _init_repo(tmp_path, nodes=28306, manifest_n=3596)
    (root / "graphify-out" / "manifest.json").write_text(
        json.dumps({f"file{i}.py": {"hash": str(i)} for i in range(335)}), encoding="utf-8"
    )
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is False  # graph.json itself never moved
    assert result["desync"] is True
    assert result["restored"] is True
    manifest = json.loads((root / "graphify-out" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3596


def test_growth_is_not_destructive(tmp_path):
    root = _init_repo(tmp_path, nodes=1000)
    _write_bundle(root, nodes=1500)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is False
    assert result["restored"] is False
    assert _node_count(root) == 1500


def test_escape_hatch_skips_restore(tmp_path, monkeypatch):
    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)
    monkeypatch.setenv("ORION_ALLOW_GRAPH_SHRINK", "1")
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is True
    assert result["escaped"] is True
    assert result["restored"] is False
    assert _node_count(root) == 2475  # left alone -- intentional re-extraction in flight


def test_check_only_reports_without_restoring(tmp_path):
    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=True)
    assert result["destructive"] is True
    assert result["restored"] is False
    assert _node_count(root) == 2475


def test_threshold_is_configurable(tmp_path):
    root = _init_repo(tmp_path, nodes=1000)
    _write_bundle(root, nodes=800)  # exactly 20% loss
    assert integrity.check(str(root), 10.0, check_only=True)["destructive"] is True
    assert integrity.check(str(root), 25.0, check_only=True)["destructive"] is False


def test_cli_json_end_to_end(tmp_path):
    import os

    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)
    env = dict(os.environ)
    env["GRAPHIFY_WORKTREE_GUARD_BACKUP_ROOT"] = str(tmp_path / "backups")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr  # never blocking
    payload = json.loads(proc.stdout)
    assert payload["restored"] is True
    assert _node_count(root) == 28306


def test_runs_on_plain_python3_without_third_party_deps():
    for path in (SCRIPT, REPO_ROOT / "scripts" / "hooks" / "session_start_graph_worktree_guard.py"):
        source = path.read_text(encoding="utf-8")
        for banned in ("import pydantic", "import yaml", "import requests", "import networkx"):
            assert banned not in source, f"{path}: unexpected dep {banned}"


def test_deleted_graph_json_is_treated_as_total_loss_and_restored(tmp_path):
    """Total loss is a WORSE instance of the same failure class this guards
    against, not a reason to skip the check."""
    root = _init_repo(tmp_path, nodes=28306)
    (root / "graphify-out" / "graph.json").unlink()
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is True
    assert result["restored"] is True
    assert _node_count(root) == 28306


def test_corrupt_graph_json_is_treated_as_total_loss_and_restored(tmp_path):
    root = _init_repo(tmp_path, nodes=28306)
    (root / "graphify-out" / "graph.json").write_text("{not valid json", encoding="utf-8")
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is True
    assert result["restored"] is True
    assert _node_count(root) == 28306


def test_restore_failure_is_reported_not_silently_treated_as_success(tmp_path, monkeypatch):
    """If `git checkout` itself fails (lock contention, permissions, ...),
    `restored` must be False -- never assumed True just because it was
    attempted."""
    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)

    real_run = integrity._run

    def _failing_checkout(args):
        if "checkout" in args:
            import subprocess as sp

            return sp.CompletedProcess(args, returncode=1, stdout="", stderr="fatal: simulated lock contention")
        return real_run(args)

    monkeypatch.setattr(integrity, "_run", _failing_checkout)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["restored"] is False
    assert "RESTORE FAILED" in result["detail"]
    assert _node_count(root) == 2475  # untouched -- checkout never actually applied

    # But the backup must still exist -- a failed restore must not also lose
    # the pre-restore snapshot of the (still-present) destroyed content.
    backup_dir = Path(result["backup_dir"])
    assert (backup_dir / "graphify-out" / "graph.json").exists()


def test_backup_failure_aborts_before_any_restore_is_attempted(tmp_path, monkeypatch):
    root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(root, nodes=2475)

    def _failing_copy2(*args, **kwargs):
        raise OSError("simulated permission error on backup root")

    monkeypatch.setattr(integrity.shutil, "copy2", _failing_copy2)
    checkout_calls = []
    real_run = integrity._run

    def _tracking_run(args):
        if "checkout" in args:
            checkout_calls.append(args)
        return real_run(args)

    monkeypatch.setattr(integrity, "_run", _tracking_run)

    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["restored"] is False
    assert "backup failed" in result["detail"]
    assert checkout_calls == []  # never even attempted
    assert _node_count(root) == 2475  # untouched


def test_combined_shrink_and_sibling_desync_in_one_pass(tmp_path):
    """The real `graphify update .` failure touches graph.json AND both
    siblings at once -- exercise that combined signature, not each file
    independently."""
    root = _init_repo(tmp_path, nodes=28306, manifest_n=3596, report_lines=4045)
    _write_bundle(root, nodes=2475, manifest_n=335, report_lines=10)
    result = integrity.check(str(root), integrity.DEFAULT_MAX_LOSS_PCT, check_only=False)
    assert result["destructive"] is True
    assert result["restored"] is True
    assert set(result["dirty"]) == {integrity.GRAPH_PATH, integrity.MANIFEST_PATH, integrity.REPORT_PATH}
    assert _node_count(root) == 28306
    manifest = json.loads((root / "graphify-out" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3596
    report_lines = (root / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8").splitlines()
    assert len(report_lines) == 4045


def test_worktree_threshold_env_var_is_independent_of_commit_gate(monkeypatch, tmp_path):
    """GRAPHIFY_WORKTREE_MAX_NODE_LOSS_PCT (this guard) must not be confused
    with check_graph_node_loss's own GRAPHIFY_COMMIT_MAX_NODE_LOSS_PCT --
    each guard point gets its own knob, same convention as
    safe_graphify_update.sh's GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT."""
    monkeypatch.setenv("GRAPHIFY_COMMIT_MAX_NODE_LOSS_PCT", "99")  # must NOT affect this guard
    assert integrity.resolve_threshold(None) == integrity.DEFAULT_MAX_LOSS_PCT
    monkeypatch.setenv("GRAPHIFY_WORKTREE_MAX_NODE_LOSS_PCT", "25")
    assert integrity.resolve_threshold(None) == 25.0
    assert integrity.resolve_threshold(5.0) == 5.0  # explicit flag wins over env


def test_hook_resolves_root_via_session_id_when_process_cwd_is_wrong(tmp_path, monkeypatch):
    """Reproduces the documented SessionStart-hook cwd bug: the harness runs
    the hook with a process cwd that is NOT necessarily this session's real
    worktree. The hook must still find the right repo via the agent board's
    session_id-tagged presence row, not the (wrong) ambient cwd."""
    real_root = _init_repo(tmp_path, nodes=28306)
    _write_bundle(real_root, nodes=2475)  # destructive drift sitting in the REAL worktree

    decoy_cwd = tmp_path / "decoy_not_a_git_repo"
    decoy_cwd.mkdir()

    board_path = tmp_path / "agent-board.jsonl"
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import agent_board_lib

    monkeypatch.setenv("ORION_AGENT_BOARD_PATH", str(board_path))
    config = agent_board_lib.board_config_from_env()
    agent_board_lib.upsert_presence(
        config, session_id="test-session-cwd-bug", worktree_path=str(real_root), branch="feat/x"
    )

    # Sanity check: plain git-rev-parse from the decoy cwd finds nothing --
    # proves this test would fail without the session_id-based fix.
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=str(decoy_cwd), capture_output=True, text=True
    )
    assert proc.returncode != 0

    hook_script = REPO_ROOT / "scripts" / "hooks" / "session_start_graph_worktree_guard.py"
    env = dict(__import__("os").environ)
    env["ORION_AGENT_BOARD_PATH"] = str(board_path)
    env["GRAPHIFY_WORKTREE_GUARD_BACKUP_ROOT"] = str(tmp_path / "backups")
    proc = subprocess.run(
        [sys.executable, str(hook_script)],
        cwd=str(decoy_cwd),
        input=json.dumps({"session_id": "test-session-cwd-bug", "cwd": str(decoy_cwd)}),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "auto-restored from HEAD" in proc.stdout, proc.stdout + proc.stderr
    assert _node_count(real_root) == 28306  # the REAL worktree got fixed, not the decoy cwd
