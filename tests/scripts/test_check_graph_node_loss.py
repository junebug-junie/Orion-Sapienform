"""Tests for the graph-shrink commit gate.

Fixtures are hand-computed, not derived from the code under test: each case
states the node counts and the percentage they must produce, worked out by
hand, so a wrong formula fails instead of agreeing with itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_graph_node_loss.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_graph_node_loss import loss_pct, _is_lfs_pointer  # noqa: E402


def _graph(path: Path, *, nodes: int, links: int = 0, hyperedges: int = 0) -> Path:
    path.write_text(
        json.dumps(
            {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": f"n{i}"} for i in range(nodes)],
                "edges": [],  # graphify emits this EMPTY; real edges are "links"
                "links": [{"source": "a", "target": "b"} for _ in range(links)],
                "hyperedges": [{"id": f"h{i}"} for i in range(hyperedges)],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "before,after,expected",
    [
        # 28306 -> 2475 is the real 2026-08-14 recurrence. Lost 25831 of 28306.
        # 25831/28306 = 0.912563... -> 91.2563%
        (28306, 2475, 91.2563),
        # Half of 200 is 100 lost -> exactly 50%.
        (200, 100, 50.0),
        # 1000 -> 901 loses 99; 99/1000 -> 9.9%, just under a 10% threshold.
        (1000, 901, 9.9),
        # 1000 -> 899 loses 101; 101/1000 -> 10.1%, just over.
        (1000, 899, 10.1),
        # Growth is not loss.
        (100, 250, 0.0),
        (100, 100, 0.0),
        # Nothing to lose.
        (0, 0, 0.0),
        (0, 50, 0.0),
        # Total wipe.
        (28306, 0, 100.0),
    ],
)
def test_loss_pct_hand_computed(before: int, after: int, expected: float):
    assert loss_pct(before, after) == pytest.approx(expected, abs=1e-4)


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=full_env,
    )


def test_catastrophic_shrink_blocks(tmp_path):
    before = _graph(tmp_path / "before.json", nodes=28306)
    after = _graph(tmp_path / "after.json", nodes=2475)
    proc = _run("--before", str(before), "--after", str(after))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "BLOCK" in proc.stdout
    assert "91.2" in proc.stdout


def test_small_shrink_passes(tmp_path):
    """A real incremental update legitimately loses a few nodes."""
    before = _graph(tmp_path / "before.json", nodes=1000)
    after = _graph(tmp_path / "after.json", nodes=960)  # 4% loss
    proc = _run("--before", str(before), "--after", str(after))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_growth_passes(tmp_path):
    before = _graph(tmp_path / "before.json", nodes=1000)
    after = _graph(tmp_path / "after.json", nodes=1200)
    assert _run("--before", str(before), "--after", str(after)).returncode == 0


def test_threshold_is_configurable(tmp_path):
    before = _graph(tmp_path / "before.json", nodes=1000)
    after = _graph(tmp_path / "after.json", nodes=800)  # exactly 20%
    assert _run("--before", str(before), "--after", str(after)).returncode == 1
    assert _run("--before", str(before), "--after", str(after), "--threshold", "25").returncode == 0
    assert _run(
        "--before", str(before), "--after", str(after),
        env={"GRAPHIFY_COMMIT_MAX_NODE_LOSS_PCT": "25"},
    ).returncode == 0


def test_escape_hatch_allows_an_intentional_reextraction(tmp_path):
    before = _graph(tmp_path / "before.json", nodes=28306)
    after = _graph(tmp_path / "after.json", nodes=100)
    proc = _run("--before", str(before), "--after", str(after), env={"ORION_ALLOW_GRAPH_SHRINK": "1"})
    assert proc.returncode == 0
    assert "ALLOWED" in proc.stderr


def test_unparseable_graph_blocks_rather_than_passing(tmp_path):
    """A graph.json we cannot parse is not one worth committing -- fail closed."""
    before = _graph(tmp_path / "before.json", nodes=1000)
    bad = tmp_path / "after.json"
    bad.write_text("{not json", encoding="utf-8")
    proc = _run("--before", str(before), "--after", str(bad))
    assert proc.returncode == 2
    assert "cannot compare" in proc.stderr


def test_counts_links_not_the_empty_edges_key(tmp_path):
    """graphify emits an EMPTY 'edges' key and puts real edges under 'links'.
    Counting 'edges' would read zero forever and hide a real collapse."""
    before = _graph(tmp_path / "before.json", nodes=10, links=500)
    after = _graph(tmp_path / "after.json", nodes=10, links=3)
    proc = _run("--before", str(before), "--after", str(after), "--json")
    payload = json.loads(proc.stdout)
    assert payload["before"]["links"] == 500
    assert payload["after"]["links"] == 3
    # Node count is unchanged, so this alone does not block -- nodes are the
    # gate. Recorded so a future change to block on link loss is deliberate.
    assert payload["blocked"] is False


def test_bad_threshold_env_falls_back_to_default(tmp_path):
    before = _graph(tmp_path / "before.json", nodes=1000)
    after = _graph(tmp_path / "after.json", nodes=500)
    proc = _run(
        "--before", str(before), "--after", str(after),
        env={"GRAPHIFY_COMMIT_MAX_NODE_LOSS_PCT": "not-a-number"},
    )
    assert proc.returncode == 1
    assert "not a number" in proc.stderr


def test_runs_on_plain_python3_without_third_party_deps():
    """The hook calls plain `python3`. If this ever needs pydantic it would
    silently skip on a bare interpreter, which is how a gate stops gating."""
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("import pydantic", "import yaml", "import requests", "import networkx"):
        assert banned not in source


# --- LFS regression coverage --------------------------------------------
#
# graphify-out/graph.json became git-LFS-tracked in fix/graph-json-lfs
# (2026-09-08). `git show <ref>:<path>` on an LFS-tracked path returns the
# ~130-byte pointer stub text, not real content -- these tests are the
# regression check that would have caught comparing two pointer stubs
# (0 nodes vs 0 nodes, i.e. never blocking) instead of real graphs.


def test_is_lfs_pointer_detects_real_pointer_text():
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\n"
        "size 12345\n"
    )
    assert _is_lfs_pointer(pointer) is True


def test_is_lfs_pointer_false_for_plain_json():
    assert _is_lfs_pointer('{"nodes": [], "links": []}') is False


def _lfs_available() -> bool:
    import shutil

    return shutil.which("git") is not None and shutil.which("git-lfs") is not None


@pytest.mark.skipif(not _lfs_available(), reason="git or git-lfs not on PATH")
def test_git_mode_smudges_lfs_pointer_for_head_and_staged(tmp_path):
    """End-to-end: a real git repo with graph.json LFS-tracked, one real commit
    (HEAD) and one staged real edit -- both sides must be read as real graph
    content, not the raw pointer stub `git show` would otherwise return."""
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@test.local"

    def run(*args):
        subprocess.run(list(args), cwd=repo, check=True, capture_output=True, text=True, env=env)

    run("git", "init", "-q", ".")
    run("git", "config", "user.email", "test@test.local")
    run("git", "config", "user.name", "Test")
    (repo / ".gitattributes").write_text(
        "graphify-out/graph.json filter=lfs diff=lfs -text\n", encoding="utf-8"
    )
    run("git", "config", "filter.lfs.clean", "git-lfs clean -- %f")
    run("git", "config", "filter.lfs.smudge", "git-lfs smudge -- %f")
    run("git", "config", "filter.lfs.process", "git-lfs filter-process")
    run("git", "config", "filter.lfs.required", "true")
    run("git", "lfs", "install", "--local")

    (repo / "graphify-out").mkdir()
    _graph(repo / "graphify-out" / "graph.json", nodes=1000)
    run("git", "add", ".gitattributes", "graphify-out/graph.json")
    run("git", "commit", "-q", "-m", "base graph, 1000 nodes")

    # Confirm the commit really did land as a pointer, not raw content --
    # otherwise this test would pass for the wrong reason.
    head_blob = subprocess.run(
        ["git", "cat-file", "-p", "HEAD:graphify-out/graph.json"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    assert head_blob.startswith("version https://git-lfs.github.com/spec/v1")

    # Stage a small (9%, under threshold) shrink -- also becomes a pointer.
    _graph(repo / "graphify-out" / "graph.json", nodes=920)
    run("git", "add", "graphify-out/graph.json")
    staged_blob = subprocess.run(
        ["git", "cat-file", "-p", ":graphify-out/graph.json"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    assert staged_blob.startswith("version https://git-lfs.github.com/spec/v1")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=repo, capture_output=True, text=True, check=False, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    # If the pointer stub had leaked through unsmudged, both sides would read
    # as an empty/unparseable graph instead of the real 1000 -> 920 change.
    assert payload["before"]["nodes"] == 1000
    assert payload["after"]["nodes"] == 920
    assert payload["blocked"] is False


@pytest.mark.skipif(not _lfs_available(), reason="git or git-lfs not on PATH")
def test_git_mode_fails_closed_on_unresolvable_lfs_pointer(tmp_path):
    """An LFS pointer this machine can't resolve (no remote configured, oid
    not in the local cache) must not hang the gate and must not silently
    report success -- it must fail closed (exit 2) with a real message.

    Deliberately built network-free and deterministic: no remote is
    configured, so `git lfs smudge` cannot even attempt a fetch (verified by
    direct repro that this makes it exit 0 and pass the pointer text through
    UNCHANGED rather than raising -- json.loads on that unchanged pointer
    text is what actually trips the gate's existing fail-closed path). The
    subprocess timeout here is a test-level backstop: if a future change
    reintroduces a hang, this test fails loudly instead of stalling CI.
    """
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@test.local"

    def run(*args):
        subprocess.run(list(args), cwd=repo, check=True, capture_output=True, text=True, env=env)

    run("git", "init", "-q", ".")
    run("git", "config", "user.email", "test@test.local")
    run("git", "config", "user.name", "Test")
    (repo / ".gitattributes").write_text(
        "graphify-out/graph.json filter=lfs diff=lfs -text\n", encoding="utf-8"
    )
    run("git", "config", "filter.lfs.clean", "git-lfs clean -- %f")
    run("git", "config", "filter.lfs.smudge", "git-lfs smudge -- %f")
    run("git", "config", "filter.lfs.process", "git-lfs filter-process")
    run("git", "config", "filter.lfs.required", "true")
    run("git", "lfs", "install", "--local")

    (repo / "graphify-out").mkdir()
    # A syntactically valid pointer whose object was never fetched and never
    # will be (no remote configured at all) -- unresolvable by construction.
    (repo / "graphify-out" / "graph.json").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\n"
        "size 999\n",
        encoding="utf-8",
    )
    run("git", "add", ".gitattributes", "graphify-out/graph.json")
    run("git", "commit", "-q", "-m", "unresolvable pointer")

    # The gate only runs when graph.json is actually staged (see
    # _staged_paths()) -- stage a real edit so HEAD's unresolvable pointer
    # actually gets compared against something.
    _graph(repo / "graphify-out" / "graph.json", nodes=50)
    run("git", "add", "graphify-out/graph.json")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=repo, capture_output=True, text=True, check=False, env=env,
        timeout=30,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert proc.stderr.strip() != ""
