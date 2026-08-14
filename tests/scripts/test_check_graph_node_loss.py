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

from check_graph_node_loss import loss_pct  # noqa: E402


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
