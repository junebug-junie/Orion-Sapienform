from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "safe_graphify_update.sh"


def _write_graph(path: Path, node_count: int) -> None:
    (path / "graphify-out").mkdir(exist_ok=True)
    (path / "graphify-out" / "graph.json").write_text(
        json.dumps({"nodes": [{"id": i} for i in range(node_count)], "edges": []}),
        encoding="utf-8",
    )
    (path / "graphify-out" / "manifest.json").write_text(
        json.dumps({"generated_at": "test"}), encoding="utf-8"
    )


def _fake_graphify(bin_dir: Path, result_node_count: int) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        f"python3 -c \"import json; json.dump({{'nodes': [{{'id': i}} for i in range({result_node_count})], 'edges': []}}, open('graphify-out/graph.json', 'w'))\"\n"
        "echo 'Code graph updated.'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)


def _run(cwd: Path, fake_bin: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return subprocess.run(
        ["sh", str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=30
    )


def test_safe_update_passes_through(tmp_path: Path) -> None:
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 999)  # trivial drop, well under threshold

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
    assert "OK" in proc.stdout
    graph = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 999


def test_destructive_update_refused_and_restored(tmp_path: Path) -> None:
    """Reproduces the exact real-incident magnitude: 999 -> 50, a ~95% loss."""
    _write_graph(tmp_path, 999)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 50)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr
    graph = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 999  # restored, not left destructive


def test_node_increase_is_fine(tmp_path: Path) -> None:
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 1200)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
    graph = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 1200


def test_threshold_is_configurable(tmp_path: Path) -> None:
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 700)  # 30% drop

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT"] = "50"
    proc = subprocess.run(
        ["sh", str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True, env=env, timeout=30
    )
    assert proc.returncode == 0  # 30% drop allowed under a 50% threshold


def test_missing_graph_file_refuses_to_run(tmp_path: Path) -> None:
    (tmp_path / "graphify-out").mkdir()
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 100)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_graphify_itself_failing_restores_backup(tmp_path: Path) -> None:
    """Regression: if `graphify update` exits nonzero (crashes, not just bad
    output), `set -e` used to abort the whole script immediately, skipping
    the restore -- graph.json could be left partially written."""
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(parents=True)
    fake = fake_bin / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        # simulate a crash that also corrupts the file mid-write
        "echo 'not json' > graphify-out/graph.json\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "graphify update itself failed" in proc.stderr
    graph = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 1000  # restored, not left as "not json"


def test_boundary_at_exactly_threshold_percent(tmp_path: Path) -> None:
    """Exactly at the 10% default threshold should NOT trigger (strictly greater-than)."""
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 900)  # exactly 10% drop

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
