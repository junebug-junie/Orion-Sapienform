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


def _write_full_artifacts(path: Path, node_count: int, report_nodes: int) -> None:
    """graph.json + manifest.json + a GRAPH_REPORT.md whose header states its
    own node count, so a leaked report is detectable by content."""
    _write_graph(path, node_count)
    (path / "graphify-out" / "GRAPH_REPORT.md").write_text(
        f"# Graph Report\n- {report_nodes} nodes\n", encoding="utf-8"
    )
    (path / "graphify-out" / ".graphify_labels.json").write_text("{}", encoding="utf-8")


def _fake_graphify_full(bin_dir: Path, result_node_count: int, report_nodes: int) -> None:
    """A graphify that rewrites the report and CREATES graph.html, like the
    real one -- not just graph.json."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        f"python3 -c \"import json; json.dump({{'nodes': [{{'id': i}} for i in range({result_node_count})], 'edges': []}}, open('graphify-out/graph.json', 'w'))\"\n"
        f"printf '# Graph Report\\n- {report_nodes} nodes\\n' > graphify-out/GRAPH_REPORT.md\n"
        "printf 'generated' > graphify-out/graph.html\n"
        "printf '{\"changed\": 1}' > graphify-out/.graphify_labels.json\n"
        "echo 'Code graph updated.'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)


def test_refused_update_also_restores_graph_report(tmp_path: Path) -> None:
    """Regression for the 2026-08-12 second incident.

    The wrapper restored graph.json but left the SHRUNKEN GRAPH_REPORT.md in
    the working tree, while printing "there is nothing to commit". Measured
    live: a 2471-node report sitting beside a 28306-node graph. CLAUDE.md
    directs every agent to read that report for architecture review.
    """
    _write_full_artifacts(tmp_path, node_count=999, report_nodes=999)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_full(fake_bin, result_node_count=50, report_nodes=50)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr

    graph = json.loads((tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 999
    report = (tmp_path / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "999 nodes" in report, "report leaked the destructive run's content"
    assert "50 nodes" not in report


def test_refused_update_removes_files_the_run_created(tmp_path: Path) -> None:
    """graph.html did not exist before the run, so restoring means deleting
    it -- otherwise every refused run leaves untracked junk behind, which is
    how graph.html kept reappearing."""
    _write_full_artifacts(tmp_path, node_count=999, report_nodes=999)
    assert not (tmp_path / "graphify-out" / "graph.html").exists()
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_full(fake_bin, result_node_count=50, report_nodes=50)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert not (tmp_path / "graphify-out" / "graph.html").exists()


def test_refused_update_restores_label_cache(tmp_path: Path) -> None:
    _write_full_artifacts(tmp_path, node_count=999, report_nodes=999)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_full(fake_bin, result_node_count=50, report_nodes=50)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert (tmp_path / "graphify-out" / ".graphify_labels.json").read_text(
        encoding="utf-8"
    ) == "{}"


def test_accepted_update_keeps_new_artifacts(tmp_path: Path) -> None:
    """The restore path must not fire on a good run: a healthy update's new
    report and graph.html are the whole point of running it."""
    _write_full_artifacts(tmp_path, node_count=1000, report_nodes=1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_full(fake_bin, result_node_count=1005, report_nodes=1005)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
    report = (tmp_path / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "1005 nodes" in report
    assert (tmp_path / "graphify-out" / "graph.html").exists()


def test_graphify_crash_also_restores_report(tmp_path: Path) -> None:
    """The crash path had the same partial-restore bug as the refusal path."""
    _write_full_artifacts(tmp_path, node_count=1000, report_nodes=1000)
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(parents=True)
    fake = fake_bin / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        "echo 'not json' > graphify-out/graph.json\n"
        "printf '# Graph Report\\n- 3 nodes\\n' > graphify-out/GRAPH_REPORT.md\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    report = (tmp_path / "graphify-out" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "1000 nodes" in report


def _fake_graphify_with_snapshot(
    bin_dir: Path, result_node_count: int, report_nodes: int, snapshot_date: str
) -> None:
    """Like _fake_graphify_full, but also drops a dated snapshot dir the way
    real graphify does -- containing a PRE-update copy.

    That ordering is established from tracked git history: in commit eae14a6c
    (which added graphify-out/2026-07-29/) the snapshot's graph.json has 32529
    nodes, the top-level in the SAME commit has 28307, and the top-level in the
    PARENT commit has 32529 -- the snapshot matches the parent, i.e. pre-update.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        f"mkdir -p graphify-out/{snapshot_date}\n"
        # snapshot captures the PRE-update state, before anything is rewritten
        f"cp graphify-out/graph.json graphify-out/{snapshot_date}/graph.json\n"
        f"cp graphify-out/GRAPH_REPORT.md graphify-out/{snapshot_date}/GRAPH_REPORT.md\n"
        f"python3 -c \"import json; json.dump({{'nodes': [{{'id': i}} for i in range({result_node_count})], 'edges': []}}, open('graphify-out/graph.json', 'w'))\"\n"
        f"printf '# Graph Report\\n- {report_nodes} nodes\\n' > graphify-out/GRAPH_REPORT.md\n"
        "printf 'generated' > graphify-out/graph.html\n"
        "echo 'Code graph updated.'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)


def test_run_created_snapshot_dir_is_reported_and_holds_pre_update_state(
    tmp_path: Path,
) -> None:
    """The scenario the suite previously never exercised: a snapshot dir
    created BY the destructive run. It must be named in output (it is
    gitignored, so `git status` will not show it) and must retain the
    PRE-update copy, which is what makes it a usable recovery source."""
    _write_full_artifacts(tmp_path, node_count=999, report_nodes=999)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_with_snapshot(fake_bin, 50, 50, "2026-08-13")

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "2026-08-13" in proc.stderr

    snap = tmp_path / "graphify-out" / "2026-08-13"
    assert snap.exists(), "snapshot dir must not be auto-deleted"
    assert "999 nodes" in (snap / "GRAPH_REPORT.md").read_text(encoding="utf-8")


def test_preexisting_snapshot_dir_is_not_named_for_deletion(tmp_path: Path) -> None:
    """Regression: an earlier version globbed EVERY dated dir and told the
    operator to remove it -- including the git-TRACKED graphify-out/2026-07-29/,
    i.e. advice to delete committed repo content. Only dirs this run created
    may be named."""
    _write_full_artifacts(tmp_path, node_count=999, report_nodes=999)
    preexisting = tmp_path / "graphify-out" / "2026-07-29"
    preexisting.mkdir()
    (preexisting / "GRAPH_REPORT.md").write_text("tracked content", encoding="utf-8")

    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_with_snapshot(fake_bin, 50, 50, "2026-08-13")

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "2026-08-13" in proc.stderr
    assert "2026-07-29" not in proc.stderr
    assert preexisting.exists()


def test_snapshot_warning_fires_on_success_path(tmp_path: Path) -> None:
    """Snapshots are gitignored now, so a HEALTHY run would otherwise add
    ~45MB to disk with no signal anywhere."""
    _write_full_artifacts(tmp_path, node_count=1000, report_nodes=1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify_with_snapshot(fake_bin, 1005, 1005, "2026-08-13")

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
    assert "2026-08-13" in proc.stderr


def test_snapshot_warning_fires_on_crash_path(tmp_path: Path) -> None:
    """The crash path restored artifacts but issued no snapshot warning."""
    _write_full_artifacts(tmp_path, node_count=1000, report_nodes=1000)
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(parents=True)
    fake = fake_bin / "graphify"
    fake.write_text(
        "#!/bin/sh\n"
        "mkdir -p graphify-out/2026-08-13\n"
        "cp graphify-out/graph.json graphify-out/2026-08-13/graph.json\n"
        "echo 'not json' > graphify-out/graph.json\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 1
    assert "2026-08-13" in proc.stderr


def test_boundary_at_exactly_threshold_percent(tmp_path: Path) -> None:
    """Exactly at the 10% default threshold should NOT trigger (strictly greater-than)."""
    _write_graph(tmp_path, 1000)
    fake_bin = tmp_path / "fake_bin"
    _fake_graphify(fake_bin, 900)  # exactly 10% drop

    proc = _run(tmp_path, fake_bin)
    assert proc.returncode == 0
