"""Gate tests for the metric lineage CI gate (phase 4).

The point of a gate is that it FAILS on real breakage. Most of these construct
the breakage explicitly and assert the failure, rather than only asserting the
green path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.metrics.gate import (
    BASELINE_PATH,
    _resolve_consumer_path,
    check_declared_consumers,
    check_orphan_ratchet,
    check_registry_integrity,
    load_baseline,
    missing_declared_consumers,
    run_gate,
    write_baseline,
)
from orion.metrics.lineage import MetricGraph, MetricNode, build_graph

REPO_ROOT = Path(__file__).resolve().parents[1]


def _node(**kw) -> MetricNode:
    base = dict(
        urn="metric://inner_state/svc/x",
        surface="inner_state",
        producer_service="svc",
        name="x",
        registry_source="orion/fake_registry.py",
    )
    base.update(kw)
    return MetricNode(**base)


# ------------------------------------------------- consumer path resolution


def test_dotted_module_consumer_resolves_to_py_path():
    got = _resolve_consumer_path(
        "services.orion-hub.app.inner_state:build_features", REPO_ROOT
    )
    assert got == REPO_ROOT / "services/orion-hub/app/inner_state.py"


def test_bare_service_consumer_resolves_to_service_dir():
    got = _resolve_consumer_path("orion-hub", REPO_ROOT)
    assert got == REPO_ROOT / "services/orion-hub"


def test_trailing_prose_is_stripped_before_resolving():
    """Real entry: '...:build_chat_stance_inputs (ctx[...] via drive_state_postgres)'."""
    got = _resolve_consumer_path(
        "services.orion-cortex-exec.app.chat_stance:build_chat_stance_inputs "
        "(ctx['chat_drive_state'] via drive_state_postgres)",
        REPO_ROOT,
    )
    assert got == REPO_ROOT / "services/orion-cortex-exec/app/chat_stance.py"


def test_wildcard_consumer_is_not_checkable():
    assert _resolve_consumer_path("*", REPO_ROOT) is None


# ----------------------------------------------------------- registry rot


def test_dangling_upstream_urn_fails():
    graph = MetricGraph(nodes={})
    node = _node(upstream=("metric://organ_signal/ghost/ghost",))
    graph.nodes[node.urn] = node
    failures = check_registry_integrity(graph)
    assert len(failures) == 1
    assert "dangling upstream" in failures[0]


def test_resolvable_upstream_passes():
    parent = _node(urn="metric://inner_state/svc/parent", name="parent")
    child = _node(urn="metric://inner_state/svc/child", name="child", upstream=(parent.urn,))
    graph = MetricGraph(nodes={parent.urn: parent, child.urn: child})
    assert check_registry_integrity(graph) == []


def test_empty_graph_fails():
    assert check_registry_integrity(MetricGraph(nodes={})) != []


# ------------------------------------------------- declared-consumer check


def test_missing_declared_consumer_fails(tmp_path):
    node = _node(declared_consumers=("orion-does-not-exist",))
    graph = MetricGraph(nodes={node.urn: node})
    failures = check_declared_consumers(graph, tmp_path)
    assert len(failures) == 1
    assert "orion-does-not-exist" in failures[0]


def test_known_missing_consumer_is_carried_as_debt_not_failure(tmp_path):
    node = _node(declared_consumers=("orion-does-not-exist",))
    graph = MetricGraph(nodes={node.urn: node})
    assert check_declared_consumers(graph, tmp_path, {"orion-does-not-exist"}) == []


def test_new_missing_consumer_fails_even_when_others_are_known(tmp_path):
    """The ratchet must still catch a NEW false claim."""
    node = _node(declared_consumers=("orion-known-gone", "orion-newly-gone"))
    graph = MetricGraph(nodes={node.urn: node})
    failures = check_declared_consumers(graph, tmp_path, {"orion-known-gone"})
    assert len(failures) == 1
    assert "orion-newly-gone" in failures[0]


def test_existing_consumer_passes(tmp_path):
    (tmp_path / "services" / "orion-real").mkdir(parents=True)
    node = _node(declared_consumers=("orion-real",))
    graph = MetricGraph(nodes={node.urn: node})
    assert check_declared_consumers(graph, tmp_path) == []


# ------------------------------------------------------------ orphan ratchet


def test_orphan_growth_fails():
    failures = check_orphan_ratchet({"organ_signal": 39}, {"organ_signal": 38})
    assert len(failures) == 1
    assert "38 -> 39" in failures[0]


def test_orphan_shrink_passes():
    assert check_orphan_ratchet({"organ_signal": 30}, {"organ_signal": 38}) == []


def test_orphan_equal_passes():
    assert check_orphan_ratchet({"organ_signal": 38}, {"organ_signal": 38}) == []


def test_brand_new_surface_with_orphans_fails():
    """A surface absent from the baseline defaults to 0, so any orphan is growth."""
    failures = check_orphan_ratchet({"new_surface": 1}, {"organ_signal": 38})
    assert len(failures) == 1
    assert "new_surface" in failures[0]


def test_absent_baseline_fails_rather_than_passing_vacuously():
    """A missing baseline must not silently green-light everything."""
    failures = check_orphan_ratchet({"organ_signal": 999}, {})
    assert len(failures) == 1
    assert "no orphan baseline" in failures[0]


# ---------------------------------------------------------------- baseline


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline({"organ_signal": 38}, {"orion-gone": "orion/bus/channels.yaml"}, path)
    loaded = load_baseline(path)
    assert loaded["orphans_by_surface"] == {"organ_signal": 38}
    assert loaded["known_missing_consumers"] == {"orion-gone": "orion/bus/channels.yaml"}


def test_committed_baseline_exists_and_parses():
    assert BASELINE_PATH.exists(), "the gate needs a committed baseline to ratchet against"
    data = load_baseline()
    assert data["orphans_by_surface"]
    assert "known_missing_consumers" in data


# ------------------------------------------------------------ live gate


def test_gate_passes_on_current_repo():
    """The gate must be green on main, with pre-existing debt in the baseline."""
    result = run_gate()
    assert result.ok, f"gate failed on a clean tree: {result.failures}"


def test_make_targets_resolve_a_real_interpreter():
    """Regression: the check-metric-lineage target shipped in PR #1603 invoked
    bare `python`, which does not exist on this host -- `make` died with
    exit 127 before running anything. A green pytest run said nothing about it
    because the tests never went through make.

    Also covers the worktree case: linked worktrees have no .venv of their own,
    so a naive lookup resolves to a python3 without pydantic exactly where
    CLAUDE.md 2 requires the work to happen.
    """
    import shutil
    import subprocess

    for target in ("check-metric-lineage", "check-metric-lineage-gate"):
        proc = subprocess.run(
            ["make", "-n", target],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"make -n {target} failed: {proc.stderr}"
        interpreter = proc.stdout.strip().split()[0]
        resolved = (
            interpreter
            if Path(interpreter).is_absolute() or "/" in interpreter
            else shutil.which(interpreter)
        )
        assert resolved and Path(resolved).exists(), (
            f"{target} invokes {interpreter!r}, which does not exist"
        )


def test_live_repo_still_has_the_three_known_missing_consumers():
    """Documents the real debt this gate is carrying.

    If one gets fixed this test fails loudly, which is the prompt to shrink the
    baseline rather than let it quietly drift.
    """
    graph = build_graph()
    missing = missing_declared_consumers(graph, REPO_ROOT)
    assert set(missing) == set(load_baseline()["known_missing_consumers"])
