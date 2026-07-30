"""Unit tests for app/producers/graph_delta.py's scheduling logic -- cold
start, publish-on-real-graphify-update, no-op-on-no-change, missing-file
handling. The pure delta/snapshot functions are already covered by
orion/structural_mass/tests/.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import graph_delta as graph_delta_module  # noqa: E402

from orion.structural_mass.graph_delta import GraphStructuralDelta  # noqa: E402
from orion.structural_mass.snapshot_history import GraphSnapshotStats  # noqa: E402


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


def _snap(commit_sha: str, node_count: int = 100) -> GraphSnapshotStats:
    return GraphSnapshotStats(node_count=node_count, edge_count=200, community_count=10, god_nodes=("A",), commit_sha=commit_sha)


@pytest.mark.asyncio
async def test_cold_start_seeds_baseline_without_publishing(fake_bus, source, stop_event, monkeypatch, tmp_path):
    graph_json = tmp_path / "graphify-out" / "graph.json"
    graph_json.parent.mkdir(parents=True)
    graph_json.write_text('{"built_at_commit": "sha1"}')

    calls: list[str] = []

    def _fake_load(repo_path, commit_sha):
        calls.append(commit_sha)
        return _snap(commit_sha)

    monkeypatch.setattr(graph_delta_module, "_load_snapshot", _fake_load)

    async def _loop():
        await graph_delta_module.graph_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_real_commit_change_publishes_real_delta(fake_bus, source, stop_event, monkeypatch, tmp_path):
    graph_dir = tmp_path / "graphify-out"
    graph_dir.mkdir(parents=True)
    graph_json = graph_dir / "graph.json"
    graph_json.write_text('{"built_at_commit": "sha1"}')

    read_calls: list[str] = []
    load_calls: list[str] = []
    # read_built_at_commit is called every tick; _load_snapshot only on a
    # real change -- these are NOT 1:1, so each needs its own call counter
    # (a single shared counter previously made this test hang: the no-op
    # tick's read never advanced the index, so the "real change" value was
    # never reached).
    commit_sequence = ["sha1", "sha1", "sha2"]  # cold start, no-op, real change

    def _fake_read_commit(path):
        idx = min(len(read_calls), len(commit_sequence) - 1)
        read_calls.append(commit_sequence[idx])
        return commit_sequence[idx]

    def _fake_load(repo_path, commit_sha):
        load_calls.append(commit_sha)
        return _snap(commit_sha, node_count=100 if commit_sha == "sha1" else 150)

    def _fake_delta(prev, curr):
        return GraphStructuralDelta(
            node_count_delta=curr.node_count - prev.node_count,
            edge_count_delta=0, community_count_delta=0, god_node_jaccard_similarity=1.0,
        )

    monkeypatch.setattr(graph_delta_module, "_read_built_at_commit", _fake_read_commit)
    monkeypatch.setattr(graph_delta_module, "_load_snapshot", _fake_load)
    monkeypatch.setattr(graph_delta_module, "graph_structural_delta", _fake_delta)

    async def _loop():
        await graph_delta_module.graph_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=read_calls)

    assert len(fake_bus.published) == 1
    channel, envelope = fake_bus.published[0]
    assert envelope.payload["domain"] == "graph"
    assert envelope.payload["graph"]["node_count_delta"] == 50
    assert envelope.payload["git"] is None
    assert envelope.payload["pr_lifecycle"] is None


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_state(fake_bus, source, stop_event, monkeypatch, tmp_path):
    """Regression guard (code review 2026-07-30): if the bus publish itself
    raises, last_snapshot must NOT advance to the new commit."""
    graph_dir = tmp_path / "graphify-out"
    graph_dir.mkdir(parents=True)
    (graph_dir / "graph.json").write_text('{"built_at_commit": "sha1"}')

    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    read_calls: list[str] = []
    commit_sequence = ["sha1", "sha1", "sha2", "sha2"]  # cold start, no-op, real change, still "changed" vs stale last_snapshot

    def _fake_read_commit(path):
        idx = min(len(read_calls), len(commit_sequence) - 1)
        read_calls.append(commit_sequence[idx])
        return commit_sequence[idx]

    def _fake_load(repo_path, commit_sha):
        return _snap(commit_sha)

    delta_calls: list[tuple] = []

    def _fake_delta(prev, curr):
        delta_calls.append((prev.commit_sha, curr.commit_sha))
        return GraphStructuralDelta(node_count_delta=1, edge_count_delta=0, community_count_delta=0, god_node_jaccard_similarity=1.0)

    monkeypatch.setattr(graph_delta_module, "_read_built_at_commit", _fake_read_commit)
    monkeypatch.setattr(graph_delta_module, "_load_snapshot", _fake_load)
    monkeypatch.setattr(graph_delta_module, "graph_structural_delta", _fake_delta)

    async def _loop():
        await graph_delta_module.graph_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=4, call_list=read_calls)

    # Every attempted diff after the commit changed to "sha2" must still be
    # against the ORIGINAL prior commit ("sha1") -- proof last_snapshot never
    # advanced past a failed publish.
    assert delta_calls
    assert all(prev == "sha1" for prev, _ in delta_calls)
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_no_commit_change_publishes_nothing(fake_bus, source, stop_event, monkeypatch, tmp_path):
    graph_dir = tmp_path / "graphify-out"
    graph_dir.mkdir(parents=True)
    (graph_dir / "graph.json").write_text('{"built_at_commit": "sha1"}')

    calls: list[str] = []

    def _fake_read_commit(path):
        calls.append(str(path))
        return "sha1"

    def _fake_load(repo_path, commit_sha):
        return _snap(commit_sha)

    monkeypatch.setattr(graph_delta_module, "_read_built_at_commit", _fake_read_commit)
    monkeypatch.setattr(graph_delta_module, "_load_snapshot", _fake_load)

    async def _loop():
        await graph_delta_module.graph_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_missing_graph_json_does_not_crash_the_loop(fake_bus, source, stop_event, tmp_path):
    """graphify-out/graph.json not present yet (e.g. graphify never run on
    this checkout) must be handled gracefully, not raise out of the tick."""
    task = asyncio.create_task(
        graph_delta_module.graph_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path=str(tmp_path), poll_interval_sec=0.01, stop=stop_event,
        )
    )
    await asyncio.sleep(0.05)  # let a few real ticks run against the missing file
    stop_event.set()
    await asyncio.wait_for(task, timeout=5.0)

    assert fake_bus.published == []
