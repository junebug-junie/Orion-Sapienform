"""Unit tests for app/producers/git_delta.py's scheduling logic -- cold start,
publish-on-real-change, no-op-on-no-change. The pure git_churn_delta()
function itself is already covered by orion/structural_mass/tests/; these
tests monkeypatch it to isolate this producer's own state-machine behavior.
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

from app.producers import git_delta as git_delta_module  # noqa: E402

from orion.structural_mass.git_delta import GitChurnDelta  # noqa: E402


def _delta(prev_sha: str, head_sha: str, lines_changed: int = 100) -> GitChurnDelta:
    return GitChurnDelta(
        prev_sha=prev_sha, head_sha=head_sha, commit_count=1,
        files_added=0, files_deleted=0, files_modified=1,
        lines_added=lines_changed, lines_removed=0,
    )


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    """Run the loop, stopping once `call_list` reaches `target_calls` entries
    -- deterministic, not timing-dependent."""

    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


@pytest.mark.asyncio
async def test_cold_start_seeds_baseline_without_publishing(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return shas[0]

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _fake_head)

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_real_sha_change_publishes_real_delta(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40, "a" * 40, "b" * 40]  # cold start, no-op, real change
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    def _fake_churn(prev_sha, head_sha, repo_path):
        return _delta(prev_sha, head_sha, lines_changed=250)

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(git_delta_module, "git_churn_delta", _fake_churn)

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert len(fake_bus.published) == 1
    channel, envelope = fake_bus.published[0]
    assert channel == "orion:substrate:codebase_delta"
    assert envelope.payload["domain"] == "git"
    assert envelope.payload["git"]["prev_sha"] == "a" * 40
    assert envelope.payload["git"]["head_sha"] == "b" * 40
    assert envelope.payload["pr_lifecycle"] is None
    assert envelope.payload["graph"] is None


@pytest.mark.asyncio
async def test_no_change_publishes_nothing(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40, "a" * 40, "a" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return shas[min(len(calls) - 1, len(shas) - 1)]

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _fake_head)

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_disabled_bus_skips_publish_without_raising(fake_bus, source, stop_event, monkeypatch):
    fake_bus.enabled = False
    shas = ["a" * 40, "a" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(git_delta_module, "git_churn_delta", lambda p, h, r: _delta(p, h))

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_state(fake_bus, source, stop_event, monkeypatch):
    """Regression guard (code review 2026-07-30): if the bus publish itself
    raises, last_sha must NOT advance -- otherwise the delta is permanently
    lost instead of being folded into the next tick's (bigger) diff."""

    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    shas = ["a" * 40, "a" * 40, "b" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    churn_calls: list[tuple] = []

    def _fake_churn(prev_sha, head_sha, repo_path):
        churn_calls.append((prev_sha, head_sha))
        return _delta(prev_sha, head_sha, lines_changed=250)

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(git_delta_module, "git_churn_delta", _fake_churn)

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=4, call_list=calls)

    # Every attempted diff after the SHA changed to "b"*40 must still be
    # against the ORIGINAL prev_sha ("a"*40) -- proof last_sha never advanced
    # past a failed publish.
    assert churn_calls
    assert all(prev == "a" * 40 for prev, _ in churn_calls)
    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_tick_exception_does_not_kill_the_loop(fake_bus, source, stop_event, monkeypatch):
    """A transient git failure on one tick must not crash the whole loop --
    the next tick should still run."""
    calls: list[str] = []

    def _flaky_head(repo_path: str) -> str:
        calls.append(repo_path)
        if len(calls) == 1:
            raise RuntimeError("transient git failure")
        return "a" * 40

    monkeypatch.setattr(git_delta_module, "_current_head_sha", _flaky_head)

    async def _loop():
        await git_delta_module.git_delta_loop(
            bus=fake_bus, channel="orion:substrate:codebase_delta", source=source,
            repo_path="/repo", poll_interval_sec=0.01, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=2, call_list=calls)

    assert len(calls) == 2
