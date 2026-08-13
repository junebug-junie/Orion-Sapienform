"""Unit tests for app/producers/doc_semantic_drift.py's scheduling logic --
cold start, publish-on-real-doc-change, no-op-on-no-change, failed-publish
non-advancement. Same cold-start-sha state machine as git_delta_loop, tested
the same way (monkeypatch the pure/IO boundaries, not the real git plumbing --
that's already covered by orion/structural_mass/tests/test_doc_semantic_drift.py).

``fork_rpc_client`` and ``_score_change`` (the real RPC embedding-request
path) are monkeypatched throughout -- exercising a real Redis RPC round trip
has no place in this fast/deterministic test lane; see
services/orion-vector-host's own tests for real embedding-host coverage.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.producers import doc_semantic_drift as doc_semantic_drift_module  # noqa: E402

from orion.schemas.doc_semantic_drift import DocSemanticDriftV1  # noqa: E402
from orion.structural_mass.doc_semantic_drift import DocHunkChange  # noqa: E402


async def _run_until_calls(loop_coro_factory, stop: asyncio.Event, target_calls: int, call_list: list) -> None:
    async def _watcher() -> None:
        while len(call_list) < target_calls:
            await asyncio.sleep(0.001)
        stop.set()

    await asyncio.wait_for(asyncio.gather(loop_coro_factory(), _watcher()), timeout=5.0)


def _change(sha: str, path: str = "README.md") -> DocHunkChange:
    return DocHunkChange(
        sha=sha, path=path, commit_subject="docs: real edit", commit_prefix="docs",
        hunk_removed="old line", hunk_added="new line",
    )


def _fake_event(change: DocHunkChange, diff: float | None = 0.5) -> DocSemanticDriftV1:
    return DocSemanticDriftV1(
        observed_at=datetime.now(timezone.utc), sha=change.sha, path=change.path,
        commit_prefix=change.commit_prefix, diff_scoped_embedding_diff=diff,
        chunk_count_removed=1, chunk_count_added=1,
        hunk_removed_len_chars=len(change.hunk_removed), hunk_added_len_chars=len(change.hunk_added),
    )


@pytest.fixture(autouse=True)
def _stub_rpc_fork(monkeypatch, fake_bus):
    """fork_rpc_client is async and would otherwise try a real bus.fork() --
    stub it to hand back the same fake_bus, matching FakeBus's own contract
    (it never actually needs RPC methods since _score_change is monkeypatched
    per-test below)."""

    async def _fake_fork(parent):
        return parent

    monkeypatch.setattr(doc_semantic_drift_module, "fork_rpc_client", _fake_fork)


@pytest.mark.asyncio
async def test_cold_start_seeds_baseline_without_publishing(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return shas[0]

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.published == []


STATE_KEY = "orion:cocreation_signals:state:doc_semantic_drift:last_sha"


@pytest.mark.asyncio
async def test_cold_start_persists_the_seeded_baseline(fake_bus, source, stop_event, monkeypatch):
    """The very first cold-start tick must persist its seeded baseline, not
    just hold it in memory -- otherwise a restart before any real doc change
    ever happens would cold-start all over again instead of resuming."""
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return "a" * 40

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key=STATE_KEY, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.redis_store.get(STATE_KEY) == "a" * 40


@pytest.mark.asyncio
async def test_real_change_persists_new_baseline_to_redis(fake_bus, source, stop_event, monkeypatch):
    """Regression guard for the live bug (2026-08-12, real PRs #1571/#1577):
    after a real, fully-published change, the new baseline must be durably
    persisted, not just held in the loop's own local variable."""
    shas = ["a" * 40, "a" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    change = _change("b" * 40)

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", lambda p, h, r: [change])
    monkeypatch.setattr(
        doc_semantic_drift_module, "_score_change",
        lambda rpc_bus, **kwargs: asyncio.sleep(0, result=_fake_event(kwargs["change"])),
    )

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key=STATE_KEY, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.redis_store.get(STATE_KEY) == "b" * 40


@pytest.mark.asyncio
async def test_failed_publish_does_not_persist_the_unpublished_sha(fake_bus, source, stop_event, monkeypatch):
    """A failed publish must not durably record a baseline past the change
    that failed to publish -- otherwise a restart right after the failure
    would permanently lose that range, not just re-cover it next tick."""
    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    shas = ["a" * 40, "a" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    change = _change("b" * 40)
    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", lambda p, h, r: [change])
    monkeypatch.setattr(
        doc_semantic_drift_module, "_score_change",
        lambda rpc_bus, **kwargs: asyncio.sleep(0, result=_fake_event(kwargs["change"])),
    )

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key=STATE_KEY, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    # Cold start seeded "a"*40 durably -- the failed publish must not have
    # advanced it to "b"*40.
    assert fake_bus.redis_store.get(STATE_KEY) == "a" * 40


@pytest.mark.asyncio
async def test_restart_resumes_from_durable_state_instead_of_cold_starting_at_new_head(
    source, stop_event, monkeypatch
):
    """The exact real bug this patch fixes: a redeploy (fresh process, fresh
    FakeBus, but the SAME Redis-backed state) landing right after a real doc
    change must resume from the last real baseline and score that change --
    not silently re-seed at the new HEAD and swallow it."""
    from conftest import FakeBus

    shared_redis_store: dict = {"orion:cocreation_signals:state:doc_semantic_drift:last_sha": "a" * 40}

    # "Process 2" -- a fresh FakeBus (fresh in-process last_sha=None) but
    # wired to the SAME durable store a real Redis-backed restart would see.
    bus_after_restart = FakeBus(redis_store=shared_redis_store)

    change = _change("b" * 40)
    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", lambda repo_path: "b" * 40)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", lambda p, h, r: [change])
    monkeypatch.setattr(
        doc_semantic_drift_module, "_score_change",
        lambda rpc_bus, **kwargs: asyncio.sleep(0, result=_fake_event(kwargs["change"], diff=0.7)),
    )

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=bus_after_restart, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key=STATE_KEY, stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=bus_after_restart.published)

    assert len(bus_after_restart.published) == 1
    _, envelope = bus_after_restart.published[0]
    assert envelope.payload["sha"] == "b" * 40
    assert envelope.payload["diff_scoped_embedding_diff"] == 0.7


@pytest.mark.asyncio
async def test_forked_rpc_bus_client_is_closed_on_loop_exit(fake_bus, source, stop_event, monkeypatch):
    """Regression guard (code review 2026-08-11): the first draft forked a
    dedicated RPC bus client and never closed it -- a real leak on every
    task cancellation/redeploy."""
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return "a" * 40

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=1, call_list=calls)

    assert fake_bus.closed is True


@pytest.mark.asyncio
async def test_real_doc_change_publishes_scored_event(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40, "a" * 40, "b" * 40]  # cold start, no-op, real change
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    change = _change("b" * 40)

    def _fake_changes(prev_sha, head_sha, repo_path):
        return [change]

    async def _fake_score_change(rpc_bus, **kwargs):
        return _fake_event(kwargs["change"], diff=0.6)

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", _fake_changes)
    monkeypatch.setattr(doc_semantic_drift_module, "_score_change", _fake_score_change)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert len(fake_bus.published) == 1
    channel, envelope = fake_bus.published[0]
    assert channel == "orion:substrate:doc_semantic_drift"
    assert envelope.payload["sha"] == "b" * 40
    assert envelope.payload["path"] == "README.md"
    assert envelope.payload["diff_scoped_embedding_diff"] == 0.6


@pytest.mark.asyncio
async def test_no_change_publishes_nothing(fake_bus, source, stop_event, monkeypatch):
    shas = ["a" * 40, "a" * 40, "a" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        calls.append(repo_path)
        return shas[min(len(calls) - 1, len(shas) - 1)]

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_no_real_doc_files_changed_publishes_nothing_but_advances_sha(
    fake_bus, source, stop_event, monkeypatch
):
    """A real HEAD change with zero *.md files touched (e.g. a code-only
    commit) must still advance last_sha -- otherwise every subsequent tick
    re-diffs a growing, entirely-irrelevant range forever."""
    shas = ["a" * 40, "a" * 40, "b" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    changes_calls: list[tuple] = []

    def _fake_changes(prev_sha, head_sha, repo_path):
        changes_calls.append((prev_sha, head_sha))
        return []

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", _fake_changes)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=4, call_list=calls)

    assert fake_bus.published == []
    # Only one real diff call, against the original baseline -- the SHA
    # change on tick 3 advanced last_sha even though it published nothing,
    # so tick 4 never re-diffs.
    assert changes_calls == [("a" * 40, "b" * 40)]


@pytest.mark.asyncio
async def test_failed_publish_does_not_advance_state(fake_bus, source, stop_event, monkeypatch):
    async def _raising_publish(channel, envelope):
        raise ConnectionError("transient redis failure")

    fake_bus.publish = _raising_publish

    shas = ["a" * 40, "a" * 40, "b" * 40, "b" * 40]
    calls: list[str] = []

    def _fake_head(repo_path: str) -> str:
        idx = min(len(calls), len(shas) - 1)
        calls.append(repo_path)
        return shas[idx]

    change = _change("b" * 40)
    changes_calls: list[tuple] = []

    def _fake_changes(prev_sha, head_sha, repo_path):
        changes_calls.append((prev_sha, head_sha))
        return [change]

    async def _fake_score_change(rpc_bus, **kwargs):
        return _fake_event(kwargs["change"])

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", _fake_changes)
    monkeypatch.setattr(doc_semantic_drift_module, "_score_change", _fake_score_change)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=4, call_list=calls)

    assert changes_calls
    assert all(prev == "a" * 40 for prev, _ in changes_calls)
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

    change = _change("b" * 40)

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _fake_head)
    monkeypatch.setattr(doc_semantic_drift_module, "doc_semantic_drift_changes", lambda p, h, r: [change])
    monkeypatch.setattr(
        doc_semantic_drift_module, "_score_change",
        lambda rpc_bus, **kwargs: asyncio.sleep(0, result=_fake_event(kwargs["change"])),
    )

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=3, call_list=calls)

    assert fake_bus.published == []


@pytest.mark.asyncio
async def test_tick_exception_does_not_kill_the_loop(fake_bus, source, stop_event, monkeypatch):
    calls: list[str] = []

    def _flaky_head(repo_path: str) -> str:
        calls.append(repo_path)
        if len(calls) == 1:
            raise RuntimeError("transient git failure")
        return "a" * 40

    monkeypatch.setattr(doc_semantic_drift_module, "_current_head_sha", _flaky_head)

    async def _loop():
        await doc_semantic_drift_module.doc_semantic_drift_loop(
            bus=fake_bus, channel="orion:substrate:doc_semantic_drift", source=source,
            repo_path="/repo", embed_request_channel="orion:embedding:generate",
            embed_collection="doc_semantic_drift", embed_timeout_sec=5.0,
            chunk_char_size=2048, poll_interval_sec=0.01,
            state_key="orion:cocreation_signals:state:doc_semantic_drift:last_sha", stop=stop_event,
        )

    await _run_until_calls(_loop, stop_event, target_calls=2, call_list=calls)

    assert len(calls) == 2


def test_cosine_similarity_orthogonal_vectors_score_one_diff() -> None:
    similarity = doc_semantic_drift_module._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_identical_vectors_score_zero_diff() -> None:
    similarity = doc_semantic_drift_module._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_returns_none_for_zero_vector() -> None:
    assert doc_semantic_drift_module._cosine_similarity([0.0, 0.0], [1.0, 1.0]) is None


def test_cosine_similarity_returns_none_for_mismatched_dims() -> None:
    assert doc_semantic_drift_module._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) is None
