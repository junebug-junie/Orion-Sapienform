"""Bus ingest worker invariants: skip own echoes, never overwrite governed state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from conftest import make_proposal
from fake_repo import FakeRepository

from app.worker import ProposalIngestWorker
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory.crystallization import governor


@dataclass
class _Decoded:
    ok: bool
    envelope: Any
    error: str | None = None


class _FakeCodec:
    def decode(self, data: Any) -> _Decoded:
        return _Decoded(ok=True, envelope=data)


class _FakeBus:
    codec = _FakeCodec()


def _make_worker(repo: FakeRepository) -> ProposalIngestWorker:
    return ProposalIngestWorker(
        bus=_FakeBus(),
        repo=repo,
        channel="orion:memory:crystallization:proposed",
        service_name="orion-memory-crystallizer",
    )


def _envelope(payload: dict, source: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="memory.crystallization.proposed.v1",
        source=ServiceRef(name=source),
        payload=payload,
    )


def _handle(worker: ProposalIngestWorker, envelope: BaseEnvelope) -> None:
    asyncio.run(worker._handle_message({"data": envelope}))


def test_worker_ingests_external_proposal() -> None:
    repo = FakeRepository()
    worker = _make_worker(repo)
    proposal = make_proposal()
    _handle(worker, _envelope(proposal.model_dump(mode="json"), source="orion-cortex-orch"))
    stored = repo.get(proposal.crystallization_id)
    assert stored is not None
    assert stored.status == "proposed"
    assert stored.governance.validation_status == "valid"
    assert [e.op for e in repo.history] == ["validate"]


def test_worker_skips_own_echo() -> None:
    repo = FakeRepository()
    worker = _make_worker(repo)
    proposal = make_proposal()
    _handle(worker, _envelope(proposal.model_dump(mode="json"), source="orion-memory-crystallizer"))
    assert repo.get(proposal.crystallization_id) is None


def test_worker_ignores_non_proposed_status() -> None:
    repo = FakeRepository()
    worker = _make_worker(repo)
    active = make_proposal().model_copy(update={"status": "active"})
    _handle(worker, _envelope(active.model_dump(mode="json"), source="elsewhere"))
    assert repo.get(active.crystallization_id) is None


def test_worker_never_overwrites_governed_state() -> None:
    repo = FakeRepository()
    worker = _make_worker(repo)
    proposal = make_proposal()
    approved, _ = governor.approve(proposal, "operator:juniper")
    repo.upsert(approved)

    # A stale re-publication of the original proposal arrives after approval.
    _handle(worker, _envelope(proposal.model_dump(mode="json"), source="orion-cortex-orch"))
    stored = repo.get(proposal.crystallization_id)
    assert stored is not None
    assert stored.status == "active"  # not reverted to proposed
