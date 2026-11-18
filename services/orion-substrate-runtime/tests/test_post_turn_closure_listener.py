from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from orion.schemas.harness_finalize import HarnessPostTurnClosureV1


def _sample_closure(*, correlation_id: str = "c-1", surprise_unresolved: bool = True) -> HarnessPostTurnClosureV1:
    return HarnessPostTurnClosureV1(
        correlation_id=correlation_id,
        outcome_molecule_id="outcome-1",
        verdict_molecule_id="verdict-1",
        grammar_event_ids=["g-1", "g-2"],
        surprise_unresolved=surprise_unresolved,
    )


@pytest.mark.asyncio
async def test_handle_post_turn_closure_message_invokes_handler() -> None:
    from app.post_turn_closure_listener import handle_post_turn_closure_message

    closure = _sample_closure()
    seen: list[HarnessPostTurnClosureV1] = []

    def _on_closure(payload: HarnessPostTurnClosureV1) -> None:
        seen.append(payload)

    await handle_post_turn_closure_message(closure, on_closure=_on_closure)
    assert seen == [closure]


@pytest.mark.asyncio
async def test_handle_post_turn_closure_bus_message_decodes_envelope() -> None:
    from app.post_turn_closure_listener import _handle_bus_message
    from app.settings import Settings
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    closure = _sample_closure(correlation_id="corr-bus")
    corr = str(uuid4())
    envelope = BaseEnvelope(
        kind="harness.post_turn.closure.v1",
        source=ServiceRef(name="orion-harness-governor"),
        correlation_id=corr,
        payload=closure.model_copy(update={"correlation_id": corr}).model_dump(mode="json"),
    )
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=True, envelope=envelope, error=None)

    seen: list[HarnessPostTurnClosureV1] = []

    await _handle_bus_message(
        bus,
        {"data": b"ignored"},
        settings=Settings(
            POSTGRES_URI="postgresql://orion:orion@localhost:5432/orion",
        ),
        on_closure=seen.append,
    )
    assert len(seen) == 1
    assert seen[0].correlation_id == corr
    assert seen[0].grammar_event_ids == ["g-1", "g-2"]


def _make_worker() -> object:
    from app.worker import BiometricsSubstrateWorker

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    return worker


class _RecordingStore:
    """Fake graph store that persists nodes and exposes a SQL engine stand-in."""

    def __init__(self) -> None:
        self.nodes: dict = {}
        self.engine = object()

    def get_node_by_id(self, node_id):
        return self.nodes.get(node_id)

    def upsert_node(self, *, identity_key, node, skip_metadata_keys=None) -> None:
        self.nodes[node.node_id] = node


def _no_pe_write(**_kwargs: object) -> None:
    raise AssertionError("harness_closure prediction_error is retired; nothing may write it")


def test_unresolved_closure_writes_no_prediction_error_node(monkeypatch) -> None:
    """Retire step 3 (2026-11-11): an unresolved closure no longer writes the
    fixed 0.65 to node:substrate.harness_closure -- with the PE-node flag ON."""
    import app.turn_referent_store as trs
    from app.worker import _PREDICTION_ERROR_NODE_FLAG

    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    worker = _make_worker()
    store = _RecordingStore()
    worker._substrate_graph_store = store
    monkeypatch.setattr(worker, "_write_prediction_error_node", _no_pe_write)
    persisted: list[str] = []
    monkeypatch.setattr(
        trs, "persist_turn_referent", lambda closure, engine: persisted.append(closure.correlation_id)
    )

    worker.handle_post_turn_closure(_sample_closure(correlation_id="corr-a", surprise_unresolved=True))
    worker.handle_post_turn_closure(_sample_closure(correlation_id="corr-b", surprise_unresolved=True))

    assert store.nodes == {}
    # The turn referent (a different artifact, read by reverie) is still persisted.
    assert persisted == ["corr-a", "corr-b"]


def test_resolved_closure_writes_nothing(monkeypatch) -> None:
    import app.turn_referent_store as trs

    worker = _make_worker()
    monkeypatch.setattr(worker, "_write_prediction_error_node", _no_pe_write)
    monkeypatch.setattr(
        trs, "persist_turn_referent",
        lambda closure, engine: (_ for _ in ()).throw(AssertionError("resolved: no referent")),
    )
    worker.handle_post_turn_closure(_sample_closure(surprise_unresolved=False))


def test_harness_closure_constant_is_gone() -> None:
    import app.worker as worker_module

    assert not hasattr(worker_module, "_HARNESS_CLOSURE_UNRESOLVED_ERROR")
