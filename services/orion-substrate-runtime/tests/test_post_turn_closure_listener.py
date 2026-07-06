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


def test_worker_handle_post_turn_closure_writes_prediction_error_when_unresolved(monkeypatch) -> None:
    from app.worker import _PREDICTION_ERROR_NODE_FLAG

    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    worker = _make_worker()
    calls: list[dict[str, object]] = []

    def _capture(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(worker, "_write_prediction_error_node", _capture)
    worker.handle_post_turn_closure(_sample_closure(surprise_unresolved=True))
    assert len(calls) == 1
    assert calls[0]["node_id"] == "harness_closure:c-1"
    assert calls[0]["reducer_key"] == "post_turn_closure"


def test_worker_handle_post_turn_closure_skips_when_flag_disabled(monkeypatch, caplog) -> None:
    import logging

    from app.worker import _PREDICTION_ERROR_NODE_FLAG

    caplog.set_level(logging.INFO, logger="orion.substrate.runtime")
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "false")
    worker = _make_worker()
    monkeypatch.setattr(
        worker,
        "_write_prediction_error_node",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not write")),
    )
    worker.handle_post_turn_closure(_sample_closure(surprise_unresolved=True))
    assert "post_turn_closure_prediction_error_skipped" in caplog.text


def test_worker_handle_post_turn_closure_skips_when_surprise_resolved(monkeypatch) -> None:
    worker = _make_worker()
    monkeypatch.setattr(worker, "_write_prediction_error_node", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not write")))
    worker.handle_post_turn_closure(_sample_closure(surprise_unresolved=False))
