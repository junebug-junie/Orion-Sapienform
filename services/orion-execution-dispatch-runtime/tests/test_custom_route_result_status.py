"""Custom RPC routes use the same result-status contract as cortex dispatches.

The reading pull is the last custom RPC route: retro leisure (PR 1) and AI Town (PR 3) left dispatch on
2026-11-05 (ladder retirement)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.result_status import status_for_stored_settlement
from app.store import ExecutionDispatchRuntimeStore
from app.worker import ExecutionDispatchRuntimeWorker
from orion.schemas.world_pulse_read_pull import WorldPulseReadPullResultV1

NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


class _Store:
    def __init__(self, existing: dict | None = None) -> None:
        self.saved: dict | None = None
        self.existing = existing
        self.canonicalized: dict | None = None

    def load_dispatch_result_by_dispatch_id(self, dispatch_id: str) -> dict | None:
        return self.existing

    def save_dispatch_result(self, **kwargs) -> None:
        self.saved = kwargs

    def canonicalize_legacy_dispatch_result_status(self, **kwargs) -> bool:
        self.canonicalized = kwargs
        return True


class _Candidate:
    dispatch_id = "dispatch:test"
    source_proposal_id = "proposal:test"
    source_decision_id = "decision:test"
    dispatch_kind = "express"
    target_id = "target:test"

    def model_copy(self, *, update: dict):
        return SimpleNamespace(**update)


def _route_worker(route: str, state: str):
    store = _Store()
    recorded: list[str] = []

    if route == "reading_pull":
        worker = object.__new__(ExecutionDispatchRuntimeWorker)
        worker._settings = SimpleNamespace(orion_dispatch_reading_pull_enabled=True)
        result = WorldPulseReadPullResultV1(
            op="act", dispatch_id=_Candidate.dispatch_id, arm="treated", state=state,
            object_ref="look:test" if state == "started" else None,
            refusal="daily_cap" if state == "refused" else None,
        )
        worker._read_pull_rpc = AsyncMock(return_value=result)
        method = worker._send_read_pull
    else:
        raise AssertionError(route)

    worker._store = store
    worker._last_proven = {}
    worker._record_dispatch_status = lambda status, live=True: recorded.append(status)
    asyncio.run(method(None, SimpleNamespace(frame_id="frame:test"), _Candidate(), now=NOW))
    assert store.saved is not None
    return store.saved["status"], recorded


@pytest.mark.parametrize("route", ["reading_pull"])
def test_started_custom_route_is_productive_for_tripwire(route: str) -> None:
    stored, recorded = _route_worker(route, "started")
    assert stored == "success"
    assert recorded == ["success"]


@pytest.mark.parametrize("route", ["reading_pull"])
def test_refused_custom_route_is_real_but_unproductive(route: str) -> None:
    stored, recorded = _route_worker(route, "refused")
    assert stored == "empty"
    assert recorded == ["empty"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [("started", "success"), ("refused", "empty"), ("rpc_error", "failed")],
)
def test_legacy_succeeded_replay_recovers_preserved_settlement(state: str, expected: str) -> None:
    row = {"status": "succeeded", "result_json": {"settlement": {"state": state}}}
    assert status_for_stored_settlement(row) == expected


def _replay_route(route: str, state: str):
    existing = {
        "result_id": "result:dispatch:test",
        "status": "succeeded",
        "result_json": {"settlement": {"state": state}},
    }
    store = _Store(existing)
    recorded: list[str] = []
    if route == "reading_pull":
        worker = object.__new__(ExecutionDispatchRuntimeWorker)
        worker._settings = SimpleNamespace(orion_dispatch_reading_pull_enabled=True)
        method = worker._send_read_pull
    else:
        raise AssertionError(route)
    worker._store = store
    worker._record_dispatch_status = lambda status, live=True: recorded.append(status)
    asyncio.run(method(None, SimpleNamespace(frame_id="frame:test"), _Candidate(), now=NOW))
    return store, recorded


@pytest.mark.parametrize("route", ["reading_pull"])
@pytest.mark.parametrize(("state", "expected"), [("started", "success"), ("refused", "empty")])
def test_legacy_replay_repairs_durable_status_before_feedback(
    route: str, state: str, expected: str
) -> None:
    store, recorded = _replay_route(route, state)
    assert store.canonicalized == {
        "result_id": "result:dispatch:test",
        "status": expected,
    }
    assert recorded == [expected]


def test_durable_repair_only_overwrites_legacy_succeeded_status() -> None:
    store = object.__new__(ExecutionDispatchRuntimeStore)
    conn = MagicMock()
    conn.execute.return_value.rowcount = 1
    store._engine = MagicMock()
    store._engine.begin.return_value.__enter__.return_value = conn

    assert store.canonicalize_legacy_dispatch_result_status(
        result_id="result:dispatch:test", status="success"
    ) is True

    statement, params = conn.execute.call_args.args
    assert "AND status = 'succeeded'" in str(statement)
    assert params == {"result_id": "result:dispatch:test", "status": "success"}
