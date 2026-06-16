"""Regression: independent reducer poll loops must not starve transport on biometrics backlog."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
SQL_WRITER_TESTS = REPO_ROOT / "services/orion-sql-writer/tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))
if str(SQL_WRITER_TESTS) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_TESTS))

from grammar_integration_helpers import bus_transport_trace_batch

from app.reducer_health import clear_health_for_tests, record_error, record_tick
from app.worker import BiometricsSubstrateWorker, REDUCER_SPECS
from orion.schemas.grammar import GrammarEventV1


@pytest.fixture(autouse=True)
def _clear_health() -> None:
    clear_health_for_tests()


def test_transport_tick_advances_while_biometrics_tick_blocked() -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_transport_bus_reducer = True
    worker._settings.enable_execution_trajectory_reducer = True
    worker._settings.reducer_poison_max_retries = 99
    worker._settings.bus_stream_depth_critical = 100_000
    worker._store = MagicMock()
    worker._store.save_receipt = MagicMock()
    worker._store.save_quarantine = MagicMock()

    from orion.substrate.transport_loop.pipeline import empty_transport_projection

    now = datetime.now(timezone.utc)
    worker._store.load_transport_bus_projection.return_value = empty_transport_projection(now=now)

    transport_trace = bus_transport_trace_batch(trace_suffix="starve01", event_count=2)
    worker._store.fetch_transport_grammar_events.return_value = transport_trace
    worker._store.fetch_biometrics_grammar_events.side_effect = RuntimeError(
        "biometrics blocked on poison backlog"
    )

    with pytest.raises(RuntimeError, match="biometrics blocked"):
        worker._tick()

    last_id = worker._transport_tick()
    assert last_id == transport_trace[-1].event_id
    worker._store.fetch_transport_grammar_events.assert_called_once()


def test_start_spawns_independent_reducer_poll_tasks() -> None:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.orion_bus_enabled = False
    worker._settings.publish_accepted_pressure_grammar = False
    worker._settings.grammar_poll_interval_sec = 60.0
    worker._stop = asyncio.Event()
    worker._tasks = []
    worker._bus = None

    created: list[str] = []

    def capture_coro(coro, *, name=None):
        created.append(name or "unnamed")
        coro.close()
        return MagicMock(name=name)

    with patch("asyncio.create_task", side_effect=capture_coro):
        asyncio.run(worker.start())

    assert "biometrics-substrate-poll" in created
    assert "execution-substrate-poll" in created
    assert "transport-substrate-poll" in created
    assert len([n for n in created if n.endswith("-poll")]) == 3


def test_biometrics_backlog_does_not_block_transport_poll_iteration() -> None:
    """One transport poll-loop iteration must complete while biometrics is failing."""
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_transport_bus_reducer = True
    worker._settings.grammar_poll_interval_sec = 0.01
    worker._settings.bus_stream_depth_critical = 100_000
    worker._stop = asyncio.Event()
    worker._store = MagicMock()
    worker._store.save_receipt = MagicMock()
    worker._store.save_quarantine = MagicMock()
    worker._store.grammar_event_created_at.return_value = datetime.now(timezone.utc)

    transport_trace = bus_transport_trace_batch(trace_suffix="starve02", event_count=2)
    worker._store.fetch_transport_grammar_events.return_value = transport_trace

    spec = REDUCER_SPECS[2]
    record_tick(spec.reducer_key, cursor_name=spec.cursor_name, enabled=True)

    async def failing_biometrics_tick():
        record_error(
            "biometrics",
            cursor_name=REDUCER_SPECS[0].cursor_name,
            enabled=True,
            event_id="gev_bio_bad",
            reason="slow poison backlog",
        )
        raise TimeoutError("biometrics stuck")

    worker._transport_tick = MagicMock(return_value=transport_trace[0].event_id)
    worker._advance_cursor = MagicMock()

    async def run_transport_once():
        enabled = spec.enabled(worker._settings)
        record_tick(spec.reducer_key, cursor_name=spec.cursor_name, enabled=enabled)
        last_event_id = await asyncio.to_thread(worker._transport_tick)
        if last_event_id:
            await asyncio.to_thread(
                worker._advance_cursor,
                spec,
                last_event_id,
                worker._store.advance_transport_cursor,
            )

    with pytest.raises(TimeoutError, match="biometrics stuck"):
        asyncio.run(failing_biometrics_tick())
    asyncio.run(run_transport_once())
    worker._transport_tick.assert_called_once()
    worker._advance_cursor.assert_called_once()
