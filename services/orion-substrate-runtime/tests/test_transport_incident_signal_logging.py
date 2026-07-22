"""Unit tests for _log_transport_incident_signals (docs/superpowers/specs/
2026-07-22-transport-bus-signal-quality-measurement-design.md item 1).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SUBSTRATE_ROOT.parents[1]
for _p in (str(_REPO_ROOT), str(_SUBSTRATE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.worker import _log_transport_incident_signals  # noqa: E402
from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1  # noqa: E402


def _bus(**overrides) -> TransportBusStateV1:
    defaults = dict(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="w1",
        source_trace_id="bus.transport:athena:w1",
    )
    defaults.update(overrides)
    return TransportBusStateV1(**defaults)


def _projection(**buses: TransportBusStateV1) -> TransportBusProjectionV1:
    from datetime import datetime, timezone

    return TransportBusProjectionV1(updated_at=datetime.now(timezone.utc), buses=buses)


def test_logs_nothing_when_all_signals_quiet(caplog) -> None:
    projection = _projection(**{"bus:athena": _bus(max_stream_depth=91, stream_depth_pressure=0.00091)})
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        _log_transport_incident_signals(projection)
    assert "transport_incident_signal" not in caplog.text


def test_logs_when_backpressure_nonzero(caplog) -> None:
    projection = _projection(**{"bus:athena": _bus(backpressure=0.5, backpressure_count=3)})
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        _log_transport_incident_signals(projection)
    assert "transport_incident_signal" in caplog.text
    assert "bus:athena" in caplog.text
    assert "backpressure" in caplog.text


def test_does_not_fire_on_stream_depth_pressure_alone() -> None:
    """stream_depth_pressure is structurally nonzero on almost every real tick
    (any nonzero queue depth divides through DEFAULT_STREAM_DEPTH_CRITICAL to a
    small positive number) -- it must not, by itself, count as an incident."""
    projection = _projection(
        **{"bus:athena": _bus(max_stream_depth=91, stream_depth_pressure=0.00091)}
    )
    logger = logging.getLogger("orion.substrate.runtime")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    try:
        _log_transport_incident_signals(projection)
    finally:
        logger.removeHandler(handler)
    assert not any("transport_incident_signal" in r for r in records)


def test_multiple_buses_each_checked_independently(caplog) -> None:
    projection = _projection(
        **{
            "bus:athena": _bus(node_id="athena", target_id="bus:athena"),
            "bus:atlas": _bus(node_id="atlas", target_id="bus:atlas", observer_failure_pressure=1.0, observer_failure_count=1),
        }
    )
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        _log_transport_incident_signals(projection)
    assert "bus:atlas" in caplog.text
    assert "bus:athena" not in caplog.text


def test_never_raises_on_malformed_input() -> None:
    class _Bad:
        buses = None

    _log_transport_incident_signals(_Bad())  # must not raise
