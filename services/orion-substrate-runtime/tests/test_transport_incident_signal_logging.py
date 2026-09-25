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
    projection = _projection(**{"bus:athena": _bus(redis_ping_ok=True)})
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        _log_transport_incident_signals(projection)
    assert "transport_incident_signal" not in caplog.text


def test_logs_when_reliability_pressure_nonzero(caplog) -> None:
    projection = _projection(**{"bus:athena": _bus(redis_ping_ok=False, reliability_pressure=1.0)})
    with caplog.at_level(logging.INFO, logger="orion.substrate.runtime"):
        _log_transport_incident_signals(projection)
    assert "transport_incident_signal" in caplog.text
    assert "bus:athena" in caplog.text
    assert "reliability_pressure" in caplog.text


def test_retired_depth_family_is_not_an_incident_field() -> None:
    """backpressure / max_stream_depth were retired 2026-09-25
    (fix/bus-observer-scope); the incident set must not name them."""
    from app.worker import _TRANSPORT_INCIDENT_FIELDS

    assert "backpressure" not in _TRANSPORT_INCIDENT_FIELDS
    assert "stream_depth_pressure" not in _TRANSPORT_INCIDENT_FIELDS


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
