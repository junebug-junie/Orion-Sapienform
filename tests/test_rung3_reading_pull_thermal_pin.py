"""Pin the world_pulse_read pull gate's thermal decision table (body economy Patch B).

Hub's reading pull refuses on `hot` and allows on `unknown` (a dead sensor is never hot).
The line (32.0 C, re-arm 30.5 C, 300 s staleness) is owned by orion/autonomy/thermal_gate.py.
The expected rows are written out by hand, so a moved line or a gate that stops reading
the owner fails here.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.world_pulse_read import pull as reading_pull  # noqa: E402


def _pipeline_cls():
    # Same import path tests/test_rung3_reading_pull.py uses for Hub's `scripts` package.
    sys.path.insert(0, str(REPO / "services" / "orion-hub"))
    return importlib.import_module("scripts.world_pulse_read_pipeline").WorldPulseReadPipeline

_ERR = object()

# reader output (or _ERR = reader raises) -> (state, allows_gpu_work, pull gate reason)
TABLE = [
    (None, ("unknown", True, None)),          # no reader output: allowed
    ((31.9, 1.0), ("elevated", True, None)),
    ((32.0, 1.0), ("hot", False, "thermal_hot")),
    ((31.0, 1.0), ("hot", False, "thermal_hot")),
    (_ERR, ("unknown", True, None)),          # dead sensor: allowed, held state stays hot
    ((30.6, 1.0), ("hot", False, "thermal_hot")),
    ((30.5, 1.0), ("elevated", True, None)),
    ((28.0, 1.0), ("normal", True, None)),
    ((34.0, 1.0), ("hot", False, "thermal_hot")),
    ((35.0, 301.0), ("unknown", True, None)), # stale: allowed
    ((30.0, 1.0), ("elevated", True, None)),
]


def _reader(row):
    def read():
        if row is _ERR:
            raise OSError("sensor down")
        return row if row is not None else (None, None)
    return read


def test_decision_table_is_pinned():
    p = object.__new__(_pipeline_cls())
    got = []
    for row, _ in TABLE:
        p._pull_thermal_reader = _reader(row)
        v = asyncio.run(p._pull_thermal())
        gate = reading_pull.pull_block_reason(reading_pull.PullGateInputs(
            enabled=True, wallet_reason=None, wallet_retry_sec=None, pulled_today=0, daily_cap=8,
            arousal_level=None, thermal=v, now=datetime(2026, 10, 27, 12, tzinfo=timezone.utc)))
        got.append((v.state, v.allows_gpu_work, gate.refusal))
        if gate.refusal:
            assert gate.retry_after_sec == 300.0
    assert got == [want for _, want in TABLE]


def test_no_reader_means_no_thermal_verdict():
    p = object.__new__(_pipeline_cls())
    p._pull_thermal_reader = None
    assert asyncio.run(p._pull_thermal()) is None
