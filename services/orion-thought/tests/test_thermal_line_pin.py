"""Pin the visual-chain thermal gate's decision table (body economy Patch B).

The hot line (32.0 C, re-arm 30.5 C) has one owner, orion/autonomy/thermal_gate.py.
This table is written out by hand, so it fails if the owner's line moves or if this
consumer stops reading it. It held before the thought-side copies
(ORION_THERMAL_HOT_C / ORION_THERMAL_HOT_REARM_C) were deleted, and it holds after.
"""

from __future__ import annotations

import asyncio

import pytest

from app import visual_chain

# (temp_c, age_sec) -> (state, allows_gpu_work). Starts from "normal"; one sequence,
# because the hysteresis is the point. `unknown` never overwrites the held state.
TABLE = [
    ((None, None), ("unknown", True)),
    ((31.9, 1.0), ("elevated", True)),
    ((32.0, 1.0), ("hot", False)),      # trip is >=
    ((31.0, 1.0), ("hot", False)),      # between re-arm and trip: still hot
    ((None, None), ("unknown", True)),  # no reading: allowed, held state stays hot
    ((30.6, 1.0), ("hot", False)),
    ((30.5, 1.0), ("elevated", True)),  # re-arm is >, so 30.5 releases hot
    ((28.1, 1.0), ("elevated", True)),
    ((28.0, 1.0), ("normal", True)),
    ((34.0, 1.0), ("hot", False)),
    ((35.0, 301.0), ("unknown", True)), # older than 300 s is no reading
    ((30.0, 1.0), ("elevated", True)),
]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(visual_chain, "_thermal_state", "normal", raising=False)
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", True)


def test_decision_table_is_pinned():
    got = []
    for (temp, age), _ in TABLE:
        v = visual_chain._apply_thermal_reading(temp, age)
        got.append((v.state, v.allows_gpu_work))
    assert got == [want for _, want in TABLE]


def test_snapshot_reports_the_owner_line(monkeypatch):
    monkeypatch.setattr(visual_chain, "read_cabinet_temp_c", lambda: (32.0, 1.0))
    snap = asyncio.run(visual_chain.thermal_gate_snapshot())
    assert (snap["hot_c"], snap["hot_rearm_c"]) == (32.0, 30.5)
    assert snap["state"] == "hot" and snap["allows_gpu_work"] is False


def test_thought_owns_no_copy_of_the_line():
    assert not hasattr(visual_chain.settings, "thermal_hot_c")
    assert not hasattr(visual_chain.settings, "thermal_hot_rearm_c")
