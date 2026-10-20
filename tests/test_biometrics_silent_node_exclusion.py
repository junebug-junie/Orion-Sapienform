"""A node in standby (stopped reporting) must not dilute node:substrate.biometrics.

hecate is powered down to BMC-only standby when not needed. Its projection entry froze
on 2026-10-15T06:31Z with 9 pressure_hints, and every tick averaged 9 zero deltas into
`biometrics_prediction_error()`. Silence is judged against each node's own report
interval, not a fixed number of seconds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1, NodeBiometricsStateV1
from orion.schemas.prediction_error_definitions import prediction_error_definition_version
from orion.substrate.biometrics_loop.report_cadence import (
    MIN_OWN_INTERVALS,
    NodeReportCadence,
    silent_nodes,
)
from orion.substrate.prediction_error import _THRESHOLD, biometrics_prediction_error

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
CADENCE = 31.0
HECATE_LAST_SEEN = datetime(2026, 10, 15, 6, 31, 3, tzinfo=timezone.utc)
# hecate's real frozen hints, substrate_node_biometrics_projection 2026-10-20.
HECATE_FROZEN = {
    "gpu": 0.0,
    "strain": 0.08708738904410249,
    "stability": 0.9996429633206283,
    "fan_pressure": 0.0,
    "disk_pressure": 0.0003219109669554181,
    "power_pressure": 0.5478641604026335,
    "memory_pressure": 0.06079577840541474,
    "thermal_pressure": 0.0,
    "disk_capacity_pressure": 0.17,
}


def _proj(nodes: dict[str, tuple[datetime, dict]]) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=T0,
        nodes={
            n: NodeBiometricsStateV1(node_id=n, last_seen_at=ts, pressure_hints=dict(h))
            for n, (ts, h) in nodes.items()
        },
    )


def _learned(nodes: list[str], reports: int = 6) -> NodeReportCadence:
    cadence = NodeReportCadence()
    for i in range(reports):
        ts = T0 - timedelta(seconds=CADENCE * (reports - 1 - i))
        cadence.observe(_proj({n: (ts, {}) for n in nodes}))
    return cadence


def test_cadence_is_learned_per_node_and_needs_enough_samples() -> None:
    cadence = NodeReportCadence()
    cadence.observe(_proj({"athena": (T0, {})}))
    cadence.observe(_proj({"athena": (T0 + timedelta(seconds=31), {})}))
    assert cadence.cadence_sec("athena") is None  # one interval is not enough
    for k in range(2, 2 + MIN_OWN_INTERVALS):
        cadence.observe(_proj({"athena": (T0 + timedelta(seconds=31 * k), {})}))
    assert cadence.cadence_sec("athena") == pytest.approx(31.0)


def test_one_long_wake_gap_does_not_become_the_cadence() -> None:
    cadence = _learned(["hecate"], reports=8)
    last = T0
    cadence.observe(_proj({"hecate": (last + timedelta(days=5), {})}))
    assert cadence.cadence_sec("hecate") == pytest.approx(CADENCE)


def test_standby_node_is_silent_and_reporting_nodes_are_not() -> None:
    cadence = _learned(["athena", "circe"])
    proj = _proj(
        {
            "athena": (T0, {}),
            "circe": (T0 - timedelta(seconds=40), {}),  # one late report: not silent
            "hecate": (HECATE_LAST_SEEN, HECATE_FROZEN),
        }
    )
    # hecate has no learned interval in this process: judged by the fleet's (31 s).
    assert silent_nodes(proj, now=T0, cadence=cadence) == frozenset({"hecate"})


def test_nothing_learned_excludes_nothing() -> None:
    proj = _proj({"hecate": (HECATE_LAST_SEEN, HECATE_FROZEN)})
    assert silent_nodes(proj, now=T0, cadence=NodeReportCadence()) == frozenset()


def test_threshold_is_relative_to_the_node_own_cadence() -> None:
    # A node that reports every 10 min is not silent after 30 min; a 31 s node is.
    cadence = NodeReportCadence()
    for i in range(6):
        cadence.observe(_proj({"slow": (T0 + timedelta(minutes=10 * i), {}),
                               "fast": (T0 + timedelta(seconds=31 * i), {})}))
    end = T0 + timedelta(minutes=50)
    proj = _proj({"slow": (end - timedelta(minutes=30), {}), "fast": (end - timedelta(minutes=30), {})})
    assert silent_nodes(proj, now=end, cadence=cadence) == frozenset({"fast"})


def test_frozen_standby_keys_no_longer_dilute_the_mean() -> None:
    prev = _proj(
        {
            "athena": (T0 - timedelta(seconds=31), {"strain": 0.20, "memory_pressure": 0.30}),
            "hecate": (HECATE_LAST_SEEN, HECATE_FROZEN),
        }
    )
    curr = _proj(
        {
            "athena": (T0, {"strain": 0.26, "memory_pressure": 0.30}),
            "hecate": (HECATE_LAST_SEEN, HECATE_FROZEN),
        }
    )
    diluted = biometrics_prediction_error(prev, curr)
    clean = biometrics_prediction_error(prev, curr, exclude_nodes={"hecate"})
    # athena alone: mean(|0.06|, 0) = 0.03; with hecate's 9 frozen zeros: 0.06 / 11.
    assert clean == pytest.approx(0.03 / _THRESHOLD)
    assert diluted == pytest.approx((0.06 / 11) / _THRESHOLD)
    assert clean > diluted


def test_hecate_counts_normally_when_up() -> None:
    cadence = _learned(["athena", "hecate"])
    prev = _proj({"athena": (T0 - timedelta(seconds=31), {"strain": 0.2}),
                  "hecate": (T0 - timedelta(seconds=31), {"strain": 0.1})})
    curr = _proj({"athena": (T0, {"strain": 0.2}), "hecate": (T0, {"strain": 0.4})})
    silent = silent_nodes(curr, now=T0, cadence=cadence)
    assert silent == frozenset()
    assert biometrics_prediction_error(prev, curr, exclude_nodes=silent) == pytest.approx(
        min(1.0, (0.3 / 2) / _THRESHOLD)
    )


def test_definition_version_moved() -> None:
    assert prediction_error_definition_version("node_biometrics") == 2
