"""A node going dark must reach Juniper, not just the projection.

The last hop of the 2026-08-29 circe work. The substrate detects absence, but during
a ~45 minute outage of the entire local GPU fleet `notify_requests` recorded nothing
at all -- the signal existed and stopped at the database.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "orion-substrate-runtime"))

from app.health_monitor import _node_availability_checks  # noqa: E402

NOW = datetime(2026, 8, 29, 0, 47, tzinfo=timezone.utc)


class _Settings:
    enable_biometrics_pressure_organ = True
    biometrics_node_stale_after_sec = 180


class _Store:
    def __init__(self, projection):
        self._p = projection

    def load_active_pressure(self, _pid):
        return self._p


def _proj(**nodes):
    return ActiveNodePressureProjectionV1(
        projection_id="p", generated_at=NOW, nodes=dict(nodes)
    )


def _node(node_id, *, pressures, impacts=(), status="online"):
    return ActiveNodePressureStateV1(
        node_id=node_id,
        availability_status=status,
        last_updated_at=NOW,
        active_pressures=list(pressures),
        capability_impacts=list(impacts),
    )


def _by_key(checks):
    return {c.key: c for c in checks}


def test_absent_node_produces_an_unhealthy_check_naming_its_capabilities() -> None:
    checks = _by_key(
        _node_availability_checks(
            _Store(
                _proj(
                    circe=_node(
                        "circe",
                        pressures=["strain", "availability"],
                        impacts=["capability:local_llm_heavy", "capability:local_llm_quick"],
                    )
                )
            ),
            _Settings(),
        )
    )
    check = checks["node_availability:circe"]
    assert check.healthy is False
    assert check.severity == "critical"
    assert "circe" in check.message
    # The operator needs to know what was LOST, not just that a box is quiet.
    assert "local_llm_heavy" in check.message and "local_llm_quick" in check.message
    assert "180" in check.message


def test_reporting_node_is_healthy() -> None:
    checks = _by_key(
        _node_availability_checks(
            _Store(_proj(circe=_node("circe", pressures=["strain"]))), _Settings()
        )
    )
    assert checks["node_availability:circe"].healthy is True


def test_suppressed_node_never_pages() -> None:
    """A decommissioned node is expected to be silent forever. atlas must never page
    again -- it is physically inside circe now."""
    checks = _by_key(
        _node_availability_checks(
            _Store(
                _proj(
                    atlas=_node("atlas", pressures=["availability"], status="suppressed"),
                    circe=_node("circe", pressures=["strain"]),
                )
            ),
            _Settings(),
        )
    )
    assert "node_availability:atlas" not in checks
    assert "node_availability:circe" in checks


def test_disabled_organ_produces_no_checks() -> None:
    class Off(_Settings):
        enable_biometrics_pressure_organ = False

    assert _node_availability_checks(_Store(_proj()), Off()) == []


def test_store_failure_does_not_raise() -> None:
    """This runs inside the health tick; a projection read failure must not take the
    whole health monitor down."""

    class Boom:
        def load_active_pressure(self, _pid):
            raise RuntimeError("db down")

    assert _node_availability_checks(Boom(), _Settings()) == []


def test_check_is_edge_triggered_by_reusing_healthmonitor() -> None:
    """Pins the reuse. These checks are only sane because HealthMonitor already
    fires on transition, debounces, survives restarts via _has_open_alert, retries,
    and emits a recovery note. A parallel notifier would have to re-derive all of it,
    and a per-tick alert on a 45-minute outage would be ~90 pages."""
    src = (
        REPO_ROOT / "services" / "orion-substrate-runtime" / "app" / "health_monitor.py"
    ).read_text()
    assert "checks.extend(_node_availability_checks(store, settings))" in src
    assert "_has_open_alert" in src
    assert "recovered=True" in src
