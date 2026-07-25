"""Regression coverage for iLO/BMC RedFish telemetry (app/ilo.py).

Added alongside real out-of-band hardware telemetry piggybacked onto
orion-biometrics' existing bus-native SystemHealthV1 heartbeat (see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md).
`fetch_ilo_snapshot()` was live-verified against athena's real HPE iLO
2026-07-25 (42 real temp sensors, 6 fans, real wattage) -- these tests mock
`requests.Session` instead of hitting real hardware, so they exercise the
parsing/skip logic deterministically in CI.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from app.ilo import IloPoller, IloSnapshot, fetch_ilo_snapshot


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], *, ok: bool = True, status: int = 200):
        self._payload = payload
        self.ok = ok
        self.status_code = status

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_session(monkeypatch: pytest.MonkeyPatch, responses: Dict[str, _FakeResponse]) -> None:
    def fake_get(self, url: str, timeout: float = 8.0):  # noqa: ANN001
        for suffix, resp in responses.items():
            if url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("requests.Session.get", fake_get, raising=True)


def test_fetch_ilo_snapshot_not_configured_without_credentials() -> None:
    snap = fetch_ilo_snapshot("", "", "")
    assert snap.error == "not_configured"
    assert snap.thermal_c == {}
    assert snap.fan_pct == {}
    assert snap.power_watts is None


def test_fetch_ilo_snapshot_parses_real_shaped_redfish_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_session(
        monkeypatch,
        {
            "/redfish/v1/Chassis/": _FakeResponse({"Members": [{"@odata.id": "/redfish/v1/Chassis/1/"}]}),
            "/redfish/v1/Chassis/1/Thermal/": _FakeResponse(
                {
                    "Temperatures": [
                        {"Name": "01-Inlet Ambient", "ReadingCelsius": 24, "Status": {"State": "Enabled"}},
                        {"Name": "02-CPU 1", "ReadingCelsius": 40, "Status": {"State": "Enabled"}},
                        # Unpopulated slot -- must be skipped, not recorded as 0.
                        {"Name": "05-PMM 1-6", "ReadingCelsius": 0, "Status": {"State": "Absent"}},
                    ],
                    "Fans": [
                        {"Name": "Fan 1", "Reading": 29, "ReadingUnits": "Percent", "Status": {"State": "Enabled"}},
                    ],
                }
            ),
            "/redfish/v1/Chassis/1/Power/": _FakeResponse(
                {"PowerControl": [{"PowerConsumedWatts": 310}]}
            ),
        },
    )

    snap = fetch_ilo_snapshot("https://192.168.1.200", "Administrator", "hunter2", timeout_sec=8.0)

    assert snap.error is None
    assert snap.thermal_c == {"01-Inlet Ambient": 24.0, "02-CPU 1": 40.0}
    assert "05-PMM 1-6" not in snap.thermal_c
    assert snap.fan_pct == {"Fan 1": 29.0}
    assert snap.power_watts == 310.0
    assert snap.fetched_at > 0


def test_fetch_ilo_snapshot_no_chassis_members_is_a_soft_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_session(monkeypatch, {"/redfish/v1/Chassis/": _FakeResponse({"Members": []})})
    snap = fetch_ilo_snapshot("https://host", "user", "pw")
    assert snap.error == "no_chassis_members"


def test_fetch_ilo_snapshot_network_failure_is_soft_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_get(self, url: str, timeout: float = 8.0):  # noqa: ANN001
        raise ConnectionError("boom")

    monkeypatch.setattr("requests.Session.get", raise_get, raising=True)
    snap = fetch_ilo_snapshot("https://host", "user", "pw")
    assert snap.error is not None
    assert "boom" in snap.error


def test_ilo_poller_disabled_without_host_returns_empty_details() -> None:
    poller = IloPoller("", "", "")
    assert poller.enabled is False
    assert poller.details() == {}


def test_ilo_poller_details_reads_cached_snapshot_not_live_network() -> None:
    poller = IloPoller("https://host", "user", "pw", interval_sec=60.0)
    assert poller.enabled is True
    # Simulate a completed background poll without actually running the loop --
    # details() must be a pure cache read, per its own docstring contract (it is
    # called from the heartbeat's bounded per-tick hook and must never block).
    poller._snapshot = IloSnapshot(
        fetched_at=123.0,
        thermal_c={"CPU 1": 40.0},
        fan_pct={"Fan 1": 29.0},
        power_watts=310.0,
    )
    details = poller.details()
    assert details["ilo_thermal_c"] == {"CPU 1": 40.0}
    assert details["ilo_fan_pct"] == {"Fan 1": 29.0}
    assert details["ilo_power_watts"] == 310.0
    assert details["ilo_fetched_at"] == 123.0
    assert "ilo_error" not in details


def test_ilo_poller_details_surfaces_error_from_last_poll() -> None:
    poller = IloPoller("https://host", "user", "pw")
    poller._snapshot = IloSnapshot(fetched_at=1.0, error="connection timeout")
    details = poller.details()
    assert details["ilo_error"] == "connection timeout"


@pytest.mark.asyncio
async def test_ilo_poller_start_background_noop_when_disabled() -> None:
    poller = IloPoller("", "", "")
    await poller.start_background()
    assert poller._task is None
    await poller.stop()
