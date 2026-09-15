"""SNMP on-battery interpretation must actually set on_battery for shutdown."""

from __future__ import annotations

import pytest

from app.ups_snmp_client import SNMPUPSClient


@pytest.mark.asyncio
async def test_output_status_onbatt_sets_on_battery_even_if_line_voltage_high(monkeypatch) -> None:
    client = SNMPUPSClient(host="192.168.1.41", community="public-2")

    async def fake_fetch():
        # Residual / stale line voltage above the 80V floor must not mask ONBATT.
        return {
            "output_status": 3,
            "batt_capacity_pct": 90,
            "input_line_voltage": 200,
        }

    monkeypatch.setattr(client, "_fetch_raw", fake_fetch)
    status = await client.get_status()
    assert status.raw_status == "ONBATT"
    assert status.on_battery is True
    assert status.battery_charge_pct == 90.0


@pytest.mark.asyncio
async def test_output_status_online_clears_on_battery(monkeypatch) -> None:
    client = SNMPUPSClient(host="192.168.1.41", community="public-2")

    async def fake_fetch():
        return {
            "output_status": 2,
            "batt_capacity_pct": 100,
            "input_line_voltage": 247,
        }

    monkeypatch.setattr(client, "_fetch_raw", fake_fetch)
    status = await client.get_status()
    assert status.raw_status == "ONLINE"
    assert status.on_battery is False
