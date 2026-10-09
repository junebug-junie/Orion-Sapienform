from datetime import datetime, timezone

from app.main import build_cooling_sample
from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1


def test_build_sample_omits_zero_fill_when_watts_missing():
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    sample = build_cooling_sample(
        node_id=2,
        controller_ready=True,
        device_online=True,
        watts=None,
        volts=None,
        amps=None,
        switch_on=None,
        device_path="/dev/zwave",
        product="Shelly Wave Plug",
        now=now,
        last_fresh_at=now,
        stale_after_sec=120.0,
    )
    assert isinstance(sample, HomeCoolingSampleV1)
    assert sample.measurements.cooling_watts is None
    # No wattage is never reported as a healthy live reading.
    assert sample.state.stale is True
    assert sample.role == "cabinet_cooling"
