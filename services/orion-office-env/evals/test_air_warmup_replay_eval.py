"""Replay a CCS811 session across a sensor restart and assert what the publisher says.

The CCS811's readings are invalid for 20 minutes after every start (datasheet run-in).
A Nano reset or a sensor re-init restarts that clock. The published series must carry
air readings only from conditioned stretches -- never the warm-up values, never a
zero -- while the SHT40 readings keep flowing the whole time.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.sample import build_sample
from app.serial_reader import parse_line

T0 = datetime(2026, 11, 3, 4, 0, 0, tzinfo=timezone.utc)
MIN = 60_000


def _line(seq: int, since_init_ms: int, eco2: int) -> bytes:
    return json.dumps(
        {
            "schema": "orion.office_env_frame.v1",
            "seq": seq,
            "uptime_ms": since_init_ms,
            "environment": {"temp_c": 21.5, "humidity_pct": 30.0},
            "air": {"eco2_ppm": eco2, "tvoc_ppb": 20, "since_init_ms": since_init_ms},
        }
    ).encode()


def test_air_series_skips_both_warmups_and_keeps_temperature() -> None:
    # Minute 0..30: first start. Minute 30: Nano reset, clock restarts. Minute 30..55.
    timeline = [(m, m * MIN) for m in range(0, 30, 5)] + [(m, (m - 30) * MIN) for m in range(30, 56, 5)]
    published = []
    for i, (minute, since_init) in enumerate(timeline):
        frame = parse_line(_line(i, since_init, 400 if since_init < 20 * MIN else 700))
        assert frame is not None
        now = T0 + timedelta(minutes=minute)
        published.append(
            (minute, build_sample(frame=frame, received_at=now, now=now, stale_after_sec=10.0,
                                  node="pan-1", sensor_id="sht40-office"))
        )

    with_air = [m for m, s in published if s.measurements.office_eco2_ppm is not None]
    warming = [m for m, s in published if s.state.air_warming]
    assert with_air == [20, 25, 50, 55]
    assert warming == [0, 5, 10, 15, 30, 35, 40, 45]
    assert all(s.measurements.office_eco2_ppm in (None, 700) for _, s in published)
    assert all(s.measurements.office_temp_c == 21.5 and not s.state.stale for _, s in published)
