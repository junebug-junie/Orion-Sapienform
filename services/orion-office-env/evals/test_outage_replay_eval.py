"""Replay a recorded serial session and assert what the publisher would have said.

Covers the failure modes the metric gate cares about: noise lines, a frame with the
SHT40 absent, a long silence (USB unplugged), then recovery. The series must contain
real readings only where a fresh frame existed -- never a repeated or zero value.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.sample import build_sample
from app.serial_reader import parse_line

T0 = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "outage_replay.ndjson"
STALE_AFTER = 10.0


def _publish_series(ticks_s: list[float]) -> list:
    events = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    out = []
    for tick in ticks_s:
        frame = received = None
        for ev in events:
            if ev["at_s"] > tick:
                break
            parsed = parse_line(ev["line"])
            if parsed is not None:
                frame, received = parsed, T0 + timedelta(seconds=ev["at_s"])
        out.append(
            build_sample(
                frame=frame, received_at=received, now=T0 + timedelta(seconds=tick),
                stale_after_sec=STALE_AFTER, node="pan-1", sensor_id="sht40-office",
            )
        )
    return out


def test_publish_series_through_noise_gap_and_recovery() -> None:
    s = _publish_series([2.5, 3.5, 20.0, 49.0, 51.0])
    # 2.5s: fresh (frame at t=2)
    assert s[0].state.stale is False and s[0].measurements.office_temp_c == 21.5
    # 3.5s: newest frame (seq 3) has NO environment -> stale, no readings, not a repeat of 21.5
    assert s[1].state.stale is True and s[1].measurements.office_temp_c is None
    # 20s and 49s: long silence -> stale, never carries the old value forward
    assert all(x.state.stale and x.measurements.office_temp_c is None for x in s[2:4])
    # 51s: recovered
    assert s[4].state.stale is False and s[4].measurements.office_temp_c == 21.9


def test_no_sample_in_the_series_reports_a_zero_reading() -> None:
    for sample in _publish_series([0.5, 2.5, 3.5, 20.0, 51.0]):
        for value in sample.measurements.model_dump(exclude_none=True).values():
            assert value != 0.0
