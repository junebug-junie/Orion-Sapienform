from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.energy.espi import EspiError, parse_espi

FIXTURE = Path(__file__).parent / "fixtures" / "espi_two_flows.xml"
RETRIEVED = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def test_parses_forward_flow_hourly_kwh() -> None:
    rows = parse_espi(FIXTURE.read_bytes(), retrieved_at=RETRIEVED, source="file_drop", source_file="x.xml")
    assert [r.energy_kwh for r in rows] == pytest.approx([1.234, 0.5, 2.0])
    assert rows[0].interval_start == datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
    assert rows[0].interval_end == datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)
    assert {r.usage_point_id for r in rows} == {"UP123"}
    assert all(r.source == "file_drop" and r.retrieved_at == RETRIEVED for r in rows)
    assert rows[0].source_file == "x.xml"


def test_reverse_flow_block_is_skipped() -> None:
    rows = parse_espi(FIXTURE.read_bytes(), retrieved_at=RETRIEVED, source="file_drop")
    assert 0.999 not in [r.energy_kwh for r in rows]


def test_power_of_ten_multiplier_applied() -> None:
    # First occurrence is the forward-flow ReadingType (RT-FWD precedes RT-REV).
    xml = FIXTURE.read_bytes().replace(
        b"<espi:powerOfTenMultiplier>0<", b"<espi:powerOfTenMultiplier>-1<", 1
    )
    rows = parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
    assert rows[0].energy_kwh == pytest.approx(0.1234)


def test_unsupported_uom_raises() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:uom>72</espi:uom>", b"<espi:uom>38</espi:uom>")
    with pytest.raises(EspiError, match="uom"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_garbage_raises_espi_error() -> None:
    with pytest.raises(EspiError):
        parse_espi(b"<not-xml", retrieved_at=RETRIEVED, source="file_drop")


def test_feed_without_readings_raises() -> None:
    empty = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with pytest.raises(EspiError, match="no forward-flow"):
        parse_espi(empty, retrieved_at=RETRIEVED, source="file_drop")
