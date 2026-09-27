from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.energy.espi import EspiError, parse_espi

FIXTURE = Path(__file__).parent / "fixtures" / "espi_two_flows.xml"
GBA_LINK_ORDER = Path(__file__).parent / "fixtures" / "espi_gba_link_order.xml"
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
    assert len(rows) == 3
    assert [r.interval_start for r in rows] == [
        datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 10, 20, 0, tzinfo=timezone.utc),
    ]


def test_missing_flow_direction_raises() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:flowDirection>19</espi:flowDirection>", b"", 1)
    with pytest.raises(EspiError, match="flowDirection"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_missing_forward_uom_raises() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:uom>72</espi:uom>", b"", 1)
    with pytest.raises(EspiError, match="uom"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_non_integer_value_raises_espi_error() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:value>1234</espi:value>", b"<espi:value>5.5</espi:value>", 1)
    with pytest.raises(EspiError):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_negative_value_raises_espi_error() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:value>1234</espi:value>", b"<espi:value>-100</espi:value>", 1)
    with pytest.raises(EspiError):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_block_without_usage_point_raises() -> None:
    xml = FIXTURE.read_bytes().replace(
        b"/UsagePoint/UP123/MeterReading/MR1/IntervalBlock/IB1",
        b"/MeterReading/MR1/IntervalBlock/IB1",
        1,
    )
    with pytest.raises(EspiError, match="UsagePoint"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


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


def test_non_integer_reading_type_uom_raises_espi_error() -> None:
    xml = FIXTURE.read_bytes().replace(b"<espi:uom>72</espi:uom>", b"<espi:uom>Wh</espi:uom>", 1)
    with pytest.raises(EspiError, match="invalid ReadingType"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_out_of_range_pow10_raises_espi_error() -> None:
    xml = FIXTURE.read_bytes().replace(
        b"<espi:powerOfTenMultiplier>0<", b"<espi:powerOfTenMultiplier>400<", 1
    )
    with pytest.raises(EspiError, match="powerOfTenMultiplier"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")


def test_gba_link_order_parses_same_forward_rows() -> None:
    rows = parse_espi(GBA_LINK_ORDER.read_bytes(), retrieved_at=RETRIEVED, source="file_drop")
    expected = parse_espi(FIXTURE.read_bytes(), retrieved_at=RETRIEVED, source="file_drop")
    assert [r.energy_kwh for r in rows] == pytest.approx([r.energy_kwh for r in expected])
    assert [r.interval_start for r in rows] == [r.interval_start for r in expected]


def test_single_reading_type_used_when_no_related_link() -> None:
    xml = FIXTURE.read_bytes()
    xml = xml.replace(
        b'<link rel="related" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-FWD"/>',
        b"",
        1,
    )
    for marker in (
        b"  <entry>\n    <id>urn:uuid:mr2</id>",
        b"  <entry>\n    <id>urn:uuid:rtr</id>",
        b"  <entry>\n    <id>urn:uuid:ib2</id>",
    ):
        start = xml.index(marker)
        end = xml.index(b"  </entry>", start) + len(b"  </entry>\n")
        xml = xml[:start] + xml[end:]
    rows = parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
    assert len(rows) == 3
    assert rows[0].energy_kwh == pytest.approx(1.234)


def test_unresolved_related_reading_type_raises() -> None:
    xml = FIXTURE.read_bytes().replace(
        b'<link rel="related" href="https://csapps.example/espi/1_1/resource/ReadingType/RT-FWD"/>',
        b'<link rel="related" href="https://csapps.example/espi/1_1/resource/ReadingType/MISSING"/>',
        1,
    )
    with pytest.raises(EspiError, match="MISSING"):
        parse_espi(xml, retrieved_at=RETRIEVED, source="file_drop")
