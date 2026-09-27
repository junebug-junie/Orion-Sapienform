"""Green Button (NAESB ESPI) Atom feed -> whole-house usage intervals.

Linking follows the ESPI href convention: an IntervalBlock's self link is
``.../UsagePoint/<up>/MeterReading/<mr>/IntervalBlock/<ib>``; the MeterReading at
``.../UsagePoint/<up>/MeterReading/<mr>`` may carry several ``related`` links
(e.g. IntervalBlock collection and ReadingType). For each MeterReading, the
ReadingType href is the first ``related`` href whose self entry is a
ReadingType present in the feed; if none match, the first ``related`` href is
kept anyway. When that href is not a feed ReadingType (e.g. only an
IntervalBlock collection link), the feed is rejected. When a MeterReading
carries no ``related`` links and the feed has exactly one ReadingType, that
ReadingType is used.

Only forward flow (delivered to the house, flowDirection 1) is kept. Reverse flow
(19, e.g. solar export) is a different quantity and is skipped, not netted.

Input is the operator's own utility export or Orion's own scraper output, parsed
with the stdlib parser (expat does not fetch external entities).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import ValidationError

from orion.schemas.energy import EnergySource, EnergyUsageIntervalV1

ATOM = "{http://www.w3.org/2005/Atom}"
ESPI = "{http://naesb.org/espi}"
UOM_WATT_HOURS = 72
FLOW_FORWARD = 1
POW10_MIN = -9
POW10_MAX = 9

_USAGE_POINT = re.compile(r"/UsagePoint/([^/]+)")
_METER_READING = re.compile(r"^(.*/UsagePoint/[^/]+/MeterReading/[^/]+)")


class EspiError(ValueError):
    """The file is not a usable ESPI usage feed."""


def _links(entry: ET.Element) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for link in entry.findall(f"{ATOM}link"):
        out.setdefault(link.get("rel", ""), []).append(link.get("href", ""))
    return out


def _int(el: Optional[ET.Element], default: Optional[int] = None) -> Optional[int]:
    if el is None or el.text is None or not el.text.strip():
        return default
    return int(el.text.strip())


def _scale(pow10: int) -> float:
    if not POW10_MIN <= pow10 <= POW10_MAX:
        raise EspiError(f"powerOfTenMultiplier {pow10} out of range ({POW10_MIN}..{POW10_MAX})")
    return 10.0 ** pow10


def _reading_type(content: ET.Element, href: str) -> Optional[dict[str, Optional[int]]]:
    rt = content.find(f"{ESPI}ReadingType")
    if rt is None:
        return None
    try:
        uom = _int(rt.find(f"{ESPI}uom"))
        pow10 = _int(rt.find(f"{ESPI}powerOfTenMultiplier"), 0)
        flow = _int(rt.find(f"{ESPI}flowDirection"))
    except (ValueError, OverflowError) as exc:
        raise EspiError(f"invalid ReadingType at {href!r}") from exc
    return {"uom": uom, "pow10": pow10 if pow10 is not None else 0, "flow": flow}


def parse_espi(
    xml_bytes: bytes,
    *,
    retrieved_at: datetime,
    source: EnergySource,
    source_file: Optional[str] = None,
) -> list[EnergyUsageIntervalV1]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise EspiError(f"not parseable XML: {exc}") from exc

    reading_types: dict[str, dict[str, Optional[int]]] = {}
    meter_related: dict[str, list[str]] = {}
    blocks: list[tuple[str, ET.Element]] = []

    for entry in root.iter(f"{ATOM}entry"):
        content = entry.find(f"{ATOM}content")
        if content is None:
            continue
        links = _links(entry)
        self_href = (links.get("self") or [""])[0]
        rt = _reading_type(content, self_href)
        if rt is not None:
            reading_types[self_href] = rt
            continue
        if content.find(f"{ESPI}MeterReading") is not None:
            related = links.get("related") or []
            if related:
                meter_related[self_href] = related
            continue
        block = content.find(f"{ESPI}IntervalBlock")
        if block is not None:
            blocks.append((self_href, block))

    meter_to_rt: dict[str, Optional[str]] = {}
    for mr_href, related_hrefs in meter_related.items():
        resolved = next((href for href in related_hrefs if href in reading_types), None)
        meter_to_rt[mr_href] = resolved if resolved is not None else (
            related_hrefs[0] if related_hrefs else None
        )

    only_rt = next(iter(reading_types.values())) if len(reading_types) == 1 else None
    rows: list[EnergyUsageIntervalV1] = []
    for self_href, block in blocks:
        up_match = _USAGE_POINT.search(self_href)
        if up_match is None:
            raise EspiError(f"no UsagePoint in block href {self_href!r}")
        mr_match = _METER_READING.match(self_href)
        rt = None
        if mr_match is not None:
            mr_href = mr_match.group(1)
            rt_href = meter_to_rt.get(mr_href)
            if rt_href is not None:
                rt = reading_types.get(rt_href)
                if rt is None:
                    raise EspiError(
                        f"ReadingType {rt_href!r} not in feed for block {self_href!r}"
                    )
            elif only_rt is not None:
                rt = only_rt
        if rt is None:
            raise EspiError(f"no ReadingType resolvable for block {self_href!r}")
        if rt["flow"] is None:
            raise EspiError(f"missing flowDirection for block {self_href!r}")
        if rt["flow"] != FLOW_FORWARD:
            continue
        if rt["uom"] is None:
            raise EspiError(f"missing uom for block {self_href!r}")
        if rt["uom"] != UOM_WATT_HOURS:
            raise EspiError(f"unsupported uom {rt['uom']} (only 72 = Wh)")
        usage_point = up_match.group(1)
        scale = _scale(rt["pow10"])
        for reading in block.findall(f"{ESPI}IntervalReading"):
            try:
                period = reading.find(f"{ESPI}timePeriod")
                start = _int(period.find(f"{ESPI}start")) if period is not None else None
                duration = _int(period.find(f"{ESPI}duration")) if period is not None else None
                value = _int(reading.find(f"{ESPI}value"))
                if start is None or not duration or value is None:
                    continue
                begin = datetime.fromtimestamp(start, tz=timezone.utc)
                rows.append(
                    EnergyUsageIntervalV1(
                        source=source,
                        usage_point_id=usage_point,
                        interval_start=begin,
                        interval_end=begin + timedelta(seconds=duration),
                        energy_kwh=value * scale / 1000.0,
                        retrieved_at=retrieved_at,
                        source_file=source_file,
                    )
                )
            except (ValueError, ValidationError, OverflowError, OSError) as exc:
                raise EspiError(f"invalid reading in block {self_href!r}") from exc
    if not rows:
        raise EspiError("no forward-flow interval readings in feed")
    rows.sort(key=lambda r: (r.usage_point_id, r.interval_start))
    return rows
