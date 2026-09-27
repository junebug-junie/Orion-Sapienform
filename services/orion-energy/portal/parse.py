"""Portal text -> bill contracts. Pure, so it is tested without a browser."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Mapping, Optional

from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

from .selectors import LOGIN_URL_MARKERS

_BLANK = {"", "-", "--", "—", "n/a"}
_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d")
_PERIOD_SPLIT = re.compile(r"\s+(?:-|–|—|to)\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_OPTIONAL_MONEY = ("energy_charge", "customer_charge", "adjustments", "fees", "taxes", "credits", "amount_due")


def parse_money(text: str) -> Optional[float]:
    s = (text or "").strip()
    if s.lower() in _BLANK:
        return None
    match = _NUMBER.search(s)
    if match is None:
        raise ValueError(f"no amount in {text!r}")
    value = float(match.group(0).replace(",", ""))
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or s.upper().endswith("CR")
    return -value if negative else value


def parse_kwh(text: str) -> Optional[float]:
    s = (text or "").strip()
    if s.lower() in _BLANK:
        return None
    match = _NUMBER.search(s)
    if match is None:
        raise ValueError(f"no kWh in {text!r}")
    return float(match.group(0).replace(",", ""))


def parse_date(text: str) -> date:
    s = (text or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {text!r}")


def parse_period(text: str) -> tuple[date, date]:
    parts = _PERIOD_SPLIT.split((text or "").strip())
    if len(parts) != 2:
        raise ValueError(f"unrecognized billing period {text!r}")
    return parse_date(parts[0]), parse_date(parts[1])


def is_login_url(url: str) -> bool:
    low = (url or "").lower()
    return any(marker in low for marker in LOGIN_URL_MARKERS)


def _required(fields: Mapping[str, str], key: str) -> str:
    value = (fields.get(key) or "").strip()
    if not value:
        raise ValueError(f"portal row has no {key}")
    return value


def bill_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = parse_period(_required(fields, "billing_period"))
    kwh = parse_kwh(_required(fields, "kwh"))
    charges = parse_money(_required(fields, "current_charges"))
    if kwh is None or charges is None:
        raise ValueError("portal row has blank kWh or current charges")
    money = {k: parse_money(fields[k]) for k in _OPTIONAL_MONEY if fields.get(k)}
    due = fields.get("due_date")
    bill = EnergyBillActualV1(
        source="rockymountain_power",
        billing_period_start=start,
        billing_period_end=end,
        kwh_billed=kwh,
        current_charges=charges,
        due_date=parse_date(due) if due else None,
        retrieved_at=retrieved_at,
        **money,
    )
    return {"kind": ENERGY_BILL_ACTUAL_KIND, **bill.model_dump(mode="json", exclude={"source_file"})}


def forecast_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = parse_period(_required(fields, "billing_period"))
    forecast = EnergyBillForecastV1(
        source="rockymountain_power",
        billing_period_start=start,
        billing_period_end=end,
        as_of=retrieved_at,
        projected_kwh=parse_kwh(fields.get("projected_kwh", "")),
        projected_total_usd=parse_money(fields.get("projected_total", "")),
        retrieved_at=retrieved_at,
    )
    return {"kind": ENERGY_BILL_FORECAST_KIND, **forecast.model_dump(mode="json", exclude={"source_file"})}
