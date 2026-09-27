"""Portal text -> bill contracts. Pure, so it is tested without a browser."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Callable, Mapping, Optional, TypeVar

from pydantic import ValidationError

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
T = TypeVar("T")


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


class PortalFieldError(ValueError):
    """A portal field failed to parse. Carries the field name and error class, never the raw text."""

    def __init__(self, field: str, cause: str) -> None:
        super().__init__(f"{cause}:{field}")
        self.field = field
        self.cause = cause


def _parsed(fields: Mapping[str, str], key: str, parse: Callable[[str], T], *, required: bool = False) -> Optional[T]:
    raw = (fields.get(key) or "").strip()
    value: Optional[T] = None
    if raw:
        try:
            value = parse(raw)
        except ValueError as exc:
            raise PortalFieldError(key, type(exc).__name__) from None
    if required and value is None:
        raise PortalFieldError(key, "ValueError")
    return value


def _build(model: Callable[..., T], **kwargs: Any) -> T:
    try:
        return model(**kwargs)
    except ValidationError as exc:
        loc = exc.errors()[0].get("loc") or ()
        raise PortalFieldError(".".join(map(str, loc)) or "model", "ValidationError") from None


def bill_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = _parsed(fields, "billing_period", parse_period, required=True)
    bill = _build(
        EnergyBillActualV1,
        source="rockymountain_power",
        billing_period_start=start,
        billing_period_end=end,
        kwh_billed=_parsed(fields, "kwh", parse_kwh, required=True),
        current_charges=_parsed(fields, "current_charges", parse_money, required=True),
        due_date=_parsed(fields, "due_date", parse_date),
        retrieved_at=retrieved_at,
        **{k: _parsed(fields, k, parse_money) for k in _OPTIONAL_MONEY if fields.get(k)},
    )
    return {"kind": ENERGY_BILL_ACTUAL_KIND, **bill.model_dump(mode="json", exclude={"source_file"})}


def forecast_payload_from_fields(fields: Mapping[str, str], *, retrieved_at: datetime) -> dict[str, Any]:
    start, end = _parsed(fields, "billing_period", parse_period, required=True)
    forecast = _build(
        EnergyBillForecastV1,
        source="rockymountain_power",
        billing_period_start=start,
        billing_period_end=end,
        as_of=retrieved_at,
        projected_kwh=_parsed(fields, "projected_kwh", parse_kwh),
        projected_total_usd=_parsed(fields, "projected_total", parse_money),
        retrieved_at=retrieved_at,
    )
    return {"kind": ENERGY_BILL_FORECAST_KIND, **forecast.model_dump(mode="json", exclude={"source_file"})}
