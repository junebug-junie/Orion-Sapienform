from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1
from portal.parse import (
    bill_payload_from_fields,
    forecast_payload_from_fields,
    is_login_url,
    parse_date,
    parse_kwh,
    parse_money,
    parse_period,
)

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text,value",
    [("$1,234.56", 1234.56), ("-$5.00", -5.0), ("($5.00)", -5.0), ("$3.20 CR", -3.2), ("", None), ("--", None)],
)
def test_parse_money(text, value) -> None:
    assert parse_money(text) == (None if value is None else pytest.approx(value))


def test_parse_money_rejects_text_without_digits() -> None:
    with pytest.raises(ValueError):
        parse_money("pending")


def test_parse_kwh_and_dates() -> None:
    assert parse_kwh("1,712 kWh") == pytest.approx(1712.0)
    assert parse_kwh("") is None
    assert parse_date("Aug 12, 2026") == date(2026, 8, 12)
    assert parse_date("08/12/2026") == date(2026, 8, 12)
    assert parse_date("2026-08-12") == date(2026, 8, 12)
    assert parse_period("Aug 12, 2026 - Sep 11, 2026") == (date(2026, 8, 12), date(2026, 9, 11))
    assert parse_period("2026-08-12 to 2026-09-11") == (date(2026, 8, 12), date(2026, 9, 11))
    with pytest.raises(ValueError):
        parse_period("Aug 12, 2026")


def test_login_detection() -> None:
    assert is_login_url("https://pacificorpb2c.b2clogin.com/x/B2C_1A_PAC_SIGNIN/oauth2")
    assert not is_login_url("https://csapps.rockymountainpower.net/secure/my-account/energy-usage")


def test_bill_row_to_payload() -> None:
    payload = bill_payload_from_fields(
        {
            "billing_period": "Aug 12, 2026 - Sep 11, 2026",
            "kwh": "712 kWh",
            "current_charges": "$101.23",
            "taxes": "$4.10",
            "due_date": "Oct 2, 2026",
        },
        retrieved_at=NOW,
    )
    assert payload["kind"] == "energy.bill.actual.v1"
    bill = EnergyBillActualV1.model_validate({k: v for k, v in payload.items() if k != "kind"})
    assert bill.source == "rockymountain_power"
    assert bill.kwh_billed == 712.0 and bill.taxes == pytest.approx(4.10)
    assert bill.energy_charge is None
    assert bill.due_date == date(2026, 10, 2)


@pytest.mark.parametrize("missing", ["billing_period", "kwh", "current_charges"])
def test_bill_row_missing_required_field_raises(missing) -> None:
    fields = {"billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh", "current_charges": "$101.23"}
    del fields[missing]
    with pytest.raises(ValueError):
        bill_payload_from_fields(fields, retrieved_at=NOW)


def test_forecast_fields_to_payload_and_empty_rejected() -> None:
    payload = forecast_payload_from_fields(
        {"billing_period": "Sep 11, 2026 - Oct 12, 2026", "projected_total": "$96.00"},
        retrieved_at=NOW,
    )
    fc = EnergyBillForecastV1.model_validate({k: v for k, v in payload.items() if k != "kind"})
    assert fc.projected_total_usd == pytest.approx(96.0) and fc.projected_kwh is None
    with pytest.raises(ValueError):
        forecast_payload_from_fields({"billing_period": "Sep 11, 2026 - Oct 12, 2026"}, retrieved_at=NOW)
