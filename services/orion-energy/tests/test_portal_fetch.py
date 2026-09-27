from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.bills import parse_bill
from portal.fetch import PortalOutcome, run_once
from portal.status import read_status, write_status

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
FIXTURE = Path(__file__).resolve().parents[3] / "orion/energy/tests/fixtures/espi_two_flows.xml"
USAGE_URL = "https://csapps.rockymountainpower.net/secure/my-account/energy-usage"
ROW = {"billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh", "current_charges": "$101.23"}
FORECAST = {"billing_period": "Sep 11, 2026 - Oct 12, 2026", "projected_total": "$96.00"}


class FakeDriver:
    def __init__(self, *, url=USAGE_URL, xml=b"", rows=None, forecast=None, raise_on=None):
        self.url, self.xml, self.rows, self.forecast, self.raise_on = url, xml, rows, forecast, raise_on
        self.days = None

    async def open_usage(self):
        if self.raise_on == "open":
            raise RuntimeError("boom")
        return self.url

    async def download_green_button(self, *, days):
        self.days = days
        return self.xml

    async def billing_rows(self):
        if self.raise_on == "billing":
            raise TimeoutError("selector")
        return list(self.rows or [])

    async def forecast_fields(self):
        return self.forecast

    async def page_html(self):
        return "<html>snapshot</html>"


def _run(tmp_path, driver) -> PortalOutcome:
    return asyncio.run(
        run_once(
            driver,
            inbox_dir=tmp_path / "inbox",
            bill_inbox_dir=tmp_path / "bills",
            raw_dir=tmp_path / "raw",
            backfill_days=3,
            now=NOW,
        )
    )


def test_login_redirect_is_reauth_and_writes_nothing(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(url="https://pacificorpb2c.b2clogin.com/B2C_1A_PAC_SIGNIN"))
    assert (out.state, out.reason) == ("reauth_required", "session_expired")
    assert not (tmp_path / "inbox").exists() and not (tmp_path / "bills").exists()


def test_good_fetch_delivers_xml_and_bills(tmp_path) -> None:
    driver = FakeDriver(xml=FIXTURE.read_bytes(), rows=[ROW], forecast=FORECAST)
    out = _run(tmp_path, driver)
    assert (out.state, out.reason) == ("ok", "fetched")
    assert driver.days == 3
    assert out.xml_file.name.startswith("rmp-portal-") and out.xml_file.suffix == ".xml"
    assert out.xml_file.read_bytes() == FIXTURE.read_bytes()
    assert len(out.bill_files) == 2
    kinds = [json.loads(p.read_text())["kind"] for p in out.bill_files]
    assert kinds == ["energy.bill.actual.v1", "energy.bill.forecast.v1"]
    bill = parse_bill(out.bill_files[0].read_bytes(), retrieved_at=NOW, source_file="x")
    assert bill.source == "rockymountain_power"
    assert not list((tmp_path / "inbox").glob(".*"))  # no leftover partial files


def test_garbage_download_is_error_and_kept_for_debugging(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=b"<html>not espi</html>", rows=[ROW]))
    assert out.state == "error" and out.reason.startswith("espi_invalid")
    assert not (tmp_path / "inbox").exists()
    assert list((tmp_path / "raw").glob("*green_button.xml"))


def test_empty_bill_table_is_error_but_usage_still_delivered(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), rows=[]))
    assert (out.state, out.reason) == ("error", "bill_rows_empty")
    assert out.xml_file is not None and out.xml_file.exists()
    assert list((tmp_path / "raw").glob("*billing.html"))


def test_billing_scrape_exception_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), raise_on="billing"))
    assert (out.state, out.reason) == ("error", "billing_scrape_failed:TimeoutError")
    assert out.xml_file is not None and out.xml_file.exists()


def test_unparseable_bill_row_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=FIXTURE.read_bytes(), rows=[{"billing_period": "soon"}]))
    assert out.state == "error" and out.reason.startswith("bill_parse_failed")
    assert out.bill_files == ()


def test_driver_crash_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(raise_on="open"))
    assert (out.state, out.reason) == ("error", "RuntimeError")


def test_status_file_keeps_last_success_across_failures(tmp_path) -> None:
    path = tmp_path / "portal" / "status.json"
    assert read_status(path) is None
    write_status(path, PortalOutcome("ok", "fetched"), now=NOW)
    later = NOW + timedelta(days=1)
    s = write_status(path, PortalOutcome("reauth_required", "session_expired"), now=later)
    assert (s.state, s.last_attempt_at, s.last_success_at) == ("reauth_required", later, NOW)
    assert read_status(path) == s
