from __future__ import annotations

import asyncio
import json
import re
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.bills import parse_bill
from portal.credentials import PortalCredentials
from portal.fetch import PortalOutcome, _day_xml_problem, run_once, usage_days
from portal.status import read_status, write_status

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
FIXTURE = Path(__file__).resolve().parents[3] / "orion/energy/tests/fixtures/espi_two_flows.xml"
USAGE_URL = "https://csapps.rockymountainpower.net/secure/my-account/energy-usage"
ROW = {"billing_period": "Aug 12, 2026 - Sep 11, 2026", "kwh": "712 kWh", "current_charges": "$101.23"}
FORECAST = {"billing_period": "Sep 11, 2026 - Oct 12, 2026", "projected_total": "$96.00"}


LOGIN_URL = "https://csapps.rockymountainpower.net/idm/login"
CREDS = PortalCredentials(username="me@example.com", password="hunter2")


FIRST_DAY, LAST_DAY = date(2026, 3, 7), date(2026, 9, 26)
_FIXTURE_FIRST_START = 1789063200  # 2026-09-10T18:00Z


def day_xml(day: date) -> bytes:
    """The hourly fixture moved to start at 08:00Z on `day`, like an RMP One Day file (02:00 MDT)."""
    target = int(datetime(day.year, day.month, day.day, 8, tzinfo=timezone.utc).timestamp())
    shift = target - _FIXTURE_FIRST_START
    return re.sub(
        rb"<espi:start>(\d+)</espi:start>",
        lambda m: b"<espi:start>%d</espi:start>" % (int(m.group(1)) + shift),
        FIXTURE.read_bytes(),
    )


class FakeDriver:
    def __init__(
        self, *, url=USAGE_URL, xml=None, rows=None, forecast=None, raise_on=None, html=None, after_login=USAGE_URL,
        day_range=(FIRST_DAY, LAST_DAY), per_day=None,
    ):
        self.url, self.xml, self.rows, self.forecast, self.raise_on = url, xml, rows, forecast, raise_on
        self.html = html or "<html>snapshot</html>"
        self.after_login = after_login
        self.day_range = day_range
        self.per_day = per_day or {}
        self.requested: list[date] = []
        self.logins: list[tuple[str, str]] = []
        self.billing_calls = 0

    async def open_usage(self):
        if self.raise_on == "open":
            raise RuntimeError("boom")
        return self.url

    async def login(self, *, username, password):
        self.logins.append((username, password))
        self.url = self.after_login
        return self.url

    async def usage_day_range(self):
        return self.day_range

    async def download_usage_day(self, day):
        self.requested.append(day)
        got = self.per_day.get(day, day_xml(day) if self.xml is None else self.xml)
        if isinstance(got, list):
            got = got.pop(0) if len(got) > 1 else got[0]
        if isinstance(got, Exception):
            raise got
        return got

    async def billing_rows(self):
        self.billing_calls += 1
        if self.raise_on == "billing":
            raise TimeoutError("selector")
        return list(self.rows or [])

    async def forecast_fields(self):
        return self.forecast

    async def page_html(self):
        return self.html


def _run(
    tmp_path, driver, *, now=NOW, raw_dir=None, bill_dir=None, credentials=None, scrape_bills=True,
) -> PortalOutcome:
    return asyncio.run(
        run_once(
            driver,
            inbox_dir=tmp_path / "inbox",
            bill_inbox_dir=bill_dir or tmp_path / "bills",
            raw_dir=raw_dir or tmp_path / "raw",
            seen_path=tmp_path / "portal" / "bills_seen.json",
            backfill_days=3,
            now=now,
            credentials=credentials,
            scrape_bills=scrape_bills,
        )
    )


def test_login_redirect_is_reauth_and_writes_nothing(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(url="https://pacificorpb2c.b2clogin.com/B2C_1A_PAC_SIGNIN"))
    assert (out.state, out.reason) == ("reauth_required", "session_expired")
    assert not (tmp_path / "inbox").exists() and not (tmp_path / "bills").exists()


def test_login_redirect_with_credentials_logs_in_once_then_fetches(tmp_path) -> None:
    driver = FakeDriver(url=LOGIN_URL)
    out = _run(tmp_path, driver, credentials=CREDS, scrape_bills=False)
    assert driver.logins == [("me@example.com", "hunter2")]
    assert (out.state, out.reason) == ("ok", "fetched_usage_only")
    assert [p.read_bytes() for p in out.xml_files] == [day_xml(date(2026, 9, d)) for d in (24, 25, 26)]


def test_rejected_login_is_reauth_after_exactly_one_submit(tmp_path) -> None:
    driver = FakeDriver(url=LOGIN_URL, after_login=LOGIN_URL)
    out = _run(tmp_path, driver, credentials=CREDS)
    assert len(driver.logins) == 1
    assert (out.state, out.reason) == ("reauth_required", "login_failed")
    assert "hunter2" not in out.reason
    assert not (tmp_path / "inbox").exists() and not (tmp_path / "bills").exists()


def test_live_session_skips_login(tmp_path) -> None:
    driver = FakeDriver()
    out = _run(tmp_path, driver, credentials=CREDS, scrape_bills=False)
    assert driver.logins == [] and out.state == "ok"


def test_usage_only_mode_never_touches_billing(tmp_path) -> None:
    driver = FakeDriver(rows=[])
    out = _run(tmp_path, driver, scrape_bills=False)
    assert (out.state, out.reason, out.bill_files) == ("ok", "fetched_usage_only", ())
    assert driver.billing_calls == 0 and not (tmp_path / "bills").exists()


def test_good_fetch_delivers_xml_and_bills(tmp_path) -> None:
    driver = FakeDriver(rows=[ROW], forecast=FORECAST)
    out = _run(tmp_path, driver)
    assert (out.state, out.reason) == ("ok", "fetched")
    assert driver.requested == [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 26)]
    assert [p.name for p in out.xml_files] == [
        f"rmp-portal-20260927T060000Z-2026-09-{d}.xml" for d in (24, 25, 26)
    ]
    assert out.xml_files[0].read_bytes() == day_xml(date(2026, 9, 24))
    assert len(out.bill_files) == 2
    kinds = [json.loads(p.read_text())["kind"] for p in out.bill_files]
    assert kinds == ["energy.bill.actual.v1", "energy.bill.forecast.v1"]
    bill = parse_bill(out.bill_files[0].read_bytes(), retrieved_at=NOW, source_file="x")
    assert bill.source == "rockymountain_power"
    assert not list((tmp_path / "inbox").glob(".*"))  # no leftover partial files


def test_garbage_download_is_error_and_kept_for_debugging(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=b"<html>not espi</html>", rows=[ROW]))
    assert out.state == "error" and out.reason.startswith("usage_days_bad:3/3:stopped:2026-09-24:espi_invalid")
    assert not (tmp_path / "inbox").exists()
    assert len(list((tmp_path / "raw").glob("*green_button-2026-09-2?.xml"))) == 3


def test_daily_grain_download_is_refused(tmp_path) -> None:
    daily = FIXTURE.read_bytes().replace(b">3600<", b">86400<")
    assert daily != FIXTURE.read_bytes()
    out = _run(tmp_path, FakeDriver(xml=daily), scrape_bills=False)
    assert (out.state, out.reason) == ("error", "usage_days_bad:3/3:stopped:2026-09-24:non_hourly_download")
    assert not (tmp_path / "inbox").exists()


def test_file_for_another_day_is_refused_and_newer_days_still_delivered(tmp_path) -> None:
    stale = day_xml(date(2026, 9, 24))  # the date entry did not take; page re-served the old day
    driver = FakeDriver(per_day={date(2026, 9, 25): stale})
    out = _run(tmp_path, driver, scrape_bills=False)
    assert (out.state, out.reason) == ("error", "usage_days_bad:1/3:2026-09-25:wrong_day_download")
    assert [p.name[-14:] for p in out.xml_files] == ["2026-09-24.xml", "2026-09-26.xml"]
    assert list((tmp_path / "raw").glob("*green_button-2026-09-25.xml"))


def test_broken_page_stops_after_three_bad_days_in_a_row(tmp_path) -> None:
    stale = day_xml(date(2026, 1, 1))  # date entry dead: every request re-serves the same old day
    driver = FakeDriver(xml=stale)
    out = asyncio.run(
        run_once(
            driver, inbox_dir=tmp_path / "inbox", bill_inbox_dir=tmp_path / "bills", raw_dir=tmp_path / "raw",
            backfill_days=60, now=NOW, scrape_bills=False,
        )
    )
    assert out.reason == "usage_days_bad:3/60:stopped:2026-07-29:wrong_day_download"
    assert len(driver.requested) == 3 and out.xml_files == ()


def test_a_good_day_resets_the_bad_day_count(tmp_path) -> None:
    junk = b"<html>not espi</html>"
    bad = {date(2026, 9, d): junk for d in (20, 21, 23, 24)}
    driver = FakeDriver(per_day=bad)
    out = asyncio.run(
        run_once(
            driver, inbox_dir=tmp_path / "inbox", bill_inbox_dir=tmp_path / "bills", raw_dir=tmp_path / "raw",
            backfill_days=7, now=NOW, scrape_bills=False,
        )
    )
    assert out.reason.startswith("usage_days_bad:4/7:2026-09-20:espi_invalid")
    assert len(driver.requested) == 7 and len(out.xml_files) == 3


def _hourly_espi(start: datetime, hours: int) -> bytes:
    """One flow of `hours` one-hour readings from `start`, the shape of an RMP One Day file."""
    readings = "".join(
        f"<espi:IntervalReading><espi:timePeriod><espi:duration>3600</espi:duration>"
        f"<espi:start>{int((start + timedelta(hours=h)).timestamp())}</espi:start></espi:timePeriod>"
        f"<espi:value>{100 + h}</espi:value></espi:IntervalReading>"
        for h in range(hours)
    )
    body = FIXTURE.read_text()
    head = body[: body.index("<espi:IntervalReading>")]
    tail = body[body.rindex("</espi:IntervalReading>") + len("</espi:IntervalReading>"):]
    return (head + readings + tail).encode()


@pytest.mark.parametrize(
    "day,start_hour_utc,hours",
    [
        (date(2026, 3, 8), 9, 23),   # spring forward, Mountain: 02:00 MST = 09:00Z, 23 hours
        (date(2026, 10, 31), 8, 25),  # fall back, Mountain: 02:00 MDT Oct 31 -> 02:00 MST Nov 1, 25 hours
        (date(2026, 1, 15), 9, 24),  # winter, Mountain
        (date(2026, 7, 4), 9, 24),   # summer, Pacific: 02:00 PDT = 09:00Z
        (date(2026, 1, 15), 10, 24),  # winter, Pacific
    ],
)
def test_real_day_shapes_across_dst_and_zones_are_accepted(day, start_hour_utc, hours) -> None:
    start = datetime(day.year, day.month, day.day, start_hour_utc, tzinfo=timezone.utc)
    xml = _hourly_espi(start, hours)
    assert _day_xml_problem(xml, day=day, now=NOW) is None
    assert _day_xml_problem(xml, day=day + timedelta(days=1), now=NOW) == "wrong_day_download"


def test_more_than_25_hours_is_refused() -> None:
    start = datetime(2026, 10, 31, 8, tzinfo=timezone.utc)
    assert _day_xml_problem(_hourly_espi(start, 26), day=date(2026, 10, 31), now=NOW) == "wrong_day_download"


def test_file_spanning_more_than_a_day_is_refused(tmp_path) -> None:
    two_days = day_xml(date(2026, 9, 25)).replace(
        b"<espi:start>%d</espi:start>" % int(datetime(2026, 9, 25, 10, tzinfo=timezone.utc).timestamp()),
        b"<espi:start>%d</espi:start>" % int(datetime(2026, 9, 26, 10, tzinfo=timezone.utc).timestamp()),
    )
    out = _run(tmp_path, FakeDriver(per_day={date(2026, 9, 25): two_days}), scrape_bills=False)
    assert out.reason == "usage_days_bad:1/3:2026-09-25:wrong_day_download"


def test_broken_login_form_is_error_not_reauth(tmp_path) -> None:
    class NoForm(FakeDriver):
        async def login(self, *, username, password):
            raise TimeoutError(f"fill {password} {username}")

    out = _run(tmp_path, NoForm(url=LOGIN_URL), credentials=CREDS)
    assert (out.state, out.reason) == ("error", "login_form_failed:TimeoutError")
    assert "hunter2" not in out.reason and "me@example.com" not in out.reason
    assert not (tmp_path / "inbox").exists()


def test_first_bad_day_stops_the_run_but_keeps_earlier_days(tmp_path) -> None:
    driver = FakeDriver(per_day={date(2026, 9, 25): TimeoutError("dead link")})
    out = _run(tmp_path, driver, scrape_bills=False)
    assert out.state == "error"
    assert out.reason == "usage_day_failed:2026-09-25:download_failed:TimeoutError:delivered=1/3"
    assert driver.requested == [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 25)]
    assert [p.name[-14:] for p in out.xml_files] == ["2026-09-24.xml"]


def test_one_dead_download_is_retried_once_after_a_reload(tmp_path) -> None:
    flaky = [TimeoutError("dead link"), day_xml(date(2026, 9, 25))]
    driver = FakeDriver(per_day={date(2026, 9, 25): flaky})
    out = _run(tmp_path, driver, scrape_bills=False)
    assert (out.state, out.reason, len(out.xml_files)) == ("ok", "fetched_usage_only", 3)
    assert driver.requested.count(date(2026, 9, 25)) == 2
    assert driver.logins == []


class ExpiringDriver(FakeDriver):
    """Session is live for the first page load, then every reload lands on the login page."""

    opens = 0

    async def open_usage(self):
        self.opens += 1
        return USAGE_URL if self.opens == 1 else LOGIN_URL


def test_session_lost_mid_run_stops_without_logging_in_again(tmp_path) -> None:
    driver = ExpiringDriver(per_day={date(2026, 9, 25): TimeoutError("dead link")})
    out = _run(tmp_path, driver, credentials=CREDS, scrape_bills=False)
    assert out.state == "error" and out.reason.startswith("usage_day_failed:2026-09-25:session_lost")
    assert driver.logins == [] and len(out.xml_files) == 1


def test_days_never_leave_the_portal_range() -> None:
    assert usage_days(FIRST_DAY, LAST_DAY, count=3) == [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 26)]
    assert usage_days(date(2026, 9, 25), LAST_DAY, count=5) == [date(2026, 9, 25), date(2026, 9, 26)]
    assert usage_days(FIRST_DAY, LAST_DAY, count=0) == []


def test_through_caps_the_newest_day_for_chunked_backfill(tmp_path) -> None:
    driver = FakeDriver()
    out = asyncio.run(
        run_once(
            driver, inbox_dir=tmp_path / "inbox", bill_inbox_dir=tmp_path / "bills", raw_dir=tmp_path / "raw",
            backfill_days=2, now=NOW, scrape_bills=False, through=date(2026, 8, 20),
        )
    )
    assert out.state == "ok"
    assert driver.requested == [date(2026, 8, 19), date(2026, 8, 20)]
    later = FakeDriver()
    asyncio.run(
        run_once(
            later, inbox_dir=tmp_path / "inbox", bill_inbox_dir=tmp_path / "bills", raw_dir=tmp_path / "raw",
            backfill_days=2, now=NOW, scrape_bills=False, through=date(2027, 1, 1),
        )
    )
    assert later.requested == [date(2026, 9, 25), date(2026, 9, 26)]


def test_empty_portal_range_is_error(tmp_path) -> None:
    driver = FakeDriver(day_range=(LAST_DAY, FIRST_DAY))
    out = _run(tmp_path, driver)
    assert (out.state, out.reason) == ("error", "no_usage_days_available")
    assert driver.requested == []


def test_empty_bill_table_is_error_but_usage_still_delivered(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(rows=[]))
    assert (out.state, out.reason) == ("error", "bill_rows_empty")
    assert out.xml_files and all(p.exists() for p in out.xml_files)
    assert list((tmp_path / "raw").glob("*billing.html"))


def test_billing_scrape_exception_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(raise_on="billing"))
    assert (out.state, out.reason) == ("error", "billing_scrape_failed:TimeoutError")
    assert out.xml_files and all(p.exists() for p in out.xml_files)


def test_unparseable_bill_row_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(rows=[{"billing_period": "soon"}]))
    assert out.state == "error" and out.reason.startswith("bill_parse_failed")
    assert out.bill_files == ()


def test_bill_parse_reason_names_field_not_raw_text(tmp_path) -> None:
    row = {**ROW, "current_charges": "pending acct SECRET"}
    out = _run(tmp_path, FakeDriver(rows=[row]))
    assert (out.state, out.reason) == ("error", "bill_parse_failed:ValueError:current_charges")
    assert "SECRET" not in out.reason and "pending" not in out.reason


def test_forecast_panel_without_parseable_fields_is_error_but_bills_delivered(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(rows=[ROW], forecast={}))
    assert out.state == "error" and out.reason.startswith("forecast_parse_failed")
    assert len(out.bill_files) == 1
    assert json.loads(out.bill_files[0].read_text())["kind"] == "energy.bill.actual.v1"


def test_zero_byte_download_is_error(tmp_path) -> None:
    out = _run(tmp_path, FakeDriver(xml=b"", rows=[ROW]))
    assert (out.state, out.reason) == ("error", "usage_days_bad:3/3:stopped:2026-09-24:empty_download")
    assert not (tmp_path / "inbox").exists()


def test_raw_save_failure_keeps_original_reason(tmp_path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    out = _run(tmp_path, FakeDriver(rows=[]), raw_dir=blocker / "raw")
    assert (out.state, out.reason) == ("error", "bill_rows_empty")
    out = _run(tmp_path, FakeDriver(xml=b"<html>not espi</html>"), raw_dir=blocker / "raw")
    assert out.state == "error" and out.reason.startswith("usage_days_bad:3/3:stopped:2026-09-24:espi_invalid")


def test_raw_html_is_scrubbed_and_private(tmp_path) -> None:
    html = (
        "<html><head><script>window.__TOKEN='sekrit-script-body';</script>"
        '<meta name="csrf-token" content="meta-csrf-111">'
        "<META content='meta-tok-222' property='og:session_token'>"
        '<meta name="description" content="keep-description">'
        "</head><body>"
        '<form><input type="hidden" name="__RequestVerificationToken" value="tok-12345">'
        "<input type='text' name='q' value='keep-me'>"
        "<INPUT value=\"tok-67890\" TYPE=HIDDEN name=csrf></form></body></html>"
    )
    out = _run(tmp_path, FakeDriver(rows=[], html=html))
    assert out.reason == "bill_rows_empty"
    [saved] = list((tmp_path / "raw").glob("*billing.html"))
    text = saved.read_text()
    assert "tok-12345" not in text and "tok-67890" not in text
    assert "meta-csrf-111" not in text and "meta-tok-222" not in text
    assert "sekrit-script-body" not in text
    assert "keep-me" in text and "__RequestVerificationToken" in text and "keep-description" in text
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "raw").stat().st_mode) == 0o700


def test_unchanged_bill_history_is_written_once(tmp_path) -> None:
    first = _run(tmp_path, FakeDriver(rows=[ROW], forecast=FORECAST))
    assert len(first.bill_files) == 2
    second = _run(
        tmp_path, FakeDriver(rows=[ROW], forecast=FORECAST),
        now=NOW + timedelta(days=1),
    )
    assert (second.state, second.reason, second.bill_files) == ("ok", "fetched", ())
    assert len(list((tmp_path / "bills").glob("*.json"))) == 2
    assert (tmp_path / "portal" / "bills_seen.json").exists()
    assert not list((tmp_path / "portal").glob(".*"))


def test_corrected_bill_is_written_again(tmp_path) -> None:
    _run(tmp_path, FakeDriver(rows=[ROW]))
    corrected = {**ROW, "current_charges": "$99.87"}
    out = _run(tmp_path, FakeDriver(rows=[corrected]), now=NOW + timedelta(days=1))
    assert len(out.bill_files) == 1
    assert json.loads(out.bill_files[0].read_text())["current_charges"] == 99.87


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


def test_same_period_rows_collapse_to_first_and_stay_quiet(tmp_path) -> None:
    rebill = {**ROW, "current_charges": "$99.87"}
    rows = [rebill, ROW]
    first = _run(tmp_path, FakeDriver(rows=rows))
    assert (first.state, len(first.bill_files)) == ("ok", 1)
    assert json.loads(first.bill_files[0].read_text())["current_charges"] == 99.87
    second = _run(tmp_path, FakeDriver(rows=rows), now=NOW + timedelta(days=1))
    assert (second.state, second.bill_files) == ("ok", ())


def test_changed_forecast_total_is_written_again(tmp_path) -> None:
    _run(tmp_path, FakeDriver(rows=[ROW], forecast=FORECAST))
    moved = {**FORECAST, "projected_total": "$120.00"}
    out = _run(
        tmp_path, FakeDriver(rows=[ROW], forecast=moved), now=NOW + timedelta(days=1),
    )
    assert len(out.bill_files) == 1
    written = json.loads(out.bill_files[0].read_text())
    assert (written["kind"], written["projected_total_usd"]) == ("energy.bill.forecast.v1", 120.0)


def test_bill_write_failure_keeps_delivered_xml(tmp_path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    out = _run(tmp_path, FakeDriver(rows=[ROW]), bill_dir=blocker / "bills")
    assert (out.state, out.reason) == ("error", "bill_write_failed:NotADirectoryError")
    assert out.xml_files and all(p.exists() for p in out.xml_files)


def test_non_object_status_json_reads_as_missing(tmp_path) -> None:
    path = tmp_path / "status.json"
    for raw in ("null", "[]", "42", '"ok"'):
        path.write_text(raw)
        assert read_status(path) is None
