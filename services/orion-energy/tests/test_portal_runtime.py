from __future__ import annotations

import asyncio
import json
import stat
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone

import pytest

from orion.energy.importer_status import PortalStatus
from portal import main as portal_main
from portal.driver import picker_date, prepare_profile_dir
from portal.fetch import PortalOutcome
from portal.settings import PortalSettings
from portal.status import read_status, write_reauth_status, write_status

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)


def _status(state: str, attempt_ago: timedelta) -> PortalStatus:
    return PortalStatus(state=state, reason=state, last_attempt_at=NOW - attempt_ago, last_success_at=None)


def test_no_previous_status_attempts_now() -> None:
    assert portal_main.seconds_until_due(None, now=NOW, interval_hours=24) == 0.0


def test_recent_attempt_waits_out_the_interval() -> None:
    wait = portal_main.seconds_until_due(_status("ok", timedelta(hours=2)), now=NOW, interval_hours=24)
    assert wait == pytest.approx(22 * 3600)


def test_reauth_required_is_not_retried_faster_than_interval() -> None:
    wait = portal_main.seconds_until_due(
        _status("reauth_required", timedelta(minutes=5)), now=NOW, interval_hours=24,
    )
    assert wait == pytest.approx(24 * 3600 - 300)


def test_old_attempt_is_due_now_and_future_attempt_is_capped() -> None:
    assert portal_main.seconds_until_due(_status("error", timedelta(days=3)), now=NOW, interval_hours=24) == 0.0
    future = _status("ok", timedelta(hours=-5))
    assert portal_main.seconds_until_due(future, now=NOW, interval_hours=24) == pytest.approx(24 * 3600)


def test_failed_status_write_does_not_raise(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert portal_main.record_status(blocker / "status.json", PortalOutcome("error", "x"), now=NOW) is None


def test_attempt_survives_browser_and_status_failures(tmp_path, monkeypatch) -> None:
    @asynccontextmanager
    async def broken_driver(**_kwargs):
        raise RuntimeError("no chromium")
        yield  # pragma: no cover

    monkeypatch.setattr(portal_main, "open_playwright_driver", broken_driver)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    settings = PortalSettings(ENERGY_PORTAL_STATUS_PATH=str(blocker / "status.json"))
    outcome = asyncio.run(portal_main.attempt(settings, days=3))
    assert (outcome.state, outcome.reason) == ("error", "browser_failed:RuntimeError")


def test_world_readable_credentials_stop_the_attempt_before_the_browser(tmp_path, monkeypatch) -> None:
    launched: list = []

    @asynccontextmanager
    async def must_not_launch(**_kwargs):
        launched.append(True)
        raise AssertionError("browser launched with a leaked password file")
        yield  # pragma: no cover

    monkeypatch.setattr(portal_main, "open_playwright_driver", must_not_launch)
    creds = tmp_path / "credentials.env"
    creds.write_text("RMP_USERNAME=me\nRMP_PASSWORD=hunter2\n")
    creds.chmod(0o644)
    status_path = tmp_path / "status.json"
    settings = PortalSettings(
        ENERGY_PORTAL_STATUS_PATH=str(status_path), ENERGY_PORTAL_CREDENTIALS_PATH=str(creds),
    )
    outcome = asyncio.run(portal_main.attempt(settings, days=3))
    assert (outcome.state, outcome.reason) == ("error", "credentials_file_too_open")
    assert launched == []
    saved = read_status(status_path)
    assert saved is not None and saved.reason == "credentials_file_too_open"
    assert "hunter2" not in status_path.read_text()


@pytest.mark.parametrize(
    "content,reason",
    [
        (b"RMP_USERNAME=me\n", "credentials_incomplete"),
        (b"RMP_USERNAME=me\nRMP_PASSWORD=hunter2\xff\n", "credentials_unreadable:UnicodeDecodeError"),
    ],
)
def test_bad_credentials_file_is_a_labelled_error_before_the_browser(tmp_path, monkeypatch, content, reason) -> None:
    @asynccontextmanager
    async def must_not_launch(**_kwargs):
        raise AssertionError("browser launched without usable credentials")
        yield  # pragma: no cover

    monkeypatch.setattr(portal_main, "open_playwright_driver", must_not_launch)
    creds = tmp_path / "credentials.env"
    creds.write_bytes(content)
    creds.chmod(0o600)
    status_path = tmp_path / "status.json"
    settings = PortalSettings(
        ENERGY_PORTAL_STATUS_PATH=str(status_path), ENERGY_PORTAL_CREDENTIALS_PATH=str(creds),
    )
    outcome = asyncio.run(portal_main.attempt(settings, days=3))
    assert (outcome.state, outcome.reason) == ("error", reason)
    assert "hunter2" not in status_path.read_text()


@pytest.mark.parametrize("form_breaks,reason", [(False, "login_failed"), (True, "login_form_failed:TimeoutError")])
def test_failed_login_never_leaks_the_password(tmp_path, monkeypatch, caplog, form_breaks, reason) -> None:
    creds = tmp_path / "credentials.env"
    creds.write_text("RMP_USERNAME=me@example.com\nRMP_PASSWORD=hunter2\n")
    creds.chmod(0o600)
    login_url = "https://csapps.rockymountainpower.net/idm/login"

    class StuckOnLogin:
        async def open_usage(self):
            return login_url

        async def login(self, *, username, password):
            if form_breaks:
                raise TimeoutError(f"fill {username} {password}")
            return login_url

    @asynccontextmanager
    async def fake_driver(**_kwargs):
        yield StuckOnLogin()

    monkeypatch.setattr(portal_main, "open_playwright_driver", fake_driver)
    status_path = tmp_path / "status.json"
    settings = PortalSettings(
        ENERGY_PORTAL_STATUS_PATH=str(status_path), ENERGY_PORTAL_CREDENTIALS_PATH=str(creds),
    )
    caplog.set_level("DEBUG")
    outcome = asyncio.run(portal_main.attempt(settings, days=3))
    assert outcome.reason == reason
    assert "hunter2" not in status_path.read_text() and "hunter2" not in caplog.text
    assert "me@example.com" not in caplog.text


def test_attempt_passes_credentials_and_bill_mode_to_the_fetch(tmp_path, monkeypatch) -> None:
    creds = tmp_path / "credentials.env"
    creds.write_text("RMP_USERNAME=me\nRMP_PASSWORD=hunter2\n")
    creds.chmod(0o600)
    seen: dict = {}

    @asynccontextmanager
    async def fake_driver(**_kwargs):
        yield object()

    async def fake_run_once(_driver, **kwargs):
        seen.update(kwargs)
        return PortalOutcome("ok", "fetched_usage_only")

    monkeypatch.setattr(portal_main, "open_playwright_driver", fake_driver)
    monkeypatch.setattr(portal_main, "run_once", fake_run_once)
    settings = PortalSettings(
        ENERGY_PORTAL_STATUS_PATH=str(tmp_path / "status.json"), ENERGY_PORTAL_CREDENTIALS_PATH=str(creds),
    )
    asyncio.run(portal_main.attempt(settings, days=3))
    assert seen["credentials"].password == "hunter2"
    assert seen["scrape_bills"] is False


def test_attempt_start_is_stamped_before_browser_launch(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"
    earlier = NOW - timedelta(days=2)
    write_status(status_path, PortalOutcome("ok", "fetched"), now=earlier)
    seen_at_launch: list = []

    @asynccontextmanager
    async def killed_mid_attempt(**_kwargs):
        seen_at_launch.append(read_status(status_path))
        raise RuntimeError("oom")
        yield  # pragma: no cover

    monkeypatch.setattr(portal_main, "open_playwright_driver", killed_mid_attempt)
    monkeypatch.setattr(portal_main, "_utcnow", lambda: NOW)
    settings = PortalSettings(ENERGY_PORTAL_STATUS_PATH=str(status_path))
    asyncio.run(portal_main.attempt(settings, days=3))
    [started] = seen_at_launch
    assert (started.state, started.reason, started.last_success_at, started.last_attempt_at) == (
        "ok", "fetched", earlier, NOW,
    )
    assert portal_main.seconds_until_due(started, now=NOW, interval_hours=24) == pytest.approx(24 * 3600)


def test_attempt_started_without_history_is_not_ok(tmp_path) -> None:
    path = tmp_path / "status.json"
    s = portal_main.record_status(path, None, now=NOW)
    assert s is not None and s.state != "ok" and s.last_success_at is None and s.last_attempt_at == NOW
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert portal_main.record_status(blocker / "status.json", None, now=NOW) is None


def test_loop_waits_before_first_attempt_after_recent_status(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"
    write_status(status_path, PortalOutcome("reauth_required", "session_expired"), now=NOW - timedelta(hours=1))
    events: list[tuple[str, float]] = []

    async def fake_attempt(_settings, *, days):
        events.append(("attempt", days))
        return PortalOutcome("reauth_required", "session_expired")

    class Stop(Exception):
        pass

    async def fake_sleep(seconds):
        events.append(("sleep", seconds))
        if len(events) >= 3:
            raise Stop

    monkeypatch.setattr(portal_main, "attempt", fake_attempt)
    settings = PortalSettings(ENERGY_PORTAL_STATUS_PATH=str(status_path), ENERGY_PORTAL_INTERVAL_HOURS=24)
    with pytest.raises(Stop):
        asyncio.run(portal_main.loop(settings, sleep=fake_sleep, clock=lambda: NOW))
    assert events[0][0] == "sleep" and events[0][1] == pytest.approx(23 * 3600)
    assert events[1] == ("attempt", settings.ENERGY_PORTAL_BACKFILL_DAYS)
    assert events[2] == ("sleep", 24 * 3600.0)


def test_loop_attempts_immediately_without_status(tmp_path, monkeypatch) -> None:
    events: list[str] = []

    async def fake_attempt(_settings, *, days):
        events.append("attempt")
        return PortalOutcome("ok", "fetched")

    class Stop(Exception):
        pass

    async def fake_sleep(_seconds):
        raise Stop

    monkeypatch.setattr(portal_main, "attempt", fake_attempt)
    settings = PortalSettings(ENERGY_PORTAL_STATUS_PATH=str(tmp_path / "status.json"))
    with pytest.raises(Stop):
        asyncio.run(portal_main.loop(settings, sleep=fake_sleep, clock=lambda: NOW))
    assert events == ["attempt"]


@pytest.mark.parametrize("arg,expected", [(None, 3), (0, 1), (-5, 1), (30, 30), (730, 730), (9999, 730)])
def test_days_are_clamped(arg, expected) -> None:
    assert portal_main.resolve_days(arg, default=3) == expected


def test_bills_seen_path_sits_next_to_status(tmp_path) -> None:
    assert portal_main.bills_seen_path(tmp_path / "p" / "status.json") == tmp_path / "p" / "bills_seen.json"


def test_picker_date_takes_the_calendar_day_as_written() -> None:
    assert picker_date("2026-09-26T00:00:00+00:00") == date(2026, 9, 26)
    assert picker_date("2026-03-07T00:00:00Z") == date(2026, 3, 7)
    with pytest.raises(ValueError):
        picker_date(None)


def test_attempt_timeout_grows_with_requested_days() -> None:
    assert portal_main.attempt_timeout_sec(300, days=3) == 300 + 3 * portal_main.PER_DAY_BUDGET_SEC
    assert portal_main.attempt_timeout_sec(300, days=60) > portal_main.attempt_timeout_sec(300, days=3)


def test_profile_dir_is_created_private(tmp_path) -> None:
    fresh = prepare_profile_dir(tmp_path / "a" / "profile")
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o700
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    prepare_profile_dir(loose)
    assert stat.S_IMODE(loose.stat().st_mode) == 0o700


def test_reauth_keeps_previous_success(tmp_path) -> None:
    path = tmp_path / "status.json"
    earlier = NOW - timedelta(days=2)
    write_status(path, PortalOutcome("ok", "fetched"), now=earlier)
    write_status(path, PortalOutcome("reauth_required", "session_expired"), now=NOW - timedelta(hours=3))
    s = write_reauth_status(path, now=NOW)
    assert (s.state, s.reason, s.last_attempt_at, s.last_success_at) == ("ok", "reauth_completed", NOW, earlier)
    assert read_status(path) == s
    assert json.loads(path.read_text())["last_success_at"] == earlier.isoformat()


def test_reauth_without_history_has_no_success(tmp_path) -> None:
    s = write_reauth_status(tmp_path / "status.json", now=NOW)
    assert (s.state, s.reason, s.last_success_at) == ("ok", "reauth_completed", None)
