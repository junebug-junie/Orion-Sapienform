"""One fetch attempt: usage XML + bills into the drop directories.

RMP keeps its login only for the life of the browser, so a login redirect is the normal
start of every attempt. With a credentials file the fetcher submits the login form once;
without one, or if that single submit does not land past the login page (wrong password,
MFA prompt, captcha), the answer is `reauth_required` -- never a retry, so a bad password
cannot lock the account. Anything empty or unparseable is an error with the raw artifact
kept for debugging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from orion.energy.espi import EspiError, parse_espi

from .credentials import PortalCredentials
from .driver import PortalDriver
from .parse import PortalFieldError, bill_payload_from_fields, forecast_payload_from_fields, is_login_url

logger = logging.getLogger("orion-energy-portal")

_STAMP = "%Y%m%dT%H%M%SZ"
_HOUR = timedelta(hours=1)
_MAX_DAY_SPAN = timedelta(hours=25)  # a DST fall-back day
# Bad files in a row mean the page itself is broken (date entry dead, button moved), not one
# day's data; stop rather than burn a long backfill against a portal that throttles (~8 downloads).
MAX_CONSECUTIVE_BAD_DAYS = 3
_ATTRS = r"""(?:[^>"']|"[^"]*"|'[^']*')*"""
_SCRIPT = re.compile(rf"(<script\b{_ATTRS}>).*?(</script\s*>|\Z)", re.IGNORECASE | re.DOTALL)
_INPUT = re.compile(rf"<input\b{_ATTRS}>", re.IGNORECASE)
_HIDDEN = re.compile(r"""(?<![\w-])type\s*=\s*["']?hidden\b""", re.IGNORECASE)
_VALUE = re.compile(r"""(?<![\w-])(value\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_META = re.compile(rf"<meta\b{_ATTRS}>", re.IGNORECASE)
_SECRET_META = re.compile(r"""(?<![\w-])(?:name|property)\s*=\s*["']?[^"'\s>]*(?:csrf|token)""", re.IGNORECASE)
_CONTENT = re.compile(r"""(?<![\w-])(content\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
# Per-retrieval stamps; a bill whose other fields are unchanged is the same bill.
_VOLATILE_KEYS = ("retrieved_at", "as_of")


@dataclass(frozen=True)
class PortalOutcome:
    state: Literal["ok", "reauth_required", "error"]
    reason: str
    xml_files: tuple[Path, ...] = ()
    bill_files: tuple[Path, ...] = ()


def usage_days(first: date, last: date, *, count: int) -> list[date]:
    """The newest `count` days RMP offers, oldest first; never outside [first, last]."""
    days = [last - timedelta(days=i) for i in range(max(0, count))]
    return sorted(d for d in days if d >= first)


def _day_xml_problem(xml: bytes, *, day: date, now: datetime) -> Optional[str]:
    """None when the download is hourly ESPI for `day`.

    A daily reading would overwrite hour 0 in the ledger. A file for another day means the
    date entry did not take (the page re-served its current day); filing it under `day`
    would report a backfill that never happened. RMP day files start 02:00 local, which is
    the same UTC calendar date for any US zone, so the first reading's UTC date must be `day`.
    """
    if not xml:
        return "empty_download"
    try:
        rows = parse_espi(xml, retrieved_at=now, source="rockymountain_power")
    except EspiError as exc:
        return f"espi_invalid:{exc}"[:120]
    if any(r.interval_end - r.interval_start != _HOUR for r in rows):
        return "non_hourly_download"
    first = min(r.interval_start for r in rows)
    last_end = max(r.interval_end for r in rows)
    if first.astimezone(timezone.utc).date() != day or last_end - first > _MAX_DAY_SPAN:
        return "wrong_day_download"
    return None


def _atomic_write(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / f".{name}.part"
    part.write_bytes(data)
    final = directory / name
    part.rename(final)
    return final


def scrub_html(html: str) -> str:
    """Drop inline script bodies, hidden-input values, and csrf/token meta content before disk."""

    def _input(match: re.Match[str]) -> str:
        tag = match.group(0)
        return _VALUE.sub(r'\1""', tag) if _HIDDEN.search(tag) else tag

    def _meta(match: re.Match[str]) -> str:
        tag = match.group(0)
        return _CONTENT.sub(r'\1""', tag) if _SECRET_META.search(tag) else tag

    html = _SCRIPT.sub(lambda m: m.group(1) + m.group(2), html)
    return _META.sub(_meta, _INPUT.sub(_input, html))


def _scrub_bytes(data: bytes) -> bytes:
    try:
        return scrub_html(data.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        return data


def _save_raw(raw_dir: Path, now: datetime, name: str, data: bytes) -> None:
    """Best effort: a failed save is logged and never replaces the caller's error reason."""
    try:
        raw_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw_dir.chmod(0o700)
        path = raw_dir / f"{now.strftime(_STAMP)}-{name}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(_scrub_bytes(data))
        path.chmod(0o600)
    except OSError as exc:
        logger.warning("energy_portal_raw_save_failed name=%s error=%s", name, type(exc).__name__)


async def _snapshot_html(driver: PortalDriver, raw_dir: Path, now: datetime) -> None:
    try:
        html = await driver.page_html()
    except Exception:  # noqa: BLE001 -- best effort debugging artifact
        return
    _save_raw(raw_dir, now, "billing.html", html.encode())


def _field_reason(prefix: str, exc: ValueError) -> str:
    if isinstance(exc, PortalFieldError):
        return f"{prefix}:{exc.cause}:{exc.field}"
    return f"{prefix}:{type(exc).__name__}"


def _natural_key(payload: dict[str, Any]) -> str:
    return f"{payload['kind']}:{payload['billing_period_start']}:{payload['billing_period_end']}"


def _content_hash(payload: dict[str, Any]) -> str:
    stable = {k: v for k, v in payload.items() if k not in _VOLATILE_KEYS}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _load_seen(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    seen = raw.get("seen") if isinstance(raw, dict) else None
    return {str(k): str(v) for k, v in seen.items()} if isinstance(seen, dict) else {}


def _store_seen(path: Optional[Path], seen: dict[str, str]) -> None:
    if path is None:
        return
    try:
        _atomic_write(path.parent, path.name, json.dumps({"version": 1, "seen": seen}, sort_keys=True).encode())
    except OSError as exc:
        logger.warning("energy_portal_bills_seen_write_failed error=%s", type(exc).__name__)


def _one_per_period(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two rows for one period would alternate in the seen-state and both be re-sent daily."""
    kept: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        # Table is newest-first (UNVERIFIED), so the first row wins, e.g. a rebill over its original.
        kept.setdefault(_natural_key(payload), payload)
    return list(kept.values())


def _write_new_bills(
    payloads: list[dict[str, Any]], *, bill_inbox_dir: Path, seen_path: Optional[Path], stamp: str
) -> tuple[Path, ...]:
    """Write only bills/forecasts whose content changed since the last delivered copy."""
    seen = _load_seen(seen_path)
    written: list[Path] = []
    for i, payload in enumerate(_one_per_period(payloads)):
        key, digest = _natural_key(payload), _content_hash(payload)
        if seen.get(key) == digest:
            continue
        written.append(
            _atomic_write(bill_inbox_dir, f"rmp-portal-{stamp}-{i:02d}.json", json.dumps(payload).encode())
        )
        seen[key] = digest
    if written:
        _store_seen(seen_path, seen)
    return tuple(written)


class _SessionLost(RuntimeError):
    pass


async def _download_day_with_reload(driver: PortalDriver, day: date) -> bytes:
    """The download link goes dead mid-run now and then (live: day 10 of 60, fine alone).

    One reload of the usage page and one more try; never touches the login form.
    """
    try:
        return await driver.download_usage_day(day)
    except Exception as first:  # noqa: BLE001
        logger.warning("energy_portal_day_retry day=%s error=%s", day.isoformat(), type(first).__name__)
        if is_login_url(await driver.open_usage()):
            raise _SessionLost() from first
        return await driver.download_usage_day(day)


def _usage_failure(
    day: date, problem: str, delivered: list[Path], bad_days: list[str], total: int
) -> PortalOutcome:
    """The page itself failed: stop. Days already written stay delivered, the rest wait for next run."""
    reason = f"usage_day_failed:{day.isoformat()}:{problem}:delivered={len(delivered)}/{total}"
    if bad_days:
        reason += f":bad={len(bad_days)}"
    return PortalOutcome("error", reason[:200], xml_files=tuple(delivered))


async def run_once(
    driver: PortalDriver,
    *,
    inbox_dir: Path,
    bill_inbox_dir: Path,
    raw_dir: Path,
    backfill_days: int,
    now: datetime,
    seen_path: Optional[Path] = None,
    credentials: Optional[PortalCredentials] = None,
    scrape_bills: bool = True,
    through: Optional[date] = None,
) -> PortalOutcome:
    stamp = now.strftime(_STAMP)
    try:
        if is_login_url(await driver.open_usage()):
            if credentials is None:
                return PortalOutcome("reauth_required", "session_expired")
            try:
                await driver.login(username=credentials.username, password=credentials.password)
            except Exception as exc:  # noqa: BLE001 -- form missing/moved: selector drift, not a bad password
                return PortalOutcome("error", f"login_form_failed:{type(exc).__name__}")
            if is_login_url(await driver.open_usage()):
                return PortalOutcome("reauth_required", "login_failed")
        first, last = await driver.usage_day_range()
        if through is not None:
            last = min(last, through)
        days = usage_days(first, last, count=backfill_days)
        if not days:
            return PortalOutcome("error", "no_usage_days_available")
        delivered: list[Path] = []
        bad_days: list[str] = []
        bad_run = 0
        stopped = False
        for day in days:
            try:
                xml = await _download_day_with_reload(driver, day)
            except _SessionLost:
                return _usage_failure(day, "session_lost", delivered, bad_days, len(days))
            except Exception as exc:  # noqa: BLE001 -- two dead downloads in a row: stop for today
                return _usage_failure(day, f"download_failed:{type(exc).__name__}", delivered, bad_days, len(days))
            problem = _day_xml_problem(xml, day=day, now=now)
            if problem is not None:
                # A bad file for one day must not block the newer days (until the cap above).
                if xml:
                    _save_raw(raw_dir, now, f"green_button-{day.isoformat()}.xml", xml)
                bad_days.append(f"{day.isoformat()}:{problem}")
                bad_run += 1
                if bad_run >= MAX_CONSECUTIVE_BAD_DAYS:
                    stopped = True
                    break
                continue
            bad_run = 0
            delivered.append(_atomic_write(inbox_dir, f"rmp-portal-{stamp}-{day.isoformat()}.xml", xml))
        xml_files = tuple(delivered)
        if bad_days:
            tag = ":stopped" if stopped else ""
            reason = f"usage_days_bad:{len(bad_days)}/{len(days)}{tag}:{bad_days[0]}"
            return PortalOutcome("error", reason[:200], xml_files=xml_files)
        if not scrape_bills:
            return PortalOutcome("ok", "fetched_usage_only", xml_files=xml_files)

        try:
            rows = await driver.billing_rows()
            forecast = await driver.forecast_fields()
        except Exception as exc:  # noqa: BLE001 -- any scrape failure is a visible error state
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", f"billing_scrape_failed:{type(exc).__name__}", xml_files=xml_files)
        if not rows:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", "bill_rows_empty", xml_files=xml_files)
        try:
            payloads = [bill_payload_from_fields(row, retrieved_at=now) for row in rows]
        except ValueError as exc:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", _field_reason("bill_parse_failed", exc), xml_files=xml_files)
        forecast_error: Optional[str] = None
        if forecast is not None:
            try:
                payloads.append(forecast_payload_from_fields(forecast, retrieved_at=now))
            except ValueError as exc:
                await _snapshot_html(driver, raw_dir, now)
                forecast_error = _field_reason("forecast_parse_failed", exc)
        try:
            bill_files = _write_new_bills(
                payloads, bill_inbox_dir=bill_inbox_dir, seen_path=seen_path, stamp=stamp
            )
        except OSError as exc:
            return PortalOutcome("error", f"bill_write_failed:{type(exc).__name__}", xml_files=xml_files)
        if forecast_error:
            return PortalOutcome("error", forecast_error, xml_files=xml_files, bill_files=bill_files)
        return PortalOutcome("ok", "fetched", xml_files=xml_files, bill_files=bill_files)
    except Exception as exc:  # noqa: BLE001 -- the loop must survive and report
        return PortalOutcome("error", type(exc).__name__)
