"""Headless fetch loop. `--once --days 730` does the two-year backfill."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

from orion.energy.importer_status import PortalStatus

from .credentials import CredentialsFileTooOpen, load_credentials
from .driver import open_playwright_driver
from .fetch import PortalOutcome, run_once
from .settings import PortalSettings, get_portal_settings
from .status import read_status, write_attempt_started, write_status

logger = logging.getLogger("orion-energy-portal")

MIN_DAYS, MAX_DAYS = 1, 730


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def resolve_days(arg: Optional[int], *, default: int) -> int:
    if arg is None:
        return default
    return max(MIN_DAYS, min(MAX_DAYS, arg))


def bills_seen_path(status_path: Path) -> Path:
    return status_path.parent / "bills_seen.json"


def seconds_until_due(previous: Optional[PortalStatus], *, now: datetime, interval_hours: float) -> float:
    """Any recorded attempt (ok, error, or reauth_required) holds the next one for a full interval.

    This is what stops a crash/restart or redeploy from hitting the portal again straight away.
    """
    interval = interval_hours * 3600.0
    if previous is None:
        return 0.0
    elapsed = (now - previous.last_attempt_at).total_seconds()
    return min(interval, max(0.0, interval - elapsed))


def record_status(path: Path, outcome: Optional[PortalOutcome], *, now: datetime) -> Optional[PortalStatus]:
    """Write the attempt's outcome, or with `outcome=None` stamp that an attempt started."""
    try:
        if outcome is None:
            return write_attempt_started(path, now=now)
        return write_status(path, outcome, now=now)
    except Exception as exc:  # noqa: BLE001 -- a status write must never crash the loop into a restart
        logger.error("energy_portal_status_write_failed path=%s error=%s", path, type(exc).__name__)
        return None


async def attempt(settings: PortalSettings, *, days: int) -> PortalOutcome:
    now = _utcnow()
    status_path = Path(settings.ENERGY_PORTAL_STATUS_PATH)
    record_status(status_path, None, now=now)
    try:
        credentials = load_credentials(Path(settings.ENERGY_PORTAL_CREDENTIALS_PATH))
    except CredentialsFileTooOpen:
        outcome = PortalOutcome("error", "credentials_file_too_open")
        record_status(status_path, outcome, now=now)
        logger.error("energy_portal_fetch state=error reason=credentials_file_too_open (chmod 600 it)")
        return outcome
    except OSError as exc:
        outcome = PortalOutcome("error", f"credentials_unreadable:{type(exc).__name__}")
        record_status(status_path, outcome, now=now)
        logger.error("energy_portal_fetch state=error reason=%s", outcome.reason)
        return outcome
    try:
        async with open_playwright_driver(
            profile_dir=settings.ENERGY_PORTAL_PROFILE_DIR,
            base_url=settings.ENERGY_PORTAL_BASE_URL,
        ) as driver:
            outcome = await asyncio.wait_for(
                run_once(
                    driver,
                    inbox_dir=Path(settings.ENERGY_INBOX_DIR),
                    bill_inbox_dir=Path(settings.ENERGY_BILL_INBOX_DIR),
                    raw_dir=Path(settings.ENERGY_PORTAL_RAW_DIR),
                    seen_path=bills_seen_path(status_path),
                    backfill_days=days,
                    now=now,
                    credentials=credentials,
                    scrape_bills=settings.ENERGY_PORTAL_SCRAPE_BILLS,
                ),
                timeout=settings.ENERGY_PORTAL_TIMEOUT_SEC,
            )
    except (TimeoutError, asyncio.TimeoutError):
        outcome = PortalOutcome("error", "timeout")
    except Exception as exc:  # noqa: BLE001
        outcome = PortalOutcome("error", f"browser_failed:{type(exc).__name__}")
    record_status(status_path, outcome, now=now)
    logger.info(
        "energy_portal_fetch state=%s reason=%s xml=%s bills=%d",
        outcome.state,
        outcome.reason,
        outcome.xml_file.name if outcome.xml_file else None,
        len(outcome.bill_files),
    )
    return outcome


async def loop(
    settings: PortalSettings,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], datetime] = _utcnow,
) -> None:
    interval = settings.ENERGY_PORTAL_INTERVAL_HOURS
    wait = seconds_until_due(
        read_status(Path(settings.ENERGY_PORTAL_STATUS_PATH)), now=clock(), interval_hours=interval,
    )
    if wait > 0:
        logger.info("energy_portal_waiting seconds=%.0f reason=recent_attempt", wait)
        await sleep(wait)
    while True:
        await attempt(settings, days=settings.ENERGY_PORTAL_BACKFILL_DAYS)
        await sleep(interval * 3600.0)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[ORION_ENERGY_PORTAL] %(asctime)s %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="fetch now, ignoring the interval")
    parser.add_argument("--days", type=int, default=None, help=f"backfill window, clamped to {MIN_DAYS}..{MAX_DAYS}")
    args = parser.parse_args()
    settings = get_portal_settings()
    if args.once:
        days = resolve_days(args.days, default=settings.ENERGY_PORTAL_BACKFILL_DAYS)
        outcome = asyncio.run(attempt(settings, days=days))
        sys.exit(0 if outcome.state == "ok" else 1)
    asyncio.run(loop(settings))


if __name__ == "__main__":
    main()
