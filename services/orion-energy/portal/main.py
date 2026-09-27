"""Headless fetch loop. `--once --days 730` does the two-year backfill."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .driver import open_playwright_driver
from .fetch import PortalOutcome, run_once
from .settings import PortalSettings, get_portal_settings
from .status import write_status

logger = logging.getLogger("orion-energy-portal")


async def attempt(settings: PortalSettings, *, days: int) -> PortalOutcome:
    now = datetime.now(timezone.utc)
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
                    backfill_days=days,
                    now=now,
                ),
                timeout=settings.ENERGY_PORTAL_TIMEOUT_SEC,
            )
    except (TimeoutError, asyncio.TimeoutError):
        outcome = PortalOutcome("error", "timeout")
    except Exception as exc:  # noqa: BLE001
        outcome = PortalOutcome("error", f"browser_failed:{type(exc).__name__}")
    write_status(Path(settings.ENERGY_PORTAL_STATUS_PATH), outcome, now=now)
    logger.info(
        "energy_portal_fetch state=%s reason=%s xml=%s bills=%d",
        outcome.state,
        outcome.reason,
        outcome.xml_file.name if outcome.xml_file else None,
        len(outcome.bill_files),
    )
    return outcome


async def loop(settings: PortalSettings) -> None:
    while True:
        await attempt(settings, days=settings.ENERGY_PORTAL_BACKFILL_DAYS)
        await asyncio.sleep(settings.ENERGY_PORTAL_INTERVAL_HOURS * 3600.0)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[ORION_ENERGY_PORTAL] %(asctime)s %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    settings = get_portal_settings()
    if args.once:
        outcome = asyncio.run(attempt(settings, days=args.days or settings.ENERGY_PORTAL_BACKFILL_DAYS))
        sys.exit(0 if outcome.state == "ok" else 1)
    asyncio.run(loop(settings))


if __name__ == "__main__":
    main()
