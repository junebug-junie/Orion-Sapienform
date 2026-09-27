"""One-time headed login into the persistent profile. MFA stays on: you complete it.

Run on a host with a display, against the same profile dir the container mounts:
  python -m portal.reauth --profile /mnt/storage-warm/orion-energy/portal/profile \
      --status /mnt/storage-warm/orion-energy/portal/status.json
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from . import selectors
from .driver import prepare_profile_dir
from .settings import get_portal_settings
from .status import write_reauth_status


async def reauth(*, profile_dir: str, base_url: str, timeout_sec: float) -> None:
    from playwright.async_api import async_playwright

    profile = prepare_profile_dir(profile_dir)
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(str(profile), headless=False)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(base_url.rstrip("/") + selectors.USAGE_PATH)
            print(f"Log in (including MFA) in the browser window. Waiting up to {int(timeout_sec)}s ...")
            await page.wait_for_selector(selectors.LOGGED_IN_MARKER, timeout=timeout_sec * 1000)
        finally:
            await context.close()


def main() -> None:
    settings = get_portal_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=settings.ENERGY_PORTAL_PROFILE_DIR)
    parser.add_argument("--status", default=settings.ENERGY_PORTAL_STATUS_PATH)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    asyncio.run(
        reauth(
            profile_dir=args.profile,
            base_url=settings.ENERGY_PORTAL_BASE_URL,
            timeout_sec=args.timeout,
        )
    )
    write_reauth_status(Path(args.status), now=datetime.now(timezone.utc))
    print("Session saved. Run the `--once` fetch now, or the loop fetches after its interval.")


if __name__ == "__main__":
    main()
