"""The only module that touches a browser. UNVERIFIED against the live portal."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Protocol

from . import selectors


class PortalDriver(Protocol):
    async def open_usage(self) -> str: ...
    async def download_green_button(self, *, days: int, now: datetime) -> bytes: ...
    async def billing_rows(self) -> list[dict[str, str]]: ...
    async def forecast_fields(self) -> Optional[dict[str, str]]: ...
    async def page_html(self) -> str: ...


def green_button_range(now: datetime, *, days: int) -> tuple[date, date]:
    end = now.date()
    return end - timedelta(days=days), end


def prepare_profile_dir(profile_dir: str | Path) -> Path:
    """The profile holds live session cookies: owner-only, even if the dir already existed."""
    path = Path(profile_dir)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


class PlaywrightDriver:
    def __init__(self, page: Any, base_url: str) -> None:
        self._page = page
        self._base = base_url.rstrip("/")

    async def open_usage(self) -> str:
        await self._page.goto(self._base + selectors.USAGE_PATH, wait_until="networkidle")
        return self._page.url

    async def download_green_button(self, *, days: int, now: datetime) -> bytes:
        page = self._page
        start, end = green_button_range(now, days=days)
        await page.click(selectors.GREEN_BUTTON_OPEN)
        await page.fill(selectors.GREEN_BUTTON_FROM, start.strftime(selectors.GREEN_BUTTON_DATE_FORMAT))
        await page.fill(selectors.GREEN_BUTTON_TO, end.strftime(selectors.GREEN_BUTTON_DATE_FORMAT))
        async with page.expect_download() as info:
            await page.click(selectors.GREEN_BUTTON_DOWNLOAD)
        download = await info.value
        return Path(await download.path()).read_bytes()

    async def _fields(self, scope: Any, spec: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, sel in spec.items():
            loc = scope.locator(sel)
            if await loc.count():
                out[name] = (await loc.first.inner_text()).strip()
        return out

    async def billing_rows(self) -> list[dict[str, str]]:
        await self._page.goto(self._base + selectors.BILLING_PATH, wait_until="networkidle")
        return [
            await self._fields(row, selectors.BILL_ROW_FIELDS)
            for row in await self._page.locator(selectors.BILL_ROW).all()
        ]

    async def forecast_fields(self) -> Optional[dict[str, str]]:
        panel = self._page.locator(selectors.FORECAST_PANEL)
        if not await panel.count():
            return None
        return await self._fields(panel.first, selectors.FORECAST_FIELDS)

    async def page_html(self) -> str:
        return await self._page.content()


@asynccontextmanager
async def open_playwright_driver(
    *, profile_dir: str, base_url: str, headless: bool = True
) -> AsyncIterator[PlaywrightDriver]:
    from playwright.async_api import async_playwright  # portal image only; tests use fakes

    profile = prepare_profile_dir(profile_dir)
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            str(profile), headless=headless, accept_downloads=True
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            yield PlaywrightDriver(page, base_url)
        finally:
            await context.close()
