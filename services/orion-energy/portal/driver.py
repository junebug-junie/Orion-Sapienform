"""The only module that touches a browser.

Login and the one-day Green Button download are verified against the live portal
(2026-09-28); billing_rows / forecast_fields are still UNVERIFIED.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Protocol

from . import selectors
from .parse import is_login_url

_PERIOD_TEXT = re.compile(r"^\s*(?:" + "|".join(map(re.escape, selectors.USAGE_PERIOD_OPTIONS)) + r")\s*$")


class PortalDriver(Protocol):
    async def open_usage(self) -> str: ...
    async def login(self, *, username: str, password: str) -> str: ...
    async def usage_day_range(self) -> tuple[date, date]: ...
    async def download_usage_day(self, day: date) -> bytes: ...
    async def billing_rows(self) -> list[dict[str, str]]: ...
    async def forecast_fields(self) -> Optional[dict[str, str]]: ...
    async def page_html(self) -> str: ...


def picker_date(raw: Optional[str]) -> date:
    """The picker's min/max are calendar days stamped at UTC midnight: take the date as written."""
    if not raw:
        raise ValueError("usage date picker has no min/max")
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()


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
        self._one_day_view = False

    async def open_usage(self) -> str:
        await self._page.goto(self._base + selectors.USAGE_PATH, wait_until="networkidle")
        self._one_day_view = False
        return self._page.url

    async def login(self, *, username: str, password: str) -> str:
        """One submit, no retry: returns the URL the page settled on, login page or not."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        frame = self._page.frame_locator(selectors.LOGIN_FRAME)
        await frame.locator(selectors.LOGIN_USERNAME).fill(username)
        await frame.locator(selectors.LOGIN_PASSWORD).fill(password)
        await frame.locator(selectors.LOGIN_SUBMIT).click()
        try:
            await self._page.wait_for_url(
                lambda url: not is_login_url(url), timeout=selectors.LOGIN_WAIT_SEC * 1000
            )
        except PlaywrightTimeout:
            pass
        return self._page.url

    async def usage_day_range(self) -> tuple[date, date]:
        picker = self._page.locator(selectors.USAGE_THROUGH_INPUT)
        return picker_date(await picker.get_attribute("min")), picker_date(await picker.get_attribute("max"))

    async def _ensure_one_day_view(self) -> None:
        """Not re-checked here: a view that did not switch yields daily readings, which
        fetch refuses as non_hourly_download."""
        if self._one_day_view:
            return
        page = self._page
        await page.locator("mat-select").filter(has_text=_PERIOD_TEXT).first.click()
        await page.locator("mat-option", has_text=selectors.USAGE_PERIOD_ONE_DAY).first.click()
        await page.wait_for_load_state("networkidle")
        self._one_day_view = True

    async def download_usage_day(self, day: date) -> bytes:
        """Hourly Green Button XML for one calendar day; `day` must be inside usage_day_range()."""
        page = self._page
        await self._ensure_one_day_view()
        picker = page.locator(selectors.USAGE_THROUGH_INPUT)
        await picker.fill(f"{day.month}/{day.day}/{day.year}")
        await picker.press("Enter")
        await picker.blur()
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(selectors.SETTLE_AFTER_DATE_SEC * 1000)
        async with page.expect_download(timeout=selectors.DOWNLOAD_WAIT_SEC * 1000) as info:
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
