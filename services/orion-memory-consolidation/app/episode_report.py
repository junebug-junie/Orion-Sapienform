"""Daily old-vs-new memory report writer (spec Stage 1 acceptance 11). Read-only on Postgres.

Once per local day it writes ``<MEMORY_EPISODE_REPORT_DIR>/<yesterday>.md`` (and ``latest.md``):
yesterday's shadow episodes, the legacy crystallization rows next to the shadow distiller's
memories. The file is the artifact; nothing is published and no one is notified. The directory is a
named Docker volume -- the report quotes Juniper's conversation and never enters the repo.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from orion.memory.episode.report import build_report

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SEC = 3600


def report_day(now: datetime, tz: ZoneInfo) -> date:
    return now.astimezone(tz).date() - timedelta(days=1)


async def write_due_report(pool: Any, settings: Any, *, now: Optional[datetime] = None) -> Optional[Path]:
    """Write yesterday's report if it does not exist yet. Returns the path written, or None."""
    tz = ZoneInfo(settings.MEMORY_EPISODE_REPORT_TZ)
    now = now or datetime.now(timezone.utc)
    day = report_day(now, tz)
    out_dir = Path(settings.MEMORY_EPISODE_REPORT_DIR)
    path = out_dir / f"{day.isoformat()}.md"
    if path.exists():
        return None
    start = datetime.combine(day, time(0), tz)
    markdown = await build_report(pool, start=start, end=start + timedelta(days=1), tz_name=settings.MEMORY_EPISODE_REPORT_TZ)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(path)
    (out_dir / "latest.md").write_text(markdown, encoding="utf-8")
    logger.info("memory_episode_report_written path=%s chars=%s", path, len(markdown))
    return path


async def run_report_loop(pool: Any, settings: Any) -> None:
    while True:
        try:
            await write_due_report(pool, settings)
        except Exception:  # noqa: BLE001 -- a report failure must never touch the live path
            logger.exception("memory_episode_report_failed")
        await asyncio.sleep(CHECK_INTERVAL_SEC)
