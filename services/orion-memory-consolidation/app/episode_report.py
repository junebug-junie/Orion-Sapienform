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

from orion.memory.episode.report import UNDISTILLED_SQL, build_report

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SEC = 3600


def report_day(now: datetime, tz: ZoneInfo) -> date:
    return now.astimezone(tz).date() - timedelta(days=1)


# After this many hours past local midnight, yesterday's report is final even if some episode was
# never distilled (the reconciler has had its retries); until then it is rewritten every pass.
FINAL_AFTER_HOURS = 12


async def write_due_report(pool: Any, settings: Any, *, now: Optional[datetime] = None) -> Optional[Path]:
    """Write (or rewrite) yesterday's report until it is final. Returns the path written, or None.

    Final = every one of yesterday's closed episodes has a distill run, or FINAL_AFTER_HOURS have
    passed since local midnight. A provisional report says how many episodes are still pending.
    """
    tz = ZoneInfo(settings.MEMORY_EPISODE_REPORT_TZ)
    now = now or datetime.now(timezone.utc)
    day = report_day(now, tz)
    out_dir = Path(settings.MEMORY_EPISODE_REPORT_DIR)
    path = out_dir / f"{day.isoformat()}.md"
    final_marker = out_dir / f".{day.isoformat()}.final"
    if final_marker.exists():
        return None
    start = datetime.combine(day, time(0), tz)
    end = start + timedelta(days=1)
    pending = int((await pool.fetchrow(UNDISTILLED_SQL, start, end))["n"])
    final = pending == 0 or now >= end + timedelta(hours=FINAL_AFTER_HOURS)
    markdown = await build_report(pool, start=start, end=end, tz_name=settings.MEMORY_EPISODE_REPORT_TZ)
    if not final:
        markdown += f"\n_Provisional: {pending} episode(s) not distilled yet; this report is rewritten until they are._\n"
    elif pending:
        markdown += f"\n_Final with {pending} episode(s) never distilled._\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(path)
    (out_dir / "latest.md").write_text(markdown, encoding="utf-8")
    if final:
        final_marker.write_text(now.isoformat(), encoding="utf-8")
    logger.info("memory_episode_report_written path=%s final=%s pending=%s", path, final, pending)
    return path


async def run_report_loop(pool: Any, settings: Any) -> None:
    while True:
        try:
            await write_due_report(pool, settings)
        except Exception:  # noqa: BLE001 -- a report failure must never touch the live path
            logger.exception("memory_episode_report_failed")
        await asyncio.sleep(CHECK_INTERVAL_SEC)
