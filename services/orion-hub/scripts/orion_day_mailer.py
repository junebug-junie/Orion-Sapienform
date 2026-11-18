"""Orion's Day mailer: email each persisted letter once.

Since living Prime Part 3 (design #3224) Hub no longer decides when a letter is written: Prime
orders it (scripts/orion_day_order.py) and orion-durable-runs writes the ``orion_day_letter`` row.
Hub only emails that row, exactly as before:

* **Quota** is ``orion_day_letter.emailed_at``: a row with it set is done.
* **Send** re-renders from the row (no model call), under a Postgres advisory lock per letter
  day, re-reading the row under the lock; ``emailed_at`` is stamped only when notify answers
  ``email_status == "sent"``.
* **Outcome unknown** (notify's reply timed out after a possible send, or the stamp failed after
  ``sent``): never resent automatically -- a duplicate letter is worse than a late one.
* **Skipped** (policy declined, or SMTP not configured): no retry can change that, so the letter
  is not retried until it leaves the sweep window (or Hub restarts).

Two triggers: Hub's durable run-state hook (``curiosity_investigation.run_state_hooks``) on a
terminal ``orion_day.letter`` state, and a slow sweep over un-emailed rows from the last
``EMAIL_LOOKBACK_DAYS`` days (``HUB_ORION_DAY_EMAIL_SWEEP_SEC``). The sweep alone converges when
the curiosity listener is off. Nothing here submits a run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from orion.orion_day.store import fetch_letter
from orion.schemas.notify import NotificationAccepted
from orion.schemas.orion_day import ORION_DAY_TIMEZONE, ORION_DAY_WORKFLOW

from scripts.orion_day_email import build_notification, load_inline_images

logger = logging.getLogger("orion-hub.orion_day_mailer")

# Gmail clips an HTML body past ~102 KB behind "View entire message" (content is kept).
GMAIL_CLIP_BYTES = 102_000
# Un-emailed letters older than this many days (local letter dates before today - N) are left alone.
EMAIL_LOOKBACK_DAYS = 3

TERMINAL = frozenset({"completed", "failed", "abandoned", "cancelled"})

# Session advisory lock per letter day around the send: a second Hub on the same database
# (a worktree deploy, a dev Hub) cannot email the same letter concurrently.
EMAIL_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtext('orion_day_email:' || $1::text))"
EMAIL_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('orion_day_email:' || $1::text))"

UNEMAILED_SINCE_SQL = """
SELECT letter_date FROM orion_day_letter
 WHERE emailed_at IS NULL AND letter_date >= $1
 ORDER BY letter_date
"""

STAMP_EMAILED_SQL = """
UPDATE orion_day_letter
   SET emailed_at = now(), email_notification_id = $2
 WHERE letter_date = $1 AND emailed_at IS NULL
"""


class OrionDayMailer:
    def __init__(
        self,
        *,
        enabled: bool,
        email_enabled: bool,
        pool_provider: Callable[[], Any],
        notify: Any,
        sweep_interval_sec: float = 900.0,
        email_retry_sec: float = 1800.0,
        lookback_days: int = EMAIL_LOOKBACK_DAYS,
        tz_name: str = ORION_DAY_TIMEZONE,
        image_dir: str = "/mnt/storage-lukewarm/orion/reverie-visual",
        max_images: int = 6,
        image_max_bytes: int = 450_000,
        source_service: str = "orion-hub",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.email_enabled = bool(email_enabled)
        self._pool_provider = pool_provider
        self.notify = notify
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.email_retry_sec = float(email_retry_sec)
        self.lookback_days = max(0, int(lookback_days))
        self.tz_name = tz_name
        self.image_dir = image_dir
        self.max_images = max(0, int(max_images))
        self.image_max_bytes = int(image_max_bytes)
        self.source_service = source_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._hook_tasks: set[asyncio.Task] = set()
        # Spam guards. Losing them on restart costs one repeat send attempt, never a wrong stamp.
        self._email_retry_after: dict[date, float] = {}
        # Dates whose email outcome is unknown (reply lost after a possible send, or the stamp
        # failed after `sent`): never resent automatically -- a duplicate letter is worse.
        self._email_outcome_unknown: set[date] = set()

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled:
            logger.info("orion_day_mailer disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("orion_day_mailer started sweep=%ss lookback_days=%s email=%s",
                    self.sweep_interval_sec, self.lookback_days, self.email_enabled)

    async def stop(self) -> None:
        tasks = [t for t in (self._task, *self._hook_tasks) if t is not None]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("orion_day_email_sweep_failed")
            await asyncio.sleep(self.sweep_interval_sec)

    async def on_run_state(self, state: Any) -> None:
        """Hook for Hub's durable run-state listener. Never blocks that listener: a terminal
        ``orion_day.letter`` state schedules one sweep in the background."""
        if not self.enabled or getattr(state, "workflow", None) != ORION_DAY_WORKFLOW:
            return
        if getattr(state, "status", None) not in TERMINAL:
            return
        detail = getattr(state, "detail", None) or {}
        logger.info("orion_day_run_terminal run=%s status=%s persist_outcome=%s",
                    state.run_id, state.status, detail.get("persist_outcome"))
        task = asyncio.create_task(self._hook_sweep())
        self._hook_tasks.add(task)
        task.add_done_callback(self._hook_tasks.discard)

    async def _hook_sweep(self) -> None:
        try:
            await self.sweep()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("orion_day_hook_sweep_failed")

    def oldest_letter_date(self) -> date:
        """The oldest letter date the sweep still emails: local today minus the lookback."""
        now = self._clock()
        aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return aware.astimezone(ZoneInfo(self.tz_name)).date() - timedelta(days=self.lookback_days)

    # --- the sweep ---------------------------------------------------------------------------

    async def sweep(self) -> dict[date, str] | str:
        """Email every un-emailed letter in the window. Returns {letter_date: outcome}, or one
        word when nothing could be read ("disabled", "no_pool", "store_unavailable")."""
        if not self.enabled:
            return "disabled"
        async with self._lock:
            pool = self._pool_provider()
            if pool is None:
                return "no_pool"
            try:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(UNEMAILED_SINCE_SQL, self.oldest_letter_date())
            except Exception as exc:  # noqa: BLE001 -- missing table (migration not applied), DB down
                logger.warning("orion_day_store_unavailable err=%s", exc)
                return "store_unavailable"
            outcomes: dict[date, str] = {}
            for row in rows:
                day = row["letter_date"]
                try:
                    outcomes[day] = await self.email_date(day)
                except Exception as exc:  # noqa: BLE001 -- one bad row must not starve the rest
                    logger.exception("orion_day_email_date_failed date=%s", day)
                    outcomes[day] = f"error:{type(exc).__name__}"
            return outcomes

    async def email_date(self, day: date) -> str:
        if not self.email_enabled:
            return "email_disabled"
        if day in self._email_outcome_unknown:
            return "email_outcome_unknown"
        retry_after = self._email_retry_after.get(day)
        if retry_after is not None and time.monotonic() < retry_after:
            return "email_backoff"
        pool = self._pool_provider()
        if pool is None:
            return "no_pool"
        async with pool.acquire() as conn:
            if not await conn.fetchval(EMAIL_LOCK_SQL, str(day)):
                return "email_locked"
            try:
                return await self._send_locked(conn, day)
            finally:
                await conn.fetchval(EMAIL_UNLOCK_SQL, str(day))

    def _build_request(self, letter):
        images = load_inline_images(letter, storage_dir=self.image_dir,
                                    max_images=self.max_images, max_bytes=self.image_max_bytes)
        return build_notification(letter, images, source_service=self.source_service), images

    async def _send_locked(self, conn, day: date) -> str:
        # Read under the lock: another process may have sent it since the sweep's read.
        letter = await fetch_letter(conn, day)
        if letter is None:
            return "no_letter"
        if letter.emailed_at is not None:
            return "quota_met"
        request, images = await asyncio.to_thread(self._build_request, letter)
        html_bytes = len((request.body_html or "").encode("utf-8"))
        if html_bytes > GMAIL_CLIP_BYTES:
            logger.warning("orion_day_email_large date=%s html_bytes=%s -- Gmail shows the rest "
                           "behind 'View entire message'", day, html_bytes)
        accepted: NotificationAccepted = await asyncio.to_thread(self.notify.send, request)
        if not (accepted.ok and accepted.email_status == "sent"):
            detail = str(accepted.detail or "")
            if not accepted.ok and "timed out" in detail.lower():
                # notify sends SMTP synchronously, so a timed-out reply may follow a real send.
                self._email_outcome_unknown.add(day)
                logger.error("orion_day_email_outcome_unknown date=%s detail=%s -- not resending; "
                             "check the inbox/notify logs, then clear by restarting Hub", day, detail)
                return "email_outcome_unknown"
            if accepted.email_status == "skipped":
                # Policy declined or SMTP not configured: no retry can change that. Wait until the
                # letter leaves the sweep window (or Hub restarts).
                self._email_retry_after[day] = float("inf")
            else:
                self._email_retry_after[day] = time.monotonic() + self.email_retry_sec
            logger.warning("orion_day_email_not_sent date=%s ok=%s email_status=%s detail=%s",
                           day, accepted.ok, accepted.email_status, detail)
            return "email_failed"
        for attempt in range(3):
            try:
                await conn.execute(STAMP_EMAILED_SQL, day, str(request.notification_id))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    self._email_outcome_unknown.add(day)
                    logger.error("orion_day_email_stamp_failed date=%s notification=%s err=%s -- "
                                 "sent but unstamped; not resending", day, request.notification_id, exc)
                    return "email_stamp_failed"
                await asyncio.sleep(0.5 * (attempt + 1))
        self._email_retry_after.pop(day, None)
        logger.info("orion_day_emailed date=%s notification=%s images=%s html_bytes=%s",
                    day, request.notification_id, len(images), html_bytes)
        return "emailed"
