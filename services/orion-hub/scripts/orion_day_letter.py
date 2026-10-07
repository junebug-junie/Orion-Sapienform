"""Orion's Day scheduler: one letter a day, retried until it lands, emailed once.

Hub only (a) submits admitted durable runs (``orion_day.letter``, orion-durable-runs) and
(b) emails a letter the run already persisted. All state lives in Postgres and in the durable
run registry -- nothing here survives a restart, and nothing needs to:

* **Quota** is ``orion_day_letter.emailed_at``: a row with it set means the day is done.
* **Attempt number** is derived from the durable registry, not memory: ``GET /runs/orion-day-
  <date>-<n>`` for n = 1, 2, ... until 404. The latest attempt's status decides the move.
* **Retry** of a failed/abandoned run is attempt n+1 (same run_id + regathered brief is refused
  by the store). Capped by ``max_attempts``; then one notice.
* **Email** of an existing row with ``emailed_at IS NULL`` re-renders from the row (no model
  call) and stamps ``emailed_at`` only when notify answers ``email_status == "sent"``.

Letter windows. Letter date L is scheduled at (L+1) HH:MM local (08:30 America/Denver by
default) and is abandoned at (L+2) HH:MM, when L+1 takes over -- the "hard stop". So between
two slots exactly one letter is active.

Latency: besides the tick, Hub's existing ``orion:durable:run:state`` listener (curiosity)
calls :meth:`OrionDayLetterLoop.on_run_state` for ``detail.line == "orion_day"`` terminals.
If curiosity is disabled that listener is not running and the tick alone still converges.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import httpx

from orion.orion_day.brief import OrionDayEmptyError, build_orion_day_brief, build_orion_day_request
from orion.orion_day.store import fetch_letter
from orion.schemas.notify import NotificationAccepted, NotificationRequest
from orion.schemas.orion_day import ORION_DAY_TIMEZONE, ORION_DAY_WORKFLOW, orion_day_run_id

from scripts.orion_day_email import build_notification, letter_notification_id, load_inline_images

logger = logging.getLogger("orion-hub.orion_day_letter")

# Gmail clips an HTML body past ~102 KB behind "View entire message" (content is kept).
GMAIL_CLIP_BYTES = 102_000

TERMINAL_RETRY = frozenset({"failed", "abandoned"})
TERMINAL_STOP = frozenset({"cancelled"})
TERMINAL_DONE = frozenset({"completed"})
TERMINAL = TERMINAL_RETRY | TERMINAL_STOP | TERMINAL_DONE

# Session advisory lock per letter day around the send: a second Hub on the same database
# (a worktree deploy, a dev Hub) cannot email the same letter concurrently.
EMAIL_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtext('orion_day_email:' || $1::text))"
EMAIL_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('orion_day_email:' || $1::text))"
MAX_SUBMIT_REFUSALS = 3

STAMP_EMAILED_SQL = """
UPDATE orion_day_letter
   SET emailed_at = now(), email_notification_id = $2
 WHERE letter_date = $1 AND emailed_at IS NULL
"""


def active_letter_date(now: datetime, *, hour: int, minute: int, tz_name: str = ORION_DAY_TIMEZONE) -> date:
    """The letter whose window contains ``now``: L is active from (L+1) HH:MM to (L+2) HH:MM local."""
    aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    local = aware.astimezone(ZoneInfo(tz_name))
    day = local.date()
    if local.time() < dtime(hour, minute):
        day -= timedelta(days=1)
    return day - timedelta(days=1)


def letter_slot(letter_date: date, *, hour: int, minute: int, tz_name: str = ORION_DAY_TIMEZONE) -> datetime:
    return datetime.combine(letter_date + timedelta(days=1), dtime(hour, minute), tzinfo=ZoneInfo(tz_name))


@dataclass(frozen=True)
class AttemptState:
    """The latest durable attempt for a letter date (None = no attempt yet)."""

    attempt: int
    status: str
    persist_outcome: str | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


class DurableUnavailable(RuntimeError):
    pass


class DurableRunsClient:
    """The two calls Hub needs from orion-durable-runs' HTTP API (services/orion-durable-runs/app/main.py)."""

    def __init__(self, base_url: str, *, timeout_sec: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_sec) as client:
                resp = await client.get(f"/runs/{run_id}")
        except httpx.HTTPError as exc:
            raise DurableUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise DurableUnavailable(f"http_{resp.status_code}")
        return resp.json()

    async def submit(self, request_json: dict[str, Any]) -> tuple[int, str]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_sec) as client:
                resp = await client.post("/runs", json=request_json)
        except httpx.HTTPError as exc:
            raise DurableUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return resp.status_code, resp.text[:300]


async def latest_attempt(client: Any, letter_date: date, *, max_probe: int) -> AttemptState | None:
    """Walk ``orion-day-<date>-1, -2, ...`` until a 404; the last one found is the latest."""
    latest: AttemptState | None = None
    for n in range(1, max_probe + 1):
        state = await client.get_run(orion_day_run_id(letter_date, n))
        if state is None:
            break
        day = state.get("orion_day") or {}
        latest = AttemptState(attempt=n, status=str(state.get("status") or ""),
                              persist_outcome=day.get("persist_outcome"), error=state.get("error"))
    return latest


class OrionDayLetterLoop:
    def __init__(
        self,
        *,
        enabled: bool,
        email_enabled: bool,
        pool_provider: Callable[[], Any],
        durable: Any,
        notify: Any,
        hour_local: int = 8,
        minute_local: int = 30,
        tz_name: str = ORION_DAY_TIMEZONE,
        tick_interval_sec: float = 300.0,
        max_attempts: int = 6,
        email_retry_sec: float = 1800.0,
        submit_refused_retry_sec: float = 3600.0,
        carry_forward_ttl_hours: float = 36.0,
        timeout_sec: float = 1800.0,
        image_dir: str = "/mnt/storage-lukewarm/orion/reverie-visual",
        max_images: int = 6,
        image_max_bytes: int = 450_000,
        source_service: str = "orion-hub",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.email_enabled = bool(email_enabled)
        self._pool_provider = pool_provider
        self.durable = durable
        self.notify = notify
        self.hour_local = int(hour_local)
        self.minute_local = int(minute_local)
        self.tz_name = tz_name
        self.tick_interval_sec = float(tick_interval_sec)
        self.max_attempts = max(1, int(max_attempts))
        self.email_retry_sec = float(email_retry_sec)
        self.submit_refused_retry_sec = float(submit_refused_retry_sec)
        self.carry_forward_ttl_hours = float(carry_forward_ttl_hours)
        self.timeout_sec = float(timeout_sec)
        self.image_dir = image_dir
        self.max_images = max(0, int(max_images))
        self.image_max_bytes = int(image_max_bytes)
        self.source_service = source_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._hook_tasks: set[asyncio.Task] = set()
        # Log-only / spam guards. Losing them on restart costs one repeat, never a wrong action.
        self._empty_dates: set[date] = set()
        self._exhausted_noticed: set[date] = set()
        self._email_retry_after: dict[date, float] = {}
        self._submit_retry_after: dict[date, float] = {}
        self._submit_refusals: dict[date, int] = {}
        # Dates whose email outcome is unknown (reply lost after a possible send, or the stamp
        # failed after `sent`): never resent automatically -- a duplicate letter is worse.
        self._email_outcome_unknown: set[date] = set()

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled:
            logger.info("orion_day_letter disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("orion_day_letter started slot=%02d:%02d %s tick=%ss max_attempts=%s email=%s",
                    self.hour_local, self.minute_local, self.tz_name, self.tick_interval_sec,
                    self.max_attempts, self.email_enabled)

    async def stop(self) -> None:
        tasks = [t for t in (self._task, *self._hook_tasks) if t is not None]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("orion_day_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def on_run_state(self, state: Any) -> None:
        """Hook for Hub's durable run-state listener. Never blocks that listener: a terminal
        ``orion_day`` state schedules one tick in the background."""
        if not self.enabled or getattr(state, "workflow", None) != ORION_DAY_WORKFLOW:
            return
        if getattr(state, "status", None) not in TERMINAL:
            return
        detail = getattr(state, "detail", None) or {}
        logger.info("orion_day_run_terminal run=%s status=%s persist_outcome=%s",
                    state.run_id, state.status, detail.get("persist_outcome"))
        task = asyncio.create_task(self._hook_tick())
        self._hook_tasks.add(task)
        task.add_done_callback(self._hook_tasks.discard)

    async def _hook_tick(self) -> None:
        try:
            await self.tick()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("orion_day_hook_tick_failed")

    def hard_stop(self, letter_date: date) -> datetime:
        """When letter L stops being active: the next letter's slot."""
        return letter_slot(letter_date + timedelta(days=1), hour=self.hour_local,
                           minute=self.minute_local, tz_name=self.tz_name)

    def _mono_until(self, when: datetime) -> float:
        return time.monotonic() + max(0.0, (when - self._clock()).total_seconds())

    # --- the tick ----------------------------------------------------------------------------

    async def tick(self) -> str:
        if not self.enabled:
            return "disabled"
        async with self._lock:
            return await self._tick()

    async def _tick(self) -> str:
        now = self._clock()
        letter_date = active_letter_date(now, hour=self.hour_local, minute=self.minute_local, tz_name=self.tz_name)
        pool = self._pool_provider()
        if pool is None:
            return "no_pool"
        try:
            async with pool.acquire() as conn:
                letter = await fetch_letter(conn, letter_date)
        except Exception as exc:  # noqa: BLE001 -- missing table (migration not applied), DB down
            logger.warning("orion_day_store_unavailable date=%s err=%s", letter_date, exc)
            return "store_unavailable"
        if letter is not None:
            if letter.emailed_at is not None:
                return "quota_met"
            return await self._send(letter)
        if letter_date in self._empty_dates:
            return "empty_day"
        retry_after = self._submit_retry_after.get(letter_date)
        if retry_after is not None and time.monotonic() < retry_after:
            return "submit_backoff"
        try:
            latest = await latest_attempt(self.durable, letter_date, max_probe=self.max_attempts + 1)
        except DurableUnavailable as exc:
            logger.warning("orion_day_durable_unavailable date=%s err=%s", letter_date, exc)
            return "durable_unavailable"
        if latest is None:
            return await self._submit(letter_date, 1)
        if not latest.terminal:
            return "run_live"
        if latest.status in TERMINAL_STOP:
            logger.info("orion_day_cancelled_by_operator date=%s attempt=%s -- not retrying",
                        letter_date, latest.attempt)
            return "cancelled"
        if latest.status in TERMINAL_DONE:
            # Completed runs persist the row; re-read in case it landed after the first read.
            async with pool.acquire() as conn:
                letter = await fetch_letter(conn, letter_date)
            if letter is not None:
                return "quota_met" if letter.emailed_at is not None else await self._send(letter)
            logger.warning("orion_day_completed_without_row date=%s attempt=%s persist_outcome=%s",
                           letter_date, latest.attempt, latest.persist_outcome)
        if latest.attempt >= self.max_attempts:
            await self._notice_exhausted(letter_date, latest)
            return "attempts_exhausted"
        logger.info("orion_day_retry date=%s after attempt=%s status=%s err=%s",
                    letter_date, latest.attempt, latest.status, (latest.error or "")[:200])
        return await self._submit(letter_date, latest.attempt + 1)

    async def _submit(self, letter_date: date, attempt: int) -> str:
        pool = self._pool_provider()
        try:
            async with pool.acquire() as conn:
                brief = await build_orion_day_brief(
                    conn, letter_date, now=self._clock(), timeout_sec=self.timeout_sec,
                    carry_forward_ttl_hours=self.carry_forward_ttl_hours,
                )
        except OrionDayEmptyError:
            self._empty_dates.add(letter_date)
            logger.info("orion_day_empty date=%s -- no letter, no email", letter_date)
            return "empty_day"
        # Deadline = Hub's own hard stop. The builder's default (local midnight starting L+2)
        # ends before it, and a run abandoned there would be resubmitted with a past deadline
        # and abandoned again until max_attempts.
        request = build_orion_day_request(brief, attempt=attempt,
                                          deadline_at=self.hard_stop(letter_date).astimezone(timezone.utc))
        try:
            status, body = await self.durable.submit(request.model_dump(mode="json"))
        except DurableUnavailable as exc:
            logger.warning("orion_day_submit_unavailable run=%s err=%s", request.run_id, exc)
            return "durable_unavailable"
        if status >= 400:
            # 409 = same run_id with a different brief; 422 = durable-runs without the
            # orion_day.letter workflow. Neither heals on the next tick; wait before regathering.
            self._submit_retry_after[letter_date] = time.monotonic() + self.submit_refused_retry_sec
            refusals = self._submit_refusals.get(letter_date, 0) + 1
            self._submit_refusals[letter_date] = refusals
            if refusals >= MAX_SUBMIT_REFUSALS:
                await self._notice_exhausted(letter_date, AttemptState(
                    attempt=attempt - 1, status=f"submit_refused_http_{status}", error=body))
            logger.warning("orion_day_submit_refused run=%s http=%s body=%s", request.run_id, status, body)
            return "submit_refused"
        logger.info("orion_day_submitted run=%s attempt=%s view_tokens=%s",
                    request.run_id, attempt, brief.llm_view.approx_tokens)
        return "submitted"

    async def _send(self, letter) -> str:
        if not self.email_enabled:
            return "email_disabled"
        day = letter.letter_date
        if day in self._email_outcome_unknown:
            return "email_outcome_unknown"
        retry_after = self._email_retry_after.get(day)
        if retry_after is not None and time.monotonic() < retry_after:
            return "email_backoff"
        pool = self._pool_provider()
        async with pool.acquire() as conn:
            if not await conn.fetchval(EMAIL_LOCK_SQL, str(day)):
                return "email_locked"
            try:
                return await self._send_locked(conn, letter)
            finally:
                await conn.fetchval(EMAIL_UNLOCK_SQL, str(day))

    def _build_request(self, letter):
        images = load_inline_images(letter, storage_dir=self.image_dir,
                                    max_images=self.max_images, max_bytes=self.image_max_bytes)
        return build_notification(letter, images, source_service=self.source_service), images

    async def _send_locked(self, conn, letter) -> str:
        day = letter.letter_date
        # Re-read under the lock: another process may have sent it since our first read.
        fresh = await fetch_letter(conn, day)
        if fresh is not None and fresh.emailed_at is not None:
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
                # Policy declined or SMTP not configured: no retry today can change that.
                self._email_retry_after[day] = self._mono_until(self.hard_stop(day))
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

    async def _notice_exhausted(self, letter_date: date, latest: AttemptState) -> None:
        if letter_date in self._exhausted_noticed:
            return
        self._exhausted_noticed.add(letter_date)
        logger.error("orion_day_attempts_exhausted date=%s attempts=%s last_status=%s err=%s",
                     letter_date, latest.attempt, latest.status, latest.error)
        request = NotificationRequest(
            notification_id=letter_notification_id(f"exhausted:{letter_date}"),
            source_service=self.source_service,
            event_kind="orion_day.letter.exhausted",
            severity="warning",
            title=f"Orion's Day for {letter_date} did not get written",
            body_text=(f"{latest.attempt} durable attempts for the {letter_date} letter ended without a "
                       f"letter. Last status: {latest.status}. Last error: {latest.error or 'none recorded'}. "
                       f"Hub stops retrying for this day; run ids orion-day-{letter_date}-1..{latest.attempt}."),
            context={"letter_date": str(letter_date), "attempts": latest.attempt, "last_status": latest.status},
            tags=["orion_day"],
            channels_requested=["in_app"],
            dedupe_key=f"orion_day:exhausted:{letter_date}",
        )
        await asyncio.to_thread(self.notify.send, request)
