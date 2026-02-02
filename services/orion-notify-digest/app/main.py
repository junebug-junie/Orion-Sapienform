import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI

from orion.core.sql_router.db import init_models
from orion.notify.client import NotifyClient
from orion.schemas.notify import NotificationRequest

from .db_models import DigestRunDB, NotificationAttemptDB, NotificationRequestDB
from .digest import (
    build_digest_content,
    fetch_topics_snapshot,
    summarize_window,
    window_bounds,
)
from .settings import settings

logger = logging.getLogger("orion-notify-digest")

app = FastAPI(
    title="Orion Notify Digest",
    version=settings.SERVICE_VERSION,
)


@app.on_event("startup")
async def on_startup() -> None:
    init_models([DigestRunDB, NotificationRequestDB, NotificationAttemptDB])
    if settings.DIGEST_ENABLED:
        app.state.scheduler_task = asyncio.create_task(_scheduler_loop())
    if settings.DRIFT_ALERTS_ENABLED:
        app.state.drift_task = asyncio.create_task(_drift_alert_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    task = getattr(app.state, "scheduler_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    drift_task = getattr(app.state, "drift_task", None)
    if drift_task:
        drift_task.cancel()
        try:
            await drift_task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "digest_enabled": settings.DIGEST_ENABLED,
    }


async def _scheduler_loop() -> None:
    if settings.DIGEST_RUN_ON_START:
        await run_digest_now(window_hours=settings.DIGEST_WINDOW_HOURS, kind="daily")

    while True:
        now = datetime.utcnow()
        next_run = _next_run_utc(now, settings.DIGEST_TIME_LOCAL)
        sleep_for = max(0, (next_run - now).total_seconds())
        logger.info("Next digest run scheduled at %s (in %.0fs)", next_run.isoformat(), sleep_for)
        await asyncio.sleep(sleep_for)
        await run_digest_now(
            window_hours=settings.DIGEST_WINDOW_HOURS,
            kind="daily",
            window_end=next_run,
        )


async def run_digest_now(
    window_hours: int,
    kind: str,
    window_end: Optional[datetime] = None,
) -> Optional[str]:
    now = datetime.utcnow()
    window_end = window_end or now
    window_start, window_end = window_bounds(window_end, window_hours)
    if _digest_already_sent(kind, window_start, window_end):
        logger.info("Digest already sent for window %s - %s", window_start, window_end)
        return None

    digest_summary = summarize_window(window_start, window_end)
    topics_snapshot = fetch_topics_snapshot(
        landing_pad_url=settings.LANDING_PAD_URL,
        window_minutes=_topics_window_minutes(),
        max_topics=settings.TOPICS_MAX_TOPICS,
        drift_min_turns=settings.TOPICS_DRIFT_MIN_TURNS,
        drift_max_sessions=settings.TOPICS_DRIFT_MAX_SESSIONS,
    )
    body_text, body_md = build_digest_content(
        digest_summary,
        topics_snapshot=topics_snapshot,
        drift_max_items=settings.DRIFT_ALERT_MAX_ITEMS,
    )

    title_date = _local_date(window_end).isoformat()
    request = NotificationRequest(
        source_service=settings.SERVICE_NAME,
        event_kind="orion.digest.daily",
        severity="info",
        title=f"Orion Daily Digest — {title_date}",
        body_text=body_text,
        body_md=body_md,
        tags=["digest", "daily"],
        recipient_group=settings.DIGEST_RECIPIENT_GROUP,
        channels_requested=["email"],
        dedupe_key=f"digest:daily:{title_date}",
    )

    client = NotifyClient(base_url=settings.NOTIFY_SERVICE_URL, api_token=settings.NOTIFY_API_TOKEN)
    response = client.send(request)

    status = "sent" if response.ok else "failed"
    _record_digest_run(kind, window_start, window_end, status)

    if response.ok:
        logger.info("Digest sent: %s", response.notification_id)
        return str(response.notification_id)

    logger.error("Digest send failed: %s", response.detail)
    return None


def _digest_already_sent(kind: str, window_start: datetime, window_end: datetime) -> bool:
    from orion.core.sql_router.db import SessionLocal

    with SessionLocal() as db:
        existing = (
            db.query(DigestRunDB)
            .filter(
                DigestRunDB.kind == kind,
                DigestRunDB.window_start == window_start,
                DigestRunDB.window_end == window_end,
                DigestRunDB.status == "sent",
            )
            .first()
        )
        return existing is not None


def _record_digest_run(kind: str, window_start: datetime, window_end: datetime, status: str) -> None:
    from orion.core.sql_router.db import SessionLocal

    with SessionLocal() as db:
        record = DigestRunDB(
            kind=kind,
            window_start=window_start,
            window_end=window_end,
            status=status,
        )
        db.add(record)
        db.commit()


def _next_run_utc(now_utc: datetime, local_time: str) -> datetime:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Denver")
    local_now = now_utc.astimezone(tz)
    hour, minute = _parse_time(local_time)
    scheduled_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if scheduled_local <= local_now:
        scheduled_local = scheduled_local + timedelta(days=1)
    return scheduled_local.astimezone(ZoneInfo("UTC"))


def _local_date(timestamp_utc: datetime) -> datetime.date:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Denver")
    return timestamp_utc.astimezone(tz).date()


def _parse_time(value: str) -> tuple[int, int]:
    try:
        hour, minute = value.split(":", 1)
        return int(hour), int(minute)
    except ValueError:
        return (7, 30)


def _topics_window_minutes() -> int:
    if settings.TOPICS_WINDOW_MINUTES:
        return settings.TOPICS_WINDOW_MINUTES
    return settings.DIGEST_WINDOW_HOURS * 60


async def _drift_alert_loop() -> None:
    await asyncio.sleep(1)
    while True:
        try:
            await _check_topic_drift()
        except Exception as exc:
            logger.warning("Drift alert check failed: %s", exc, exc_info=True)
        await asyncio.sleep(max(settings.DRIFT_CHECK_INTERVAL_SECONDS, 30))


async def _check_topic_drift() -> None:
    if not settings.LANDING_PAD_URL:
        logger.info("Drift alerts skipped: LANDING_PAD_URL not set")
        return
    if settings.DRIFT_ALERT_SEVERITY not in {"warning", "error"}:
        logger.warning("Invalid DRIFT_ALERT_SEVERITY=%s; expected warning|error", settings.DRIFT_ALERT_SEVERITY)
        return

    topics_snapshot = fetch_topics_snapshot(
        landing_pad_url=settings.LANDING_PAD_URL,
        window_minutes=_topics_window_minutes(),
        max_topics=settings.TOPICS_MAX_TOPICS,
        drift_min_turns=settings.TOPICS_DRIFT_MIN_TURNS,
        drift_max_sessions=settings.TOPICS_DRIFT_MAX_SESSIONS,
    )
    if not topics_snapshot.drift_items:
        logger.info("No drift items found for window=%s", topics_snapshot.window_minutes)
        return

    top_items = topics_snapshot.drift_items[: settings.DRIFT_ALERT_MAX_ITEMS]
    exceeds_threshold = any(item.score >= settings.DRIFT_ALERT_THRESHOLD for item in top_items)
    if not exceeds_threshold:
        logger.info("Drift below threshold=%.2f; skipping alert", settings.DRIFT_ALERT_THRESHOLD)
        return

    window_minutes = topics_snapshot.window_minutes
    topic_lines = [f"- {item.label}: {item.score:.2f}" for item in top_items]
    hub_hint = "/topics"
    body_text = "\n".join(
        [
            "Topic drift exceeded threshold.",
            f"Window (minutes): {window_minutes}",
            f"Threshold: {settings.DRIFT_ALERT_THRESHOLD:.2f}",
            "",
            "Top drifting topics:",
            *topic_lines,
            "",
            f"Hub: {hub_hint} (if available)",
        ]
    )

    title_date = _local_date(datetime.utcnow()).isoformat()
    request = NotificationRequest(
        source_service=settings.SERVICE_NAME,
        event_kind=settings.DRIFT_ALERT_EVENT_KIND,
        severity=settings.DRIFT_ALERT_SEVERITY,
        title="Topic Drift Detected",
        body_text=body_text,
        tags=["topics", "drift"],
        recipient_group=settings.DIGEST_RECIPIENT_GROUP,
        channels_requested=["in_app"],
        dedupe_key=f"topic_drift:{title_date}:{window_minutes}",
        dedupe_window_seconds=settings.DRIFT_ALERT_DEDUPE_WINDOW_SECONDS,
        context={"hub_path": hub_hint, "window_minutes": window_minutes},
    )

    client = NotifyClient(base_url=settings.NOTIFY_SERVICE_URL, api_token=settings.NOTIFY_API_TOKEN)
    response = client.send(request)
    if response.ok:
        logger.info("Drift alert sent: %s", response.notification_id)
    else:
        logger.warning("Drift alert failed: %s", response.detail)
