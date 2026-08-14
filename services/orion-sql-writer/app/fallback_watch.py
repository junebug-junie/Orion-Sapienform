"""Watcher: alert when the ``bus_fallback_log`` backlog crosses 5, 10, 15, ...

## Why this exists

``bus_fallback_log`` is where every event that ``handle_envelope`` could not
route ends up. It is the safety net that means a routing mistake loses the
*destination* rather than the *data* -- and until now nothing watched it.

Twice on 2026-08-13/14 this service ran for hours writing live events there
instead of to their tables. Both times everything looked healthy: container up,
producer ticking, ``PUBSUB NUMSUB`` reporting a subscriber, one WARNING per
event and no error. Both were found by someone happening to run a row count.

``app/route_coverage.py`` (PR #1648) closed half of that: it compares the
subscribe list against the route map at startup, so a *config drift* between a
durable ``.env`` and the code is now loud at boot. It cannot see the other half.
A kind that arrives on a subscribed channel but was never in the route map at
all is invisible to it -- there is no drift to detect, the config is
self-consistent and simply incomplete. That is not hypothetical: ``legacy.message``
has been landing in the fallback log since at least 2026-07-24 (80 of the 87 rows
present when this was written), carrying ``prompt`` / ``response`` /
``reasoning_trace`` payloads, and route coverage reports all-clear.

So: coverage catches drift at boot, this catches arrival at runtime. Neither
subsumes the other.

## The escalation rule

Thresholds are multiples of ``step`` (5, 10, 15, ...). The alert fires when the
windowed count first *reaches* a multiple higher than the highest already
alerted, and the high-water mark ratchets back down as the backlog drains, which
re-arms the lower levels. Consequences worth being explicit about:

- One alert per level, not one per poll. A backlog sitting at 7 for a week is
  one email, not 2,016 of them.
- A jump straight past several levels sends ONE alert naming the level actually
  reached, not one per level skipped.
- Recovery is what re-arms. A backlog that drains below ``step`` and climbs again
  alerts again -- that is a new incident, not a repeat of the old one.

``count >= threshold`` rather than a literal ``count > 5``. Off-by-one in the
noisy direction: the alert names the level it reached, and reaching 5 is what
"above 5" means to the person who asked for it.

## Severity, and why it is `error`

Read off the live policy (``services/orion-notify/app/policy/rules.yaml``), not
chosen by feel:

- ``error`` and ``critical`` -> ``channels: ["email", "in_app"]``
- ``warning`` -> ``in_app`` only, no email
- ``info`` -> ``channels: []``, delivered nowhere

and ``Policy.evaluate`` drops channels to ``[]`` during quiet hours (22:00-07:00
America/Denver) for anything that is not ``critical`` or ``error``. Hub *and*
email, at any hour, therefore means ``error`` at minimum. Deliberately not
escalating to ``critical`` at higher thresholds: it buys no additional channel
and no additional urgency in the policy, so it would be decoration.

## Privacy

The alert carries kinds and counts ONLY -- never payloads. Fallback rows hold
whatever the undelivered event held, and for the largest current contributor
that is Orion's own prompts, responses, and reasoning traces. Those do not
belong in an email or a Hub card. The alert says "``legacy.message`` x8"; the
operator goes and looks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("sql-writer.fallback_watch")

STATE_ROW_ID = 1

# Kinds shown by name in the alert body. Enough to identify the culprit at a
# glance; bounded so a pathological spread of kinds cannot produce a
# thousand-line email.
_MAX_KINDS_IN_BODY = 10


def reached_threshold(count: int, step: int) -> int:
    """Highest multiple of ``step`` that ``count`` has reached (0 if none).

    Pure. This and :func:`next_alert_threshold` hold the entire escalation
    rule, with no clock, database, or network in reach, so the behaviour that
    decides whether Juniper gets an email is testable directly.
    """
    if step <= 0 or count < step:
        return 0
    return (count // step) * step


def next_alert_threshold(count: int, step: int, last_alerted: int) -> tuple[int | None, int]:
    """``(threshold_to_alert_or_None, new_high_water_mark)``.

    The second element is returned even when no alert fires, because the
    ratchet-down on recovery is exactly as important as the alert itself: it is
    what re-arms the lower levels. Folding it into the same call keeps the
    caller from persisting one without the other.
    """
    reached = reached_threshold(count, step)
    if reached > last_alerted:
        return reached, reached
    return None, reached


def format_alert(count: int, threshold: int, window_seconds: int, by_kind: list[tuple[str, int]]) -> tuple[str, str]:
    """``(title, body)`` for the notification. Kinds and counts only -- no payloads."""
    hours = window_seconds / 3600.0
    window_label = f"{hours:.0f}h" if hours >= 1 and hours == int(hours) else f"{window_seconds}s"

    title = f"[Orion] Bus fallback backlog crossed {threshold} ({count} in {window_label})"

    lines = [
        f"{count} event(s) landed in bus_fallback_log in the last {window_label}, "
        f"crossing the {threshold} threshold.",
        "",
        "These are events orion-sql-writer received but had no route for. The payloads are "
        "preserved in the table; their destination tables never got them.",
        "",
        "By kind:",
    ]
    shown = by_kind[:_MAX_KINDS_IN_BODY]
    for kind, kind_count in shown:
        lines.append(f"  {kind}: {kind_count}")
    remaining = len(by_kind) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more kind(s)")

    lines += [
        "",
        "Payloads are deliberately not included -- fallback rows can contain prompts, "
        "responses, and reasoning traces.",
        "",
        "To look:",
        "  SELECT kind, count(*) FROM bus_fallback_log",
        f"  WHERE created_at_ts > now() - interval '{window_seconds} seconds'",
        "  GROUP BY 1 ORDER BY 2 DESC;",
        "",
        "Two usual causes: this service is running an image that predates a route "
        "(check the route_coverage_missing lines at startup), or the kind was never "
        "routed at all and nobody noticed.",
    ]
    return title, "\n".join(lines)


def count_backlog(session, window_seconds: int, now: datetime | None = None) -> tuple[int, list[tuple[str, int]]]:
    """``(total, [(kind, count)])`` over the trailing window, busiest kind first.

    Counts on ``created_at_ts`` (the real timestamptz column), not ``created_at``
    -- that one is declared ``String`` with a ``func.now()`` default and cannot
    be range-scanned. Rows with a NULL ``created_at_ts`` are therefore invisible
    here; ``_write_fallback`` has always set it explicitly (``worker.py``), and
    there are zero such rows live, but a row inserted by any other path would be
    missed rather than counted as recent.
    """
    from sqlalchemy import func as sa_func

    from app.models.fallback_log import BusFallbackLog

    now = now or datetime.now(timezone.utc)
    since = now - timedelta(seconds=window_seconds)

    rows = (
        session.query(BusFallbackLog.kind, sa_func.count().label("n"))
        .filter(BusFallbackLog.created_at_ts.isnot(None))
        .filter(BusFallbackLog.created_at_ts > since)
        .group_by(BusFallbackLog.kind)
        .all()
    )
    by_kind = sorted(((r[0] or "<unknown>", int(r[1])) for r in rows), key=lambda kv: (-kv[1], kv[0]))
    return sum(n for _, n in by_kind), by_kind


def _load_state(session):
    from app.models.fallback_alert_state import BusFallbackAlertState

    state = session.get(BusFallbackAlertState, STATE_ROW_ID)
    if state is None:
        state = BusFallbackAlertState(id=STATE_ROW_ID, last_alerted_threshold=0)
        session.add(state)
        session.flush()
    return state


def _send_alert(settings, title: str, body: str, threshold: int, count: int, window_seconds: int) -> str:
    """Post to the notify service. Returns a short status string for the state row.

    Never raises: a monitoring path that can take down the writer it monitors is
    a worse bug than the one it reports.
    """
    from orion.notify.client import NotifyClient
    from orion.schemas.notify import NotificationRequest

    base_url = (settings.notify_service_url or "").strip()
    if not base_url:
        logger.error(
            "fallback_watch_notify_unconfigured threshold=%d count=%d -- NOTIFY_SERVICE_URL is "
            "empty, so this crossing is logged and nothing else",
            threshold,
            count,
        )
        return "unconfigured"

    request = NotificationRequest(
        source_service="orion-sql-writer",
        event_kind="sqlwriter.fallback_backlog",
        # See module docstring: `error` is the lowest severity the live policy
        # routes to BOTH email and in_app, and the lowest that survives quiet hours.
        severity="error",
        title=title,
        body_text=body,
        context={
            "threshold": threshold,
            "count": count,
            "window_seconds": window_seconds,
        },
        tags=["sql-writer", "bus-fallback", "backlog"],
        recipient_group="juniper_primary",
        # Recorded for whenever notify learns to honour it; it does NOT suppress
        # anything today (see BusFallbackAlertState). The real dedupe is the
        # persisted high-water mark.
        dedupe_key=f"bus-fallback-backlog:{threshold}",
        dedupe_window_seconds=window_seconds,
    )

    try:
        client = NotifyClient(
            base_url=base_url,
            api_token=(settings.notify_api_token or "").strip() or None,
        )
        response = client.send(request)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("fallback_watch_notify_raised threshold=%d error=%s", threshold, exc)
        return "error"

    if getattr(response, "ok", False):
        logger.warning(
            "fallback_watch_alert_sent threshold=%d count=%d window_sec=%d",
            threshold,
            count,
            window_seconds,
        )
        return "sent"

    logger.error(
        "fallback_watch_alert_failed threshold=%d count=%d detail=%s",
        threshold,
        count,
        getattr(response, "detail", None),
    )
    return "failed"


def evaluate_once(settings, session_factory, now: datetime | None = None) -> dict:
    """One evaluation. Returns a summary dict (used by tests and the log line).

    The high-water mark is committed whether or not an alert was sent, and the
    send happens BEFORE the commit is finalised only in the sense that a failed
    send still advances the mark. That is deliberate: retrying a failed send on
    the next poll would turn one unreachable notify service into an alert every
    ``interval`` for as long as it stays down. The failure is recorded in
    ``last_alert_status`` and shouted at ERROR instead.
    """
    from app.db import remove_session

    session = session_factory()
    try:
        count, by_kind = count_backlog(session, settings.sql_writer_fallback_watch_window_sec, now=now)
        state = _load_state(session)
        step = int(settings.sql_writer_fallback_watch_threshold_step)
        previous = int(state.last_alerted_threshold or 0)

        threshold, new_mark = next_alert_threshold(count, step, previous)

        state.last_count = count
        state.last_evaluated_at = now or datetime.now(timezone.utc)
        state.last_alerted_threshold = new_mark

        status = None
        if threshold is not None:
            title, body = format_alert(
                count, threshold, settings.sql_writer_fallback_watch_window_sec, by_kind
            )
            status = _send_alert(
                settings, title, body, threshold, count, settings.sql_writer_fallback_watch_window_sec
            )
            state.last_alert_sent_at = state.last_evaluated_at
            state.last_alert_status = status
        elif new_mark < previous:
            logger.info(
                "fallback_watch_rearmed count=%d high_water=%d previous=%d", count, new_mark, previous
            )

        session.commit()
        return {
            "count": count,
            "threshold": threshold,
            "high_water": new_mark,
            "previous": previous,
            "status": status,
            "by_kind": by_kind,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        try:
            session.close()
        finally:
            remove_session()


async def fallback_watch_loop(settings, session_factory=None) -> None:
    """Poll forever. Started from ``main.py``'s lifespan when enabled."""
    if session_factory is None:
        from app.db import get_session

        session_factory = get_session

    interval = max(30, int(settings.sql_writer_fallback_watch_interval_sec))
    logger.info(
        "fallback_watch_started interval_sec=%d window_sec=%d step=%d",
        interval,
        settings.sql_writer_fallback_watch_window_sec,
        settings.sql_writer_fallback_watch_threshold_step,
    )

    while True:
        # Sleep first: at startup the schema may still be being created, and an
        # immediate evaluation buys nothing a poll later does not.
        await asyncio.sleep(interval)
        try:
            result = await asyncio.to_thread(evaluate_once, settings, session_factory)
            logger.info(
                "fallback_watch_tick count=%d high_water=%d alerted=%s",
                result["count"],
                result["high_water"],
                result["threshold"],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never let a monitoring failure kill the loop -- a watcher that dies
            # on the first bad poll is worse than no watcher, because its silence
            # reads as "nothing wrong".
            logger.error("fallback_watch_tick_failed error=%s", exc)
