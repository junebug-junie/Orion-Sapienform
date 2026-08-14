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
- Only FULL recovery re-arms. A backlog that drains below ``step`` and climbs
  again alerts again -- that is a new incident. A backlog that merely dips from
  15 to 11 and comes back does not, because the window is trailing and that
  happens constantly on its own. See :func:`next_alert_threshold`.

``count >= threshold`` rather than a literal ``count > 5``. Off-by-one in the
noisy direction: the alert names the level it reached, and reaching 5 is what
"above 5" means to the person who asked for it.

## Severity, and why it is `error`

Traced through the handler this actually calls -- ``POST /notify``
(``services/orion-notify/app/main.py:194``) -- rather than through the policy
file, which looks authoritative and is not on this path. An earlier version of
this docstring justified the choice from ``rules.yaml`` and was wrong about the
mechanism while right about the answer:

- **email** is gated ONLY by ``should_send_email()``
  (``app/email_delivery.py:15``): true when severity is ``error`` or
  ``critical``, or when ``channels_requested`` contains ``email``.
- **in_app** is published unconditionally at ``main.py:233`` whenever the bus
  and ``NOTIFY_IN_APP_ENABLED`` are up -- "we do this blindly as a router",
  per the comment there. Severity is not consulted.
- ``Policy.evaluate`` -- and therefore the ``channels`` lists, the
  ``throttle: max_per_window`` blocks, and quiet hours -- runs only on
  ``/attention/request`` (``main.py:274``). None of it applies here.

So ``error`` is what makes the email go out, and it is also the floor that
survives quiet hours on the paths where quiet hours exist. Both
``channels_requested`` and the severity are set, so the alert survives an edit
to either one. Deliberately not escalating to ``critical`` at higher thresholds:
it buys no additional channel on this path, so it would be decoration.

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
    re-arm on recovery is exactly as important as the alert itself. Folding it
    into the same call keeps the caller from persisting one without the other.

    Full recovery is the ONLY thing that re-arms
    --------------------------------------------
    The mark drops back to 0 when the count falls below ``step`` entirely. It
    does NOT track the count downward level by level.

    The obvious implementation -- ratchet the mark down to whatever level the
    count currently sits at -- looks equivalent and is not, because the count is
    taken over a TRAILING window and therefore moves in both directions all day
    as rows age out. A backlog oscillating across a level boundary would re-arm
    and re-alert that same level on every crossing.

    Not theoretical. Review replayed these functions over the real 87
    ``created_at_ts`` values in the live ``bus_fallback_log``, at the shipped
    300s/86400s/step-5 settings: 12 alerts across 20 days, six of them "crossed
    5" and three "crossed 10". A count alternating 14/15 between polls produced
    an alert every other poll -- roughly 144 emails a day. That is precisely the
    alert fatigue this module's docstring claims to prevent, and the first
    version of this function shipped it.
    """
    reached = reached_threshold(count, step)
    if reached > last_alerted:
        return reached, reached
    if reached == 0:
        # Genuinely drained: below the first threshold. Re-arm everything, so a
        # later climb is reported as the new incident it is.
        return None, 0
    # Still elevated, just not at a new high. Hold the mark -- a dip from 15 to
    # 11 and back is one incident, not two.
    return None, last_alerted


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


def notify_is_configured(settings) -> bool:
    return bool((getattr(settings, "notify_service_url", "") or "").strip())


def build_alert_request(title: str, body: str, threshold: int, count: int, window_seconds: int):
    """The outbound ``NotificationRequest``. Split out so the delivery-critical
    fields can be asserted against the real ``should_send_email()`` gate without
    a network call -- a test on the source text would not have caught a severity
    the gate rejects."""
    from orion.schemas.notify import NotificationRequest

    return NotificationRequest(
        source_service="orion-sql-writer",
        event_kind="sqlwriter.fallback_backlog",
        # See module docstring: `error` is what should_send_email() gates the
        # email on, and the floor that survives quiet hours where they apply.
        severity="error",
        title=title,
        body_text=body,
        # Three integers. No payload, no correlation_id -- see the privacy note
        # in the module docstring.
        context={
            "threshold": threshold,
            "count": count,
            "window_seconds": window_seconds,
        },
        tags=["sql-writer", "bus-fallback", "backlog"],
        recipient_group="juniper_primary",
        # Belt and braces with the severity above: should_send_email returns
        # true for EITHER an explicit email channel or severity in
        # {error, critical}, so the mail survives a later "let's be less
        # alarmist" edit to one of them.
        channels_requested=["email", "in_app"],
        # Recorded for whenever notify learns to honour it; it does NOT suppress
        # anything today (see BusFallbackAlertState). The real dedupe is the
        # persisted high-water mark.
        dedupe_key=f"bus-fallback-backlog:{threshold}",
        dedupe_window_seconds=window_seconds,
    )


def _send_alert(settings, title: str, body: str, threshold: int, count: int, window_seconds: int) -> str:
    """Post to the notify service. Returns a short status string for the state row.

    Never raises. Everything is inside the try -- including the imports and the
    request construction, which an earlier version left outside it. That gap was
    not hypothetical: ``orion/notify/client.py`` imports ``requests`` at module
    level and this service's requirements.txt did not list it, so the very first
    alert would have raised ``ModuleNotFoundError`` from the import line,
    propagated through ``evaluate_once``'s ``rollback; raise``, and discarded the
    diagnostic state write along with the alert -- 288 ERROR lines a day, no
    alerts, and an empty state row implying the watcher had never run.
    """
    try:
        from orion.notify.client import NotifyClient

        request = build_alert_request(title, body, threshold, count, window_seconds)
        client = NotifyClient(
            base_url=settings.notify_service_url.strip(),
            api_token=(settings.notify_api_token or "").strip() or None,
        )
        response = client.send(request)
    except Exception as exc:
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

    Ordering matters here in two ways.

    The mark is committed BEFORE the HTTP send, and a failed send does not roll
    it back. Retrying on the next poll would turn one unreachable notify service
    into an alert attempt every ``interval`` for as long as it stays down, then
    deliver the whole backlog of them the moment it recovered. The failure is
    recorded in ``last_alert_status`` and shouted at ERROR instead.

    Committing first also releases the pool connection and the row lock on
    ``bus_fallback_alert_state`` before a blocking 10s HTTP call, rather than
    holding both across it.

    An UNCONFIGURED notify service is the one case that does not consume the
    crossing: there is no network cost to retrying, and silently burning the
    only alert for a level because ``NOTIFY_SERVICE_URL`` was empty at that
    moment would lose it permanently.
    """
    from app.db import remove_session

    window_sec = settings.sql_writer_fallback_watch_window_sec
    step = int(settings.sql_writer_fallback_watch_threshold_step)
    if step <= 0:
        logger.error(
            "fallback_watch_disabled_by_config step=%d -- SQL_WRITER_FALLBACK_WATCH_THRESHOLD_STEP "
            "must be positive; this watcher is inert until it is",
            step,
        )

    session = session_factory()
    try:
        count, by_kind = count_backlog(session, window_sec, now=now)
        state = _load_state(session)
        previous = int(state.last_alerted_threshold or 0)
        threshold, new_mark = next_alert_threshold(count, step, previous)

        configured = notify_is_configured(settings)
        if threshold is not None and not configured:
            logger.error(
                "fallback_watch_notify_unconfigured threshold=%d count=%d -- NOTIFY_SERVICE_URL is "
                "empty, so this crossing is logged and NOT consumed; it will alert once configured",
                threshold,
                count,
            )
            threshold, new_mark = None, previous

        evaluated_at = now or datetime.now(timezone.utc)
        state.last_count = count
        state.last_evaluated_at = evaluated_at
        state.last_alerted_threshold = new_mark
        if threshold is None and new_mark < previous:
            logger.info(
                "fallback_watch_rearmed count=%d high_water=%d previous=%d", count, new_mark, previous
            )
        session.commit()
    except Exception:
        session.rollback()
        session.close()
        remove_session()
        raise

    status = None
    if threshold is not None:
        title, body = format_alert(count, threshold, window_sec, by_kind)
        status = _send_alert(settings, title, body, threshold, count, window_sec)
        try:
            state = _load_state(session)
            state.last_alert_status = status
            # Only a real send stamps `last_alert_sent_at` -- a column named
            # "sent_at" that is set on failure lies to whoever queries it.
            if status == "sent":
                state.last_alert_sent_at = evaluated_at
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("fallback_watch_status_write_failed status=%s error=%s", status, exc)

    try:
        session.close()
    finally:
        remove_session()

    return {
        "count": count,
        "threshold": threshold,
        "high_water": new_mark,
        "previous": previous,
        "status": status,
        "by_kind": by_kind,
    }


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
