from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class BusFallbackAlertState(Base):
    """Single-row alert bookkeeping for the ``bus_fallback_log`` backlog watcher.

    Holds the highest threshold already alerted on, so a rising backlog produces
    ONE alert per level crossed (5, then 10, then 15, ...) rather than one per
    poll. See ``app/fallback_watch.py`` for the escalation rule itself.

    Why this is a durable table and not an in-process variable
    ---------------------------------------------------------
    ``orion-sql-writer`` restarts often -- every deploy, and this repo has 100+
    active worktrees any of which may redeploy it. In-memory state would reset
    the high-water mark on every restart and re-send every threshold alert that
    had already been sent, which is precisely the alert fatigue that trains
    people to filter these mails into a folder they never read.

    It cannot lean on the notify service for that instead: ``dedupe_key`` and
    ``dedupe_window_seconds`` are accepted by ``NotificationRequest`` and stored
    on the record, but nothing in ``services/orion-notify`` ever reads them back
    to suppress a duplicate -- verified by search, 2026-08-14. The policy rules
    carry ``dedupe_window_seconds: 60`` on the critical and error paths, and it
    does nothing today. This row is the only real dedupe in the path.

    Single row by construction (``id`` pinned to 1). Not a log -- the history of
    what fired lives in the notify service's own records.
    """

    __tablename__ = "bus_fallback_alert_state"

    # Pinned to STATE_ROW_ID by every writer; a PK rather than a bare column so
    # a second concurrent writer collides loudly instead of silently forking
    # the high-water mark into two rows that each under-report.
    id = Column(Integer, primary_key=True)

    # Highest threshold already alerted on, in rows. 0 means "armed, nothing
    # sent yet". Ratchets DOWN as the backlog drains (see fallback_watch.
    # next_alert_threshold) so recovery re-arms the lower levels.
    last_alerted_threshold = Column(Integer, nullable=False, default=0)

    # Purely diagnostic: what the last evaluation actually saw. Lets an operator
    # answer "is the watcher running, and what does it think the backlog is?"
    # with one query, without reading logs.
    last_count = Column(Integer, nullable=True)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)
    last_alert_sent_at = Column(DateTime(timezone=True), nullable=True)
    last_alert_status = Column(String, nullable=True)

    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
