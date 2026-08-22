"""Object permanence: a persisted per-(stream, label) inventory, updated on a
timer, separate from the per-frame path.

**Why this cannot be event-triggered.** The council only re-interprets a
window when the observed LABEL SET changes (`reason=stable_scene` otherwise),
so a pure count change emits no `vision_events` row, and a departure is a
non-event by nature -- nothing fires when a thing stops being there. The only
way to notice "I have not seen the box in an hour" is something that wakes up
on a clock and asks, which is what this module and its loop wrapper are.

**The graduated threshold is the whole design.** A bare "gone for N minutes"
threshold cannot be right for both a coffee cup (there for ten minutes, gone
for one is genuinely gone) and a desk (there for three days, gone for one
sweep cycle is a detector miss, not a departure). `_absence_threshold_sec`
scales the grace period to how long the object was established before it
stopped being seen, bounded on both ends: a floor so a single missed sweep
cycle never reads as departure, a ceiling so nothing waits literally forever.

**Reads `counts`, the per-frame max already fixed in `orion-vision-window`'s
`projection.py`** -- never `label_detections`, which scales with frame rate
and would make "seen this window" depend on how many frames happened to land
in the sweep's lookback, not on what was actually there.

Pure functions only. No Postgres, no asyncio -- `vision_object_permanence_loop.py`
is the thin async wrapper that calls these from a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# Tuned, not derived: no live incident yet motivates a different number, so
# these start at round values and are meant to be revisited once real
# departures/arrivals have been observed and can be checked against what a
# person would actually call "gone".
DEFAULT_ABSENCE_FRACTION = 0.1
DEFAULT_MIN_ABSENCE_SEC = 3600.0     # >= 2 sweep cycles at the 1800s default
DEFAULT_MAX_ABSENCE_SEC = 24 * 3600.0


@dataclass(frozen=True)
class InventoryRow:
    stream_id: str
    label: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_count: int
    state: str            # "present" | "departed"
    state_since: datetime


@dataclass(frozen=True)
class Transition:
    """One human-legible thing that happened this sweep. Logged, not (yet)
    published -- see the module docstring in the loop wrapper for why a bus
    channel with no real consumer would be exactly the kind of concept
    AGENTS.md 0A calls junk."""

    stream_id: str
    label: str
    kind: str              # "arrived" | "departed" | "count_changed"
    detail: str


@dataclass(frozen=True)
class SweepResult:
    updated: Dict[str, InventoryRow]     # label -> new row, this stream's full state
    transitions: List[Transition]


def _absence_threshold_sec(
    established_sec: float,
    *,
    absence_fraction: float,
    min_absence_sec: float,
    max_absence_sec: float,
) -> float:
    raw = absence_fraction * max(0.0, established_sec)
    return min(max_absence_sec, max(min_absence_sec, raw))


def apply_sweep(
    *,
    stream_id: str,
    window_max_counts: Dict[str, int],
    existing: Dict[str, InventoryRow],
    now: datetime,
    absence_fraction: float = DEFAULT_ABSENCE_FRACTION,
    min_absence_sec: float = DEFAULT_MIN_ABSENCE_SEC,
    max_absence_sec: float = DEFAULT_MAX_ABSENCE_SEC,
) -> SweepResult:
    """One stream's sweep tick. Deterministic; the only external input besides
    the arguments is `now`, which callers must pass explicitly (this module
    never reads the clock itself, so it stays trivially testable)."""
    updated: Dict[str, InventoryRow] = {}
    transitions: List[Transition] = []

    all_labels = set(window_max_counts) | set(existing)
    for label in all_labels:
        seen_this_window = label in window_max_counts
        prior = existing.get(label)

        if seen_this_window:
            count = window_max_counts[label]
            if prior is None or prior.state == "departed":
                # First-ever sighting, or a comeback after being marked gone.
                # A comeback keeps its ORIGINAL first_seen_at only if the
                # label row still exists (state='departed' rows are not
                # deleted -- see the loop wrapper) -- otherwise this is
                # genuinely new.
                first_seen = prior.first_seen_at if prior is not None else now
                new_row = InventoryRow(
                    stream_id=stream_id, label=label, first_seen_at=first_seen,
                    last_seen_at=now, last_count=count, state="present", state_since=now,
                )
                transitions.append(Transition(
                    stream_id=stream_id, label=label, kind="arrived",
                    detail=f"count={count}" + ("" if prior is None else " (returned)"),
                ))
            elif count != prior.last_count:
                new_row = replace(prior, last_seen_at=now, last_count=count)
                transitions.append(Transition(
                    stream_id=stream_id, label=label, kind="count_changed",
                    detail=f"{prior.last_count} -> {count}",
                ))
            else:
                # Quiet refresh: still there, same count. No transition logged
                # -- this is the common case every sweep and must not spam.
                new_row = replace(prior, last_seen_at=now)
            updated[label] = new_row
            continue

        # Not seen this window.
        if prior is None:
            continue  # never tracked, still not seen -- nothing to do
        if prior.state == "departed":
            updated[label] = prior  # already gone, stays gone, no re-logging
            continue

        established_sec = (prior.last_seen_at - prior.first_seen_at).total_seconds()
        threshold = _absence_threshold_sec(
            established_sec,
            absence_fraction=absence_fraction,
            min_absence_sec=min_absence_sec,
            max_absence_sec=max_absence_sec,
        )
        gap_sec = (now - prior.last_seen_at).total_seconds()
        if gap_sec > threshold:
            updated[label] = replace(prior, state="departed", state_since=now)
            transitions.append(Transition(
                stream_id=stream_id, label=label, kind="departed",
                detail=(
                    f"established {established_sec:.0f}s, absent {gap_sec:.0f}s "
                    f"(threshold {threshold:.0f}s)"
                ),
            ))
        else:
            updated[label] = prior  # still within grace; leave last_seen_at alone

    return SweepResult(updated=updated, transitions=transitions)


# ---------------------------------------------------------------------------
# Postgres access. Synchronous by design -- same reasoning as
# grammar_retention_loop's run_one_retention_cycle: sql-writer's event loop is
# also draining the bus, so this MUST run in a worker thread, never inline.
# ---------------------------------------------------------------------------


def run_one_sweep_cycle(
    *,
    postgres_uri: str,
    lookback_ceiling_sec: float,
    absence_fraction: float = DEFAULT_ABSENCE_FRACTION,
    min_absence_sec: float = DEFAULT_MIN_ABSENCE_SEC,
    max_absence_sec: float = DEFAULT_MAX_ABSENCE_SEC,
    now: Optional[datetime] = None,
) -> dict:
    """One full sweep: every stream with a cursor or recent census activity.

    Blocking SQLAlchemy. Callers on an event loop MUST wrap this in
    `asyncio.to_thread`.

    Returns a small summary dict for logging -- streams swept, rows read,
    transitions per kind -- never raises past its own connection setup;
    per-stream failures are caught and logged so one bad stream cannot block
    the others in the same cycle.
    """
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    summary = {"streams": 0, "census_rows_read": 0, "arrived": 0, "departed": 0, "count_changed": 0}

    engine = create_engine(postgres_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            stream_ids = {
                r[0] for r in conn.execute(
                    text("SELECT DISTINCT stream_id FROM vision_scene_inventory "
                         "WHERE observed_at > now() - make_interval(secs => :ceiling) "
                         "AND stream_id IS NOT NULL"),
                    {"ceiling": lookback_ceiling_sec},
                ).fetchall()
            }
            stream_ids |= {
                r[0] for r in conn.execute(
                    text("SELECT stream_id FROM vision_object_permanence_cursor")
                ).fetchall()
            }

        for stream_id in sorted(stream_ids):
            try:
                _sweep_one_stream(
                    engine, stream_id, ts,
                    lookback_ceiling_sec=lookback_ceiling_sec,
                    absence_fraction=absence_fraction,
                    min_absence_sec=min_absence_sec,
                    max_absence_sec=max_absence_sec,
                    summary=summary,
                )
                summary["streams"] += 1
            except Exception as exc:
                logger = __import__("logging").getLogger("sql-writer.vision_object_permanence")
                logger.warning("sweep_failed stream=%s error=%s", stream_id, exc)
    finally:
        engine.dispose()

    return summary


def _sweep_one_stream(
    engine, stream_id: str, now: datetime, *,
    lookback_ceiling_sec: float, absence_fraction: float,
    min_absence_sec: float, max_absence_sec: float, summary: dict,
) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        cursor_row = conn.execute(
            text("SELECT last_swept_at FROM vision_object_permanence_cursor WHERE stream_id = :s"),
            {"s": stream_id},
        ).fetchone()
        # First-ever sweep for this stream: bound the initial lookback so a
        # stream with a long history does not scan every row it has ever
        # produced on its first tick.
        since = cursor_row[0] if cursor_row else now - timedelta(seconds=lookback_ceiling_sec)

        census_rows = conn.execute(
            text("SELECT counts FROM vision_scene_inventory "
                 "WHERE stream_id = :s AND observed_at > :since AND observed_at <= :now"),
            {"s": stream_id, "since": since, "now": now},
        ).fetchall()
        summary["census_rows_read"] += len(census_rows)

        window_max: Dict[str, int] = {}
        for (counts,) in census_rows:
            for label, count in (counts or {}).items():
                window_max[label] = max(window_max.get(label, 0), int(count))

        existing_rows = conn.execute(
            text("SELECT label, first_seen_at, last_seen_at, last_count, state, state_since "
                 "FROM vision_object_inventory WHERE stream_id = :s"),
            {"s": stream_id},
        ).fetchall()
        existing = {
            r[0]: InventoryRow(
                stream_id=stream_id, label=r[0], first_seen_at=r[1], last_seen_at=r[2],
                last_count=r[3], state=r[4], state_since=r[5],
            )
            for r in existing_rows
        }

        result = apply_sweep(
            stream_id=stream_id, window_max_counts=window_max, existing=existing, now=now,
            absence_fraction=absence_fraction, min_absence_sec=min_absence_sec,
            max_absence_sec=max_absence_sec,
        )

        for row in result.updated.values():
            conn.execute(
                text("""
                    INSERT INTO vision_object_inventory
                        (stream_id, label, first_seen_at, last_seen_at, last_count, state, state_since, updated_at)
                    VALUES (:stream_id, :label, :first_seen_at, :last_seen_at, :last_count, :state, :state_since, now())
                    ON CONFLICT (stream_id, label) DO UPDATE SET
                        first_seen_at = EXCLUDED.first_seen_at,
                        last_seen_at = EXCLUDED.last_seen_at,
                        last_count = EXCLUDED.last_count,
                        state = EXCLUDED.state,
                        state_since = EXCLUDED.state_since,
                        updated_at = now()
                """),
                {
                    "stream_id": row.stream_id, "label": row.label,
                    "first_seen_at": row.first_seen_at, "last_seen_at": row.last_seen_at,
                    "last_count": row.last_count, "state": row.state, "state_since": row.state_since,
                },
            )

        conn.execute(
            text("""
                INSERT INTO vision_object_permanence_cursor (stream_id, last_swept_at, updated_at)
                VALUES (:s, :now, now())
                ON CONFLICT (stream_id) DO UPDATE SET last_swept_at = EXCLUDED.last_swept_at, updated_at = now()
            """),
            {"s": stream_id, "now": now},
        )

    logger = __import__("logging").getLogger("sql-writer.vision_object_permanence")
    for t in result.transitions:
        summary[t.kind] = summary.get(t.kind, 0) + 1
        logger.info("[VISION_PERMANENCE] %s stream=%s label=%s %s", t.kind, t.stream_id, t.label, t.detail)
