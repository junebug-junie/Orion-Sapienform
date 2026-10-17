"""Clocked reducer for walkway motion stories (app/vision_motion.py).

Each tick, per stream in ``VISION_MOTION_STORY_STREAMS``:

1. Read the cursor ``vision_motion_story_cursor.processed_until`` (first run:
   ``now - VISION_MOTION_STORY_INITIAL_LOOKBACK_SEC``).
2. ``horizon = min(now - settle, processed_until + max_step)``: crops land a
   few seconds after ``observed_at`` (live p99.9 = 3.4 s, 2026-10-16), so the
   newest ``settle`` seconds wait; a long outage is caught up in steps.
3. Link every crop in ``(processed_until - window, horizon]``. A passage is
   closed when its last box is at least ``link_gap`` before ``horizon`` (no
   later box can join it). Emit the closed passages whose last box is after
   the old cursor, then move the cursor to ``horizon - link_gap`` in the same
   transaction. ``story_id`` is derived from the first crop, and the insert is
   ``ON CONFLICT DO NOTHING``, so a re-run never duplicates.
4. Prune stories older than ``VISION_MOTION_STORY_RETENTION_DAYS``.

Trace: one ``[MOTION_STORY]`` line per stream per tick with the counts and
the cursor.

HUB ONLY (Juniper, 2026-10-16): nothing in Orion's cognition reads this
table. The Hub's "Walkway stories" card shows the stories and records her
spot-check verdicts. Switching stories on for Orion is gated on that tally
(>= 50 checked, >= 80% fully right, zero wrong names).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.settings import Settings
from app.vision_motion import MotionConfig, MotionCrop, Story, build_stories, link_passages

logger = logging.getLogger("sql-writer.vision_motion_loop")

MOTION_LOCK_KEY = 0x0A1C_0003
MAX_BACKOFF_SEC = 3600.0


class MigrationMissing(RuntimeError):
    """vision_motion_story tables do not exist yet. The loop backs off instead of crash-looping."""


@dataclass(frozen=True)
class LoopConfig:
    motion: MotionConfig = field(default_factory=MotionConfig)
    streams: Tuple[str, ...] = ("walkway",)
    settle_sec: float = 30.0
    window_sec: float = 900.0
    max_step_sec: float = 1800.0
    initial_lookback_sec: float = 3600.0
    retention_days: float = 90.0
    clean_since: Optional[datetime] = None
    prop_individual_ids: Tuple[str, ...] = ()


def parse_utc(value: str) -> Optional[datetime]:
    v = (value or "").strip()
    if not v:
        return None
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def loop_config(settings: Settings) -> LoopConfig:
    return LoopConfig(
        motion=MotionConfig(
            frame_w=float(settings.vision_motion_frame_width),
            frame_h=float(settings.vision_motion_frame_height),
            link_gap_sec=float(settings.vision_motion_story_close_gap_sec),
            min_detections=int(settings.vision_motion_story_min_detections),
        ),
        streams=tuple(s.strip() for s in str(settings.vision_motion_story_streams).split(",") if s.strip()),
        settle_sec=float(settings.vision_motion_story_settle_sec),
        initial_lookback_sec=float(settings.vision_motion_story_initial_lookback_sec),
        retention_days=float(settings.vision_motion_story_retention_days),
        clean_since=parse_utc(str(settings.vision_motion_clean_since)),
        prop_individual_ids=tuple(
            s.strip() for s in str(settings.vision_motion_prop_individual_ids).split(",") if s.strip()),
    )


def is_missing_relation(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "undefinedtable" in type(getattr(exc, "orig", exc)).__name__.lower() or (
        "does not exist" in msg and "relation" in msg
    )


# ---------------------------------------------------------------------------
# Reads (also used by the replay eval, read-only)
# ---------------------------------------------------------------------------

CROPS_SQL = (
    "SELECT crop_id, observed_at, label, score, box_xyxy, individual_id, thumb_ref "
    "FROM vision_crop_observation WHERE stream_id = :s AND observed_at > :a AND observed_at <= :b "
    "AND label = ANY(:kinds) ORDER BY observed_at, crop_id"
)


def read_crops(conn, stream_id: str, after: datetime, until: datetime) -> List[MotionCrop]:
    from sqlalchemy import text

    from app.vision_motion import KINDS

    rows = conn.execute(text(CROPS_SQL), {"s": stream_id, "a": after, "b": until, "kinds": list(KINDS)}).fetchall()
    out = []
    for r in rows:
        box = r[4] or []
        if len(box) != 4:
            continue
        out.append(MotionCrop(crop_id=r[0], t=r[1], label=r[2], score=float(r[3] or 0.0),
                              box=tuple(float(v) for v in box), individual_id=r[5], thumb_ref=r[6]))
    return out


def read_context(conn, stream_id: str, individual_ids: Sequence[str], prop_ids: Sequence[str]
                 ) -> Tuple[Dict[str, str], set, List[Tuple[str, List[float]]]]:
    """(labels by individual, stationary individual ids, prop boxes) for one stream.

    Prop boxes: the still box of every stationary individual on the stream,
    plus the still box and the distinct last boxes of every configured prop
    individual (the patio skeleton's sightings span two box shapes, a partial
    and a whole one)."""
    from sqlalchemy import text

    labels: Dict[str, str] = {}
    if individual_ids:
        for iid, label in conn.execute(text(
                "SELECT individual_id, label FROM vision_individual WHERE individual_id = ANY(:ids) "
                "AND label IS NOT NULL AND btrim(label) <> ''"), {"ids": list(individual_ids)}).fetchall():
            labels[iid] = label.strip()
    stationary = set()
    boxes: List[Tuple[str, List[float]]] = []
    for iid, kind, box in conn.execute(text(
            "SELECT individual_id, kind, still_box FROM vision_individual WHERE stream_id = :s AND stationary"),
            {"s": stream_id}).fetchall():
        stationary.add(iid)
        if box and len(box) == 4:
            boxes.append((kind, [float(v) for v in box]))
    if prop_ids:
        for kind, box in conn.execute(text(
                "SELECT kind, still_box FROM vision_individual WHERE individual_id = ANY(:ids) "
                "AND still_box IS NOT NULL"), {"ids": list(prop_ids)}).fetchall():
            if box and len(box) == 4:
                boxes.append((kind, [float(v) for v in box]))
        for kind, box in conn.execute(text(
                "SELECT DISTINCT ON (round(s.last_box_xyxy[1]::numeric, -1), round(s.last_box_xyxy[4]::numeric, -1)) "
                "i.kind, s.last_box_xyxy FROM vision_individual_sighting s JOIN vision_individual i "
                "USING (individual_id) WHERE s.individual_id = ANY(:ids) AND s.last_box_xyxy IS NOT NULL "
                "LIMIT 50"), {"ids": list(prop_ids)}).fetchall():
            if box and len(box) == 4:
                boxes.append((kind, [float(v) for v in box]))
    return labels, stationary, boxes


# ---------------------------------------------------------------------------
# One stream, one step (pure apart from the reads passed in)
# ---------------------------------------------------------------------------


def close_window(crops: Sequence[MotionCrop], *, stream_id: str, processed_until: datetime,
                 horizon: datetime, window_start: datetime, cfg: LoopConfig, zones,
                 context: Tuple[Dict[str, str], set, List[Tuple[str, List[float]]]]) -> Tuple[List[Story], datetime]:
    """Stories closed in this step and the new cursor."""
    gap = cfg.motion.link_gap_sec
    new_cursor = horizon - timedelta(seconds=gap)
    passages = link_passages(crops, cfg.motion)
    emit = {i for i, p in enumerate(passages) if processed_until < p.last.t <= new_cursor}
    too_long = {i for i in emit if (passages[i].first.t - window_start).total_seconds() <= gap}
    labels, stationary, boxes = context
    stories = build_stories(
        passages, stream_id=stream_id, zones=zones, cfg=cfg.motion, labels=labels,
        stationary_ids=stationary, prop_ids=set(cfg.prop_individual_ids), prop_boxes=boxes,
        clean_since=cfg.clean_since, emit=emit, too_long=too_long)
    return stories, new_cursor


INSERT_SQL = (
    "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
    "used_detections, edge_clipped_count, speed_class, speed_bh_per_sec, net_travel_bh, path_bh, dir_lateral, "
    "dir_depth, zone_path, individual_id, name_label, name_withheld, companion_kinds, together_story_ids, "
    "published, drop_reason, clean, sentence, crop_ids, evidence, rules_version) VALUES (:story_id, :stream_id, "
    ":kind, :started_at, :ended_at, :detection_count, :used_detections, :edge_clipped_count, :speed_class, "
    ":speed_bh_per_sec, :net_travel_bh, :path_bh, :dir_lateral, :dir_depth, :zone_path, :individual_id, "
    ":name_label, :name_withheld, :companion_kinds, :together_story_ids, :published, :drop_reason, :clean, "
    ":sentence, :crop_ids, CAST(:evidence AS jsonb), :rules_version) ON CONFLICT (story_id) DO NOTHING"
)


def story_params(s: Story) -> Dict[str, Any]:
    d = dict(s.__dict__)
    d["evidence"] = json.dumps(s.evidence)
    return d


def _check_tables(conn) -> None:
    from sqlalchemy import text

    missing = conn.execute(text(
        "SELECT t FROM unnest(ARRAY['vision_crop_observation','vision_individual','vision_individual_sighting',"
        "'vision_motion_story','vision_motion_story_cursor']) AS t WHERE to_regclass(t) IS NULL")).fetchall()
    if missing:
        raise MigrationMissing(
            "missing tables " + ",".join(r[0] for r in missing)
            + " -- apply services/orion-sql-db/manual_migration_vision_motion_story_v1.sql")


def _one_stream(engine, stream_id: str, ts: datetime, cfg: LoopConfig, zones, summary: Dict[str, Any]) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT processed_until FROM vision_motion_story_cursor WHERE stream_id = :s FOR UPDATE"),
            {"s": stream_id}).fetchone()
        pu = row[0] if row and row[0] else ts - timedelta(seconds=cfg.initial_lookback_sec)
        horizon = min(ts - timedelta(seconds=cfg.settle_sec), pu + timedelta(seconds=cfg.max_step_sec))
        if horizon - timedelta(seconds=cfg.motion.link_gap_sec) <= pu:
            summary["streams_idle"] += 1
            return
        window_start = pu - timedelta(seconds=cfg.window_sec)
        crops = read_crops(conn, stream_id, window_start, horizon)
        ids = sorted({c.individual_id for c in crops if c.individual_id})
        context = read_context(conn, stream_id, ids, cfg.prop_individual_ids)
        stories, new_cursor = close_window(
            crops, stream_id=stream_id, processed_until=pu, horizon=horizon, window_start=window_start,
            cfg=cfg, zones=zones, context=context)
        written = 0
        for s in stories:
            written += conn.execute(text(INSERT_SQL), story_params(s)).rowcount or 0
        conn.execute(text(
            "INSERT INTO vision_motion_story_cursor (stream_id, processed_until, updated_at) VALUES (:s, :p, now()) "
            "ON CONFLICT (stream_id) DO UPDATE SET processed_until = EXCLUDED.processed_until, updated_at = now()"),
            {"s": stream_id, "p": new_cursor})
    counts: Dict[str, int] = {}
    for s in stories:
        key = "published" if s.published else (s.drop_reason or "dropped")
        counts[key] = counts.get(key, 0) + 1
    logger.info(
        "[MOTION_STORY] stream=%s crops=%d closed=%d written=%d published=%d glimpses=%d standing=%d "
        "parked=%d dropped_props=%d edge_clipped=%d too_long=%d out_of_scope=%d companion=%d cursor=%s",
        stream_id, len(crops), len(stories), written, counts.get("published", 0), counts.get("glimpse", 0),
        counts.get("standing", 0), counts.get("parked", 0), counts.get("prop", 0),
        counts.get("edge_clipped", 0), counts.get("too_long", 0), counts.get("out_of_scope", 0),
        counts.get("companion", 0), new_cursor.isoformat())
    summary["streams"] += 1
    summary["crops"] += len(crops)
    summary["closed"] += len(stories)
    summary["written"] += written
    for k, v in counts.items():
        summary[k] = summary.get(k, 0) + v


def run_one_motion_cycle(*, postgres_uri: str, cfg: LoopConfig, zones_by_stream: Dict[str, list],
                         now: Optional[datetime] = None) -> Dict[str, Any]:
    """One tick over every configured stream. Blocking: run in a worker thread."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    summary: Dict[str, Any] = {"streams": 0, "streams_idle": 0, "crops": 0, "closed": 0, "written": 0,
                               "pruned": 0}
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=1)
    lock_conn = None
    try:
        lock_conn = engine.connect()
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": MOTION_LOCK_KEY}).scalar():
            summary["skipped"] = "another instance holds the motion-story lock"
            return summary
        lock_conn.commit()
        with engine.begin() as conn:
            _check_tables(conn)
        for stream_id in cfg.streams:
            try:
                _one_stream(engine, stream_id, ts, cfg, zones_by_stream.get(stream_id, []), summary)
            except Exception as exc:
                if is_missing_relation(exc):
                    raise MigrationMissing(str(exc)) from exc
                logger.warning("motion_story_stream_failed stream=%s error=%s", stream_id, exc)
        if cfg.retention_days > 0:
            with engine.begin() as conn:
                summary["pruned"] = conn.execute(text(
                    "DELETE FROM vision_motion_story s WHERE s.ended_at < :cut AND NOT EXISTS "
                    "(SELECT 1 FROM vision_motion_story_verdict v WHERE v.story_id = s.story_id)"),
                    {"cut": ts - timedelta(days=cfg.retention_days)}).rowcount or 0
    finally:
        if lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": MOTION_LOCK_KEY})
                lock_conn.close()
            except Exception:
                pass
        engine.dispose()
    return summary


def next_backoff(interval: float, current: Optional[float]) -> float:
    return min(MAX_BACKOFF_SEC, max(interval, (current or interval) * 2.0))


async def vision_motion_loop(settings: Settings) -> None:
    enabled = bool(getattr(settings, "vision_motion_stories_enabled", False))
    interval = float(getattr(settings, "vision_motion_story_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if not enabled or interval <= 0 or not postgres_uri:
        logger.info("vision_motion_loop DISABLED (VISION_MOTION_STORIES_ENABLED=%s interval=%s)", enabled, interval)
        return
    from orion.vision.zones import load_zones

    try:
        zones = load_zones()
    except Exception as exc:
        logger.error("vision_zones_load_failed error=%s -- stories will carry no zones", exc)
        zones = {}
    cfg = loop_config(settings)
    logger.info("vision_motion_loop starting interval_sec=%.0f cfg=%s", interval, cfg)
    backoff: Optional[float] = None
    while True:
        await asyncio.sleep(backoff or interval)
        try:
            summary = await asyncio.to_thread(
                run_one_motion_cycle, postgres_uri=postgres_uri, cfg=cfg, zones_by_stream=zones)
            backoff = None
            logger.debug("[MOTION_STORY] tick_complete %s", summary)
        except asyncio.CancelledError:
            raise
        except MigrationMissing as exc:
            backoff = next_backoff(interval, backoff)
            logger.error("[MOTION_STORY] migration not applied: %s -- retrying in %.0fs", exc, backoff)
        except Exception as exc:
            logger.warning("vision_motion_cycle_failed error=%s", exc)
