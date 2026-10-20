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
4. Fill late individual links: an evidence entry with a null individual_id
   whose crop the individuals reducer has since assigned gets that id
   (``relink_stories``, null-only, stories that ended in the last 6 h).
5. Prune stories older than ``VISION_MOTION_STORY_RETENTION_DAYS``.

Pictures of the moment (app/vision_story_strip.py, 2026-10-17), when
``VISION_STORY_STRIPS_ENABLED``: before step 1 every tick saves a padded crop
and a small full frame for each new walker crop while its frame still exists
(the shared frames directory keeps a walkway frame ~60 s); after step 3 every
published story without a strip gets one (``evidence_strip``, ``frame_luminance``,
``scene_luminance``, ``scene_contrast``, ``dark``). With
``VISION_STORY_STRIP_RETENTION_DAYS=0`` (default since 2026-10-20) a strip is
kept forever: no age rule deletes it, story retention skips a story that still
has its picture, and the orphan-directory sweep is off. With a positive value
a strip is kept until that many days after its story ended AND the story has a
verdict ("Can't tell" counts), or until story retention deletes the row.

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
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.settings import Settings
from app.vision_motion import PUBLISH_KINDS, MotionConfig, MotionCrop, Story, build_stories, link_passages
from app.vision_story_strip import StripConfig

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
    strip: Optional[StripConfig] = None
    # VISION_STORY_STRIP_RETENTION_DAYS <= 0: pictures are kept forever, so story
    # retention never deletes a story that still has one -- even with strips off.
    strip_keep_forever: bool = True


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
        strip=StripConfig(
            frames_dir=str(settings.vision_story_frames_dir),
            strip_dir=str(settings.vision_story_strip_dir),
            dark_contrast=float(settings.vision_story_strip_dark_contrast),
            pending_sec=float(settings.vision_story_strip_pending_sec),
            retention_days=float(settings.vision_story_strip_retention_days),
            frame_w=float(settings.vision_motion_frame_width),
            frame_h=float(settings.vision_motion_frame_height),
        ) if bool(settings.vision_story_strips_enabled) else None,
        strip_keep_forever=float(settings.vision_story_strip_retention_days) <= 0,
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


# ---------------------------------------------------------------------------
# Pictures of the moment (app/vision_story_strip.py)
# ---------------------------------------------------------------------------

STRIP_COLUMNS = ("evidence_strip", "frame_luminance", "scene_luminance", "scene_contrast", "dark")
# How far back the first capture tick looks. Frames older than ~60 s are gone
# anyway; a longer look only writes "frame_gone" markers.
CAPTURE_FIRST_LOOKBACK_SEC = 120.0
ORPHAN_SWEEP_EVERY_SEC = 3600.0
ORPHAN_MIN_AGE_SEC = 3600.0
STRIPS_PER_TICK = 20

# Per process: crops are captured once per crop id (the pending file is the
# real dedupe), so losing this on restart costs nothing but a re-scan.
_capture_cursor: Dict[str, datetime] = {}
# None = never swept this process. Not 0.0: time.monotonic() counts from host
# boot, so on a host up < ORPHAN_SWEEP_EVERY_SEC (a fresh CI runner, a rebooted
# box) a 0.0 sentinel would postpone the first sweep by up to an hour.
_last_orphan_sweep: List[Optional[float]] = [None]

# created_at is the inserting transaction's start time: a row can commit after
# a newer one was already read. Each read re-covers this overlap behind the
# cursor (the pending files dedupe), and never reads the last second.
CAPTURE_OVERLAP_SEC = 10.0
CAPTURE_SQL = (
    "SELECT crop_id, observed_at, label, box_xyxy, camera_id, created_at, individual_id "
    "FROM vision_crop_observation WHERE stream_id = :s AND created_at > :a "
    "AND created_at <= now() - interval '1 second' AND label = ANY(:kinds) "
    "ORDER BY created_at, crop_id LIMIT 2000"
)
# A failed strip is retried each tick; after this share of the pending window
# the story gets a final error record instead of staying NULL forever.
ASSEMBLE_GIVE_UP_SHARE = 0.5


def strip_columns_missing(conn) -> List[str]:
    from sqlalchemy import text

    have = {r[0] for r in conn.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'vision_motion_story' "
        "AND column_name = ANY(:c)"), {"c": list(STRIP_COLUMNS)}).fetchall()}
    return [c for c in STRIP_COLUMNS if c not in have]


def capture_step(engine, stream_id: str, ts: datetime, cfg: LoopConfig) -> Dict[str, int]:
    """Save crops + small frames for walker crops that landed since the last tick."""
    from sqlalchemy import text

    from app.vision_story_strip import PendingCrop, capture_pending, mark_skipped

    from app.vision_motion import box_iou

    strip = cfg.strip
    assert strip is not None
    cursor = _capture_cursor.get(stream_id)
    after = (cursor - timedelta(seconds=CAPTURE_OVERLAP_SEC)) if cursor else \
        ts - timedelta(seconds=CAPTURE_FIRST_LOOKBACK_SEC)
    with engine.connect() as conn:
        rows = conn.execute(text(CAPTURE_SQL), {"s": stream_id, "a": after,
                                                "kinds": sorted(PUBLISH_KINDS)}).fetchall()
        # Props (the patio skeleton, stationary things) are detected every frame and
        # never become stories. A fresh crop has no individual_id yet, so match by box.
        prop_boxes = read_context(conn, stream_id, [], cfg.prop_individual_ids)[2] if rows else []
    empty = {"saved": 0, "frame_gone": 0, "already": 0, "failed": 0, "prop": 0}
    if not rows:
        return empty
    props = set(cfg.prop_individual_ids)
    crops, skipped = [], 0
    for r in rows:
        if not r[3] or len(r[3]) != 4:
            continue
        box = tuple(float(v) for v in r[3])
        c = PendingCrop(crop_id=r[0], t=r[1], label=r[2], box=box, camera_id=r[4])
        if r[6] in props or any(kind == r[2] and box_iou(box, b) >= cfg.motion.prop_iou for kind, b in prop_boxes):
            skipped += 1
            mark_skipped(c, "prop_overlap", strip)
            continue
        crops.append(c)
    counts = {**empty, **capture_pending(crops, stream_id=stream_id, cfg=strip), "prop": skipped}
    _capture_cursor[stream_id] = max(cursor or rows[-1][5], rows[-1][5])
    return counts


def assemble_step(engine, stream_id: str, ts: datetime, cfg: LoopConfig) -> int:
    """A strip for every published story that has none, while its pending pictures can exist."""
    from sqlalchemy import text

    from app.vision_story_strip import assemble_strip, failed_record, remove_strip_dir

    strip = cfg.strip
    assert strip is not None
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT story_id, evidence, created_at FROM vision_motion_story WHERE stream_id = :s AND published "
            "AND evidence_strip IS NULL AND created_at > :since ORDER BY created_at LIMIT :n"),
            {"s": stream_id, "since": ts - timedelta(seconds=strip.pending_sec), "n": STRIPS_PER_TICK}).fetchall()
    done = 0
    for story_id, evidence, created_at in rows:
        ev = evidence if isinstance(evidence, list) else json.loads(evidence or "[]")
        try:
            rec = assemble_strip(story_id, stream_id, ev, cfg=strip)
        except Exception as exc:
            logger.warning("story_strip_assemble_failed story_id=%s error=%s", story_id, exc)
            if created_at is None or created_at > ts - timedelta(seconds=strip.pending_sec * ASSEMBLE_GIVE_UP_SHARE):
                continue
            # Final: the row says why there is no picture, and the half-written folder goes.
            remove_strip_dir(strip, story_id)
            rec = failed_record(str(exc), strip)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE vision_motion_story SET evidence_strip = CAST(:j AS jsonb), frame_luminance = :fl, "
                "scene_luminance = :sl, scene_contrast = :sc, dark = :d "
                "WHERE story_id = :id AND evidence_strip IS NULL"),
                {"j": json.dumps(rec), "fl": rec["frame_luminance"], "sl": rec["scene_luminance"],
                 "sc": rec["scene_contrast"], "d": rec["dark"], "id": story_id})
        done += 1
        logger.info(
            "[STORY_STRIP] story_id=%s crops_saved=%d crops_missing=%d frame=%s frame_lum=%s scene_lum=%s "
            "scene_contrast=%s dark=%s",
            story_id, sum(1 for c in rec["crops"] if "file" in c), sum(1 for c in rec["crops"] if "missing" in c),
            bool(rec["frame"]), rec["frame_luminance"], rec["scene_luminance"], rec["scene_contrast"], rec["dark"])
    return done


# ---------------------------------------------------------------------------
# Late individual links (2026-10-20). A story snapshots each crop's
# individual_id when it closes; the individuals reducer often assigns the
# crops a few seconds later, so the frozen evidence says "no individual" for
# crops that ARE linked (live: 101 of 147 person stories over 5 days). Each
# tick fills those nulls from vision_crop_observation while the crop rows
# still exist (VISION_CROP_RETENTION_DAYS, 7) -- in the story row itself, so
# the link outlives the crop rows. Null-only: an id already in the evidence is
# never changed.
# ---------------------------------------------------------------------------

RELINK_LOOKBACK_SEC = 6 * 3600.0
RELINK_MAX_STORIES = 500
RELINK_CANDIDATES_SQL = (
    "SELECT story_id, evidence FROM vision_motion_story WHERE stream_id = ANY(:streams) AND ended_at >= :since "
    "AND jsonb_typeof(evidence) = 'array' "
    "AND jsonb_path_exists(evidence, '$[*] ? (!exists(@.individual_id) || @.individual_id == null)') "
    "ORDER BY ended_at DESC LIMIT :n")
RELINK_CROPS_SQL = (
    "SELECT crop_id, individual_id FROM vision_crop_observation "
    "WHERE crop_id = ANY(:ids) AND individual_id IS NOT NULL")
RELINK_UPDATE_SQL = (
    "UPDATE vision_motion_story SET evidence = CAST(:new AS jsonb) "
    "WHERE story_id = :id AND evidence = CAST(:old AS jsonb)")


def unlinked_crop_ids(evidence: Any) -> List[str]:
    """Crop ids of evidence entries whose individual_id is null."""
    if not isinstance(evidence, list):
        return []
    return [str(e["crop_id"]) for e in evidence
            if isinstance(e, dict) and e.get("crop_id") and e.get("individual_id") is None]


def fill_evidence_links(evidence: Any, links: Dict[str, str]) -> Tuple[Any, int]:
    """Pure: (evidence with null individual_ids filled from ``links``, how many filled).
    Never changes an entry that already has an id, and never removes or reorders one."""
    if not isinstance(evidence, list):
        return evidence, 0
    out: List[Any] = []
    filled = 0
    for e in evidence:
        if isinstance(e, dict) and e.get("individual_id") is None and links.get(str(e.get("crop_id") or "")):
            e = {**e, "individual_id": links[str(e["crop_id"])]}
            filled += 1
        out.append(e)
    return out, filled


def _evidence_value(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def relink_stories(conn, since: datetime, streams: Sequence[str], *,
                   limit: int = RELINK_MAX_STORIES) -> Dict[str, int]:
    """Fill null evidence individual_ids of ``streams``' stories that ended at/after ``since``
    (filtered by stream so the (stream_id, ended_at) index applies).
    Returns {"stories": rows rewritten, "crops": entries filled}."""
    from sqlalchemy import text

    rows = [(r[0], _evidence_value(r[1])) for r in conn.execute(
        text(RELINK_CANDIDATES_SQL), {"streams": list(streams), "since": since, "n": int(limit)}).fetchall()]
    ids = sorted({cid for _, ev in rows for cid in unlinked_crop_ids(ev)})
    if not ids:
        return {"stories": 0, "crops": 0}
    links = {r[0]: r[1] for r in conn.execute(text(RELINK_CROPS_SQL), {"ids": ids}).fetchall()}
    stories = crops = 0
    for story_id, ev in rows:
        new, n = fill_evidence_links(ev, links)
        if not n:
            continue
        stories += conn.execute(text(RELINK_UPDATE_SQL), {
            "id": story_id, "new": json.dumps(new), "old": json.dumps(ev)}).rowcount or 0
        crops += n
    return {"stories": stories, "crops": crops}


STORY_PRUNE_SQL = (
    "DELETE FROM vision_motion_story s WHERE s.ended_at < :cut AND NOT EXISTS "
    "(SELECT 1 FROM vision_motion_story_verdict v WHERE v.story_id = s.story_id)")
# A strip that saved at least one picture file and was not pruned (failed_record has no files).
_KEEP_STORY_WITH_PICTURE = (
    " AND NOT (s.evidence_strip IS NOT NULL AND NOT (s.evidence_strip ? 'pruned_at') AND ("
    "jsonb_typeof(s.evidence_strip->'frame') = 'object' OR EXISTS (SELECT 1 FROM jsonb_array_elements("
    "CASE WHEN jsonb_typeof(s.evidence_strip->'crops') = 'array' THEN s.evidence_strip->'crops' "
    "ELSE '[]'::jsonb END) c WHERE c ? 'file')))")


def story_prune_sql(cfg: LoopConfig, *, strip_columns: bool = True) -> str:
    """Story retention. When strips are kept forever (strip retention 0), a story
    that still has a saved picture is never deleted, so the picture keeps its
    story (the Hub reads a strip through its story row). Without the strip
    columns (migration not applied) no story can have a picture: plain prune."""
    return STORY_PRUNE_SQL + (_KEEP_STORY_WITH_PICTURE if cfg.strip_keep_forever and strip_columns else "")


def prune_strips(engine, ts: datetime, cfg: LoopConfig) -> int:
    """Delete strips of stories that ended > retention ago AND have a verdict; never an unchecked story's.
    Once an hour, also delete strip directories whose story row is gone (story retention).
    Strip retention 0 = keep forever: only stale pending files go, no strip directory is ever removed."""
    from sqlalchemy import text

    from app.vision_story_strip import prune_pending, remove_strip_dir, strip_dirs

    strip = cfg.strip
    assert strip is not None
    prune_pending(strip)
    removed = 0
    if strip.retention_days <= 0:
        return removed  # keep forever: no strip directory is ever removed
    with engine.connect() as conn:
        ids = [r[0] for r in conn.execute(text(
            "SELECT s.story_id FROM vision_motion_story s WHERE s.ended_at < :cut "
            "AND s.evidence_strip IS NOT NULL AND NOT (s.evidence_strip ? 'pruned_at') "
            "AND EXISTS (SELECT 1 FROM vision_motion_story_verdict v WHERE v.story_id = s.story_id) "
            "LIMIT 200"), {"cut": ts - timedelta(days=strip.retention_days)}).fetchall()]
    for sid in ids:
        remove_strip_dir(strip, sid)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE vision_motion_story SET evidence_strip = evidence_strip || "
                "jsonb_build_object('pruned_at', CAST(:at AS text)) WHERE story_id = :id"),
                {"at": ts.isoformat(), "id": sid})
        removed += 1
    now_mono = time.monotonic()
    last = _last_orphan_sweep[0]
    if last is None or now_mono - last >= ORPHAN_SWEEP_EVERY_SEC:
        _last_orphan_sweep[0] = now_mono
        dirs = strip_dirs(strip)
        if dirs:
            with engine.connect() as conn:
                alive = {r[0] for r in conn.execute(text(
                    "SELECT story_id FROM vision_motion_story WHERE story_id = ANY(:ids)"), {"ids": dirs}).fetchall()}
            for sid in dirs:
                if sid in alive:
                    continue
                try:
                    age = time.time() - os.stat(os.path.join(strip.strip_dir, sid)).st_mtime
                except OSError:
                    continue
                if age >= ORPHAN_MIN_AGE_SEC and remove_strip_dir(strip, sid):
                    removed += 1
    return removed


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
            missing = strip_columns_missing(conn)
            strips_on = cfg.strip is not None
            if strips_on:
                if missing:
                    strips_on = False
                    summary["strips"] = "migration missing: " + ",".join(missing)
                    logger.error("[STORY_STRIP] missing columns %s -- apply services/orion-sql-db/"
                                 "manual_migration_vision_motion_story_strip_v2.sql; stories still run",
                                 ",".join(missing))
        # Capture first: the frames of crops that landed since the last tick are
        # the ones closest to being pruned.
        if strips_on:
            for stream_id in cfg.streams:
                try:
                    c = capture_step(engine, stream_id, ts, cfg)
                    for k, v in c.items():
                        summary["capture_" + k] = summary.get("capture_" + k, 0) + v
                except Exception as exc:
                    logger.warning("story_strip_capture_failed stream=%s error=%s", stream_id, exc)
        for stream_id in cfg.streams:
            try:
                _one_stream(engine, stream_id, ts, cfg, zones_by_stream.get(stream_id, []), summary)
            except Exception as exc:
                if is_missing_relation(exc):
                    raise MigrationMissing(str(exc)) from exc
                logger.warning("motion_story_stream_failed stream=%s error=%s", stream_id, exc)
            if strips_on:
                try:
                    # A crop that landed after the first capture read can belong to a
                    # story closed just now: catch it before the strip is built.
                    capture_step(engine, stream_id, ts, cfg)
                    summary["strips_written"] = summary.get("strips_written", 0) + assemble_step(
                        engine, stream_id, ts, cfg)
                except Exception as exc:
                    logger.warning("story_strip_assemble_step_failed stream=%s error=%s", stream_id, exc)
        try:
            with engine.begin() as conn:
                linked = relink_stories(conn, ts - timedelta(seconds=RELINK_LOOKBACK_SEC), cfg.streams)
            summary["relinked_stories"], summary["relinked_crops"] = linked["stories"], linked["crops"]
            if linked["stories"]:
                logger.info("[MOTION_STORY] relinked stories=%d crops=%d", linked["stories"], linked["crops"])
        except Exception as exc:
            logger.warning("motion_story_relink_failed error=%s", exc)
        if strips_on:
            try:
                summary["strips_pruned"] = prune_strips(engine, ts, cfg)
            except Exception as exc:
                logger.warning("story_strip_prune_failed error=%s", exc)
        if cfg.retention_days > 0:
            try:
                with engine.begin() as conn:
                    summary["pruned"] = conn.execute(text(story_prune_sql(cfg, strip_columns=not missing)),
                        {"cut": ts - timedelta(days=cfg.retention_days)}).rowcount or 0
            except Exception as exc:
                logger.warning("motion_story_prune_failed error=%s", exc)
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
