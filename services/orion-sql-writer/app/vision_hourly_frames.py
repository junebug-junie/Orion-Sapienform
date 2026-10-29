"""One full frame per camera per hour, kept for N days, with a small index.

Why (2026-10-29): the walkway detector went half-blind on 2026-10-22..23
(scores slid under the 0.35 box cut-off; likely grey, wet weather,
UNVERIFIED) and recovered on 10-24 with no change. The shared frames
directory keeps a frame for ~5 min, so nobody could look at a sunny hour next
to a grey hour, or re-run the detector on them. This keeps that evidence.

Each tick (``VISION_HOURLY_FRAMES_INTERVAL_SEC``), for every configured
stream: if this UTC hour has no sample yet, copy the newest frame of this
hour from the shared frames directory (untouched bytes, full resolution) to

    <VISION_HOURLY_FRAMES_DIR>/<YYYY-MM-DD>/<stream>_<HH>Z.jpg

and append one JSON line to ``<YYYY-MM-DD>/index.jsonl``:

- ``frame_ts``, ``width``/``height``, ``bytes``
- ``luma_mean`` / ``luma_contrast`` (0..1; contrast = std of luma, the same
  measure the story strips use for "dark") -- a cheap sunny-vs-grey reading
- ``detections``: what the detector reported in the ``span_sec`` before the
  frame (windows / windows that saw something / per-label window counts from
  ``vision_scene_inventory``; per-label score n / median / max from
  ``vision_crop_observation``, which only holds crop labels on walkway and
  doorbell) and ``frame_crops``: crop rows for this exact frame. An empty
  ``frame_crops`` means none was stored for it, not "nothing there": most
  frames are never sent to the detector.

The doorbell has frames only during a Ring event, so it gets a sample only in
hours with an event. Pruning: this module owns only ``YYYY-MM-DD`` directories
under its own root; a day older than ``retention_days`` is removed whole.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from app.vision_story_strip import FrameIndex, luma_stats

logger = logging.getLogger("sql-writer.vision_hourly_frames")

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INDEX_NAME = "index.jsonl"


@dataclass(frozen=True)
class HourlyFramesConfig:
    frames_dir: str = "/mnt/telemetry/vision/frames"
    out_dir: str = "/mnt/telemetry/vision/hourly_frames"
    retention_days: float = 14.0  # 0 = keep forever
    # (stream_id, frame file prefix in frames_dir)
    streams: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    span_sec: float = 600.0
    # A frame younger than this may still be being written (cv2.imwrite is not atomic).
    min_age_sec: float = 2.0


def parse_streams(raw: str) -> Tuple[Tuple[str, str], ...]:
    """``walkway=walkway_walkway,cam0=frame`` -> pairs. A bare ``doorbell``
    means the retina naming ``<camera>_<stream>`` with camera == stream."""
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        stream, _, prefix = part.partition("=")
        stream, prefix = stream.strip(), prefix.strip()
        out.append((stream, prefix or f"{stream}_{stream}"))
    return tuple(out)


def hour_start(t: datetime) -> datetime:
    return t.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def sample_path(out_dir: str, stream: str, t: datetime) -> Path:
    h = hour_start(t)
    return Path(out_dir) / h.strftime("%Y-%m-%d") / f"{stream}_{h.strftime('%H')}Z.jpg"


def pick_frame(index: FrameIndex, prefix: str, now: datetime, min_age_sec: float) -> Optional[Tuple[int, str]]:
    """Newest frame of this UTC hour that is at least ``min_age_sec`` old."""
    lo = int(hour_start(now).timestamp() * 1000)
    hi = int((now.timestamp() - min_age_sec) * 1000)
    best = None
    for ms, path in index.by_prefix.get(prefix, ()):
        if lo <= ms <= hi and (best is None or ms > best[0]):
            best = (ms, path)
    return best


Summarize = Callable[[str, datetime], Dict[str, Any]]


def take_samples(cfg: HourlyFramesConfig, now: datetime, summarize: Optional[Summarize] = None) -> Dict[str, int]:
    """One pass over every stream. Never raises for one stream's failure."""
    counts = {"saved": 0, "already": 0, "no_frame": 0, "failed": 0}
    index: Optional[FrameIndex] = None
    for stream, prefix in cfg.streams:
        dest = sample_path(cfg.out_dir, stream, now)
        if dest.exists():
            counts["already"] += 1
            continue
        if index is None:
            index = FrameIndex(cfg.frames_dir)
        picked = pick_frame(index, prefix, now, cfg.min_age_sec)
        if picked is None:
            counts["no_frame"] += 1
            continue
        ms, src = picked
        try:
            entry = _save(src, dest, stream, ms)
        except FileNotFoundError:
            counts["no_frame"] += 1  # pruned between listing and copying; next tick retries
            continue
        except OSError as exc:
            counts["failed"] += 1
            logger.warning("hourly_frame_save_failed stream=%s error=%s", stream, exc)
            continue
        if summarize is not None:
            try:
                entry["detections"] = summarize(stream, datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc))
            except Exception as exc:  # the picture is the point; a DB blip must not cost it
                entry["detections"] = {"error": f"{exc.__class__.__name__}: {str(exc)[:200]}"}
        try:
            with open(dest.parent / INDEX_NAME, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("hourly_frame_index_failed stream=%s error=%s", stream, exc)
        counts["saved"] += 1
    return counts


def _save(src: str, dest: Path, stream: str, ms: int) -> Dict[str, Any]:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jpg.tmp")
    shutil.copyfile(src, tmp)
    try:
        with Image.open(tmp) as im:
            w, h = im.size
            im.draft("L", (max(1, w // 8), max(1, h // 8)))  # JPEG: decode at 1/8 scale, cheap
            mean, _median, contrast = luma_stats(im)
    except (OSError, ValueError) as exc:  # torn or corrupt frame: do not keep it
        tmp.unlink(missing_ok=True)
        raise OSError(f"unreadable frame {os.path.basename(src)}: {exc}") from exc
    os.replace(tmp, dest)
    return {
        "stream": stream,
        "hour": hour_start(datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)).isoformat(),
        "frame_ts": datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(),
        "file": dest.name,
        "source": os.path.basename(src),
        "bytes": dest.stat().st_size,
        "width": w,
        "height": h,
        "luma_mean": round(mean, 4),
        "luma_contrast": round(contrast, 4),
    }


def prune(cfg: HourlyFramesConfig, now: datetime) -> int:
    """Remove whole day directories older than retention. Only ``YYYY-MM-DD``
    directories directly under ``out_dir`` are ever touched."""
    if cfg.retention_days <= 0:
        return 0
    cutoff = (now - timedelta(days=cfg.retention_days)).astimezone(timezone.utc).date()
    removed = 0
    try:
        entries = list(os.scandir(cfg.out_dir))
    except FileNotFoundError:
        return 0
    for e in entries:
        if not DAY_RE.match(e.name) or not e.is_dir(follow_symlinks=False):
            continue
        try:
            day = datetime.strptime(e.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if day < cutoff:
            shutil.rmtree(e.path, ignore_errors=True)
            removed += 1
    return removed


# ----------------------------------------------------------------- database

SCENE_SQL = (
    "SELECT count(*), count(*) FILTER (WHERE counts <> '{}'::jsonb), coalesce(sum(frame_count), 0) "
    "FROM vision_scene_inventory WHERE stream_id = :s AND observed_at > :lo AND observed_at <= :t"
)
SCENE_LABELS_SQL = (
    "SELECT k, count(*) FROM vision_scene_inventory v, jsonb_object_keys(v.counts) k "
    "WHERE v.stream_id = :s AND v.observed_at > :lo AND v.observed_at <= :t GROUP BY k"
)
CROP_SCORES_SQL = (
    "SELECT label, count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY score), max(score) "
    "FROM vision_crop_observation WHERE stream_id = :s AND observed_at > :lo AND observed_at <= :t GROUP BY label"
)
FRAME_CROPS_SQL = (
    "SELECT label, score FROM vision_crop_observation WHERE stream_id = :s "
    "AND observed_at >= :t - interval '2 milliseconds' AND observed_at <= :t + interval '2 milliseconds' "
    "ORDER BY score DESC"
)


def db_summarizer(engine, span_sec: float) -> Summarize:
    from sqlalchemy import text

    def summarize(stream: str, t: datetime) -> Dict[str, Any]:
        p = {"s": stream, "t": t, "lo": t - timedelta(seconds=span_sec)}
        with engine.connect() as conn:
            conn.execute(text("SET LOCAL statement_timeout = 10000"))
            windows, detected, frames = conn.execute(text(SCENE_SQL), p).one()
            labels = {str(k): int(n) for k, n in conn.execute(text(SCENE_LABELS_SQL), p).fetchall()}
            scores = {
                str(lab): {"n": int(n), "p50": round(float(med), 3), "max": round(float(mx), 3)}
                for lab, n, med, mx in conn.execute(text(CROP_SCORES_SQL), p).fetchall()
            }
            frame = [{"label": str(lab), "score": round(float(sc), 3)}
                     for lab, sc in conn.execute(text(FRAME_CROPS_SQL), {"s": stream, "t": t}).fetchall()]
        return {
            "span_sec": span_sec,
            "windows": int(windows),
            "windows_with_detection": int(detected),
            "detector_frames": int(frames),
            "label_windows": labels,
            "crop_scores": scores,
            "frame_crops": frame,
        }

    return summarize


def config_from_settings(settings) -> HourlyFramesConfig:
    return HourlyFramesConfig(
        frames_dir=str(settings.vision_story_frames_dir),
        out_dir=str(settings.vision_hourly_frames_dir),
        retention_days=float(settings.vision_hourly_frames_retention_days),
        streams=parse_streams(str(settings.vision_hourly_frames_streams)),
    )


def run_one_tick(cfg: HourlyFramesConfig, summarize: Optional[Summarize], now: Optional[datetime] = None) -> Dict[str, int]:
    ts = now or datetime.now(timezone.utc)
    out = take_samples(cfg, ts, summarize)
    out["pruned_days"] = prune(cfg, ts)
    return out


async def vision_hourly_frames_loop(settings) -> None:
    interval = float(getattr(settings, "vision_hourly_frames_interval_sec", 0.0) or 0.0)
    if not bool(getattr(settings, "vision_hourly_frames_enabled", False)) or interval <= 0:
        logger.info("vision hourly frames DISABLED (VISION_HOURLY_FRAMES_ENABLED=false or interval 0)")
        return
    cfg = config_from_settings(settings)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    engine = None
    summarize: Optional[Summarize] = None
    if postgres_uri:
        from sqlalchemy import create_engine

        engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
        summarize = db_summarizer(engine, cfg.span_sec)
    logger.info("vision_hourly_frames_loop starting interval_sec=%.0f cfg=%s", interval, cfg)
    try:
        while True:
            # Tick first, then sleep: a restart loop still takes samples.
            try:
                summary = await asyncio.to_thread(run_one_tick, cfg, summarize)
                if summary.get("saved") or summary.get("failed") or summary.get("pruned_days"):
                    logger.info("[VISION_HOURLY_FRAMES] %s", summary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("vision_hourly_frames_tick_failed error=%s", exc)
            await asyncio.sleep(interval)
    finally:
        if engine is not None:
            engine.dispose()
