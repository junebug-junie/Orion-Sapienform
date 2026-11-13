"""`dig look [<camera>]`: open the newest frame of one camera so Orion can see it.

Design: docs/superpowers/specs/2026-11-13-camera-look-design.md.

The cameras already save a frame every half second (orion-vision-retina, ``RETINA_FRAME_MODE=local``,
``RETINA_FPS=2.0``) to ``<frames>/<camera>_<stream>_<epoch_ms>_<sha8>.jpg`` and sweep them after
5 minutes (services/orion-vision-retina/app/frame_store.py ``save_frame``). So "look now" is the
newest file for that stream: at most about half a second old while the camera is up. Nothing here
asks the camera for anything.

The frame is copied into ``$ORION_DIG_HOME/look/<day>/`` (the sandbox, outside the repo checkout,
like chat attachments), so the retina's sweep cannot delete it mid-turn and the turn's Read tool
can open it. Every look appends one line to ``look/looks.jsonl``: that is the trace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from orion.situational.camera_eyes import describe_camera

ZONE = ZoneInfo("America/Denver")
DEFAULT_FRAMES_DIR = "/mnt/telemetry/vision/frames"  # the retinas' FRAME_STORAGE_DIR on athena
LIVE_MAX_AGE_SEC = 5.0  # 2 fps: a live camera's newest frame is about 0.5 s old
LOOKS_PER_HOUR = 20  # each look puts an image (~1-3k tokens) into the turn
KEEP_DAYS = 7
LOG_NAME = "looks.jsonl"
# Plain words Orion (or Juniper) may use, beside the stream ids themselves.
CAMERA_WORDS = {"office": "cam0", "walkway": "walkway", "doorbell": "doorbell", "door": "doorbell"}
STREAMS = ("cam0", "walkway", "doorbell")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LookError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    path: Path
    stream_id: str
    epoch_ms: int

    @property
    def taken_at(self) -> datetime:
        return datetime.fromtimestamp(self.epoch_ms / 1000.0, tz=timezone.utc)


def frames_dir() -> Path:
    return Path(os.environ.get("ORION_DIG_FRAMES_DIR") or DEFAULT_FRAMES_DIR)


def resolve_stream(word: str) -> str:
    key = word.strip().lower()
    if key in CAMERA_WORDS:
        return CAMERA_WORDS[key]
    if key in STREAMS:
        return key
    names = ", ".join(sorted(CAMERA_WORDS))
    raise LookError(f"unknown camera {word!r}; say one of: {names}")


def _frame_re(stream_id: str) -> re.Pattern[str]:
    # The retina names a frame <camera>_<stream>_<epoch_ms>_<sha8>.jpg (frame_store.save_frame).
    return re.compile(rf"^[A-Za-z0-9._-]+_{re.escape(stream_id)}_(\d+)_[0-9a-f]{{8}}\.jpg$")


def newest_frame(directory: Path, stream_id: str) -> Frame | None:
    pattern = _frame_re(stream_id)
    best: Frame | None = None
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise LookError(
            f"cannot read the frames folder {directory}: {exc.strerror or exc}. It is mounted read-only "
            "into the harness (services/orion-harness-governor/docker-compose.yml); tell Juniper."
        ) from exc
    for entry in entries:
        m = pattern.match(entry.name)
        if m is None:
            continue
        ms = int(m.group(1))
        if best is None or ms > best.epoch_ms:
            best = Frame(path=Path(entry.path), stream_id=stream_id, epoch_ms=ms)
    return best


def _age_words(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _read_log(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def looks_in_last_hour(log: Path, now: datetime) -> int:
    since = now - timedelta(hours=1)
    n = 0
    for row in _read_log(log):
        try:
            at = datetime.fromisoformat(str(row.get("looked_at")))
        except ValueError:
            continue
        if at >= since:
            n += 1
    return n


def sweep(look_root: Path, today: date) -> int:
    """Remove day folders older than KEEP_DAYS. Only folders named like a date are touched."""
    if not look_root.is_dir():
        return 0
    removed = 0
    for child in look_root.iterdir():
        if not (child.is_dir() and _DAY_RE.match(child.name)):
            continue
        try:
            day = date.fromisoformat(child.name)
        except ValueError:
            continue
        if (today - day).days > KEEP_DAYS:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def look(word: str, *, home: Path, directory: Path | None = None, now: datetime | None = None) -> tuple[list[str], dict]:
    """Copy one camera's newest frame into ``home/look`` and say how to see it."""
    stream_id = resolve_stream(word)
    now = now or datetime.now(timezone.utc)
    directory = directory or frames_dir()
    look_root = home / "look"
    log = look_root / LOG_NAME
    used = looks_in_last_hour(log, now)
    if used >= LOOKS_PER_HOUR:
        raise LookError(f"{used} looks in the last hour; the cap is {LOOKS_PER_HOUR}. Look again later.")
    desc = describe_camera(stream_id)
    frame = newest_frame(directory, stream_id)
    if frame is None:
        raise LookError(
            f"no saved frame for {desc.name} ({stream_id}) in {directory}. Frames are kept 5 minutes, "
            + ("and this camera only records after motion or a ring." if desc.event_driven
               else "so this camera has saved nothing for at least that long: it may be down.")
        )
    age = max(0.0, (now - frame.taken_at).total_seconds())
    day = frame.taken_at.astimezone(ZONE).date()
    dest_dir = look_root / day.isoformat()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stream_id}-{frame.epoch_ms}.jpg"
    try:
        shutil.copyfile(frame.path, dest)
    except OSError as exc:
        raise LookError(f"could not copy {frame.path}: {exc.strerror or exc}") from exc
    data = dest.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    live = age <= LIVE_MAX_AGE_SEC
    record = {
        "looked_at": now.isoformat(),
        "camera": stream_id,
        "frame_at": frame.taken_at.isoformat(),
        "age_sec": round(age, 2),
        "live": live,
        "sha256": sha,
        "bytes": len(data),
        "path": str(dest),
        "source": str(frame.path),
    }
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    sweep(look_root, now.astimezone(ZONE).date())

    when = frame.taken_at.astimezone(ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    if live:
        state = "live"
    elif desc.event_driven:
        state = "NOT LIVE: this camera only records after motion or a ring; this is its last event frame"
    else:
        state = "NOT LIVE: the camera has saved nothing newer; it may be down"
    lines = [
        f"look: {desc.name} ({stream_id})" + (f", which sees {desc.view}" if desc.view else ""),
        f"frame: {when}, {_age_words(age)} old ({state})",
        f"saved: {dest} ({len(data) // 1024} KB, sha256 {sha[:12]})",
        "To see it, open that path with your Read tool. Say what you see in it, not what you expect.",
        "If Read gives a note instead of a picture, the model serving this turn cannot see images: say so.",
        f"looks in the last hour: {used + 1} of {LOOKS_PER_HOUR}",
    ]
    return lines, record


def cameras(*, directory: Path | None = None, now: datetime | None = None) -> tuple[list[str], dict]:
    """`dig look` with no camera: each camera's newest frame age."""
    now = now or datetime.now(timezone.utc)
    directory = directory or frames_dir()
    words = {}
    for word, sid in CAMERA_WORDS.items():
        words.setdefault(sid, word)
    lines, payload = [], {}
    for sid in STREAMS:
        desc = describe_camera(sid)
        frame = newest_frame(directory, sid)
        if frame is None:
            lines.append(f"{words[sid]}: {desc.name}, no frame in the last 5 min")
            payload[sid] = None
            continue
        age = max(0.0, (now - frame.taken_at).total_seconds())
        live = "live" if age <= LIVE_MAX_AGE_SEC else "not live"
        lines.append(f"{words[sid]}: {desc.name}, newest frame {_age_words(age)} old ({live})")
        payload[sid] = {"frame_at": frame.taken_at.isoformat(), "age_sec": round(age, 2)}
    lines.append("Look at one: python -P -m orion.dig look <camera>")
    return lines, payload
