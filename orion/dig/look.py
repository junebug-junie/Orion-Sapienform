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

Every look is also an act on the act ledger (actor ``orion.look``, verb ``look``, object
``camera:<stream>``; issue #3320, design docs/superpowers/specs/2026-11-13-look-acts-and-checks-design.md).
Its expectation: a live frame (at most ``LIVE_MAX_AGE_SEC`` old) comes back. A look that finds the
camera down, stale or mid-write closes 0. The act carries the turn's id as ``session_id`` (joinable)
but no ``correlation_id``: the ledger charges an act every GPU lease of its correlation id, and a
look is one image inside a turn, not the turn. Cost is left unmeasured, which the ledger reads as
"not measured", never zero. A look refused by the hourly cap or an unknown camera word is no act.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
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

ACTOR = "orion.look"
ACT_SOURCE_SERVICE = "orion-harness-governor"  # where the dig runs (a registered grammar producer)
ACT_SOURCE_COMPONENT = "dig.look"
ACT_HORIZON_SEC = 60.0
ACT_BASIS = (
    "artifact_produced = a live frame (<= 5 s old) by the look itself; the retinas save one every 0.5 s, "
    "so an older newest frame means the camera stopped (down, stale, or event-driven between events)"
)
PUBLISH_TIMEOUT_SEC = 3.0
# == orion.harness.fcc_warm_relay.CORRELATION_HEADER (tests pin it): the motor puts the turn's id on
# every model call through ANTHROPIC_CUSTOM_HEADERS, and the turn's Bash inherits that environment.
TURN_ID_HEADER = "X-Orion-Correlation-Id"

Publisher = Callable[[Sequence[Any]], None]


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


def newest_frames(directory: Path, stream_id: str, limit: int = 3) -> list[Frame]:
    """This stream's frames, newest first."""
    pattern = _frame_re(stream_id)
    found: list[Frame] = []
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
        found.append(Frame(path=Path(entry.path), stream_id=stream_id, epoch_ms=int(m.group(1))))
    found.sort(key=lambda f: f.epoch_ms, reverse=True)
    return found[:limit]


def newest_frame(directory: Path, stream_id: str) -> Frame | None:
    frames = newest_frames(directory, stream_id, limit=1)
    return frames[0] if frames else None


def is_complete_jpeg(data: bytes) -> bool:
    """The retina writes each frame in place (cv2.imwrite, no rename), so the newest file can be
    caught mid-write: a whole JPEG starts with SOI and ends with EOI."""
    return data[:2] == b"\xff\xd8" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9"


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
            if at >= since:
                n += 1
        except (ValueError, TypeError):  # unparseable, or a naive time beside an aware one
            continue
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


def turn_id(environ: dict[str, str] | None = None) -> str | None:
    """The turn's correlation id, from the header the motor sets on every model call."""
    env = os.environ if environ is None else environ
    for line in (env.get("ANTHROPIC_CUSTOM_HEADERS") or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == TURN_ID_HEADER.lower() and value.strip():
            return value.strip()
    return None


def _publish_bus(events: Sequence[Any]) -> None:
    from orion.core.bus.async_service import OrionBusAsync
    from orion.grammar.publish import publish_grammar_event

    url = os.environ.get("ORION_BUS_URL")
    if not url:
        raise RuntimeError("ORION_BUS_URL is unset")

    async def go() -> None:
        bus = OrionBusAsync(url=url)
        await bus.connect()
        try:
            for event in events:  # open, then close: one trace_id, so the writer keeps the order
                await publish_grammar_event(bus, event, source_name=ACT_SOURCE_SERVICE)
        finally:
            await bus.close()

    asyncio.run(asyncio.wait_for(go(), timeout=PUBLISH_TIMEOUT_SEC))


def look_act_events(record: dict[str, Any], *, now: datetime, session_id: str | None) -> tuple[str, list[Any]]:
    """act_opened at the look, act_closed with live (1) or not (0)."""
    from orion.grammar.acts import act_closed_event, act_opened_event, build_act

    camera = record["camera"]
    act = build_act(
        actor=ACTOR, verb="look", key=f"{camera}:{record['looked_at']}", signal_id="artifact_produced",
        horizon_sec=ACT_HORIZON_SEC, basis=ACT_BASIS, object_ref=f"camera:{camera}",
        test_before=f"newest_frame:{camera}",
    )
    common = dict(source_service=ACT_SOURCE_SERVICE, source_component=ACT_SOURCE_COMPONENT, session_id=session_id)
    note = record.get("reason") or f"newest frame {record.get('age_sec')} s old, sha256 {str(record.get('sha256'))[:12]}"
    opened = act_opened_event(act, at=now, **common)
    closed = act_closed_event(act, produced=bool(record["live"]), note=note, at=now, **common)
    return act.act_id, [opened, closed]


def _record_look(log: Path, record: dict[str, Any], *, now: datetime, publish: Publisher) -> str:
    """Append the look to looks.jsonl and put it on the act ledger. The act line for the printout."""
    try:
        act_id, events = look_act_events(record, now=now, session_id=turn_id())
        publish(events)
        record["act_id"] = act_id
        line = f"act: {act_id} on your act ledger (open it: python -P -m orion.dig act {act_id})"
    except Exception as exc:  # noqa: BLE001 -- the look stands; only its ledger entry is missing
        record["act_error"] = f"{type(exc).__name__}: {exc}"[:300]
        line = f"act: not recorded ({record['act_error']})"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return line


def look(word: str, *, home: Path, directory: Path | None = None, now: datetime | None = None,
         publish: Publisher | None = None) -> tuple[list[str], dict]:
    """Copy one camera's newest frame into ``home/look`` and say how to see it."""
    publish = publish or _publish_bus
    stream_id = resolve_stream(word)
    now = now or datetime.now(timezone.utc)
    directory = directory or frames_dir()
    look_root = home / "look"
    log = look_root / LOG_NAME
    used = looks_in_last_hour(log, now)
    if used >= LOOKS_PER_HOUR:
        raise LookError(f"{used} looks in the last hour; the cap is {LOOKS_PER_HOUR}. Look again later.")
    desc = describe_camera(stream_id)

    def failed(reason: str) -> LookError:
        record = {"looked_at": now.isoformat(), "camera": stream_id, "live": False, "reason": reason[:300]}
        act_line = _record_look(log, record, now=now, publish=publish)
        return LookError(f"{reason}\n{act_line}")

    candidates = newest_frames(directory, stream_id)
    if not candidates:
        raise failed(
            f"no saved frame for {desc.name} ({stream_id}) in {directory}. Frames are kept 5 minutes, "
            + ("and this camera only records after motion or a ring." if desc.event_driven
               else "so this camera has saved nothing for at least that long: it may be down.")
        )
    frame, data = None, b""
    for candidate in candidates:
        try:
            data = candidate.path.read_bytes()
        except OSError:
            continue  # swept between the scan and the read
        if is_complete_jpeg(data):
            frame = candidate
            break
    if frame is None:
        raise failed(f"the newest {len(candidates)} frames of {desc.name} ({stream_id}) are incomplete or "
                     "gone; look again in a second.")
    age = max(0.0, (now - frame.taken_at).total_seconds())
    day = frame.taken_at.astimezone(ZONE).date()
    dest_dir = look_root / day.isoformat()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stream_id}-{frame.epoch_ms}.jpg"
    dest.write_bytes(data)
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
    act_line = _record_look(log, record, now=now, publish=publish)
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
        act_line,
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
