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
Its expectation: a live frame (at most ``LIVE_MAX_AGE_SEC`` old) comes back. A look that finds a
live camera down or stale closes 0. Not graded (logged, no act): an event camera (the doorbell)
between events, and a look that caught every recent frame mid-write (a read race, not a fault; it
also costs no slot of the hourly cap). A look refused by the cap, or an unknown camera word, is no act.

The act carries the turn's id as ``session_id`` (``ORION_TURN_CORRELATION_ID``, set for per-turn
spawns and warm slots alike) but no ``correlation_id``: the ledger charges an act every GPU lease of
its correlation id, and a look is one image inside a turn, not the turn. The producer states no
cost; the ledger fills ``seconds`` from the real open and close times (the look's own wall time,
milliseconds). The image's model cost (~1,000 prompt tokens, measured 2026-11-13) stays with the turn.
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
# == orion.harness.fcc_motor.TURN_CORRELATION_ENV (tests pin it): set on per-turn spawns and in a warm
# slot's per-turn env file. The header above is the fallback (a spawn from before this key existed).
TURN_ID_ENV = "ORION_TURN_CORRELATION_ID"

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
            if at >= since and not row.get("cap_exempt"):
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
    if (env.get(TURN_ID_ENV) or "").strip():
        return env[TURN_ID_ENV].strip()
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

    sent = 0

    async def go() -> None:
        nonlocal sent
        bus = OrionBusAsync(url=url)
        await bus.connect()
        try:
            for event in events:  # open, then close: one trace_id, so the writer keeps the order
                await publish_grammar_event(bus, event, source_name=ACT_SOURCE_SERVICE)
                sent += 1
        finally:
            await bus.close()

    try:
        asyncio.run(asyncio.wait_for(go(), timeout=PUBLISH_TIMEOUT_SEC))
    except Exception as exc:  # TimeoutError included; never a KeyboardInterrupt
        raise PartialPublish(sent, exc) from exc


class PartialPublish(Exception):
    """``sent`` of the act's events reached the bus before ``cause`` stopped the rest."""

    def __init__(self, sent: int, cause: BaseException) -> None:
        super().__init__(f"{sent} sent; {cause!r}")
        self.sent, self.cause = sent, cause


def camera_word(stream_id: str) -> str:
    """The plain word for a stream (cam0 -> office). Letters only, so self-monitor's faculty name keeps it."""
    return next((w for w, sid in CAMERA_WORDS.items() if sid == stream_id), stream_id)


def look_act_events(record: dict[str, Any], *, opened_at: datetime, closed_at: datetime,
                    session_id: str | None) -> tuple[str, list[Any]]:
    """act_opened when the look began, act_closed when its frame was saved: live (1) or not (0).
    The two real times are what the ledger fills ``seconds`` from (the look's own wall time)."""
    from orion.grammar.acts import act_closed_event, act_opened_event, build_act

    camera = record["camera"]
    act = build_act(
        actor=ACTOR, verb="look", key=f"{camera}:{record['looked_at']}", signal_id="artifact_produced",
        horizon_sec=ACT_HORIZON_SEC, basis=ACT_BASIS, object_ref=f"camera:{camera_word(camera)}",
        test_before=f"newest_frame:{camera}",
    )
    common = dict(source_service=ACT_SOURCE_SERVICE, source_component=ACT_SOURCE_COMPONENT, session_id=session_id)
    note = record.get("reason") or f"newest frame {record.get('age_sec')} s old, sha256 {str(record.get('sha256'))[:12]}"
    opened = act_opened_event(act, at=opened_at, **common)
    closed = act_closed_event(act, produced=bool(record["live"]), note=note, at=max(opened_at, closed_at), **common)
    return act.act_id, [opened, closed]


# --- what Orion's vision records said about this frame (#3347 part 2a) --------------------------

RECORDS_BEFORE_SEC = 90   # a record is written after its frames: council windows run ~20 s behind
# Only records that claim what one frame shows. Rhythm and handoff verdicts (expected_absent,
# arrived_as_expected, handoff_*), attention_worthy and doorbell_motion are not checkable from a frame.
FRAME_RECORD_TYPES = ("visual_observation", "camera_caption", "person_presence")
CHECK_MAX_AGE_SEC = 2 * 3600  # a verdict is filed in the turn that looked, not days later
RECORDS_SHOWN = 6
RECORD_TEXT_CHARS = 160

Records = Callable[[str, datetime], list[dict[str, Any]]]


def records_sql(stream_id: str, frame_at: datetime) -> str:
    """vision_events rows for this stream nearest the frame. Literals only from a fixed set."""
    if stream_id not in STREAMS:
        raise LookError(f"unknown stream {stream_id!r}")
    ts = frame_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00")
    return (
        "SELECT event_id, event_type, round(confidence::numeric, 2) AS confidence, entities::text AS entities, "
        f"left(regexp_replace(coalesce(narrative, ''), '[[:space:]]+', ' ', 'g'), {RECORD_TEXT_CHARS}) AS narrative, "
        f"round(extract(epoch FROM created_at - timestamptz '{ts}')::numeric, 1) AS offset_sec "
        f"FROM vision_events WHERE stream_id = '{stream_id}' "
        f"AND event_type IN ({', '.join(repr(t) for t in FRAME_RECORD_TYPES)}) "
        f"AND created_at BETWEEN timestamptz '{ts}' - interval '{RECORDS_BEFORE_SEC} seconds' "
        f"AND timestamptz '{ts}' + interval '5 seconds' "
        f"ORDER BY abs(extract(epoch FROM created_at - timestamptz '{ts}')) LIMIT {RECORDS_SHOWN}"
    )


def read_records(stream_id: str, frame_at: datetime) -> list[dict[str, Any]]:
    """Read-only, as Orion's own role (orion_readonly may read vision_events)."""
    from orion.body_atlas.sources import psql_rows
    from orion.evals.model_replay.sandbox import ReadOnlyPsql

    return psql_rows(ReadOnlyPsql(timeout_sec=20.0), records_sql(stream_id, frame_at))


def _num(value: Any) -> float | None:
    """psql CSV gives every field as text."""
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _record_line(r: dict[str, Any]) -> str:
    ents = str(r.get("entities") or "").strip()
    ents = "" if ents in ("", "[]", "null") else f" entities {ents}"
    return (f"  {r['id']} {r['type']} conf {r.get('confidence')} ({r.get('offset_sec')} s from the frame)"
            f"{ents}: {r.get('text') or '(no text)'}")


def _publish_into(entry: dict[str, Any], build: Callable[[], tuple[str, list[Any]]], publish: Publisher) -> str:
    """Build and publish one act's two events; write ``act_id`` / ``act_error`` into ``entry``.
    Returns the act line for the printout. A bus failure never fails the caller's own record."""
    act_id = None
    try:
        act_id, events = build()
        publish(events)
        entry["act_id"] = act_id
        return f"act: {act_id} on your act ledger (open it: python -P -m orion.dig act {act_id})"
    except PartialPublish as exc:
        entry["act_error"] = f"{exc.sent} of 2 events sent; {exc.cause!r}"[:300]
        if exc.sent:
            # The open is on the ledger without its close: the resolver will expire it at 0.
            entry["act_id"] = act_id
            return f"act: {act_id} OPENED BUT NOT CLOSED ({entry['act_error']}); it will read as a 0"
        return f"act: not recorded ({entry['act_error']})"
    except Exception as exc:  # noqa: BLE001
        entry["act_error"] = repr(exc)[:300]
        return f"act: not recorded ({entry['act_error']})"


def _record_look(log: Path, record: dict[str, Any], *, now: datetime, publish: Publisher | None,
                 clock: Callable[[], datetime]) -> str:
    """Append the look to looks.jsonl and, when ``publish`` is given, put it on the act ledger.
    The act line for the printout."""
    if publish is None:
        line = f"act: none ({record.get('ungraded', 'not graded')})"
    else:
        line = _publish_into(record, lambda: look_act_events(
            record, opened_at=now, closed_at=clock(), session_id=turn_id()), publish)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return line


def look(word: str, *, home: Path, directory: Path | None = None, now: datetime | None = None,
         publish: Publisher | None = None, clock: Callable[[], datetime] | None = None,
         records: Records | None = None) -> tuple[list[str], dict]:
    """Copy one camera's newest frame into ``home/look`` and say how to see it."""
    publish = publish or _publish_bus
    records = records or read_records
    clock = clock or (lambda: datetime.now(timezone.utc))
    stream_id = resolve_stream(word)
    now = now or datetime.now(timezone.utc)
    directory = directory or frames_dir()
    look_root = home / "look"
    log = look_root / LOG_NAME
    used = looks_in_last_hour(log, now)
    if used >= LOOKS_PER_HOUR:
        raise LookError(f"{used} looks in the last hour; the cap is {LOOKS_PER_HOUR}. Look again later.")
    desc = describe_camera(stream_id)

    def failed(reason: str, *, ungraded: str | None = None) -> LookError:
        """A look that got no frame. Graded 0 (the camera is down), unless ``ungraded`` says why a
        missing frame is the expected state: then logged, no act, and not counted against the cap."""
        record = {"looked_at": now.isoformat(), "camera": stream_id, "live": False, "reason": reason[:300]}
        if ungraded:
            record.update(ungraded=ungraded, cap_exempt=True)
        act_line = _record_look(log, record, now=now, publish=None if ungraded else publish, clock=clock)
        return LookError(f"{reason}\n{act_line}")

    candidates = newest_frames(directory, stream_id)
    if not candidates:
        raise failed(
            f"no saved frame for {desc.name} ({stream_id}) in {directory}. Frames are kept 5 minutes, "
            + ("and this camera only records after motion or a ring." if desc.event_driven
               else "so this camera has saved nothing for at least that long: it may be down."),
            ungraded="an event camera with no recent event is waiting, not broken" if desc.event_driven else None,
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
                     "gone; look again in a second.",
                     ungraded="caught the camera mid-write: a read race, not a camera fault")
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
    # An event camera's stale frame is its last event, not a failure: shown, not graded.
    if not live and desc.event_driven:
        record["ungraded"] = "an event camera between events shows its last event frame"
    # What the vision pipeline wrote about this moment, so Orion can check it against the picture.
    # Kept on the row: `look check` accepts only a record this look showed.
    near: list[dict[str, Any]] = []
    if live:
        try:
            near = [{"id": str(r.get("event_id")), "type": r.get("event_type"), "confidence": _num(r.get("confidence")),
                     "entities": r.get("entities"), "text": r.get("narrative"), "offset_sec": _num(r.get("offset_sec"))}
                    for r in records(stream_id, frame.taken_at)]
            record["records"] = near
        except Exception as exc:  # noqa: BLE001 -- the look stands without them
            record["records_error"] = repr(exc)[:300]
    act_line = _record_look(log, record, now=now, clock=clock,
                            publish=None if (not live and desc.event_driven) else publish)
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
    ]
    if record.get("records_error"):
        lines.append(f"records: unread ({record['records_error']})")
    elif live and not near:
        lines.append(f"records: none for this camera in the {RECORDS_BEFORE_SEC} s before the frame")
    elif near:
        lines.append(f"What your vision records said in the {RECORDS_BEFORE_SEC} s before this frame (your pipeline's "
                     "words, not yours; someone may have left since):")
        lines.extend(_record_line(r) for r in near)
        if record.get("act_id"):
            lines.append(f'Check one against the picture: python -P -m orion.dig look check {record["act_id"]} '
                         'matches|differs|cannot-tell "<what you see>" because <record id>')
        else:
            lines.append("These cannot be checked: this look is not on your act ledger (see the act line).")
    lines += [f"looks in the last hour: {used + 1} of {LOOKS_PER_HOUR}", act_line]
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


# --- look check: one record against what Orion saw (#3347 part 2a) ---------------------------

CHECK_LOG = "checks.jsonl"
CHECK_ACTOR = "orion.look_check"
CHECK_COMPONENT = "dig.look_check"
VERDICTS = ("matches", "differs", "cannot-tell")
SEEN_MIN, SEEN_MAX = 10, 400
CHECK_BASIS = (
    "artifact_produced = the vision record matched what Orion saw in the frame (1) or did not (0), "
    "filed by Orion after reading the frame; cannot-tell is never graded. The cell's mean is the "
    "records' agreement rate with Orion's own look"
)


def check(act_id: str, verdict: str, seen: str, record_id: str, *, home: Path,
          publish: Publisher | None = None, now: datetime | None = None,
          clock: Callable[[], datetime] | None = None) -> tuple[list[str], dict]:
    """File one verdict on one record a look showed. matches/differs: an act (1/0); cannot-tell: logged."""
    from orion.grammar.acts import act_closed_event, act_opened_event, build_act

    publish = publish or _publish_bus
    now = now or datetime.now(timezone.utc)
    clock = clock or (lambda: datetime.now(timezone.utc))
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        raise LookError(f"verdict must be one of {', '.join(VERDICTS)}, not {verdict!r}")
    seen = " ".join(seen.split())
    if not SEEN_MIN <= len(seen) <= SEEN_MAX:
        raise LookError(f"say what you see in {SEEN_MIN}-{SEEN_MAX} characters (got {len(seen)})")
    root = home / "look"
    row = next((r for r in _read_log(root / LOG_NAME) if r.get("act_id") == act_id), None)
    if row is None:
        raise LookError(f"no look with act {act_id} in {root / LOG_NAME}")
    if not row.get("live") or not row.get("path"):
        raise LookError(f"look {act_id} saw no live frame: nothing to check")
    shown = row.get("records") or []
    matches = [r for r in shown if r.get("id") == record_id or (len(record_id) >= 8 and str(r.get("id")).startswith(record_id))]
    if len(matches) != 1:
        ids = ", ".join(str(r.get("id")) for r in shown) or "none"
        raise LookError(f"record {record_id!r} is not one this look showed (shown: {ids})")
    rec = matches[0]
    log = root / CHECK_LOG
    # A check whose act never reached the ledger may be filed again (same act id: the ledger upserts).
    if any(c.get("look_act_id") == act_id and c.get("record_id") == rec["id"]
           and (c.get("act_id") or c.get("verdict") == "cannot-tell") for c in _read_log(log)):
        raise LookError(f"record {rec['id']} was already checked against look {act_id}")
    try:
        looked = datetime.fromisoformat(str(row.get("looked_at")))
    except ValueError:
        looked = None
    if looked is not None and (now - looked).total_seconds() > CHECK_MAX_AGE_SEC:
        raise LookError(f"look {act_id} is over {CHECK_MAX_AGE_SEC // 3600} h old: check what you see now in a new look")
    entry = {
        "checked_at": now.isoformat(), "look_act_id": act_id, "record_id": rec["id"], "record_type": rec.get("type"),
        "record_confidence": rec.get("confidence"), "record_text": rec.get("text"), "verdict": verdict,
        "seen": seen, "camera": row.get("camera"), "frame_at": row.get("frame_at"), "frame_path": row.get("path"),
        "frame_sha256": row.get("sha256"),
    }
    if verdict == "cannot-tell":
        line = "act: none (cannot-tell is logged, never graded)"
    else:
        def build() -> tuple[str, list[Any]]:
            act = build_act(
                actor=CHECK_ACTOR, verb="infer", key=f"{act_id}:{rec['id']}", signal_id="artifact_produced",
                horizon_sec=ACT_HORIZON_SEC, basis=CHECK_BASIS,
                # The middle part names the self-monitor faculty: one per camera and record type.
                object_ref=f"vision_record:{camera_word(str(row.get('camera')))}_{rec.get('type')}:{rec['id']}",
                test_before=f"record:{rec.get('type')}", parent_act_id=act_id,
            )
            common = dict(source_service=ACT_SOURCE_SERVICE, source_component=CHECK_COMPONENT, session_id=turn_id())
            # Fixed text at both ends: self-monitor reads a note starting "abandoned"/"failed:"/... or
            # ending " died" as a run that never happened (orion/self_monitor/adapters.py act_outcome).
            note = f"{verdict}: {seen} (look check)"
            return act.act_id, [act_opened_event(act, at=now, **common),
                                act_closed_event(act, produced=verdict == "matches", note=note,
                                                 at=max(now, clock()), **common)]
        line = _publish_into(entry, build, publish)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    lines = [
        f"check: {verdict} on {rec.get('type')} {rec['id']} (conf {rec.get('confidence')}) for look {act_id}",
        f"you saw: {seen}",
        line,
    ]
    return lines, entry
