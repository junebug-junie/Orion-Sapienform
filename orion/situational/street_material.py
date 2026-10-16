"""The street, as material for Orion's mind (walkway camera, 2026-10-15).

Pure, stdlib-only helpers shared by every consumer that shows Orion what the
walkway camera saw: the situation brief (`perception_reader.fetch_street_summary`),
reverie (`services/orion-thought`), dreams (`services/orion-dream`) and
curiosity study material (`orion/curiosity/study_material.py`). Each consumer
does its own read (they use different drivers and pools); the rules for what
counts as a caption, how duplicates collapse, and how a graded prediction is
worded live here once.

ROOM vs STREET. `vision_events` holds every camera's narratives. The room
readers (`perception_reader._room_percept_stmt`, orion-thought's
`read_recent_vision_events`) read room streams only; these helpers are for the
walkway streams only, and every rendering labels the text as the street. The
two are never merged into one list.

PRIVACY. Narrative text only, the same contract as the room percept: no
`entities`, no image refs. Subject labels on graded expectations are what the
rhythm reducer wrote (`vision_percept_expectation.subject_label`): a
Juniper-given name or a kind ("a dog"), never a face match.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

# Rows the walkway REDUCERS write into vision_events (orion-sql-writer
# vision_rhythm.py / attention-worthy reducer). They are grades and flags, not
# captions of the scene: graded predictions reach consumers through
# `vision_percept_expectation` instead, with their subject and time.
WALKWAY_REDUCER_EVENT_TYPES: frozenset[str] = frozenset(
    {"arrived_as_expected", "expected_absent", "attention_worthy"}
)

CAPTION_MAX_CHARS = 160

# Rows orion-vision-window writes for a detector-agreed VLM caption
# (services/orion-vision-window/app/caption_events.py, 2026-10-16). The text
# is the captioning model's description of ONE frame -- Qwen2-VL invents
# motion ("is pushing", "walks", PR #2737) -- so every rendering labels it as
# a camera caption with the detector boxes that let it through, never as a
# checked fact. Council narratives (any other event_type) render unlabelled,
# as before.
CAMERA_CAPTION_EVENT_TYPE = "camera_caption"
_DETECTOR_TAG_PREFIX = "detector:"


def _tags(raw: Any) -> list[str]:
    """`vision_events.tags` as read by any driver: list, JSON text, or None."""
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(t) for t in raw if str(t).strip()]


def detector_labels_from_tags(raw: Any) -> list[str]:
    """`detector:<label>` tags of a camera_caption row -> ["person", ...]."""
    out: list[str] = []
    for tag in _tags(raw):
        if tag.startswith(_DETECTOR_TAG_PREFIX):
            label = tag[len(_DETECTOR_TAG_PREFIX):].strip()
            if label and label not in out:
                out.append(label)
    return out


# Labels named in the caption label; the rest collapse to "+N" so a busy
# frame (all 7 gate labels) cannot push the brief past the chat-stance clip.
CAPTION_LABELS_NAMED = 3


def label_camera_caption(text: str, detector_labels: Sequence[str], *, at: str | None = None) -> str:
    """'camera caption[ logged HH:MM] (one frame; detector saw person,
    stroller; motion not verified): "A person is pushing a stroller…"'.
    `at` is the row's write time (seconds after the frame), hence "logged"."""
    when = f" logged {at}" if at else ""
    labels = list(detector_labels)
    named = ", ".join(labels[:CAPTION_LABELS_NAMED])
    if len(labels) > CAPTION_LABELS_NAMED:
        named += f" +{len(labels) - CAPTION_LABELS_NAMED}"
    saw = f"detector saw {named}; " if labels else ""
    return f'camera caption{when} (one frame; {saw}motion not verified): "{text}"'
GRADED_STATUSES: tuple[str, ...] = ("met", "missed")
CALIBRATION_LOOKBACK_DAYS = 14


def parse_stream_ids(raw: Any, default: str = "walkway") -> list[str]:
    """Comma-separated stream ids. None -> default; an explicit empty string
    -> [] (the consumer is switched off, never "every camera")."""
    if raw is None:
        raw = default
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _utc(ts: Any) -> datetime | None:
    """datetime (Postgres drivers) or ISO string (SQLite text() reads) -> aware UTC."""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts, datetime):
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _clean(text: Any, limit: int = CAPTION_MAX_CHARS) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _norm_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()


def dedupe_street_captions(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    max_age_sec: float,
    limit: int,
    max_chars: int = CAPTION_MAX_CHARS,
    max_camera_captions: int = 1,
) -> list[dict[str, Any]]:
    """Walkway caption rows in (any order), the newest distinct fresh ones out.

    Each row needs `narrative` and `created_at`; `event_type` and `stream_id`
    are optional. Dropped: blank narratives, reducer rows
    (`WALKWAY_REDUCER_EVENT_TYPES`), rows older than `max_age_sec`, rows with
    no timestamp (age unknown cannot pass a freshness gate), and repeats of a
    caption already kept (case/punctuation-insensitive). Freshness and
    dedupe apply before `limit`, so a repeat never crowds out a distinct one.

    A `camera_caption` row (orion-vision-window, a detector-agreed VLM
    caption) comes back with its narrative already labelled as a caption of
    one frame (`label_camera_caption`), so no consumer can show it as a fact.
    At most `max_camera_captions` of them are kept (newest first): Qwen words
    each frame differently, so text dedupe cannot stop a busy street from
    filling the list with captions and pushing out the council's narrative.

    Returns `{narrative, observed_at, age_sec, stream_id, event_id,
    event_type, caption_text, detector_labels}`, newest first; `caption_text`
    and `detector_labels` are the unlabelled parts (camera captions only).
    """
    if limit <= 0:
        return []
    now = _utc(now) or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=max(0.0, float(max_age_sec)))
    parsed: list[tuple[datetime, Mapping[str, Any]]] = []
    for row in rows or ():
        if str(row.get("event_type") or "") in WALKWAY_REDUCER_EVENT_TYPES:
            continue
        ts = _utc(row.get("created_at"))
        if ts is None or ts < cutoff:
            continue
        parsed.append((ts, row))
    parsed.sort(key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    camera_kept = 0
    for ts, row in parsed:
        text = _clean(row.get("narrative"), max_chars)
        key = _norm_key(text)
        if not key or key in seen:
            continue
        event_type = str(row.get("event_type") or "")
        is_caption = event_type == CAMERA_CAPTION_EVENT_TYPE
        if is_caption:
            if camera_kept >= max_camera_captions:
                continue
            camera_kept += 1
        seen.add(key)
        labels = detector_labels_from_tags(row.get("tags")) if is_caption else []
        out.append(
            {
                "narrative": label_camera_caption(text, labels) if is_caption else text,
                "observed_at": ts,
                "age_sec": max(0.0, (now - ts).total_seconds()),
                "stream_id": str(row.get("stream_id") or ""),
                "event_id": str(row.get("event_id") or ""),
                "event_type": event_type,
                "caption_text": text if is_caption else None,
                "detector_labels": labels,
            }
        )
        if len(out) >= limit:
            break
    return out


def street_caption_line(
    captions: Sequence[Mapping[str, Any]], *, camera: str = "walkway", tz: Any = None
) -> str | None:
    """One labelled sentence for the brief: "On the walkway just now: X; Y."
    None when there is nothing -- never an empty sentence. A camera caption
    renders with its frame's local time when `tz` is given."""
    texts: list[str] = []
    for c in captions or ():
        if c.get("event_type") == CAMERA_CAPTION_EVENT_TYPE and c.get("caption_text"):
            at = None
            observed = _utc(c.get("observed_at"))
            if observed is not None and tz is not None:
                at = f"{observed.astimezone(tz):%H:%M}"
            texts.append(label_camera_caption(str(c["caption_text"]), c.get("detector_labels") or [], at=at))
        else:
            texts.append(str(c.get("narrative") or "").strip().rstrip("."))
    texts = [t for t in texts if t]
    if not texts:
        return None
    camera = (camera or "").strip() or "street camera"
    return f"On the {camera} just now: " + "; ".join(texts) + "."


# --- Graded predictions -------------------------------------------------------


@dataclass(frozen=True)
class GradedExpectation:
    """One met/missed walkway expectation, ready to render."""

    stream_id: str
    subject_key: str
    subject: str
    status: str  # "met" | "missed"
    around: str | None  # expected peak, local HH:MM
    window_start: datetime | None
    scored_at: datetime | None
    expectation_id: str = ""

    @property
    def came(self) -> bool:
        return self.status == "met"


@dataclass(frozen=True)
class SubjectCalibration:
    met: int
    missed: int

    @property
    def decided(self) -> int:
        return self.met + self.missed

    @property
    def hit_rate(self) -> float | None:
        return round(self.met / self.decided, 2) if self.decided else None


def _hhmm(minute_of_day: Any) -> str | None:
    try:
        m = int(minute_of_day)
    except (TypeError, ValueError):
        return None
    return f"{m // 60:02d}:{m % 60:02d}" if 0 <= m < 1440 else None


def build_graded_expectations(rows: Iterable[Mapping[str, Any]]) -> list[GradedExpectation]:
    """Rows from `vision_percept_expectation` in, met/missed only out, newest
    scored first. Open and `unscorable` rows are dropped: an ungraded window
    is not a result, and "could not tell" says nothing about the street."""
    out: list[GradedExpectation] = []
    for row in rows or ():
        status = str(row.get("status") or "")
        label = _clean(row.get("subject_label"), 80)
        key = str(row.get("subject_key") or "").strip()
        if status not in GRADED_STATUSES or not label or not key:
            continue
        out.append(
            GradedExpectation(
                stream_id=str(row.get("stream_id") or ""),
                subject_key=key,
                subject=label,
                status=status,
                around=_hhmm(row.get("peak_minute")),
                window_start=_utc(row.get("window_start")),
                scored_at=_utc(row.get("scored_at")),
                expectation_id=str(row.get("expectation_id") or ""),
            )
        )
    out.sort(key=lambda g: g.scored_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return out


def build_calibration(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], SubjectCalibration]:
    """(stream_id, subject_key) -> 14-day met/missed counts. Rows with no
    decided outcome are dropped: "no evidence" must not read as "always wrong"."""
    out: dict[tuple[str, str], SubjectCalibration] = {}
    for row in rows or ():
        key = str(row.get("subject_key") or "").strip()
        try:
            met, missed = int(row.get("met") or 0), int(row.get("missed") or 0)
        except (TypeError, ValueError):
            continue
        if not key or met + missed <= 0:
            continue
        out[(str(row.get("stream_id") or ""), key)] = SubjectCalibration(met=met, missed=missed)
    return out


def calibration_phrase(cal: SubjectCalibration | None) -> str:
    if cal is None or not cal.decided:
        return ""
    return f"over the last {CALIBRATION_LOOKBACK_DAYS} days it came {cal.met} of {cal.decided} times"


def graded_expectation_line(
    g: GradedExpectation,
    *,
    tz: Any,
    calibration: Mapping[tuple[str, str], SubjectCalibration] | None = None,
) -> str:
    """"On the walkway I expected the black dog around 07:40 (Wed 15 Oct); it
    came. Over the last 14 days it came 4 of 5 times." Plain, one line."""
    camera = g.stream_id or "street camera"
    when = ""
    anchor = g.window_start or g.scored_at
    if anchor is not None:
        when = f" ({anchor.astimezone(tz):%a %d %b})"
    at = f" around {g.around}" if g.around else ""
    outcome = "it came" if g.came else "it did not come"
    line = f"On the {camera} I expected {g.subject}{at}{when}; {outcome}."
    cal = (calibration or {}).get((g.stream_id, g.subject_key))
    phrase = calibration_phrase(cal)
    if phrase:
        line += f" {phrase[:1].upper()}{phrase[1:]}."
    return line


def graded_expectation_lines(
    rows: Iterable[Mapping[str, Any]],
    calibration_rows: Iterable[Mapping[str, Any]] | None,
    *,
    tz: Any,
    limit: int,
) -> list[str]:
    """Newest `limit` graded expectations as plain lines. [] when none."""
    graded = build_graded_expectations(rows)[: max(0, int(limit))]
    cal = build_calibration(calibration_rows or ())
    return [graded_expectation_line(g, tz=tz, calibration=cal) for g in graded]


# SQL, sqlalchemy `text()` style (:named params, expanding :stream_ids). The
# Hub's asyncpg reader keeps its own $-numbered copy in study_material.py.

def _sql_str_list(values: frozenset[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in sorted(values))


# Reducer rows excluded IN SQL (before LIMIT), so a burst of them can never
# crowd distinct captions out of the window; dedupe_street_captions repeats
# the check as a backstop. Built from WALKWAY_REDUCER_EVENT_TYPES: one source.
REDUCER_EVENT_TYPES_SQL = (
    f"COALESCE(event_type, '') NOT IN ({_sql_str_list(WALKWAY_REDUCER_EVENT_TYPES)})"
)

# Camera captions and council narratives are fetched as two LIMITed halves
# (CAMERA_CAPTION_SQL_LIMIT newest captions + :limit newest others): one
# caption every ~8 s on a busy street would otherwise fill the LIMIT and the
# council's row would never be read at all. Shared by every SQL copy.
CAMERA_CAPTION_SQL_LIMIT = 2
CAMERA_CAPTION_SQL = f"event_type = '{CAMERA_CAPTION_EVENT_TYPE}'"
NOT_CAMERA_CAPTION_SQL = f"COALESCE(event_type, '') <> '{CAMERA_CAPTION_EVENT_TYPE}'"


def split_caption_sql(select_where: str, *, order: str, other_limit: str) -> str:
    """`SELECT ... WHERE ...` (no ORDER/LIMIT) -> newest `other_limit`
    non-caption rows UNION ALL newest CAMERA_CAPTION_SQL_LIMIT camera
    captions, newest first. Subqueries keep it valid on Postgres and SQLite."""
    return (
        f"SELECT * FROM ({select_where} AND {NOT_CAMERA_CAPTION_SQL} {order} LIMIT {other_limit}) AS council_rows "
        f"UNION ALL "
        f"SELECT * FROM ({select_where} AND {CAMERA_CAPTION_SQL} {order} LIMIT {CAMERA_CAPTION_SQL_LIMIT}) AS caption_rows "
        f"{order}"
    )


STREET_CAPTIONS_SQL = split_caption_sql(
    "SELECT stream_id, event_type, narrative, tags, created_at FROM vision_events "
    "WHERE stream_id IN :stream_ids "
    "AND narrative IS NOT NULL AND narrative <> '' "
    f"AND {REDUCER_EVENT_TYPES_SQL} "
    "AND created_at >= :since",
    order="ORDER BY created_at DESC",
    other_limit=":limit",
)

GRADED_EXPECTATIONS_SQL = (
    "SELECT expectation_id, stream_id, subject_key, subject_label, status, "
    "       peak_minute, window_start, scored_at "
    "FROM vision_percept_expectation "
    "WHERE stream_id IN :stream_ids AND status IN ('met', 'missed') "
    "AND scored_at >= :since AND scored_at < :until "
    "ORDER BY scored_at DESC LIMIT :limit"
)

CALIBRATION_SQL = (
    "SELECT stream_id, subject_key, "
    "       count(*) FILTER (WHERE status = 'met') AS met, "
    "       count(*) FILTER (WHERE status = 'missed') AS missed "
    "FROM vision_percept_expectation "
    "WHERE stream_id IN :stream_ids AND status IN ('met', 'missed') "
    "AND window_start > :cal_since "
    "GROUP BY stream_id, subject_key"
)
