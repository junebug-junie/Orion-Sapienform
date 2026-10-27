"""Walkway rollups: what the outdoor camera saw, by hour, day, week and month.

Juniper (2026-10-18): "build the whole daily walkway summary, but actually roll
that into hour intervals, and also have a day version, a week version, and a
month version." Before this, walkway content reached Orion only as "right now"
(street line, last few captions); nothing remembered the street over time, and
the nightly walkway journal invented "the walkway has been empty on at least
five occasions" with no rows behind it.

Deterministic, no LLM. ``orion-sql-writer`` (app/vision_rollup_loop.py) reads
the raw tables and writes one ``vision_rollup`` row per (stream, grain, period).
This module is the pure part: periods, the hour build from raw rows, the parent
build (day from hours, week and month from days, so every grain agrees), and the
one-line plain-English template.

What is counted (and what is not):

* A **passing** is one individual seen in one clock hour. A person who walks by
  at 08:10 and back at 17:30 is 2 passings and 1 distinct individual. Several
  sightings of one individual in the same hour are 1 passing, so a detector
  that flickers on one box cannot inflate the count. Day passings are the sum
  of its hours; week and month passings the sum of their days.
* Not counted, and tallied under ``excluded`` so the gap is visible:
  - ``stationary``: a sighting or individual the individuals reducer marked
    stationary (a parked car, a prop);
  - ``prop``: individuals listed in ``VISION_MOTION_PROP_INDIVIDUAL_IDS``
    (the patio skeleton);
  - ``patio``: sightings in a no-embed zone (``config/vision_zones.yaml``
    ``embed: false``; the street line leaves them out too). None today: the
    patio was the only one, until Juniper lifted it on 2026-10-15;
  - ``parked_repeat``: a vehicle individual seen in ``parked_min_hours`` or more
    different clock hours of the trailing 24 h. Its first appearances count; a
    car that keeps reappearing all day is a parked car the stationary rule
    missed (live 2026-10-16: one vehicle individual, 54 sightings in a day).
* **Passings are an upper bound.** The individuals reducer can split one person
  into several appearance clusters (live 2026-10-16 12:00 MDT: 79 person
  clusters in one hour while no frame held more than 5 people). So every hour
  also records ``peak_at_once``: the most of each kind detected in a single
  vision window (``vision_scene_inventory.counts``) -- a lower bound from a
  different producer than the sightings. The two together bracket the truth.
* **People are always a range (rules_version 2, 2026-10-20).** Juniper: every
  Orion-facing people count prints "at least <most at once>, at most <clusters>",
  never a bare cluster count. Each rollup with people carries
  ``kinds.person.count_range = {low, high, method_low, method_high}``:
  low is the most people in one vision window (``peak_at_once.person``; for a day,
  week or month the largest hourly low), less any prop individual of kind person
  (the patio skeleton is detected as a person) sighted that hour, and never below 1
  when a person was sighted; high is the person clusters (passings), raised to low
  when a person was in frame but excluded from passings (stationary). So
  ``low <= high`` at every grain. One-frame glimpse stories do not count (spec
  decision 6 is open). Rows written before version 2 have no ``count_range``:
  ``person_count_range`` derives it from their fields and ``display_line``
  re-renders their line, so a reader never shows the old bare count.
* **Named people come from the person record (rules_version 3, 2026-10-24).** Juniper:
  the day line should know people, not cluster names. ``named`` is keyed by the person's
  ``display_name`` when the sighted individual has an active ``vision_individual`` link
  (``person_link``, PR #2852) to an active ``person``; several clusters of one person merge
  into one entry (times and sightings unioned, ``individual_ids`` kept, ``person_ids``
  added). The record decides which clusters are one person; names are never re-matched
  here. An individual linked to a ``not_a_person`` row is not named. A labelled non-person
  cluster is not named either: when it has an active ``vision_belonging`` link to an active
  person it goes under ``belongings`` (the line says "<name>'s bicycle"), else it is left
  out (a vehicle answered "no"). An unlinked person cluster keeps its typed label. When the
  person tables cannot be read, ``persons`` is None and the rollup groups by label as
  before (rules_version 2 behavior). Hours store ``person_record`` (a digest of the links)
  so a changed link rebuilds the recent hours once. Older rows keep their stored line.
* **Camera minutes** come from ``vision_scene_inventory``: one row per vision
  window, written whether or not anything was seen, so a minute with no row is a
  minute the camera did not deliver frames. An hour the camera saw for fewer
  than ``DOWN_MAX_MINUTES`` reads "camera was down", never "nobody came".
  Decode corruption is not stored anywhere (only the retina's live /health has
  it); corrupt frames the retina drops show up here as missing minutes.

Times are local (``America/Denver`` by default). Hours are UTC hours (Denver's
offsets are whole hours, so they are also local hours, and the repeated
01:00 on the fall-back night is two separate hours). Days run local midnight to
local midnight (23 or 25 hours across a DST change), weeks Monday to Monday
(ISO), months first to first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

SCHEMA = "vision_rollup.v1"
RULES_VERSION = 3
# The first rules_version whose ``named`` is keyed by the person record (person_link).
PERSON_RECORD_RULES_VERSION = 3
# The first rules_version whose stored ``line`` shows people as a range. Older rows are
# re-rendered at read time (display_line).
COUNT_RANGE_RULES_VERSION = 2
# Kinds counted as a range. Vehicles keep their count until they are checked the same way.
RANGE_KINDS = ("person",)
METHOD_LOW = "most_in_one_window"            # vision_scene_inventory.counts, max per hour
METHOD_LOW_LESS_PROP = "most_in_one_window_less_props"
METHOD_LOW_SIGHTED = "one_sighting"          # no window count, but a person was sighted
METHOD_LOW_UNCORRECTED = "most_in_one_window_uncorrected"  # pre-v2 row: props not taken out
METHOD_HIGH = "appearance_clusters"          # individuals per hour (passings), summed
METHOD_HIGH_RAISED = "raised_to_low"         # clusters below the window count (stationary)
SPLIT_NOTE = "the camera splits one person into several"
GRAINS = ("hour", "day", "week", "month")
KINDS = ("person", "dog", "bicycle", "stroller", "vehicle")
KIND_PLURAL = {"person": "people", "dog": "dogs", "bicycle": "bicycles", "stroller": "strollers",
               "vehicle": "vehicles", "other": "other"}
KIND_SINGULAR = {"person": "person", "dog": "dog", "bicycle": "bicycle", "stroller": "stroller",
                 "vehicle": "vehicle", "other": "other"}

# Camera coverage of one hour, from vision_scene_inventory minutes.
DOWN_MAX_MINUTES = 15       # fewer seen minutes than this: the camera was down
PARTIAL_MAX_MINUTES = 50    # fewer than this (but not down): partial hour
# Baselines need at least this many earlier periods, else "no baseline yet".
MIN_BASELINE_PERIODS = 3
HOUR_BASELINE_DAYS = 7
DAY_BASELINE_DAYS = 7
WEEK_BASELINE_WEEKS = 4
MONTH_BASELINE_MONTHS = 3
# A day counts an unnamed individual as a regular when it was also seen on an
# earlier day of the trailing week (so: seen on 2+ days of the last 7).
REGULAR_LOOKBACK_DAYS = 6
MAX_HOUR_CAPTIONS = 3
MAX_PARENT_CAPTIONS = 5
MAX_HOUR_STORIES = 3
MAX_DAY_STORIES = 5
MAX_NAMED_TIMES = 12
MAX_NAMED_DAYS = 31
DEFAULT_PARKED_MIN_HOURS = 4


# --- periods --------------------------------------------------------------------------------------


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def local_midnight(d: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, time.min, tzinfo=tz).astimezone(timezone.utc)


@dataclass(frozen=True)
class Period:
    grain: str
    start: datetime  # UTC
    end: datetime    # UTC, exclusive

    @property
    def key(self) -> tuple[str, datetime]:
        return self.grain, self.start


def hour_period(ts: datetime) -> Period:
    t = _utc(ts).replace(minute=0, second=0, microsecond=0)
    return Period("hour", t, t + timedelta(hours=1))


def day_period(d: date, tz: ZoneInfo) -> Period:
    return Period("day", local_midnight(d, tz), local_midnight(d + timedelta(days=1), tz))


def week_start_date(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_period(d: date, tz: ZoneInfo) -> Period:
    mon = week_start_date(d)
    return Period("week", local_midnight(mon, tz), local_midnight(mon + timedelta(days=7), tz))


def month_first(d: date) -> date:
    return d.replace(day=1)


def next_month_first(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def month_period(d: date, tz: ZoneInfo) -> Period:
    first = month_first(d)
    return Period("month", local_midnight(first, tz), local_midnight(next_month_first(first), tz))


def local_date(ts: datetime, tz: ZoneInfo) -> date:
    return _utc(ts).astimezone(tz).date()


def period_for(grain: str, ts: datetime, tz: ZoneInfo) -> Period:
    if grain == "hour":
        return hour_period(ts)
    d = local_date(ts, tz)
    return {"day": day_period, "week": week_period, "month": month_period}[grain](d, tz)


def child_periods(p: Period, tz: ZoneInfo) -> list[Period]:
    """Hours of a day; days of a week or month."""
    out: list[Period] = []
    if p.grain == "day":
        t = p.start
        while t < p.end:
            out.append(Period("hour", t, t + timedelta(hours=1)))
            t += timedelta(hours=1)
    elif p.grain in ("week", "month"):
        d = local_date(p.start, tz)
        while local_midnight(d, tz) < p.end:
            out.append(day_period(d, tz))
            d += timedelta(days=1)
    return out


def child_grain(grain: str) -> Optional[str]:
    return {"day": "hour", "week": "day", "month": "day"}.get(grain)


def _ambiguous(local: datetime) -> bool:
    """True for a wall time that happens twice (the fall-back night)."""
    other = local.replace(fold=1 - local.fold)
    return other.utcoffset() != local.utcoffset()


def hhmm(ts: datetime, tz: ZoneInfo) -> str:
    local = _utc(ts).astimezone(tz)
    s = local.strftime("%H:%M")
    return f"{s} {local.tzname()}" if _ambiguous(local) else s


def period_label(p: Period, tz: ZoneInfo) -> str:
    start = p.start.astimezone(tz)
    if p.grain == "hour":
        return f"{start:%Y-%m-%d} {hhmm(p.start, tz)}"
    if p.grain == "day":
        return f"{start:%Y-%m-%d}"
    if p.grain == "week":
        iso = start.date().isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return f"{start:%Y-%m}"


def local_fields(p: Period, tz: ZoneInfo) -> dict[str, Any]:
    """The clock and calendar words a line uses, so every number in a line is a field."""
    start = p.start.astimezone(tz)
    last = (p.end - timedelta(seconds=1)).astimezone(tz)
    return {"date": start.date().isoformat(), "last_date": last.date().isoformat(), "year": start.year,
            "start": hhmm(p.start, tz), "end": hhmm(p.end, tz)}


def _day_words(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b}"


def _when_words(p: Period, tz: ZoneInfo) -> str:
    start = p.start.astimezone(tz)
    if p.grain == "hour":
        return f"{hhmm(p.start, tz)}–{hhmm(p.end, tz)} ({_day_words(start.date())})"
    if p.grain == "day":
        return f"{_day_words(start.date())} {start.year}"
    if p.grain == "week":
        return f"week of {_day_words(start.date())} {start.year}"
    return f"{start:%B %Y}"


# --- raw rows -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SightingRow:
    sighting_id: str
    individual_id: str
    started_at: datetime
    kind: str
    label: Optional[str] = None
    zone_no_embed: bool = False
    stationary: bool = False             # the sighting
    individual_stationary: bool = False  # the individual


@dataclass(frozen=True)
class CaptionRow:
    event_id: str
    created_at: datetime
    text: str
    detector: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoryRow:
    story_id: str
    started_at: datetime
    kind: str
    sentence: str
    name_label: Optional[str] = None


SOURCE_INDIVIDUAL = "vision_individual"
SOURCE_BELONGING = "vision_belonging"


@dataclass(frozen=True)
class PersonRef:
    """One active ``person_link`` for an individual, with its person row."""
    person_id: str
    display_name: str
    status: str = "active"            # active | not_a_person
    source_kind: str = SOURCE_INDIVIDUAL


def person_record_digest(persons: Optional[Mapping[str, PersonRef]]) -> Optional[str]:
    """Short digest of the links the rollup grouped by; None when the record was unreadable."""
    if persons is None:
        return None
    import hashlib

    raw = "\n".join(f"{iid}|{r.person_id}|{r.display_name}|{r.status}|{r.source_kind}"
                    for iid, r in sorted(persons.items()))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _name_bucket(iid: str, kind: str, label: Optional[str],
                 persons: Optional[Mapping[str, PersonRef]]) -> Optional[tuple[str, str, Optional[str]]]:
    """(bucket, key, person_id) for a sighted individual: bucket "named" or "belongings";
    None when it is not named. ``persons`` None = no person record: group by label."""
    if persons is None:
        return ("named", label, None) if label else None
    ref = persons.get(iid)
    if ref is not None:
        if ref.status != "active" or not ref.display_name:
            return None
        if ref.source_kind == SOURCE_INDIVIDUAL:
            return "named", ref.display_name, ref.person_id
        if ref.source_kind == SOURCE_BELONGING and kind != "person":
            return "belongings", ref.display_name, ref.person_id
        return None
    if label and kind == "person":
        return "named", label, None
    return None


@dataclass
class HourInputs:
    stream_id: str
    period: Period
    sightings: Sequence[SightingRow] = ()
    camera_minutes: int = 0
    peak_at_once: Mapping[str, int] = field(default_factory=dict)
    captions: Sequence[CaptionRow] = ()
    stories: Sequence[StoryRow] = ()
    story_gate: Mapping[str, Any] = field(default_factory=dict)
    parked_ids: frozenset = frozenset()
    prop_ids: frozenset = frozenset()
    baseline: Optional[Mapping[str, Any]] = None
    # individual_id -> active person link. None: the person tables were unreadable.
    persons: Optional[Mapping[str, PersonRef]] = None


_VEHICLE_WORDS = ("vehicle", "car", "truck", "bus", "van", "motorcycle", "suv", "pickup")


def _kind(raw: Any) -> str:
    """Detector/individual label -> one of KINDS or "other". Compound labels from the live
    scene counts ("mail truck", "vehicle truck", "bicycle vehicle") map by their words:
    bicycle first (a bicycle box often also fires "vehicle"), then the vehicle words."""
    k = str(raw or "").strip().lower()
    if k in KINDS:
        return k
    words = set(k.replace("_", " ").replace("-", " ").split())
    for kind in ("bicycle", "stroller", "dog", "person"):
        if kind in words:
            return kind
    if words & set(_VEHICLE_WORDS):
        return "vehicle"
    return "other"


def parked_vehicle_ids(trailing: Iterable[SightingRow], min_hours: int = DEFAULT_PARKED_MIN_HOURS) -> frozenset:
    """Vehicle individuals seen in ``min_hours`` or more different clock hours of the trailing rows."""
    hours: dict[str, set] = {}
    for s in trailing:
        if _kind(s.kind) != "vehicle":
            continue
        hours.setdefault(s.individual_id, set()).add(hour_period(s.started_at).start)
    return frozenset(i for i, h in hours.items() if len(h) >= max(1, int(min_hours)))


def _peaks(raw: Mapping[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        kind = _kind(k)
        if n > out.get(kind, 0):
            out[kind] = n
    return out


def window_peak(counts: Mapping[str, Any] | None, kind: str = "person") -> int:
    """How many of ``kind`` one vision window held (``vision_scene_inventory.counts``), with
    compound detector labels mapped like the rollup does (max, never a sum)."""
    return _peaks(counts).get(kind, 0)


def _max_peaks(rows: Iterable[Mapping[str, Any] | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for k, v in ((r or {}).get("peak_at_once") or {}).items():
            out[k] = max(out.get(k, 0), int(v))
    return out


def count_range(peak: Any, passings: Any, *, props: int = 0) -> dict[str, Any]:
    """People as a range for one hour. ``peak``: most in one vision window; ``passings``: the
    person clusters; ``props``: prop individuals of this kind sighted in the hour (each may be
    one of the window's detections, so the floor drops by that many)."""
    peak, high, props = max(0, int(peak or 0)), max(0, int(passings or 0)), max(0, int(props or 0))
    low, method_low = peak, METHOD_LOW
    if props and peak:
        low, method_low = max(0, peak - props), METHOD_LOW_LESS_PROP
    if high and low < 1:
        low, method_low = 1, METHOD_LOW_SIGHTED
    method_high = METHOD_HIGH
    if high < low:
        high, method_high = low, METHOD_HIGH_RAISED
    return {"low": low, "high": high, "method_low": method_low, "method_high": method_high}


def person_count_range(rollup: Mapping[str, Any] | None, kind: str = "person") -> Optional[dict[str, Any]]:
    """The stored range, or (a row from before rules_version 2) one derived from its fields:
    ``peak_at_once`` (already the max over hours for a parent) and the kind's passings. None
    when the kind did not pass."""
    if not isinstance(rollup, Mapping):
        return None
    k = (rollup.get("kinds") or {}).get(kind) or {}
    stored = k.get("count_range")
    if isinstance(stored, Mapping) and "low" in stored and "high" in stored:
        return dict(stored)
    passings = int(k.get("passings", 0) or 0)
    if passings <= 0:
        return None
    r = count_range(((rollup.get("peak_at_once") or {}).get(kind)) or 0, passings)
    if r["method_low"] == METHOD_LOW:
        r["method_low"] = METHOD_LOW_UNCORRECTED
    return r


def _parent_range(rows: Sequence[Mapping[str, Any]], passings: int, kind: str = "person") -> dict[str, Any]:
    """Low: the largest child low (most at once in any one hour). High: the sum of the child
    highs (a child whose clusters were raised to its low keeps that raise), never below the
    summed clusters or the low."""
    kids = [r for r in (person_count_range(c, kind) for c in rows) if r]
    low, method_low = max(((r["low"], r["method_low"]) for r in kids), default=(0, METHOD_LOW))
    clusters = max(0, int(passings))
    high = max(clusters, sum(int(r["high"]) for r in kids))
    method_high = METHOD_HIGH if high == clusters else METHOD_HIGH_RAISED
    if high < low:
        high, method_high = low, METHOD_HIGH_RAISED
    return {"low": int(low), "high": high, "method_low": method_low, "method_high": method_high}


def range_phrase(r: Mapping[str, Any] | None, kind: str = "person") -> str:
    """"people: at least 5, at most 79 (the camera splits one person into several)"."""
    if not r:
        return ""
    low, high = int(r.get("low", 0)), int(r.get("high", 0))
    s = f"{KIND_PLURAL[kind]}: at least {low}, at most {high}"
    return s + (f" ({SPLIT_NOTE})" if high > low else "")


def display_line(line: Any, rollup: Any) -> str:
    """The line a reader may show. The stored line only when the row is known to be
    rules_version COUNT_RANGE_RULES_VERSION or later; an older row is re-rendered from its own
    fields; anything unreadable (no rollup, bad JSON, render error) is "" -- never the stored
    line, which may hold a bare people count."""
    import json as _json

    r = rollup
    if isinstance(r, (str, bytes)):
        try:
            r = _json.loads(r)
        except ValueError:
            return ""
    if not isinstance(r, Mapping):
        return ""
    try:
        version = int(r.get("rules_version") or 1)
    except (TypeError, ValueError):
        version = 1
    if version >= COUNT_RANGE_RULES_VERSION:
        return str(line or "").strip()
    try:
        return render_line(r)
    except Exception:  # noqa: BLE001 -- an unreadable old row shows nothing, not a bare count
        return ""


def _peak_phrase(peaks: Mapping[str, int]) -> str:
    """"most at once 2 dogs" -- the kind with the highest single-window count. Vehicles are
    left out: the scene counts include parked cars, so their peak says nothing about passing.
    People are left out too: their window count is the low end of their range."""
    best = max(((int(v), k) for k, v in (peaks or {}).items()
                if int(v) > 0 and k not in ("vehicle", "other", *RANGE_KINDS)), default=None)
    if best is None:
        return ""
    n, k = best
    return f"most at once {n} {KIND_SINGULAR[k] if n == 1 else KIND_PLURAL[k]}"


def camera_status(up: int, expected: int, *, saw_something: bool = False) -> str:
    """ok / partial / down from seen minutes. Too few minutes but sightings anyway is
    "unknown": the frame record is incomplete (inventory writer behind or not yet
    running), not a camera that was down -- something WAS seen."""
    if expected <= 0:
        return "unknown"
    scale = expected / 60.0
    if up < DOWN_MAX_MINUTES * scale:
        return "unknown" if saw_something else "down"
    if up < PARTIAL_MAX_MINUTES * scale:
        return "partial"
    return "ok"


def _norm_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()


def _add_kind(kinds: dict, kind: str, *, passings: int = 0, distinct: int = 0, sightings: int = 0) -> None:
    k = kinds.setdefault(kind, {"passings": 0, "distinct": 0, "sightings": 0})
    k["passings"] += passings
    k["distinct"] += distinct
    k["sightings"] += sightings


def _totals(kinds: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {f: sum(int(v.get(f, 0)) for v in kinds.values()) for f in ("passings", "distinct", "sightings")}


def _story_block(gate: Mapping[str, Any], stories: Sequence[StoryRow], max_sentences: int, tz: ZoneInfo,
                 ) -> dict[str, Any]:
    g = dict(gate.get("gate") or {}) if gate else {}
    block: dict[str, Any] = {"gate_passed": bool(g.get("passed")), "checked": int(gate.get("checked", 0) or 0)
                             if gate else 0}
    if not block["gate_passed"]:
        return block  # Hub only until Juniper's spot-check passes: no story content here
    by_kind: dict[str, int] = {}
    for s in stories:
        by_kind[_kind(s.kind)] = by_kind.get(_kind(s.kind), 0) + 1
    block["count"] = len(stories)
    block["by_kind"] = by_kind
    block["sentences"] = [{"at": hhmm(s.started_at, tz), "text": s.sentence} for s in
                          sorted(stories, key=lambda s: s.started_at)[:max_sentences] if s.sentence]
    return block


def build_hour(inp: HourInputs, tz: ZoneInfo) -> dict[str, Any]:
    p = inp.period
    excluded = {"stationary": 0, "prop": 0, "patio": 0, "parked_repeat": 0}
    prop_seen: dict[str, set] = {}
    per_ind: dict[str, dict[str, Any]] = {}
    for s in sorted(inp.sightings, key=lambda r: r.started_at):
        if not (p.start <= _utc(s.started_at) < p.end):
            continue
        if s.individual_id in inp.prop_ids:
            excluded["prop"] += 1
            prop_seen.setdefault(_kind(s.kind), set()).add(s.individual_id)
            continue
        if s.stationary or s.individual_stationary:
            excluded["stationary"] += 1
            continue
        if s.zone_no_embed:
            excluded["patio"] += 1
            continue
        if s.individual_id in inp.parked_ids:
            excluded["parked_repeat"] += 1
            continue
        at = hhmm(s.started_at, tz)
        ind = per_ind.setdefault(s.individual_id, {
            "kind": _kind(s.kind), "label": (s.label or None), "sightings": 0, "first": at, "times": []})
        ind["sightings"] += 1
        ind["times"].append(at)
    kinds: dict[str, dict[str, int]] = {}
    buckets: dict[str, dict[str, dict[str, Any]]] = {"named": {}, "belongings": {}}
    for iid, ind in per_ind.items():
        _add_kind(kinds, ind["kind"], passings=1, distinct=1, sightings=ind["sightings"])
        hit = _name_bucket(iid, ind["kind"], ind["label"], inp.persons)
        if hit is None:
            continue
        bucket, name, pid = hit
        n = buckets[bucket].setdefault(name, {"times": [], "sightings": 0, "individual_ids": []})
        n["times"].extend(ind["times"])
        n["sightings"] += ind["sightings"]
        n["individual_ids"].append(iid)
        if pid:
            n.setdefault("person_ids", set()).add(pid)
        if bucket == "belongings":
            n.setdefault("kinds", set()).add(ind["kind"])
    for group in buckets.values():
        for n in group.values():
            n["times"] = sorted(dict.fromkeys(n["times"]))[:MAX_NAMED_TIMES]
            n["individual_ids"] = sorted(n["individual_ids"])
            for k in ("person_ids", "kinds"):
                if k in n:
                    n[k] = sorted(n[k])
    named, belongings = buckets["named"], buckets["belongings"]
    individuals = {iid: {"kind": v["kind"], "label": v["label"], "sightings": v["sightings"], "first": v["first"]}
                   for iid, v in sorted(per_ind.items())}

    captions: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for c in sorted(inp.captions, key=lambda r: r.created_at):
        key = _norm_text(c.text or "")
        if not key or key in seen_text or not (p.start <= _utc(c.created_at) < p.end):
            continue
        seen_text.add(key)
        captions.append({"at": hhmm(c.created_at, tz), "text": " ".join(c.text.split()), "detector": list(c.detector)})
        if len(captions) >= MAX_HOUR_CAPTIONS:
            break

    peaks = _peaks(inp.peak_at_once)
    for k in RANGE_KINDS:
        if k in kinds:
            kinds[k]["count_range"] = count_range(peaks.get(k, 0), kinds[k]["passings"],
                                                  props=len(prop_seen.get(k, ())))
    up = max(0, min(60, int(inp.camera_minutes)))
    status = camera_status(up, 60, saw_something=bool(per_ind) or any(excluded.values()))
    rollup = {
        "schema": SCHEMA, "rules_version": RULES_VERSION, "stream_id": inp.stream_id, "grain": "hour",
        "period_start": p.start.isoformat(), "period_end": p.end.isoformat(), "label": period_label(p, tz),
        "tz": tz.key, "local": local_fields(p, tz), "complete": True,
        "kinds": kinds, "total": _totals(kinds), "excluded": excluded,
        "peak_at_once": peaks,
        "named": named, "belongings": belongings, "individuals": individuals,
        "person_record": {"available": inp.persons is not None, "digest": person_record_digest(inp.persons)},
        "camera": {"up_minutes": up, "expected_minutes": 60, "status": status,
                   "down_spans": [f"{hhmm(p.start, tz)}–{hhmm(p.end, tz)}"] if status == "down" else []},
        "captions": captions,
        "stories": _story_block(inp.story_gate, [s for s in inp.stories if p.start <= _utc(s.started_at) < p.end],
                                MAX_HOUR_STORIES, tz),
        "baseline": dict(inp.baseline) if inp.baseline else None,
    }
    return rollup


# --- parents (day from hours; week and month from days) -------------------------------------------


def _spread(items: Sequence[Any], n: int) -> list[Any]:
    """Up to ``n`` items spread evenly over the sequence (first and last included)."""
    if len(items) <= n:
        return list(items)
    if n <= 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def _merge_spans(children: Sequence[tuple[Period, Optional[Mapping[str, Any]]]], tz: ZoneInfo,
                 reason: str) -> list[str]:
    """Consecutive children with ``reason`` -- "down" (the camera did not deliver frames) or
    "missing" (no row: not built yet, or the reducer was behind) -- as "HH:MM–HH:MM" or day spans."""
    spans: list[list[Period]] = []
    for per, row in children:
        if row is None:
            hit = reason == "missing"
        else:
            hit = reason == "down" and (row.get("camera") or {}).get("status") == "down"
        if hit:
            if spans and spans[-1][-1].end == per.start:
                spans[-1].append(per)
            else:
                spans.append([per])
    out = []
    for run in spans:
        a, b = run[0], run[-1]
        if a.grain == "hour":
            out.append(f"{hhmm(a.start, tz)}–{hhmm(b.end, tz)}")
        else:
            da, db = local_date(a.start, tz), local_date(b.start, tz)
            out.append(_day_words(da) if da == db else f"{_day_words(da)}–{_day_words(db)}")
    return out


def _merge_named(covered: Sequence[tuple[Period, Optional[Mapping[str, Any]]]], key: str, grain: str,
                 tz: ZoneInfo) -> dict[str, dict[str, Any]]:
    """Union the children's ``named`` (or ``belongings``) entries by name: sightings summed,
    individual ids, person ids and kinds unioned; times (day) or days (week/month)."""
    merged: dict[str, dict[str, Any]] = {}
    for per, r in covered:
        if r is None:
            continue
        day_key = local_date(per.start, tz).isoformat()
        for label, v in (r.get(key) or {}).items():
            v = v or {}
            n = merged.setdefault(label, {"sightings": 0, "individual_ids": set(), "person_ids": set(),
                                          "kinds": set(), "times": [], "days": []})
            n["sightings"] += int(v.get("sightings", 0) or 0)
            n["individual_ids"].update(v.get("individual_ids") or [])
            n["person_ids"].update(v.get("person_ids") or [])
            n["kinds"].update(v.get("kinds") or [])
            if grain == "day":
                n["times"].extend(t for t in (v.get("times") or []) if t not in n["times"])
            else:
                for d in (v.get("days") or [day_key]):
                    if d not in n["days"]:
                        n["days"].append(d)
    out: dict[str, dict[str, Any]] = {}
    for label, n in sorted(merged.items()):
        entry: dict[str, Any] = {"sightings": n["sightings"], "individual_ids": sorted(n["individual_ids"])}
        for k in ("person_ids", "kinds"):
            if n[k]:
                entry[k] = sorted(n[k])
        if grain == "day":
            entry["times"] = n["times"][:MAX_NAMED_TIMES]
        else:
            entry["days"] = sorted(n["days"])[:MAX_NAMED_DAYS]
        out[label] = entry
    return out


def _belonging_name(name: str, v: Mapping[str, Any]) -> str:
    """"Emerald's bicycle"."""
    kinds = [KIND_SINGULAR.get(k, str(k)) for k in (v.get("kinds") or [])] or ["thing"]
    return f"{name}'s {' and '.join(kinds)}"


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 2) if values else None


def build_parent(
    stream_id: str,
    period: Period,
    children: Sequence[tuple[Period, Optional[Mapping[str, Any]]]],
    tz: ZoneInfo,
    *,
    covered_until: Optional[datetime] = None,
    history: Sequence[Mapping[str, Any]] = (),
    story_gate: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Combine child rollups. ``children`` lists every child period of ``period`` up to
    ``covered_until`` (a partial parent stops there), each with its row or None when the
    child has no row (not computed, or the reducer was behind).

    ``history`` holds earlier rollups of the same grain, newest first: earlier days for a
    day (hourly baselines, regulars), earlier weeks/months for those (total baseline).
    """
    grain = period.grain
    # A partial parent covers its children that started before ``covered_until``; the last
    # one may itself be partial (today's day row inside this week), and says so in its row.
    covered = [(p, r) for p, r in children if covered_until is None or p.start < covered_until]
    rows = [r for _, r in covered if r is not None]
    complete = covered_until is None or covered_until >= period.end
    complete = complete and len(rows) == len(child_periods(period, tz)) and all(r.get("complete") for r in rows)

    kinds: dict[str, dict[str, int]] = {}
    excluded = {"stationary": 0, "prop": 0, "patio": 0, "parked_repeat": 0}
    for r in rows:
        for k, v in (r.get("kinds") or {}).items():
            _add_kind(kinds, k, passings=int(v.get("passings", 0)), sightings=int(v.get("sightings", 0)))
        for k, v in (r.get("excluded") or {}).items():
            excluded[k] = excluded.get(k, 0) + int(v or 0)

    # Distinct individuals across the period, and (week/month) on how many days each was seen.
    ind_info: dict[str, dict[str, Any]] = {}
    for per, r in covered:
        if r is None:
            continue
        day_key = local_date(per.start, tz).isoformat()
        for iid, v in (r.get("individuals") or r.get("individual_days") or {}).items():
            info = ind_info.setdefault(iid, {"kind": v.get("kind") or "other", "label": v.get("label"),
                                             "sightings": 0, "days": set(), "first": v.get("first")})
            info["sightings"] += int(v.get("sightings", 0) or 0)
            if "days" in v and isinstance(v["days"], list):
                info["days"].update(v["days"])
            else:
                info["days"].add(day_key)
            if v.get("label") and not info["label"]:
                info["label"] = v["label"]
    for info in ind_info.values():
        _add_kind(kinds, info["kind"], distinct=1)
    for k in RANGE_KINDS:
        if int((kinds.get(k) or {}).get("passings", 0)) > 0:
            kinds[k]["count_range"] = _parent_range(rows, kinds[k]["passings"], k)

    named_out = _merge_named(covered, "named", grain, tz)
    belongings_out = _merge_named(covered, "belongings", grain, tz)

    # Regulars: unnamed individuals seen on 2+ days (day: this day plus an earlier day of
    # the trailing week, from earlier day rows; week/month: 2+ days inside the period).
    regular_ids: list[str] = []
    if grain == "day":
        earlier: set[str] = set()
        floor = period.start - timedelta(days=REGULAR_LOOKBACK_DAYS + 1)
        for h in list(history)[:REGULAR_LOOKBACK_DAYS]:
            try:
                if _parse(str(h.get("period_start"))) < floor:
                    continue
            except (TypeError, ValueError):
                continue
            earlier.update((h.get("individuals") or {}).keys())
        regular_ids = sorted(i for i, v in ind_info.items() if not v["label"] and i in earlier)
    else:
        regular_ids = sorted(i for i, v in ind_info.items() if not v["label"] and len(v["days"]) >= 2)
    reg_by_kind: dict[str, int] = {}
    for i in regular_ids:
        k = ind_info[i]["kind"]
        reg_by_kind[k] = reg_by_kind.get(k, 0) + 1

    up = sum(int((r.get("camera") or {}).get("up_minutes", 0)) for r in rows)
    def _expected(per: Period, row: Optional[Mapping[str, Any]]) -> int:
        cam = (row or {}).get("camera") or {}
        if "expected_minutes" in cam:
            return int(cam["expected_minutes"])
        end = min(per.end, covered_until) if covered_until else per.end
        return max(0, int((end - per.start).total_seconds() // 60))

    expected = sum(_expected(per, r) for per, r in covered)
    # Status over the children that HAVE a row: a child with no row is "no record" (reducer
    # off or behind), never "camera down". No row at all -> "missing".
    row_expected = sum(_expected(per, r) for per, r in covered if r is not None)
    if not rows or not row_expected:
        status = "missing"
    else:
        status = camera_status(round(up * 60 / row_expected), 60,
                               saw_something=any((r.get("total") or {}).get("sightings") for r in rows))

    # Per-child series: the busiest / quietest child the camera actually watched.
    series = []
    for per, r in covered:
        cam = (r or {}).get("camera") or {}
        entry = {
            "start": per.start.isoformat(),
            "at": hhmm(per.start, tz) if per.grain == "hour" else local_date(per.start, tz).isoformat(),
            "passings": int(((r or {}).get("total") or {}).get("passings", 0)),
            "up_minutes": int(cam.get("up_minutes", 0)),
            "expected_minutes": _expected(per, r),
            "status": cam.get("status") if r is not None else "missing",
            "peak_at_once": dict((r or {}).get("peak_at_once") or {}),
            "people": person_count_range(r),
        }
        if per.grain == "hour":
            entry["hour"] = per.start.astimezone(tz).hour
        series.append(entry)
    # Only children the camera watched in full: a partial day (today, inside its week) is
    # never "the quietest day".
    watched = [e for (per, r), e in zip(covered, series) if e["status"] == "ok" and (r or {}).get("complete")]

    hour_baselines: dict[int, Optional[float]] = {}
    if grain == "day":
        by_hour: dict[int, list[float]] = {}
        for h in list(history)[:HOUR_BASELINE_DAYS]:
            for e in h.get("series") or []:
                if e.get("status") == "ok" and "hour" in e:
                    by_hour.setdefault(int(e["hour"]), []).append(float(e.get("passings", 0)))
        hour_baselines = {hr: (_mean(v) if len(v) >= MIN_BASELINE_PERIODS else None) for hr, v in by_hour.items()}

    def _point(e: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        if e is None:
            return None
        out = {"at": e["at"], "passings": e["passings"], "peak_at_once": e.get("peak_at_once") or {},
               "people": e.get("people")}
        if grain == "day":
            out["usual"] = hour_baselines.get(int(e["hour"]))
        return out

    # "busiest ... (0)" says nothing: only when something passed in a watched child.
    top = max(watched, key=lambda e: (e["passings"], e["start"])) if watched else None
    busiest = _point(top) if top and top["passings"] > 0 else None
    quietest = (_point(min(watched, key=lambda e: (e["passings"], e["start"])))
                if busiest and len(watched) >= 2 else None)

    lookback = {"day": DAY_BASELINE_DAYS, "week": WEEK_BASELINE_WEEKS, "month": MONTH_BASELINE_MONTHS}[grain]
    prior = [float((h.get("total") or {}).get("passings", 0)) for h in list(history)[:lookback]
             if h.get("complete") and (h.get("camera") or {}).get("status") == "ok"]
    baseline = {"mean_passings": _mean(prior) if len(prior) >= MIN_BASELINE_PERIODS else None,
                "periods": len(prior)}

    cap_pool = [c for _, r in covered if r is not None for c in (r.get("captions") or [])]
    seen: set[str] = set()
    distinct_caps = []
    for c in cap_pool:
        key = _norm_text(str(c.get("text") or ""))
        if key and key not in seen:
            seen.add(key)
            distinct_caps.append(c)
    captions = _spread(distinct_caps, MAX_PARENT_CAPTIONS)

    gate = story_gate if story_gate is not None else {}
    stories: dict[str, Any] = {"gate_passed": bool((gate.get("gate") or {}).get("passed")),
                               "checked": int(gate.get("checked", 0) or 0)}
    if stories["gate_passed"]:
        by_kind: dict[str, int] = {}
        sentences: list[dict[str, Any]] = []
        for r in rows:
            sb = r.get("stories") or {}
            for k, v in (sb.get("by_kind") or {}).items():
                by_kind[k] = by_kind.get(k, 0) + int(v)
            sentences.extend(sb.get("sentences") or [])
        stories["count"] = sum(by_kind.values())
        stories["by_kind"] = by_kind
        if grain == "day":
            stories["sentences"] = _spread(sentences, MAX_DAY_STORIES)

    individuals_key = "individuals" if grain == "day" else "individual_days"
    if grain == "day":
        ind_out = {i: {"kind": v["kind"], "label": v["label"], "sightings": v["sightings"], "first": v["first"]}
                   for i, v in sorted(ind_info.items())}
    else:
        ind_out = {i: {"kind": v["kind"], "label": v["label"], "sightings": v["sightings"],
                       "days": sorted(v["days"])} for i, v in sorted(ind_info.items())}

    return {
        "schema": SCHEMA, "rules_version": RULES_VERSION, "stream_id": stream_id, "grain": grain,
        "period_start": period.start.isoformat(), "period_end": period.end.isoformat(),
        "label": period_label(period, tz), "tz": tz.key, "local": local_fields(period, tz),
        "complete": bool(complete),
        "covered_until": (min(covered_until, period.end) if covered_until else period.end).isoformat(),
        "kinds": kinds, "total": _totals(kinds), "excluded": excluded,
        "peak_at_once": _max_peaks(rows),
        "named": named_out, "belongings": belongings_out, individuals_key: ind_out,
        "regulars": {"count": len(regular_ids), "by_kind": reg_by_kind, "ids": regular_ids},
        "camera": {"up_minutes": up, "expected_minutes": expected, "status": status,
                   "up_hours": round(up / 60.0, 1), "expected_hours": round(expected / 60.0, 1),
                   "down_spans": _merge_spans(covered, tz, "down"),
                   "missing_spans": _merge_spans(covered, tz, "missing")},
        "children": {"expected": len(child_periods(period, tz)), "covered": len(covered), "present": len(rows),
                     "complete": sum(1 for r in rows if r.get("complete"))},
        "series": series,
        "busiest": busiest, "quietest": quietest, "baseline": baseline,
        "captions": captions, "stories": stories,
    }


# --- template line --------------------------------------------------------------------------------


def _count_phrase(kinds: Mapping[str, Mapping[str, int]], field_name: str = "passings", *,
                  skip: Sequence[str] = (), with_distinct: bool = False) -> str:
    parts = []
    for k in (*KINDS, "other"):
        if k in skip:
            continue
        n = int((kinds.get(k) or {}).get(field_name, 0))
        if n:
            s = f"{n} {KIND_SINGULAR[k] if n == 1 else KIND_PLURAL[k]}"
            if with_distinct:
                s += f" ({int((kinds.get(k) or {}).get('distinct', 0))} distinct)"
            parts.append(s)
    return ", ".join(parts)


def _upper(n: Any) -> str:
    """A passings total: an upper bound (one passer can be split into several clusters)."""
    n = int(n or 0)
    return "0 passings" if n == 0 else f"at most {n} passing{'s' if n != 1 else ''}"


def _range_short(r: Mapping[str, Any] | None) -> str:
    return f"people at least {int(r['low'])}, at most {int(r['high'])}" if r else ""


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _parse(ts: str) -> datetime:
    return _utc(datetime.fromisoformat(ts))


def render_line(rollup: Mapping[str, Any]) -> str:
    """One short plain-English line. Every number in it is a field of ``rollup``."""
    tz = ZoneInfo(str(rollup.get("tz") or "America/Denver"))
    grain = str(rollup.get("grain"))
    p = Period(grain, _parse(rollup["period_start"]), _parse(rollup["period_end"]))
    stream = str(rollup.get("stream_id") or "walkway")
    head = f"{stream[:1].upper()}{stream[1:]} {_when_words(p, tz)}"
    if not rollup.get("complete") and grain != "hour":
        cu = rollup.get("covered_until")
        if cu and _parse(cu) < p.end:
            cut = _parse(cu)
            head += (f" (so far, through {hhmm(cut, tz)})" if grain == "day"
                     else f" (so far, through {_day_words(local_date(cut - timedelta(seconds=1), tz))})")
    cam = rollup.get("camera") or {}
    total = rollup.get("total") or {}
    passings = int(total.get("passings", 0))
    up, expected = int(cam.get("up_minutes", 0)), int(cam.get("expected_minutes", 0))

    if grain == "hour":
        if cam.get("status") == "down":
            return f"{head}: camera was down ({up} of {expected} minutes seen), so no count."
        unknown_note = (f". Camera record incomplete ({up} of {expected} minutes logged), so this may be low"
                        if cam.get("status") == "unknown" else "")
        if passings == 0:
            body = ("nobody passed" if cam.get("status") == "ok"
                    else f"nobody seen in the {up} minutes the camera was up")
        else:
            kinds = rollup.get("kinds") or {}
            peak = _peak_phrase(rollup.get("peak_at_once") or {})
            others = _count_phrase(kinds, skip=RANGE_KINDS)
            segs = [range_phrase(person_count_range(rollup))]
            if others:
                segs.append(f"{others} passing" + (f" ({peak})" if peak else ""))
            elif peak:
                segs.append(peak)
            body = "; ".join(x for x in segs if x)
        named = [(name, v) for name, v in (rollup.get("named") or {}).items()]
        named += [(_belonging_name(name, v), v) for name, v in (rollup.get("belongings") or {}).items()]
        if named:
            body += "; " + ", ".join(f"{name} at {', '.join(v.get('times') or [])}" for name, v in named)
        base = (rollup.get("baseline") or {}).get("mean_passings")
        if base is not None and cam.get("status") == "ok":
            body += f" (usual for this hour at most {_fmt_num(base)} passing{'' if base == 1 else 's'})"
        if cam.get("status") == "partial":
            body += f". Camera up {up} of {expected} minutes"
        return f"{head}: {body}{unknown_note}."

    if cam.get("status") == "missing":
        return f"{head}: no record (nothing was counted for this period yet)."
    if expected and cam.get("status") == "down" and passings == 0:
        return f"{head}: camera was down ({_fmt_hours(up)} of {_fmt_hours(expected)} seen), so no count."
    if passings == 0:
        parts = ["nobody passed while the camera was watching"]
    else:
        kinds = rollup.get("kinds") or {}
        others = _count_phrase(kinds, skip=RANGE_KINDS, with_distinct=True)
        parts = [x for x in (range_phrase(person_count_range(rollup)), f"{others} passing" if others else "") if x]
    named = [(name, v) for name, v in (rollup.get("named") or {}).items()]
    named += [(_belonging_name(name, v), v) for name, v in (rollup.get("belongings") or {}).items()]
    if named:
        if grain == "day":
            parts.append("named: " + ", ".join(f"{n} ({', '.join((v.get('times') or [])[:4])})"
                                               for n, v in named))
        else:
            parts.append("named: " + ", ".join(f"{n} on {len(v.get('days') or [])} "
                                               f"day{'s' if len(v.get('days') or []) != 1 else ''}"
                                               for n, v in named))
    reg = rollup.get("regulars") or {}
    if int(reg.get("count", 0)):
        where = "seen on another day this week too" if grain == "day" else f"seen on 2+ days this {grain}"
        n = int(reg["count"])
        parts.append(f"{n} unnamed regular{'s' if n != 1 else ''} ({where})")
    b, q = rollup.get("busiest"), rollup.get("quietest")
    if b:
        s = f"busiest {_at_words(b['at'], grain)} ({_upper(b['passings'])}"
        if grain == "day" and b.get("usual") is not None:
            s += f", usual at most {_fmt_num(b['usual'])}"
        people = _range_short(b.get("people"))
        if people:
            s += f"; {people}"
        peak = _peak_phrase(b.get("peak_at_once") or {})
        if peak:
            s += f", {peak}"
        s += ")"
        if q and q["at"] != b["at"]:
            s += f", quietest {_at_words(q['at'], grain)} ({_upper(q['passings'])}"
            if grain == "day" and q.get("usual") is not None:
                s += f", usual at most {_fmt_num(q['usual'])}"
            s += ")"
        parts.append(s)
    base = (rollup.get("baseline") or {}).get("mean_passings")
    if base is not None:
        parts.append(f"usual {grain} at most {_fmt_num(base)} passing{'' if base == 1 else 's'}")
    line = f"{head}: " + "; ".join(parts) + "."
    gaps = []
    for word, key in (("Camera down", "down_spans"), ("No record", "missing_spans")):
        spans = cam.get(key) or []
        if spans:
            gaps.append(f"{word} {', '.join(spans[:4])}{' …' if len(spans) > 4 else ''}")
    if gaps:
        line += " " + "; ".join(gaps) + f"; counts cover {_fmt_hours(up)} of {_fmt_hours(expected)}."
    return line


def _at_words(at: str, grain: str) -> str:
    if grain == "day":
        return at
    try:
        return _day_words(date.fromisoformat(at))
    except ValueError:
        return at


def _fmt_hours(minutes: int) -> str:
    h = minutes / 60.0
    return f"{_fmt_num(round(h, 1))} h"


# --- the journal guard: phrase, never invent ------------------------------------------------------

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "hundred": 100, "dozen": 12, "dozens": 24, "twice": 2,
}
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
# Clock times, ISO dates/timestamps, "17 Oct" / "Oct 17", and years are not counts: they are
# taken out of the prose AND the facts before numbers are compared, so a date part (17) or a
# minute (04) can never vouch for an invented count. Times are checked separately, whole.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b")
_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+)?(?:Z|[+-]\d{2}:?\d{2})?")
_DAY_MONTH_RE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}|\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
                           re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO_WEEK_RE = re.compile(r"\b\d{4}-W\d{2}\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")


def _strip_calendar(text: str) -> str:
    for rx in (_ISO_WEEK_RE, _ISO_RE, _TIME_RE, _DAY_MONTH_RE, _YEAR_RE):
        text = rx.sub(" ", text)
    return text


def _times(text: str) -> set[str]:
    out = set()
    for m in _TIME_RE.findall(text or ""):
        h, mm = m.split(":")[:2]
        out.add(f"{int(h):02d}:{mm}")
    return out


def allowed_numbers(facts: Any) -> set[float]:
    """Counts ``facts`` vouches for: every numeric VALUE (never a key, never digits inside a
    free string such as a timestamp or a caption), plus the numbers in its template ``lines``
    after clock times and dates are taken out."""
    out: set[float] = set()

    def walk(v: Any, key: str = "") -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            out.add(round(float(v), 1))
        elif isinstance(v, str):
            if key == "lines":
                for m in _NUM_RE.findall(_strip_calendar(v)):
                    out.add(round(float(m), 1))
        elif isinstance(v, Mapping):
            for k, x in v.items():
                walk(x, str(k))
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                walk(x, key)

    walk(facts)
    return out


def allowed_times(facts: Any) -> set[str]:
    """Every clock time (HH:MM) that appears anywhere in ``facts``."""
    out: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, str):
            out.update(_times(v))
        elif isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                walk(x)

    walk(facts)
    return out


def unsupported_numbers(prose: str, facts: Any) -> list[str]:
    """Counts and clock times in ``prose`` that ``facts`` does not hold.

    Counts: digits and number words two..hundred ("one" is not checked: "no one", "one of
    them" are not counts), after dates and times are taken out. Times: each HH:MM must be a
    time the facts name. Empty result = every number the prose claims is backed."""
    text = prose or ""
    allowed = allowed_numbers(facts)
    times = allowed_times(facts)
    bad: list[str] = [t for t in sorted(_times(text)) if t not in times]
    stripped = _strip_calendar(text)
    for m in _NUM_RE.findall(stripped):
        if round(float(m), 1) not in allowed:
            bad.append(m)
    for w in _WORD_RE.findall(stripped.lower()):
        n = _NUMBER_WORDS.get(w)
        if n is not None and float(n) not in allowed:
            bad.append(w)
    return bad


# A people count in prose with no range word before it ("79 people walked by"). The rollup
# says people only as "at least N, at most M"; a composed entry that states one number of
# people is refused even when the number is in the facts (it is the top of the range).
_PEOPLE_NOUN = r"(?:people|persons?|passers-?by|pedestrians|walkers)"
_COUNT_WORD = "|".join(sorted((w for w in _NUMBER_WORDS if w != "twice"), key=len, reverse=True))
_BARE_PEOPLE_RE = re.compile(rf"\b(?:\d+|{_COUNT_WORD})\s+(?:[a-z-]+\s+)?{_PEOPLE_NOUN}\b", re.IGNORECASE)
# A range word right before the number ("at most 79 people"), or the low end of a pair right
# before it ("at least 5, at most 79 people", "between 5 and 79", "2 to 5 people"), or a range
# word right after it ("79 people at most"). Anything further away belongs to another phrase.
_RANGE_BEFORE_RE = re.compile(
    r"(?:\b(?:at least|at most|up to|no more than|no fewer than)\s+"
    r"|\d+\s*(?:,|and|to|-|–)\s*(?:at most\s+)?)$", re.IGNORECASE)
_RANGE_AFTER_RE = re.compile(r"^\s*(?:at most|or fewer|or less|at the most)\b", re.IGNORECASE)


def bare_people_counts(prose: str) -> list[str]:
    """People counts in ``prose`` stated as one number, not as a range."""
    text = _strip_calendar(prose or "")
    out: list[str] = []
    for m in _BARE_PEOPLE_RE.finditer(text):
        if not (_RANGE_BEFORE_RE.search(text[max(0, m.start() - 40):m.start()])
                or _RANGE_AFTER_RE.match(text[m.end():])):
            out.append(" ".join(m.group(0).split()))
    return out
