"""Office sessions (2026-11-05, office camera design PR #3127 step B).

Each visit to a room camera (``cam0``, the office) becomes ONE row in
``vision_office_session``: when it began and ended, who was probably there, and
an ordered timeline of what they did. Two consumers in this patch:

* the Temporal Self (orion-durable-runs ``office_session`` source) labels the
  office ``company`` arc from the closed row;
* an unknown person opens the existing "Who is this?" card (``orion_ask``,
  ``vision_individual``) right away, instead of after the street rule (10
  sightings over 5 days).

Inputs, all rows other writers already store:

* the visit: ``vision_presence_transition`` (orion-vision-window presence.py),
  one presence stretch = one session. Not individual records: one 8-minute
  visit on 11-05 was split into 10 records.
* what happened: ``vision_events`` with ``event_type = 'camera_caption'`` on the
  stream (Qwen2-VL captions of the person crop). Council ``visual_observation``
  rows (furniture lists) are not activity and are ignored.
* who: the person records sighted in the visit (``vision_individual_sighting``
  + ``vision_individual.label``) and the person model's guesses
  (``vision_person_attribution``: band, and ``basis`` prior | evidence).
* how many at once: ``vision_scene_inventory.counts.person`` (most in one window).

The honest-wording rule (``who_of_session``): the office camera's prior
(``config/person_model/camera_priors.*.yaml``, Juniper 0.97) covers at most ONE
person. "Someone I don't know" is said only when more people stood in one
window than the people accounted for, or when a record's own evidence points
away from the household. A lone stranger therefore reads "probably Juniper":
that is the prior's known limit, and the label says "(probably)", never a bare
name.

Pure functions first, then the database cycle. No LLM: the timeline is the
captions' own words with the "The person in the image is" opener removed.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("sql-writer.vision_office_sessions")

TABLE = "vision_office_session"
OFFICE_LOCK_KEY = 0x0A1C_0051
# Words for the place a room stream looks at ("Juniper (probably) in the office").
PLACE_WORDS = {"cam0": "the office"}
# evidence_refs marker on a "Who is this?" ask opened by this rule: its own daily
# cap counts these, and the slow street rule's cap (ASKED_TODAY_SQL) skips them.
OFFICE_RULE_REF = "rule:office_session"
PEOPLE_KINDS = ("person", "man", "woman", "child", "kid")
LABEL_MAX = 300           # orion.schemas.temporal_self.LABEL_MAX (the arc label it becomes)
TIMELINE_CAP = 200        # entries kept per session (2 fps captions: ~18 per 8 minutes live)
IMAGE_REFS_CAP = 6
# A caption or a sighting this long before the 'present' row still belongs to the visit: the
# presence state machine needs a few windows to say "present" (live 11-05: first caption 01:23:48,
# present 01:24:07).
PRE_SLACK_SEC = 60.0
POST_SLACK_SEC = 30.0


class MigrationMissing(RuntimeError):
    """vision_office_session is not there yet (manual_migration_vision_office_session_v1.sql)."""


# --------------------------------------------------------------------------- spans


@dataclass
class Span:
    """One presence stretch on one stream."""

    session_id: str
    stream_id: str
    started_at: datetime
    last_seen_at: datetime
    ended_at: Optional[datetime] = None
    end_reason: Optional[str] = None
    refs: List[str] = field(default_factory=list)


def build_spans(rows: Iterable[Mapping[str, Any]]) -> List[Span]:
    """Presence rows (one stream, oldest first) -> visits, the same rules as the Temporal Self
    company arc (``orion/temporal_self/arcs.py`` ``_fold_presence``), so a session and its arc
    cover the same stretch:

    * opens on ``present`` with no open visit;
    * ``recent`` (the grace period) and same-state keepalives extend it;
    * ``absent`` closes it at the last sighting (the ``recent`` row when it came from recent);
    * ``unknown`` (camera dark) closes it at that row;
    * a row with ``from_state`` NULL / absent / unknown while a visit is open means the closing
      row was lost (a producer restart): the visit ends at its last row.

    A stretch whose opening row is before the rows given is not a visit here (its start is
    unknown)."""
    out: List[Span] = []
    cur: Optional[Span] = None
    left_at: Optional[datetime] = None   # the 'recent' row: when the person was last seen
    for r in rows:
        to_state, frm, t = r.get("to_state"), r.get("from_state"), r.get("occurred_at")
        ref = str(r.get("transition_id") or "")
        if t is None:
            continue
        if cur is not None and frm is not None and frm == to_state:
            cur.last_seen_at = max(cur.last_seen_at, t)
            continue
        if cur is not None and frm in (None, "absent", "unknown"):
            cur.ended_at, cur.end_reason = cur.last_seen_at, "producer_restart"
            out.append(cur)
            cur, left_at = None, None
        if to_state == "present":
            if cur is None:
                cur = Span(session_id=f"{r.get('presence_id')}-{ref}", stream_id=str(r.get("presence_id")),
                           started_at=t, last_seen_at=t, refs=[ref])
            else:
                cur.last_seen_at = max(cur.last_seen_at, t)
                cur.refs.append(ref)
            left_at = None
        elif cur is None:
            continue
        elif to_state == "recent":
            cur.last_seen_at = max(cur.last_seen_at, t)
            cur.refs.append(ref)
            left_at = t
        elif to_state == "absent":
            cur.refs.append(ref)
            cur.ended_at = left_at if (frm == "recent" and left_at is not None) else t
            cur.end_reason = "left"
            out.append(cur)
            cur, left_at = None, None
        elif to_state == "unknown":
            cur.refs.append(ref)
            cur.last_seen_at = max(cur.last_seen_at, t)
            cur.ended_at, cur.end_reason = t, "camera_dark"
            out.append(cur)
            cur, left_at = None, None
    if cur is not None:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- timeline

_CAPTION_OPENER = re.compile(
    r"^\s*(?:in (?:the|this) (?:image|picture|frame|photo),?\s+)?"
    r"(?:the|a|one)\s+(?:person|man|woman|child|kid|individual)"
    r"(?:\s+in (?:the|this) (?:image|picture|frame|photo))?\s+(?:is|appears to be|seems to be)\s+",
    re.I)


def caption_phrase(narrative: Any) -> str:
    """"The person in the image is sitting at a desk, opening a box." -> "sitting at a desk, opening
    a box". A caption without that opener is kept whole (first letter lowered). Empty -> ""."""
    s = " ".join(str(narrative or "").split()).strip()
    if not s:
        return ""
    s = _CAPTION_OPENER.sub("", s, count=1).strip().rstrip(".").strip()
    return (s[:1].lower() + s[1:]) if s else ""


def caption_time(row: Mapping[str, Any]) -> Optional[datetime]:
    """The frame's own time (``frame_ts:`` in evidence_refs), else when the row was written."""
    refs = row.get("evidence_refs")
    if isinstance(refs, str):
        try:
            refs = json.loads(refs)
        except ValueError:
            refs = []
    for ref in refs or []:
        if isinstance(ref, str) and ref.startswith("frame_ts:"):
            try:
                t = datetime.fromisoformat(ref[len("frame_ts:"):])
                return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
            except ValueError:
                break
    return row.get("created_at")


def build_timeline(captions: Iterable[Mapping[str, Any]]) -> List[dict]:
    """Captions (any order) -> [{"at", "text", "repeats", "event_id"}], oldest first, consecutive
    repeats (same phrase, case-insensitive) folded into one entry with a count."""
    rows = []
    for c in captions:
        text_ = caption_phrase(c.get("narrative"))
        at = caption_time(c)
        if text_ and at is not None:
            rows.append((at, str(c.get("event_id") or ""), text_))
    rows.sort(key=lambda r: (r[0], r[1]))
    out: List[dict] = []
    for at, eid, text_ in rows:
        if out and out[-1]["text"].casefold() == text_.casefold():
            out[-1]["repeats"] += 1
            continue
        out.append({"at": at.isoformat(), "text": text_, "repeats": 1, "event_id": eid})
    return out[:TIMELINE_CAP]


def short_activity(phrase: str) -> str:
    """A caption phrase cut to its first two comma clauses (Qwen's captions read "where, doing what,
    and a hedge"): "sitting at a desk, working on a computer, and appears to be organizing items"
    -> "sitting at a desk, working on a computer". The timeline keeps the whole phrase."""
    return ",".join(phrase.split(",")[:2]).strip()


def activities_text(timeline: Sequence[Mapping[str, Any]], budget: int) -> str:
    """The distinct activities in time order, "; "-joined, at most ``budget`` characters. When they
    do not all fit, the most that fit spread evenly over the visit (the first and the last always
    among them), then "(+N more)": a long visit is not summed up by its first minutes."""
    seen: set[str] = set()
    parts: List[str] = []
    for e in timeline:
        short = short_activity(str(e.get("text") or ""))
        k = short.casefold()
        if k and k not in seen:
            seen.add(k)
            parts.append(short)
    n = len(parts)
    if not n or budget <= 0:
        return ""
    whole = "; ".join(parts)
    if len(whole) <= budget:
        return whole
    for k in range(n - 1, 0, -1):
        idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
        text_ = "; ".join(parts[i] for i in idx) + f" (+{n - len(idx)} more)"
        if len(text_) <= budget:
            return text_
    return parts[0][: max(0, budget - 3)] + "..."


# --------------------------------------------------------------------------- who


@dataclass(frozen=True)
class Record:
    """One person record sighted in the visit."""

    individual_id: str
    label: Optional[str] = None          # the name Juniper gave this record
    guess: Optional[str] = None          # the person model's latest guess
    band: Optional[str] = None           # accept | unsure
    basis: Optional[str] = None          # prior | evidence (NULL = evidence)
    score: Optional[float] = None
    ruled_out: frozenset = frozenset()   # casefolded names Juniper said it is not


BASIS_WORDS = {"named": "", "matched": " (matched)", "prior": " (probably)", "likely": " (maybe)"}
_RANK = {"named": 0, "matched": 1, "prior": 2, "likely": 3}


def _dominant(prior: Mapping[str, float], dominance: float) -> Optional[str]:
    top = max(prior.items(), key=lambda kv: kv[1], default=(None, 0.0))
    return top[0] if top[0] is not None and top[1] >= dominance else None


def who_of_session(records: Sequence[Record], *, peak_people: Optional[int], props: int = 0,
                   prior: Optional[Mapping[str, float]] = None, dominance: float = 0.5) -> dict:
    """Who was probably in the visit.

    Each unnamed record is classified: a name Juniper gave (``named``), a confident album match
    (``matched``, band accept), the camera prior's guess (``prior``), an unsure score for a household
    name (``likely``; household = the names in this camera's prior), or a ``stranger`` (an unsure
    score for a name outside the household, or no guess at all when the prior cannot cover it).
    A record with no guess yet is covered by the prior's dominant person unless Juniper ruled that
    person out for it.

    Heads = the most people in one window, less stationary people (at least 1: presence saw
    someone). The prior covers ONE head that no name or stranger explains: two people in view never
    read as two "probably Juniper"s, and a lone stranger record is not also "probably Juniper".
    Unknown people = (any stranger record ? 1 : 0) + the heads still unexplained.
    """
    spelled = {str(k).strip().casefold(): str(k).strip() for k in (prior or {}) if str(k).strip()}
    prior = {str(k).strip().casefold(): float(v) for k, v in (prior or {}).items() if str(k).strip()}
    household = set(prior)
    dominant = _dominant(prior, dominance)
    # casefolded name -> (rank, display name, individual ids)
    people: Dict[str, List[Any]] = {}
    strangers: List[str] = []
    covered: List[str] = []

    def add(name: str, kind: str, iid: Optional[str]) -> None:
        key = name.strip().casefold()
        cur = people.get(key)
        if cur is None:
            people[key] = [kind, name.strip(), [iid] if iid else []]
            return
        if _RANK[kind] < _RANK[cur[0]]:
            cur[0], cur[1] = kind, name.strip()
        if iid and iid not in cur[2]:
            cur[2].append(iid)

    for r in records:
        if r.label and r.label.strip():
            add(r.label, "named", r.individual_id)
        elif r.guess and r.guess.strip():
            g = r.guess.strip().casefold()
            if r.band == "accept":
                add(r.guess, "matched", r.individual_id)
            elif (r.basis or "evidence") == "prior":
                add(r.guess, "prior", r.individual_id)
            elif g in household:
                add(r.guess, "likely", r.individual_id)
            else:
                strangers.append(r.individual_id)
        elif dominant is not None and dominant not in r.ruled_out:
            covered.append(r.individual_id)
        else:
            strangers.append(r.individual_id)
    # Heads: the most people in one window, less stationary people. A session exists because the
    # presence state machine saw someone, so at least one (also when no window was read).
    heads = max(1, int(peak_people) - max(0, int(props))) if peak_people is not None else 1
    # A stranger record is one unknown head (several stranger records may be one split person).
    stranger_heads = 1 if strangers else 0
    if dominant is not None and (covered or (dominant not in people and heads > len(people) + stranger_heads)):
        # The prior covers ONE person: the records with no guess yet, or a person in view with no
        # record at all (visits before the office crop pipeline, PR #3125). Spelled as the config
        # spells it, or as a record already has it.
        display = people[dominant][1] if dominant in people else spelled[dominant]
        add(display, "prior", None)
        for iid in covered:
            add(display, "prior", iid)
    by_count = max(0, heads - len(people) - stranger_heads)
    unknown = stranger_heads + by_count
    ordered = sorted(people.values(), key=lambda p: (_RANK[p[0]], p[1].casefold()))
    phrases = [f"{p[1]}{BASIS_WORDS[p[0]]}" for p in ordered]
    if unknown == 1:
        phrases.append("someone I don't know")
    elif unknown > 1:
        phrases.append(f"{unknown} people I don't know")
    if not phrases:
        phrases = ["someone"]
    text_ = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return {
        "text": text_,
        "people": [{"name": p[1], "basis": p[0], "individual_ids": sorted(p[2])} for p in ordered],
        "unknown": unknown,
        # Why someone is unknown: "count" (more people in one window than names), "evidence" (a
        # record's own scores point outside the household, or nothing can cover it), or both.
        "unknown_reason": "+".join(w for w, on in (("count", by_count > 0), ("evidence", bool(strangers))) if on) or None,
        "stranger_ids": sorted(strangers),
        "peak_people": peak_people,
        "props": int(props),
        "records": len(records),
        "prior_dominant": dominant,
    }


def session_label(who_text: str, place: str, timeline: Sequence[Mapping[str, Any]]) -> str:
    """"Juniper (probably) in the office: sitting at a desk, opening a box; reaching into a cabinet"."""
    head = f"{who_text[:1].upper()}{who_text[1:]} in {place}"
    if not timeline:
        return head[:LABEL_MAX]
    acts = activities_text(timeline, LABEL_MAX - len(head) - 2)
    return f"{head}: {acts}" if acts else head


# --------------------------------------------------------------------------- the ask


def ask_candidates(records: Sequence[Record], who: Mapping[str, Any],
                   peak_overlap: Mapping[str, bool]) -> List[str]:
    """Which records to show on the "Who is this?" card, best first. Strangers (their own evidence
    says not the household) first; then, when the unknown came from the people count, the unnamed
    records sighted while the most people were in view, least like the guessed name first; then the
    other unnamed records. Named records never."""
    unnamed = [r for r in records if not (r.label and r.label.strip())]
    strangers = set(who.get("stranger_ids") or [])
    first = [r.individual_id for r in unnamed if r.individual_id in strangers]
    rest = [r for r in unnamed if r.individual_id not in strangers]
    rest.sort(key=lambda r: (not peak_overlap.get(r.individual_id, False),
                             r.score if r.score is not None else -1.0, r.individual_id))
    return first + [r.individual_id for r in rest]


def _hhmm(t: datetime, tz: ZoneInfo) -> str:
    return t.astimezone(tz).strftime("%H:%M")


def office_ask_question(*, place: str, started_at: datetime, ended_at: Optional[datetime], tz: ZoneInfo,
                        others: Sequence[str], unknown: int) -> str:
    span = (f"from {_hhmm(started_at, tz)} to {_hhmm(ended_at, tz)}" if ended_at is not None
            else f"since {_hhmm(started_at, tz)}")
    one = unknown <= 1
    verb = ("was" if one else "were") if ended_at is not None else ("is" if one else "are")
    who = "someone I don't recognize" if one else f"{unknown} people I don't recognize"
    text_ = f"I think {who} {verb} in {place} {span}."
    if others:
        text_ += f" Also there: {', '.join(others)}."
    return text_ + " Do you know who this is?"


# --------------------------------------------------------------------------- database cycle


@dataclass(frozen=True)
class OfficeSessionsConfig:
    streams: Tuple[str, ...] = ("cam0",)
    settle_sec: float = 660.0
    lookback_hours: float = 24.0
    ask_daily_cap: int = 4
    local_tz: str = "America/Denver"


def table_present(conn) -> bool:
    from sqlalchemy import text

    return conn.execute(text("SELECT to_regclass(:t)"), {"t": TABLE}).scalar() is not None


def _present(conn, table: str) -> bool:
    from sqlalchemy import text

    return conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar() is not None


def _transitions(conn, stream: str, lo: datetime, hi: datetime) -> List[dict]:
    """Presence rows in [lo, hi] (hi = the pass's own clock, so a replay never sees its future)."""
    from sqlalchemy import text

    return [dict(r._mapping) for r in conn.execute(text(
        "SELECT transition_id, presence_id, from_state, to_state, occurred_at FROM vision_presence_transition "
        "WHERE presence_id = :s AND occurred_at >= :lo AND occurred_at <= :hi ORDER BY occurred_at, transition_id"),
        {"s": stream, "lo": lo, "hi": hi})]


def _captions(conn, stream: str, lo: datetime, hi: datetime) -> List[dict]:
    from sqlalchemy import text

    # created_at bounds the read (indexed); caption_time then uses the frame's own time.
    rows = [dict(r._mapping) for r in conn.execute(text(
        "SELECT event_id, narrative, evidence_refs, created_at FROM vision_events "
        "WHERE stream_id = :s AND event_type = 'camera_caption' AND created_at >= :lo AND created_at < :hi"),
        {"s": stream, "lo": lo, "hi": hi + timedelta(seconds=POST_SLACK_SEC)})]
    out = []
    for r in rows:
        t = caption_time(r)
        if t is not None and lo <= t <= hi:
            out.append(r)
    return out


def _records(conn, stream: str, lo: datetime, hi: datetime) -> Tuple[List[Record], int, Dict[str, List[Tuple]]]:
    """(person records sighted in [lo, hi], stationary person records, record -> sighting rows)."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT s.individual_id, s.sighting_id, s.started_at, s.ended_at, s.stationary AS s_stat, "
        "i.label, i.stationary AS i_stat "
        "FROM vision_individual_sighting s JOIN vision_individual i USING (individual_id) "
        "WHERE s.stream_id = :s AND s.started_at <= :hi AND s.ended_at >= :lo AND i.kind = ANY(:k) "
        "ORDER BY s.started_at"), {"s": stream, "lo": lo, "hi": hi, "k": list(PEOPLE_KINDS)}).fetchall()
    moving: Dict[str, Optional[str]] = {}
    stationary: set[str] = set()
    sightings: Dict[str, List[Tuple]] = {}
    for r in rows:
        if r.s_stat or r.i_stat:
            stationary.add(r.individual_id)
            continue
        moving.setdefault(r.individual_id, r.label)
        sightings.setdefault(r.individual_id, []).append((r.sighting_id, r.started_at, r.ended_at))
    guesses: Dict[str, Tuple] = {}
    ruled: Dict[str, set] = {}
    ids = sorted(i for i, lbl in moving.items() if not (lbl and lbl.strip()))
    if ids and _present(conn, "vision_person_attribution"):
        from app.vision_person_model import _blocked, attribution_has_basis

        basis_col = "basis" if attribution_has_basis(conn) else "NULL"
        for g in conn.execute(text(
                f"SELECT DISTINCT ON (individual_id) individual_id, person_label, band, score, {basis_col} AS basis "
                "FROM vision_person_attribution WHERE individual_id = ANY(:ids) AND revoked_at IS NULL "
                "ORDER BY individual_id, decided_at DESC"), {"ids": ids}).fetchall():
            guesses[g.individual_id] = (g.person_label, g.band, g.basis, g.score)
        if _present(conn, "vision_person_exemplar"):
            ruled = _blocked(conn)
    records = []
    for iid, lbl in moving.items():
        g = guesses.get(iid)
        records.append(Record(
            individual_id=iid, label=lbl, guess=g[0] if g else None, band=g[1] if g else None,
            basis=g[2] if g else None, score=float(g[3]) if g and g[3] is not None else None,
            ruled_out=frozenset(ruled.get(iid, set()))))
    return records, len(stationary - set(moving)), sightings


def _peak(conn, stream: str, lo: datetime, hi: datetime) -> Tuple[Optional[int], List[Tuple[datetime, datetime]]]:
    """(most people in one window in [lo, hi], or None when no window was read; the windows that
    held more than one person, as (start, end))."""
    from sqlalchemy import text

    if not _present(conn, "vision_scene_inventory"):
        return None, []
    rows = conn.execute(text(
        "SELECT observed_at, window_start_ts, window_end_ts, counts FROM vision_scene_inventory "
        "WHERE stream_id = :s AND observed_at >= :lo AND observed_at <= :hi"), {"s": stream, "lo": lo, "hi": hi}).fetchall()
    if not rows:
        return None, []
    peak = 0
    crowded: List[Tuple[datetime, datetime]] = []
    for r in rows:
        counts = r.counts if isinstance(r.counts, dict) else json.loads(r.counts or "{}")
        try:
            n = int(counts.get("person") or 0)
        except (TypeError, ValueError):
            n = 0
        peak = max(peak, n)
        if n > 1:
            a = datetime.fromtimestamp(r.window_start_ts, timezone.utc) if r.window_start_ts else r.observed_at
            b = datetime.fromtimestamp(r.window_end_ts, timezone.utc) if r.window_end_ts else r.observed_at
            crowded.append((a, b))
    return peak, crowded


def _overlaps(sightings: Sequence[Tuple], windows: Sequence[Tuple[datetime, datetime]]) -> bool:
    return any(s[1] <= b and s[2] >= a for s in sightings for a, b in windows)


def _stored(conn, stream: str, lo: datetime) -> Dict[str, dict]:
    from sqlalchemy import text

    return {r.session_id: dict(r._mapping) for r in conn.execute(text(
        f"SELECT session_id, status, ask_id FROM {TABLE} WHERE stream_id = :s AND started_at >= :lo"),
        {"s": stream, "lo": lo - timedelta(seconds=PRE_SLACK_SEC)})}


UPSERT_SQL = f"""
INSERT INTO {TABLE} (session_id, stream_id, status, started_at, last_seen_at, ended_at, end_reason, closed_at,
    who_text, who, unknown_people, timeline, label, individual_ids, image_refs, evidence_refs, updated_at)
VALUES (:sid, :stream, :status, :started, :last, :ended, :reason, :closed, :who_text, CAST(:who AS jsonb), :unknown,
    CAST(:timeline AS jsonb), :label, CAST(:ids AS jsonb), CAST(:imgs AS jsonb), CAST(:ev AS jsonb), :now)
ON CONFLICT (session_id) DO UPDATE SET status = EXCLUDED.status, last_seen_at = EXCLUDED.last_seen_at,
    ended_at = EXCLUDED.ended_at, end_reason = EXCLUDED.end_reason, closed_at = EXCLUDED.closed_at,
    who_text = EXCLUDED.who_text, who = EXCLUDED.who, unknown_people = EXCLUDED.unknown_people,
    timeline = EXCLUDED.timeline, label = EXCLUDED.label, individual_ids = EXCLUDED.individual_ids,
    image_refs = EXCLUDED.image_refs, evidence_refs = EXCLUDED.evidence_refs, updated_at = EXCLUDED.updated_at
WHERE {TABLE}.status = 'open'
"""

OFFICE_ASKED_TODAY_SQL = (
    "SELECT count(*) FROM orion_ask WHERE source_kind = 'vision_individual' AND created_at >= :d "
    "AND evidence_refs @> CAST(:office AS jsonb)"
)


def compute_session(conn, span: Span, now: datetime, cfg: OfficeSessionsConfig,
                    prior: Mapping[str, float]) -> Tuple[dict, List[Record], Dict[str, List[Tuple]], Dict[str, bool]]:
    """Read the visit's rows and build the row to store (no write)."""
    end = span.ended_at or now
    lo = span.started_at - timedelta(seconds=PRE_SLACK_SEC)
    hi = end + timedelta(seconds=POST_SLACK_SEC)
    timeline = build_timeline(_captions(conn, span.stream_id, lo, hi))
    records, props, sightings = _records(conn, span.stream_id, lo, hi)
    peak, crowded = _peak(conn, span.stream_id, lo, hi)
    from app.vision_person_model import PRIOR_DOMINANCE

    who = who_of_session(records, peak_people=peak, props=props, prior=prior, dominance=PRIOR_DOMINANCE)
    place = PLACE_WORDS.get(span.stream_id, f"view of {span.stream_id}")
    overlap = {iid: _overlaps(s, crowded) for iid, s in sightings.items()}
    settled = span.ended_at is not None and now >= span.ended_at + timedelta(seconds=cfg.settle_sec)
    row = {
        "sid": span.session_id, "stream": span.stream_id, "status": "closed" if settled else "open",
        "started": span.started_at, "last": span.last_seen_at, "ended": span.ended_at, "reason": span.end_reason,
        "closed": now if settled else None, "who_text": who["text"], "who": json.dumps(who, default=str),
        "unknown": int(who["unknown"]), "timeline": json.dumps(timeline), "label": session_label(who["text"], place, timeline),
        "ids": json.dumps(sorted(sightings)), "ev": json.dumps([f"vision_presence_transition:{r}" for r in span.refs]),
        "now": now, "_place": place, "_who": who,
    }
    return row, records, sightings, overlap


def _session_images(conn, order: Sequence[str], now: datetime, icfg) -> List[str]:
    from app.vision_individuals import _images, _sighting_rows

    out: List[str] = []
    for iid in order:
        for ref in _images(_sighting_rows(conn, iid), now, icfg, limit=2):
            if ref not in out:
                out.append(ref)
        if len(out) >= IMAGE_REFS_CAP:
            break
    return out[:IMAGE_REFS_CAP]


def open_office_ask(conn, row: dict, records: Sequence[Record], overlap: Mapping[str, bool], now: datetime,
                    cfg: OfficeSessionsConfig, icfg) -> Optional[dict]:
    """The existing "Who is this?" card for an unknown person, right away: one per session, under
    its own daily cap (``ask_daily_cap``), the shared cooldown and "no picture, no ask"."""
    from sqlalchemy import text

    from app.vision_individuals import (
        _NOT_ASKED_SQL, SOURCE_KIND_INDIVIDUAL, _images, _insert_ask, _sighting_rows, local_day_start,
        remaining_ask_budget,
    )
    from orion.schemas.ask import OrionAskV1

    who = row["_who"]
    if int(who["unknown"]) <= 0 or cfg.ask_daily_cap <= 0:
        return None
    tz = ZoneInfo(cfg.local_tz)
    asked = conn.execute(text(OFFICE_ASKED_TODAY_SQL), {"d": local_day_start(now, tz),
                                                       "office": json.dumps([OFFICE_RULE_REF])}).scalar() or 0
    if remaining_ask_budget(daily_cap=cfg.ask_daily_cap, asked_today=int(asked)) <= 0:
        return None
    cool = now - timedelta(days=icfg.ask_cooldown_days)
    for iid in ask_candidates(records, who, overlap):
        ok = conn.execute(text(
            "SELECT 1 FROM vision_individual i WHERE i.individual_id = :iid AND i.label IS NULL "
            f"AND {_NOT_ASKED_SQL.format(kind=':sk', ref='i.individual_id')} "
            "AND NOT EXISTS (SELECT 1 FROM orion_ask p WHERE p.status = 'open' AND p.source_kind IN "
            "('vision_person_confirm', 'vision_same_individual') AND (split_part(p.source_ref, '|', 1) = i.individual_id "
            "OR p.source_ref LIKE '%|' || i.individual_id))"),
            {"iid": iid, "sk": SOURCE_KIND_INDIVIDUAL, "cool": cool}).scalar()
        if not ok:
            continue
        srows = _sighting_rows(conn, iid)
        images = _images(srows, now, icfg)
        if not images:
            continue  # no picture, no ask
        # Everyone else accounted for; a person who IS only this record is not "also there".
        others = [f"{p['name']}{BASIS_WORDS[p['basis']]}" for p in who["people"] if p["individual_ids"] != [iid]]
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=office_ask_question(place=row["_place"], started_at=row["started"], ended_at=row["ended"], tz=tz,
                                         others=others, unknown=int(who["unknown"])),
            evidence_refs=[OFFICE_RULE_REF, f"office_session:{row['sid']}"] + [f"sighting:{r[0]}" for r in srows[:5]],
            image_ref=images[0], image_refs=images, created_at=now,
            expires_at=now + timedelta(days=icfg.ask_expiry_days),
            source_kind=SOURCE_KIND_INDIVIDUAL, source_ref=iid,
        )
        if _insert_ask(conn, ask):
            conn.execute(text(f"UPDATE {TABLE} SET ask_id = :a WHERE session_id = :s AND ask_id IS NULL"),
                         {"a": ask.ask_id, "s": row["sid"]})
            return ask.model_dump(mode="json")
    return None


def run_office_sessions(conn, now: datetime, cfg: OfficeSessionsConfig, icfg,
                        prior_by_stream: Mapping[str, Mapping[str, float]]) -> Tuple[dict, List[dict]]:
    """One pass on an open connection (the caller commits): every visit in the lookback that is
    not final yet is rebuilt and upserted; a final row is never rewritten (its ask_id excepted)."""
    if not table_present(conn):
        raise MigrationMissing(f"{TABLE} missing: apply services/orion-sql-db/manual_migration_vision_office_session_v1.sql")
    summary = {"office_sessions_seen": 0, "office_sessions_written": 0, "office_sessions_closed": 0,
               "office_unknown_sessions": 0, "office_asks_opened": 0}
    opened: List[dict] = []
    lo = now - timedelta(hours=cfg.lookback_hours)
    for stream in cfg.streams:
        stored = _stored(conn, stream, lo)
        for span in build_spans(_transitions(conn, stream, lo, now)):
            summary["office_sessions_seen"] += 1
            prev = stored.get(span.session_id)
            if prev is not None and prev["status"] == "closed":
                continue
            row, records, sightings, overlap = compute_session(conn, span, now, cfg, prior_by_stream.get(stream) or {})
            order = ask_candidates(records, row["_who"], overlap) + [
                r.individual_id for r in records if r.label and r.label.strip()]
            row["imgs"] = json.dumps(_session_images(conn, order, now, icfg))
            from sqlalchemy import text

            conn.execute(text(UPSERT_SQL), {k: v for k, v in row.items() if not k.startswith("_")})
            summary["office_sessions_written"] += 1
            summary["office_sessions_closed"] += int(row["status"] == "closed")
            if row["unknown"] > 0:
                summary["office_unknown_sessions"] += 1
                if not (prev and prev.get("ask_id")):
                    ask = open_office_ask(conn, row, records, overlap, now, cfg, icfg)
                    if ask:
                        opened.append(ask)
                        summary["office_asks_opened"] += 1
            if row["status"] == "closed":
                logger.info("[VISION_OFFICE_SESSION] closed session_id=%s %s-%s who=%r timeline=%d label=%r",
                            span.session_id, span.started_at.isoformat(), (span.ended_at or now).isoformat(),
                            row["who_text"], len(json.loads(row["timeline"])), row["label"])
    return summary, opened


def run_one_office_cycle(*, postgres_uri: str = "", cfg: OfficeSessionsConfig, icfg, now: Optional[datetime] = None,
                         engine: Any = None, prior_by_stream: Optional[Mapping[str, Mapping[str, float]]] = None,
                         ) -> Tuple[dict, List[dict]]:
    """One bounded pass in one transaction, under an advisory lock (one writer)."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    if prior_by_stream is None:
        from app.vision_person_model import _camera_priors

        prior_by_stream = _camera_priors()
    own = engine is None
    eng = engine or create_engine(postgres_uri, pool_pre_ping=True)
    try:
        with eng.begin() as conn:
            if not conn.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": OFFICE_LOCK_KEY}).scalar():
                return {"skipped": "another instance holds the office sessions lock"}, []
            return run_office_sessions(conn, ts, cfg, icfg, prior_by_stream)
    finally:
        if own:
            eng.dispose()
