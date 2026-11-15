"""Two more topic kinds for the outreach choice point: people Orion has seen, and open chat threads.

Personality spec rev 4, move 1 (docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md).
The background tick's choice point (``scripts/outreach_choice.py``) showed only Orion's own material
(the newest daydream, reads waiting to be told). Juniper (2026-11-15): the choices should be about the
world -- the people around the house and the threads of her own conversations -- not Orion's machinery.

PEOPLE (``kind="person"``, subject ``person:<person_id>``). Source: the person record
(``person`` / ``person_link``, PR #2852) joined to camera sightings (``vision_individual_sighting``).
Only a person Juniper named: a link exists only from a name she typed on a camera card (or
``people.yaml``), never from a picture match, and a copied label (``label_copied``) is left out. Only
``role='other'`` (not Juniper, not Orion), ``status='active'``, merged names resolved to the person they
were merged into. Office camera streams are left out (any stream with a ``vision_office_session`` row):
office visits keep their own door (``scripts/office_reach_out.py``).

What is new about the person, from real rows only (``person_topics``):

* sightings less than ``VISIT_GAP`` apart are one visit;
* the newest visit must have started in the last ``FRESH_HOURS``;
* "back": the visit before it ended at least ``RETURN_GAP_DAYS`` ago, or the person had no visit in the
  whole ``HISTORY_DAYS`` window (said as "since <date>", the date clamped to the oldest named sighting the
  rows hold, so it never claims the cameras watched before their records begin);
* "unusual time": at least ``MIN_PRIOR_VISITS`` earlier visits on ``MIN_PRIOR_DAYS`` days, and none of
  them started within ``UNUSUAL_HOUR_MARGIN`` hours (time of day, local) of this one.

A visit that is neither is not a topic. A camera match is a cluster match, not proof of who it was
(clusters hold lookalikes, PR #2859), so a visit also needs ``MIN_VISIT_OBSERVATIONS`` frames or
``MIN_VISIT_DWELL_SEC`` of dwell, and is dropped when another named person matched on the same camera
within ``AMBIGUOUS_WINDOW``; the text says "a camera match", never "the cameras saw". The version is the visit's first sighting id, so a later visit
that is new again comes back after the 7-day subject ledger; the same visit never does. The topic cites
that sighting (``evidence_ref = vision_individual_sighting:<id>``).

CHAT THREADS (``kind="chat_thread"``, subject ``chat_thread:<correlation_id>``). Source:
``chat_history_log`` rows Juniper typed in Hub (``hub_orion`` / ``hub_ws`` / ``hub_http``). A thread is
open when Orion's reply ended on a question to her (the last paragraph holds a sentence ending in "?";
see ``last_question``)
and the chat then went quiet: Juniper's next message came ``QUIET_GAP`` or more later, or not yet. The
reply must be at least ``QUIET_GAP`` old and at most ``THREAD_LOOKBACK_DAYS`` old. The text says exactly
that ("the chat went quiet after"), never that she did not answer: she may have answered later in other
words, and nothing here can tell.

Not used, on purpose (checked live 2026-11-15):
* the promise ledger (``harness_turn_trace...promises_made``): Orion Prime already carries each promise
  back into its thread (``orion/prime/intention_sources/turn_promise.py``), and two doors for one
  promise would double it;
* temporal-self "open loop" concern arcs: their ``subject_label`` is empty (no words to show).

Machinery filter. Juniper does not want plumbing. A thread whose question or her message has the shape
of machinery -- a code span (backtick), a URL, a file path, an issue/PR number (``#123``), an
identifier (``snake_case``), or a word that mixes letters and digits (``qwen38``) -- is left out. This
is a shape rule, not a word list.

``person_topics`` and ``chat_thread_topics`` are pure (callers pass rows and ``now``). The fetchers
read through the Hub's asyncpg pool (``app.state.memory_pg_pool``), SELECT only, and never raise.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from scripts.outreach_choice import OutreachTopic, clip

logger = logging.getLogger("orion-hub.outreach_world_topics")

PERSON_KIND = "person"
CHAT_THREAD_KIND = "chat_thread"

# --- people -----------------------------------------------------------------------------------------
HISTORY_DAYS = 60
FRESH_HOURS = 36.0
VISIT_GAP = timedelta(minutes=90)
RETURN_GAP_DAYS = 4.0
MIN_PRIOR_VISITS = 6
MIN_PRIOR_DAYS = 4
UNUSUAL_HOUR_MARGIN = 1.5
# A camera match is a cluster match, not proof of who it was (clusters are impure: PR #2859). A visit
# counts only when the camera held it (MIN_VISIT_OBSERVATIONS frames or MIN_VISIT_DWELL_SEC), and not
# when another named person matched on the same camera within AMBIGUOUS_WINDOW of it (live 2026-11-14:
# three different named people matched on the doorbell inside 81 s, almost surely one visitor).
MIN_VISIT_OBSERVATIONS = 3
MIN_VISIT_DWELL_SEC = 5.0
AMBIGUOUS_WINDOW = timedelta(minutes=2)
# A sighting this far past ``now`` is clock skew: left out, never allowed to hide the real newest visit.
FUTURE_SKEW = timedelta(minutes=5)

STREAM_PLACES = {"walkway": "the walkway", "doorbell": "the front door"}

PEOPLE_SQL = """
SELECT p.person_id, p.display_name, s.sighting_id, s.stream_id, s.started_at, s.ended_at,
       s.dwell_sec, s.observation_count
FROM person_link l
JOIN person lp ON lp.person_id = l.person_id
JOIN person p ON p.person_id = coalesce(lp.merged_into, lp.person_id)
JOIN vision_individual i ON i.individual_id = l.source_ref
JOIN vision_individual_sighting s ON s.individual_id = l.source_ref
WHERE l.revoked_at IS NULL
  AND l.source_kind = 'vision_individual'
  AND lp.status = 'active'
  AND p.status = 'active' AND p.role = 'other' AND p.merged_into IS NULL
  AND i.kind = 'person' AND NOT i.label_copied
  AND s.started_at >= $1
  AND s.stream_id NOT IN (SELECT DISTINCT stream_id FROM vision_office_session)
ORDER BY p.person_id, s.started_at
"""

# --- chat threads -----------------------------------------------------------------------------------
THREAD_LOOKBACK_DAYS = 7
QUIET_GAP = timedelta(hours=2)
JUNIPER_CHAT_SOURCES = ("hub_orion", "hub_ws", "hub_http")
SAID_CHARS = 140
QUESTION_CHARS = 200

CHAT_SQL = """
SELECT correlation_id, session_id, created_at, prompt, response
FROM chat_history_log
WHERE source = ANY($2::text[]) AND created_at >= $1 AND coalesce(prompt, '') <> ''
ORDER BY session_id, created_at
"""


def _aware(value: Any) -> Optional[datetime]:
    """Rows hold aware datetimes (vision) or naive UTC (chat_history_log). None when unreadable."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _zone(tz_name: str) -> Any:
    try:
        return ZoneInfo(str(tz_name or "UTC"))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return timezone.utc


def _when(at: datetime, now: datetime, tz: Any) -> str:
    """"today 3:12 pm", "yesterday 9:05 am", "Thu Nov 13 4:40 am" (local)."""
    local, today = at.astimezone(tz), now.astimezone(tz).date()
    clock = local.strftime("%I:%M %p").lstrip("0").lower()
    if local.date() == today:
        return f"today {clock}"
    if local.date() == today - timedelta(days=1):
        return f"yesterday {clock}"
    return f"{local.strftime('%a %b')} {local.day} {clock}"


def _days(delta: timedelta) -> str:
    n = max(1, int(round(delta.total_seconds() / 86400.0)))
    return f"{n} day{'s' if n != 1 else ''}"


@dataclass(frozen=True)
class _Visit:
    first_sighting_id: str
    stream_id: str
    start: datetime
    end: datetime
    observations: int = 0
    dwell_sec: float = 0.0
    streams: frozenset = frozenset()


def _visits(rows: Sequence[Mapping[str, Any]]) -> List[_Visit]:
    """Sightings (one person, any order) folded into visits: under ``VISIT_GAP`` apart is one visit."""
    clean = []
    for r in rows:
        start, end = _aware(r.get("started_at")), _aware(r.get("ended_at"))
        sid = str(r.get("sighting_id") or "")
        if start is None or not sid:
            continue
        obs = int(r.get("observation_count") or 0)
        dwell = float(r.get("dwell_sec") or 0.0)
        clean.append((start, max(end or start, start), sid, str(r.get("stream_id") or ""), obs, dwell))
    clean.sort()
    visits: List[_Visit] = []
    for start, end, sid, stream, obs, dwell in clean:
        if visits and start - visits[-1].end < VISIT_GAP:
            last = visits[-1]
            visits[-1] = _Visit(last.first_sighting_id, last.stream_id, last.start, max(last.end, end),
                                last.observations + obs, last.dwell_sec + dwell, last.streams | {stream})
        else:
            visits.append(_Visit(sid, stream, start, end, obs, dwell, frozenset({stream})))
    return visits


def _hour_distance(a: datetime, b: datetime, tz: Any) -> float:
    """Hours between two times of day (local), around the clock: 23:30 and 00:30 are 1 hour apart."""
    la, lb = a.astimezone(tz), b.astimezone(tz)
    ha = la.hour + la.minute / 60.0
    hb = lb.hour + lb.minute / 60.0
    d = abs(ha - hb) % 24.0
    return min(d, 24.0 - d)


def person_topics(rows: Iterable[Mapping[str, Any]], now: datetime, *, tz_name: str = "UTC",
                  history_days: int = HISTORY_DAYS) -> List[OutreachTopic]:
    """One topic per named person whose newest visit is new (see the module docstring). Pure.

    ``rows``: ``PEOPLE_SQL`` rows (``person_id``, ``display_name``, ``sighting_id``, ``stream_id``,
    ``started_at``, ``ended_at``). The SQL keeps out unnamed clusters; a row with no ``person_id`` or
    no ``display_name`` is skipped here too."""
    tz = _zone(tz_name)
    rows_list = [r for r in rows if (_aware(r.get("started_at")) or now) <= now + FUTURE_SKEW]
    by_person: Dict[str, List[Mapping[str, Any]]] = {}
    names: Dict[str, str] = {}
    for r in rows_list:
        pid, name = str(r.get("person_id") or "").strip(), " ".join(str(r.get("display_name") or "").split())
        if not pid or not name:
            continue
        by_person.setdefault(pid, []).append(r)
        names[pid] = name
    # Other named people's sightings per stream, for the ambiguity check.
    by_stream: Dict[str, List[Tuple[datetime, str]]] = {}
    for pid, prs in by_person.items():
        for r in prs:
            at = _aware(r.get("started_at"))
            if at is not None:
                by_stream.setdefault(str(r.get("stream_id") or ""), []).append((at, pid))
    window_floor = now - timedelta(days=history_days)
    topics: List[OutreachTopic] = []
    for pid in sorted(by_person):
        visits = _visits(by_person[pid])
        if not visits:
            continue
        visit, prior = visits[-1], visits[:-1]
        if visit.start > now + FUTURE_SKEW or now - visit.start > timedelta(hours=FRESH_HOURS):
            continue
        if visit.observations < MIN_VISIT_OBSERVATIONS and visit.dwell_sec < MIN_VISIT_DWELL_SEC:
            continue  # a glimpse: too thin to say who it was
        if any(other != pid and visit.start - AMBIGUOUS_WINDOW <= at <= visit.end + AMBIGUOUS_WINDOW
               for stream in visit.streams for at, other in by_stream.get(stream, ())):
            continue  # another named person matched on the same camera at the same moment
        name = names[pid]
        place = STREAM_PLACES.get(visit.stream_id, f"the {visit.stream_id} camera")
        when = _when(visit.start, now, tz)
        if not prior:
            # "Since <date>": the later of the window start and the oldest OTHER named sighting (the camera
            # records only reach back so far; live 2026-10-15). No such date before this visit: no "since".
            oldest = min((a for o, prs in by_person.items() if o != pid
                          for a in (_aware(r.get("started_at")) for r in prs) if a is not None), default=None)
            since = max(window_floor, oldest) if oldest else None
            if since is not None and since < visit.start - timedelta(days=1):
                ws = since.astimezone(tz)
                what = f"the first camera match for {name} since {ws.strftime('%b')} {ws.day}"
            else:
                what = f"the first camera match for {name} on record"
        elif visit.start - prior[-1].end >= timedelta(days=RETURN_GAP_DAYS):
            what = f"the first camera match for {name} in {_days(visit.start - prior[-1].end)}"
        else:
            prior_days = {v.start.astimezone(tz).date() for v in prior}
            if (len(prior) < MIN_PRIOR_VISITS or len(prior_days) < MIN_PRIOR_DAYS
                    or any(_hour_distance(v.start, visit.start, tz) <= UNUSUAL_HOUR_MARGIN for v in prior)):
                continue
            what = (f"a time of day {name} had not matched at in {len(prior)} earlier visits "
                    f"over {len(prior_days)} days")
        evidence = f"vision_individual_sighting:{visit.first_sighting_id}"
        option = clip(f"{name} (a camera match for a name Juniper gave) at {place}, {when}: {what}.", 400)
        topics.append(OutreachTopic(
            subject=f"person:{pid}",
            kind=PERSON_KIND,
            version=visit.first_sighting_id,
            observed_at=visit.end,
            option_text=option,
            prompt_text=clip(
                f"Someone Juniper named for your cameras: {option} The camera recorded a match and a time, "
                f"nothing more. A match can be wrong (the cameras group lookalikes), so say it is a camera "
                f"match, and do not invent what {name} was doing.", 600),
            evidence_ref=evidence,
        ))
    return topics


# --- chat threads -----------------------------------------------------------------------------------
_MACHINERY = re.compile(
    r"`|https?://|www\.|#\d|\b\w+_\w+\b"
    # a path: two slashes ("docs/specs/x") or a file name ("thing.md"); "and/or", "24/7" are prose
    r"|\b[\w.~-]+/[\w.~-]+/[\w.~-]+|\b[\w-]+\.(?:py|md|ya?ml|json|sql|sh|js|ts|toml|txt|env|log)\b"
    # letters and digits mixed in one word ("qwen38", "v2b"); ordinals, times, decades, units are prose
    r"|\b(?!\d+(?:st|nd|rd|th|s|am|pm|k|h|m|kg|lb|mg|g)\b)(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]{3,}\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TRAILING = " \t*_\"'”’)]<3♥❤"


def looks_like_machinery(text: str) -> bool:
    """True when ``text`` has the shape of plumbing (code span, URL, path, #123, snake_case, qwen38)."""
    return bool(_MACHINERY.search(str(text or "")))


def last_question(reply: str) -> Optional[str]:
    """The last sentence ending in "?" in the reply's last paragraph, or None.

    The question need not be the very last sentence: Orion usually adds one line after it ("So what's
    the plan for 44? February's cold here -- I'm picturing something indoor."), and requiring the
    question last dropped both real threads live on 2026-11-15. A question in an earlier paragraph, with
    a new paragraph after it, is not where the chat stopped. Double quotes inside the question become
    single quotes, so it can be quoted again."""
    text = str(reply or "").strip()
    if not text:
        return None
    paragraph = " ".join(re.split(r"\n\s*\n", text)[-1].split())
    sentences = [x.strip() for x in _SENTENCE_SPLIT.split(paragraph) if x.strip()]
    for i in range(len(sentences) - 1, -1, -1):
        q = sentences[i].rstrip(_TRAILING).strip()
        if not q.endswith("?"):
            continue
        # "e.g. what do you think?" splits after "e.g."; join a lowercase tail back to its lead.
        if i >= 1 and q[:1].islower():
            q = f"{sentences[i - 1]} {q}"
        return q.replace('"', "'").replace("\u201c", "'").replace("\u201d", "'")
    return None


def chat_thread_topics(rows: Iterable[Mapping[str, Any]], now: datetime, *, tz_name: str = "UTC",
                       lookback_days: int = THREAD_LOOKBACK_DAYS) -> List[OutreachTopic]:
    """Open threads from Juniper's chats (see the module docstring). Pure.

    ``rows``: ``CHAT_SQL`` rows (``correlation_id``, ``session_id``, ``created_at``, ``prompt``,
    ``response``): Juniper's own messages with Orion's reply. Juniper's next message is read across
    every session: she is one person, and a message in a new session still means she came back."""
    tz = _zone(tz_name)
    turns = []
    for r in rows:
        at = _aware(r.get("created_at"))
        cid = str(r.get("correlation_id") or "").strip()
        if at is None or not cid or not str(r.get("prompt") or "").strip():
            continue
        turns.append((at, cid, r))
    turns.sort(key=lambda t: (t[0], t[1]))
    oldest = now - timedelta(days=lookback_days)
    topics: List[OutreachTopic] = []
    for i, (at, cid, r) in enumerate(turns):
        if at < oldest or now - at < QUIET_GAP:
            continue
        nxt = turns[i + 1][0] if i + 1 < len(turns) else None
        if nxt is not None and nxt - at < QUIET_GAP:
            continue
        question = last_question(str(r.get("response") or ""))
        said = " ".join(str(r.get("prompt") or "").split())
        if not question or looks_like_machinery(question) or looks_like_machinery(said[:SAID_CHARS]):
            continue
        quiet = (f"the chat went quiet for {_days(nxt - at) if nxt - at >= timedelta(days=1) else _hours(nxt - at)} after"
                 if nxt is not None else "the chat has been quiet since")
        when = _when(at, now, tz)
        option = clip(
            f'An open thread with Juniper, {when}: she said "{clip(said.replace(chr(34), chr(39)), SAID_CHARS)}" and you asked '
            f'"{clip(question, QUESTION_CHARS)}"; {quiet}.', 400)
        topics.append(OutreachTopic(
            subject=f"chat_thread:{cid}",
            kind=CHAT_THREAD_KIND,
            observed_at=at,
            option_text=option,
            prompt_text=clip(
                f"{option} Picking this means going back to that thread with her, in plain words; she may "
                f"have answered it since in other words, so do not tell her she ignored it.", 700),
            evidence_ref=f"chat_history_log:{cid}",
        ))
    return topics


def _hours(delta: timedelta) -> str:
    n = max(1, int(round(delta.total_seconds() / 3600.0)))
    return f"{n} hour{'s' if n != 1 else ''}"


# --- fetchers (I/O) ---------------------------------------------------------------------------------
TopicSource = Callable[[], Awaitable[List[OutreachTopic]]]


def people_source(pool_provider: Callable[[], Any], *, tz_name: str,
                  clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> TopicSource:
    """An async topic source for ``EndogenousOutreach.topic_sources``. Never raises: [] on any failure."""

    async def source() -> List[OutreachTopic]:
        pool = pool_provider()
        if pool is None:
            return []
        now = clock()
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(PEOPLE_SQL, now - timedelta(days=HISTORY_DAYS))
            return person_topics([dict(r) for r in rows], now, tz_name=tz_name)
        except Exception:  # noqa: BLE001 -- the tick's other topics still show
            logger.warning("outreach_people_topics_failed", exc_info=True)
            return []

    source.__qualname__ = "people_source"
    return source


def chat_thread_source(pool_provider: Callable[[], Any], *, tz_name: str,
                       clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> TopicSource:
    """An async topic source for ``EndogenousOutreach.topic_sources``. Never raises: [] on any failure.
    chat_history_log.created_at is naive UTC, so the cutoff is passed naive."""

    async def source() -> List[OutreachTopic]:
        pool = pool_provider()
        if pool is None:
            return []
        now = clock()
        cutoff = (now - timedelta(days=THREAD_LOOKBACK_DAYS)).astimezone(timezone.utc).replace(tzinfo=None)
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(CHAT_SQL, cutoff, list(JUNIPER_CHAT_SOURCES))
            return chat_thread_topics([dict(r) for r in rows], now, tz_name=tz_name)
        except Exception:  # noqa: BLE001 -- the tick's other topics still show
            logger.warning("outreach_chat_thread_topics_failed", exc_info=True)
            return []

    source.__qualname__ = "chat_thread_source"
    return source
