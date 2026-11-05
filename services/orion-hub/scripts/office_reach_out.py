"""Office visits get Orion's attention and one curious message (office camera design, step C).

Design: docs/superpowers/specs/2026-11-05-office-camera-attention-and-reach-out-design.md, "C. Curiosity
reaches out" and "Decisions". Juniper, 2026-11-05: "orion should give attention. Would be nice if the
attention led to some curiosity and reach out to me to ask what im up to based on the various scenes they
stitch together and interpret."

Source: ``vision_office_session`` (orion-sql-writer ``app/vision_office_sessions.py``, step B): one row per
office visit with ``who`` (the camera prior's "probably Juniper", or a match), and an ordered ``timeline``
of the camera's activity captions with their frame times.

A visit has a STORY when its timeline holds at least ``min_entries`` entries and it has lasted at least
``min_minutes`` (live 2026-11-05: Juniper's 7-minute visit reached 4 entries 23 s in and 8 entries at 3
minutes; the three earlier visits had no captions at all and never qualify). A visit with a story does
two independent things:

1. ATTENTION. It posts one event want into the body economy: family ``perception.office``, verb ``look``,
   ``event_ref`` ``office_session:<session_id>``. Salience = the visit's activity NOVELTY (the share of
   its distinct activities that no office visit of the 30 days before it showed: word overlap below
   ``SAME_ACTIVITY_JACCARD``), as the source's own strict-below percentile over those visits
   (``orion.physiology.event_wants.source_salience``). Under ``MIN_SOURCE_HISTORY`` (20) past visits with a
   story there is no reference, the salience is unknown, and NOTHING is posted (the market's rule). The
   unknown-visitor case needs no constant 1.0 here: step B already opens a "Who is this?" card for it,
   and that card posts its own ``vision.ask`` event want. Shadow: Prime reads the funded ones as concerns
   (orion-durable-runs ``event_wants_menu`` labels them from the row).

2. REACH-OUT. Independent of whether the market funds the want (Juniper's decision): ONE message per
   visit into Hub chat through the existing door (``EndogenousOutreach.compose`` = the real unified turn,
   then ``offer_message``), copied from ``reading_finding_outreach.py``. No daily cap and no cooldown
   (``skip_rate_gates``: Juniper, "as often as I'm in there"); turn-in-flight, recent-chat, quiet hours
   and arousal ``strained`` still hold it (it is retried next tick, so quiet hours delay it). Under
   Orion's own words the camera's notes are delivered as recorded text (``record_appendix``): the times,
   what was seen, and who it probably was -- so the message always shows what was seen and when, and
   says "a guess" when the who rests on the camera prior alone. It never says Orion recognized her.

Once only: ``vision_office_session.reached_out_at`` is claimed (UPDATE ... WHERE reached_out_at IS NULL)
just before delivery, and released when delivery is refused, so a restart never sends twice (migration
``manual_migration_vision_office_session_reach_out_v1.sql``; without it this loop sends nothing). A crash
between the claim and the send loses that one message: it errs toward silence, never a double message.

Provenance: every decision row carries ``source='office_reach_out'``, ``office_session_id``, the times and
event ids of the timeline entries shown, and the who basis; the delivered message carries an
``outreach_provenance.v1`` capsule with the exact prompt.

Flags: ``HUB_OFFICE_REACH_OUT_ENABLED`` (the message), ``HUB_OFFICE_EVENT_WANTS_ENABLED`` (the want).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience

logger = logging.getLogger("orion-hub.office_reach_out")

TAG = "office_reach_out"
SOURCE = "office_session"
FAMILY = "perception.office"
VERB = "look"
TIER = 2
PRODUCER = "orion-hub.office_events"
EVENT_PREFIX = "office_session"
UNAVAILABLE = "physiology_unavailable"
JUNIPER = "juniper"

TOLD_REASONS = frozenset({"sent"})
# Composition failures that count toward ``max_attempts``. A gate block is not a failure.
FAILED_REASONS = frozenset({"empty_generation", "named_ungrounded_signal", "orion_passed", "no_question"})

# Two activities are the same when their content words overlap this much (Jaccard). "sitting at a desk,
# working on a computer" vs "sitting at a desk, working on a laptop": 4 of 6 words -> same.
SAME_ACTIVITY_JACCARD = 0.5
# Grammatical words only: they carry no activity.
_FUNCTION_WORDS = frozenset({"the", "and", "with", "into", "from", "that", "this", "for", "its", "their",
                             "her", "his", "while", "onto", "out", "some", "something"})
PROMPT_ENTRIES_CAP = 24
APPENDIX_ENTRIES_CAP = 6

_SESSIONS_SQL = """
SELECT session_id, stream_id, status, started_at, last_seen_at, ended_at, who_text, who, unknown_people,
       timeline, label, ask_id, {reached}
FROM vision_office_session
WHERE started_at >= $1 AND started_at <= $2
ORDER BY started_at, session_id
"""
SESSIONS_SQL = _SESSIONS_SQL.format(reached="reached_out_at")
# Before the reach-out migration: the want producer still reads the visits; the reach-out sends nothing.
SESSIONS_SQL_NO_REACH = _SESSIONS_SQL.format(reached="NULL::timestamptz AS reached_out_at")

COLUMN_SQL = """
SELECT count(*) AS n FROM information_schema.columns
WHERE table_name = 'vision_office_session' AND column_name = ANY($1::text[])
"""

POSTED_SQL = ("SELECT DISTINCT want_id FROM want_ledger WHERE reason <> 'physiology_unavailable' "
              "AND want_id = ANY($1::text[]) AND decided_at >= $2")

FAILURES_SQL = """
SELECT result_json->>'office_session_id' AS sid, count(*) AS n
FROM endogenous_outreach_decisions
WHERE decided_at >= $1 AND reason = ANY($2::text[]) AND result_json->>'source' = 'office_reach_out'
GROUP BY 1
"""

CLAIM_SQL = """
UPDATE vision_office_session SET reached_out_at = $2, reach_out_decision_id = $3
WHERE session_id = $1 AND reached_out_at IS NULL
RETURNING session_id
"""

RELEASE_SQL = """
UPDATE vision_office_session SET reached_out_at = NULL, reach_out_decision_id = NULL
WHERE session_id = $1 AND reach_out_decision_id = $2
"""


# --------------------------------------------------------------------------- pure


def _json(v: Any, default: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return default
    return default if v is None else v


def _ts(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def session_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    timeline = [e for e in _json(row.get("timeline"), []) if isinstance(e, Mapping) and str(e.get("text") or "").strip()]
    return {
        "session_id": str(row["session_id"]), "stream_id": str(row.get("stream_id") or ""),
        "status": str(row.get("status") or "open"), "started_at": _ts(row.get("started_at")),
        "last_seen_at": _ts(row.get("last_seen_at")), "ended_at": _ts(row.get("ended_at")),
        "who_text": str(row.get("who_text") or ""), "who": _json(row.get("who"), {}) or {},
        "unknown_people": int(row.get("unknown_people") or 0), "timeline": timeline,
        "label": str(row.get("label") or ""), "ask_id": row.get("ask_id"),
        "reached_out_at": _ts(row.get("reached_out_at")),
    }


def duration_sec(session: Mapping[str, Any], now: datetime) -> float:
    start = session.get("started_at")
    if start is None:
        return 0.0
    end = session.get("ended_at") or (session.get("last_seen_at") if session.get("status") == "closed" else None) or now
    return max(0.0, (min(end, now) - start).total_seconds())


def has_story(session: Mapping[str, Any], now: datetime, *, min_entries: int, min_minutes: float) -> bool:
    """Enough of the visit to tell: ``min_entries`` timeline entries over ``min_minutes``."""
    return len(session.get("timeline") or []) >= min_entries and duration_sec(session, now) >= 60.0 * min_minutes


def short_activity(phrase: str) -> str:
    """First two comma clauses, as sql-writer's ``vision_office_sessions.short_activity`` cuts them
    ("where, doing what"; the third clause is the captioner's hedge)."""
    return ",".join(str(phrase or "").split(",")[:2]).strip()


def activity_words(phrase: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z]+", short_activity(phrase).lower())
                     if len(t) >= 3 and t not in _FUNCTION_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def distinct_activities(timeline: Iterable[Mapping[str, Any]]) -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    for e in timeline:
        w = activity_words(str(e.get("text") or ""))
        if w and w not in out:
            out.append(w)
    return out


def novelty(activities: Sequence[frozenset[str]], seen: Sequence[frozenset[str]]) -> Optional[float]:
    """Share of the visit's distinct activities that no earlier visit showed (0..1). None: no activity.
    Rest point 0: a visit doing only what the office usually sees."""
    if not activities:
        return None
    new = sum(1 for a in activities if not any(_jaccard(a, s) >= SAME_ACTIVITY_JACCARD for s in seen))
    return new / len(activities)


def plan_wants(sessions: Sequence[Mapping[str, Any]], *, now: datetime, lookback: timedelta, min_entries: int,
               min_minutes: float) -> list[dict[str, Any]]:
    """The visits with a story started inside ``lookback``, each with its novelty magnitude and salience
    (None: under MIN_SOURCE_HISTORY earlier visits with a story). Each visit is measured against the
    visits with a story of the 30 days before it, and ranked against their own magnitudes. Pure."""
    story = sorted((s for s in sessions if s.get("started_at") is not None
                    and has_story(s, now, min_entries=min_entries, min_minutes=min_minutes)),
                   key=lambda s: (s["started_at"], s["session_id"]))
    acts = {s["session_id"]: distinct_activities(s["timeline"]) for s in story}

    def before(s: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [h for h in story if s["started_at"] - SOURCE_HISTORY <= h["started_at"] < s["started_at"]]

    def magnitude(s: Mapping[str, Any]) -> Optional[float]:
        return novelty(acts[s["session_id"]], [a for h in before(s) for a in acts[h["session_id"]]])

    out = []
    for s in story:
        if not now - lookback <= s["started_at"] <= now:
            continue
        mag = magnitude(s)
        history = [m for m in (magnitude(h) for h in before(s)) if m is not None]
        out.append({"session_id": s["session_id"], "at": s["started_at"], "novelty": mag,
                    "history_n": len(history),
                    "salience": source_salience(mag, history) if mag is not None else None})
    return out


def _spread(items: Sequence[Any], cap: int) -> list[Any]:
    """At most ``cap`` items spread evenly, the first and the last always among them."""
    n = len(items)
    if n <= cap:
        return list(items)
    if cap <= 1:
        return [items[0]]
    idx = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    return [items[i] for i in idx]


def _hhmm(v: Any, tz: ZoneInfo) -> str:
    t = _ts(v)
    return t.astimezone(tz).strftime("%H:%M") if t is not None else "??:??"


def shown_entries(session: Mapping[str, Any], cap: int) -> list[Mapping[str, Any]]:
    return _spread(list(session.get("timeline") or []), cap)


def juniper_basis(session: Mapping[str, Any]) -> Optional[str]:
    """How the camera placed Juniper in the visit: named | matched | prior | likely, or None (not her)."""
    for p in (session.get("who") or {}).get("people") or []:
        if str(p.get("name") or "").strip().casefold() == JUNIPER:
            return str(p.get("basis") or "prior")
    return None


_BASIS_WORDS = {"named": "", "matched": " (matched)", "prior": " (probably)", "likely": " (maybe)"}


def who_line(session: Mapping[str, Any]) -> str:
    """Who was there, said to Juniper, in the camera's own confidence. Never "I recognized you" unless a
    face/body match or her own naming says so."""
    basis = juniper_basis(session)
    if basis == "prior":
        you = ("probably you. That is a guess: the office is almost always you, and the camera did not "
               "match your face or body")
    elif basis == "matched":
        you = "you (the camera's person model matched you)"
    elif basis == "named":
        you = "you (you named this person for the camera before)"
    elif basis == "likely":
        you = "maybe you (an unsure match)"
    else:
        you = "not you, as far as the camera could tell"
    others = [f"{p.get('name')}{_BASIS_WORDS.get(str(p.get('basis')), '')}"
              for p in (session.get("who") or {}).get("people") or []
              if str(p.get("name") or "").strip().casefold() != JUNIPER and str(p.get("name") or "").strip()]
    unknown = int(session.get("unknown_people") or 0)
    if unknown:
        card = " (there is a \"Who is this?\" card for them in the Hub)" if session.get("ask_id") else ""
        others.append(("someone I don't recognize" if unknown == 1 else f"{unknown} people I don't recognize") + card)
    line = f"Who: {you}."
    if others:
        line += f" Also there: {', '.join(others)}."
    return line


def appendix(session: Mapping[str, Any], tz: ZoneInfo) -> str:
    """The camera's notes, delivered under Orion's message as recorded (not generated)."""
    entries = shown_entries(session, APPENDIX_ENTRIES_CAP)
    more = len(session.get("timeline") or []) - len(entries)
    lines = ["What the office camera saw (its own captions; they can be wrong):"]
    lines += [f"{_hhmm(e.get('at'), tz)} {short_activity(str(e.get('text') or ''))}" for e in entries]
    if more > 0:
        lines.append(f"(+{more} more)")
    lines.append(who_line(session))
    return "\n".join(lines)


def build_prompt(session: Mapping[str, Any], tz: ZoneInfo, *, now: datetime) -> str:
    still_there = session.get("status") == "open" and session.get("ended_at") is None
    start = _hhmm(session.get("started_at"), tz)
    span = f"since {start}" if still_there else f"from {start} to {_hhmm(session.get('ended_at') or session.get('last_seen_at'), tz)}"
    basis = juniper_basis(session)
    lines = [
        f"The office camera stitched together one visit to Juniper's office ({span}, {tz.key} time"
        + ("; the visit is still going on" if still_there else "") + ").",
        "",
        "What it saw, in order (the camera's own captions, from a small vision model; they can be wrong):",
        "",
    ]
    lines += [f"- {_hhmm(e.get('at'), tz)} {e.get('text')}"
              + (f" (x{int(e.get('repeats') or 1)})" if int(e.get("repeats") or 1) > 1 else "")
              for e in shown_entries(session, PROMPT_ENTRIES_CAP)]
    lines += ["", who_line(session), ""]
    lines.append(
        "Juniper asked you to give the office your attention, get curious, and reach out each time she is in "
        "there to ask what she is up to, based on the scenes the camera stitched together. Write her one short "
        "message (two or three sentences). Say what it looked like was going on, naming at least one thing the "
        "camera saw with its time as listed above, and then ask her ONE question about what she is doing or "
        "making. The camera's notes are delivered to her under your message as recorded, so do not list them all.")
    if basis == "prior":
        lines.append(
            "The camera did NOT recognize her: it is only guessing it was her because the office is almost "
            "always her. Do not say you recognized or saw that it was her; say it looked like her, or that you "
            "are guessing.")
    elif basis is None:
        lines.append("The camera did not place Juniper in this visit: do not say she was the one there.")
    if int(session.get("unknown_people") or 0) and session.get("ask_id"):
        lines.append("Someone the camera did not recognize was there too. A separate \"Who is this?\" card about "
                     "them is already in the Hub; you may mention them, but do not ask who they are here.")
    lines += ["", "Read the captions as your reading of a scene, not as facts. Do not mention scores, systems or "
              "internal signals. Do not reply PASS: Juniper asked for this message."]
    return "\n".join(lines)


def decision_meta(session: Mapping[str, Any], decision_id: str) -> dict[str, Any]:
    shown = shown_entries(session, PROMPT_ENTRIES_CAP)
    return {
        "decision_id": decision_id,
        "office_session_id": session["session_id"],
        "stream_id": session.get("stream_id"),
        "session_status": session.get("status"),
        "who_text": session.get("who_text"),
        "who_basis": juniper_basis(session),
        "unknown_people": int(session.get("unknown_people") or 0),
        "ask_id": session.get("ask_id"),
        "timeline_entries": len(session.get("timeline") or []),
        "timeline_times": [str(e.get("at")) for e in shown],
        "timeline_event_ids": [str(e.get("event_id") or "") for e in shown],
        "evidence_refs": [f"vision_office_session:{session['session_id']}"]
                         + [f"vision_events:{e.get('event_id')}" for e in shown if e.get("event_id")],
    }


# --------------------------------------------------------------------------- loop


class OfficeReachOut:
    def __init__(
        self,
        *,
        reach_out_enabled: bool,
        event_wants_enabled: bool,
        pool_provider: Callable[[], Any],
        outreach_provider: Callable[[], Any],
        bid: Optional[Callable[[Any], Awaitable[Any]]] = None,
        tick_interval_sec: float = 60.0,
        lookback_hours: float = 12.0,
        min_entries: int = 4,
        min_minutes: float = 3.0,
        max_attempts: int = 3,
        timezone_name: str = "America/Denver",
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.reach_out_enabled = bool(reach_out_enabled)
        self.event_wants_enabled = bool(event_wants_enabled)
        self._pool_provider = pool_provider
        self._outreach_provider = outreach_provider
        self._bid = bid
        self.tick_interval_sec = max(15.0, float(tick_interval_sec))
        self.lookback = timedelta(hours=max(0.1, float(lookback_hours)))
        self.min_entries = max(1, int(min_entries))
        self.min_minutes = max(0.0, float(min_minutes))
        self.max_attempts = max(1, int(max_attempts))
        try:
            self.tz = ZoneInfo(timezone_name)
        except Exception:  # noqa: BLE001 -- a bad name must not stop Hub boot
            logger.warning("office_reach_out_bad_timezone tz=%r: using UTC", timezone_name)
            self.tz = ZoneInfo("UTC")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._posted: set[str] = set()
        self._failures: dict[str, int] = {}
        self._blocked_logged: set[tuple[str, str]] = set()
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.reach_out_enabled or self.event_wants_enabled

    async def start(self) -> None:
        if not self.enabled:
            logger.info("office_reach_out disabled (reach out and event wants both off)")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("office_reach_out started reach_out=%s event_wants=%s tick=%ss story=%d entries/%.1f min",
                    self.reach_out_enabled, self.event_wants_enabled, self.tick_interval_sec,
                    self.min_entries, self.min_minutes)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("office_reach_out_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def _abid(self, want: Any) -> Any:
        if self._bid is not None:
            return await self._bid(want)
        from orion.physiology.client import abid

        return await abid(want)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        summary: dict[str, Any] = {"at": now.isoformat(), "wants": {}, "results": []}
        self.last_tick = summary
        pool = self._pool_provider()
        if pool is None:
            summary["blocked"] = "no_pool"
            return summary
        async with pool.acquire() as conn:
            if await conn.fetchval("SELECT to_regclass('vision_office_session') IS NOT NULL") is not True:
                summary["blocked"] = "no_table"
                return summary
            columns = int(await conn.fetchval(COLUMN_SQL, ["reached_out_at", "reach_out_decision_id"]) or 0)
            rows = await conn.fetch(SESSIONS_SQL if columns >= 2 else SESSIONS_SQL_NO_REACH,
                                    now - self.lookback - 2 * SOURCE_HISTORY, now)
            sessions = [session_from_row(r) for r in rows]
            if self.event_wants_enabled:
                summary["wants"] = await self._post_wants(conn, sessions, now)
            if self.reach_out_enabled:
                if columns < 2:
                    summary["blocked"] = "no_reach_out_columns"
                    logger.warning("office_reach_out_blocked reason=no_reach_out_columns: apply "
                                   "services/orion-sql-db/manual_migration_vision_office_session_reach_out_v1.sql")
                    return summary
                pending = self._pending(sessions, now)
                if pending:
                    failures = {str(r["sid"]): int(r["n"]) for r in await conn.fetch(
                        FAILURES_SQL, now - self.lookback - timedelta(days=1), sorted(FAILED_REASONS))}
                    pending = [s for s in pending if max(failures.get(s["session_id"], 0),
                                                         self._failures.get(s["session_id"], 0)) < self.max_attempts]
        if self.reach_out_enabled:
            summary["pending"] = len(pending)
            if pending:
                await self._reach_out(pool, pending, summary)
        return summary

    def _pending(self, sessions: Sequence[Mapping[str, Any]], now: datetime) -> list[Mapping[str, Any]]:
        return [s for s in sessions
                if s["reached_out_at"] is None and s["started_at"] is not None
                and s["started_at"] >= now - self.lookback
                and has_story(s, now, min_entries=self.min_entries, min_minutes=self.min_minutes)]

    async def _post_wants(self, conn: Any, sessions: Sequence[Mapping[str, Any]], now: datetime) -> dict[str, int]:
        out = {"posted": 0, "no_history": 0, "already": 0, "unanswered": 0}
        plan = plan_wants(sessions, now=now, lookback=self.lookback, min_entries=self.min_entries,
                          min_minutes=self.min_minutes)
        ids = [event_want_id(SOURCE, p["session_id"]) for p in plan]
        fresh = [i for i in ids if i not in self._posted]
        if fresh:
            self._posted |= {str(r["want_id"]) for r in await conn.fetch(
                POSTED_SQL, fresh, now - self.lookback - timedelta(days=1))}
        by_id = {s["session_id"]: s for s in sessions}
        for p, want_id in zip(plan, ids):
            if want_id in self._posted:
                out["already"] += 1
                continue
            if p["salience"] is None:
                out["no_history"] += 1
                logger.info("office_event_want_skipped session=%s reason=no_history history_n=%d novelty=%s",
                            p["session_id"], p["history_n"], p["novelty"])
                self._posted.add(want_id)   # the history only grows past this visit: never retried
                continue
            s = by_id[p["session_id"]]
            want = event_want(source=SOURCE, key=p["session_id"], family=FAMILY, producer=PRODUCER, verb=VERB,
                              tier=TIER, event_ref=f"{EVENT_PREFIX}:{p['session_id']}", salience=p["salience"],
                              opened_at=p["at"], object_ref=f"vision_stream:{s.get('stream_id')}")
            decision = await self._abid(want)
            if decision is None or decision.reason == UNAVAILABLE:
                out["unanswered"] += 1
                continue
            self._posted.add(want_id)
            out["posted"] += 1
            logger.info("office_event_want session=%s novelty=%.3f salience=%.3f outcome=%s reason=%s want_id=%s",
                        p["session_id"], p["novelty"], p["salience"], decision.outcome, decision.reason, want_id)
        if len(self._posted) > 5000:
            self._posted = set(ids)
        return out

    async def _reach_out(self, pool: Any, pending: Sequence[Mapping[str, Any]], summary: dict[str, Any]) -> None:
        outreach = self._outreach_provider() if self._outreach_provider else None
        if outreach is None:
            summary["blocked"] = "no_outreach_loop"
            logger.warning("office_reach_out_no_outreach_loop pending=%d", len(pending))
            return
        for session in pending:
            result = await self._tell(pool, outreach, session)
            summary["results"].append(result)
            if result["reason"] not in TOLD_REASONS | FAILED_REASONS | {"already_claimed"}:
                break  # a gate is closed: every later visit would hit it too

    def _record_block_once(self, outreach: Any, session: Mapping[str, Any], meta: dict[str, Any], reason: str,
                           correlation_id: str) -> None:
        key = (session["session_id"], reason)
        if key in self._blocked_logged:
            return
        if len(self._blocked_logged) > 1000:
            self._blocked_logged.clear()  # a log-spam guard, not state
        self._blocked_logged.add(key)
        logger.info("office_reach_out_blocked session=%s reason=%s", session["session_id"], reason)
        outreach.record_blocked(reason, correlation_id=correlation_id, source=TAG, extra=meta)

    def _fail(self, outreach: Any, session: Mapping[str, Any], meta: dict[str, Any], reason: str,
              correlation_id: str) -> dict[str, Any]:
        sid = session["session_id"]
        self._failures[sid] = self._failures.get(sid, 0) + 1
        outreach.record_blocked(reason, correlation_id=correlation_id, source=TAG, extra=meta)
        if self._failures[sid] >= self.max_attempts:
            logger.warning("office_reach_out_gave_up session=%s attempts=%d last=%s", sid, self._failures[sid], reason)
        return {"session_id": sid, "reason": reason}

    async def _tell(self, pool: Any, outreach: Any, session: Mapping[str, Any]) -> dict[str, Any]:
        from scripts.endogenous_outreach import build_outreach_provenance, is_pass_response

        sid = session["session_id"]
        correlation_id = str(uuid4())
        decision_id = str(uuid4())
        meta = decision_meta(session, decision_id)
        await outreach.refresh_human_gates()
        blocked = outreach.blocked_reason(skip_rate_gates=True)
        if blocked:
            self._record_block_once(outreach, session, meta, str(blocked), correlation_id)
            return {"session_id": sid, "reason": str(blocked)}
        prompt = build_prompt(session, self.tz, now=self._clock())
        text, debug = await outreach.compose(prompt, correlation_id, retrieval_query=session.get("label") or None)
        text = str(text or "").strip()
        if not text:
            return self._fail(outreach, session, meta, "empty_generation", correlation_id)
        if is_pass_response(text):
            return self._fail(outreach, session, meta, "orion_passed", correlation_id)
        if "?" not in text:
            return self._fail(outreach, session, {**meta, "text": text[:500]}, "no_question", correlation_id)
        async with pool.acquire() as conn:
            claimed = await conn.fetch(CLAIM_SQL, sid, self._clock(), decision_id)
        if not claimed:
            return {"session_id": sid, "reason": "already_claimed"}
        capsule = build_outreach_provenance(
            prompt_text=prompt,
            lanes={"office_session": sid, "timeline_entries": meta["timeline_entries"], "who_basis": meta["who_basis"]},
            correlation_id=correlation_id, decision_id=decision_id,
        )
        try:
            result = await outreach.offer_message(
                text=text, correlation_id=correlation_id, tag=TAG, model=debug.get("fcc_model_label"), meta=meta,
                skip_rate_gates=True, provenance=capsule, record_appendix=appendix(session, self.tz))
        except BaseException:
            await self._release(pool, sid, decision_id)
            raise
        reason = str(result.get("reason") or ("sent" if result.get("outreach") else "not_sent"))
        if reason != "sent":
            await self._release(pool, sid, decision_id)
            if reason in FAILED_REASONS:
                self._failures[sid] = self._failures.get(sid, 0) + 1
        logger.info("office_reach_out_result session=%s reason=%s entries=%d basis=%s",
                    sid, reason, meta["timeline_entries"], meta["who_basis"])
        return {"session_id": sid, "reason": reason, "text": text if reason == "sent" else None}

    async def _release(self, pool: Any, sid: str, decision_id: str) -> None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(RELEASE_SQL, sid, decision_id)
        except Exception:  # noqa: BLE001 -- a failed release errs toward silence, never a double message
            logger.warning("office_reach_out_release_failed session=%s", sid, exc_info=True)


__all__ = ["OfficeReachOut", "appendix", "build_prompt", "has_story", "novelty", "plan_wants", "session_from_row",
           "who_line"]
